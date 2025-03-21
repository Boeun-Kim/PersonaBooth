import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import sys
sys.path.append('./dependency/TMR')
sys.path.append('./dependency/MDM')


from dependency.TMR.src.config import read_config
from dependency.TMR.src.load import load_model_from_cfg
from dependency.TMR.src.data.text_CLIP import TokenEmbeddings
from dependency.TMR.src.data.collate import collate_x_dict, collate_x_dict_both

from dependency.MDM.utils.parser_util import train_args, generate_args
from dependency.MDM.utils.dist_util import setup_dist
from dependency.MDM.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from dependency.MDM.diffusion.resample import create_named_schedule_sampler, LossAwareSampler
from dependency.MDM.model.cfg_sampler import ClassifierFreeSampleModel
from model.loss import contrastive_loss

class Personality(nn.Module):
    def __init__(self, args, mdm_args, device='gpu', is_train=True):
        super().__init__()
        
        self.pass_args = mdm_args
        self.device = device
        self.is_train = is_train
        self.mdm_path = args.MDM_path

        # CFG params
        if is_train:
            self.token_renorm = 1.0
            self.motion_renorm = 1.0
            self.w_txt = 1.0
            self.guide_motion = 1.0
            self.guide_txt = 1.0
        else:
            self.token_renorm = args.token_renorm
            self.motion_renorm = args.motion_renorm
            self.w_txt = args.w_txt
            self.guide_motion = args.guide_motion
            self.guide_txt = args.guide_txt

        # Load pretrained motion clip (TMR)
        self.motion_clip, self.token_emb = self.load_motion_clip(args.TMR_config)

        # Projections
        self.clip_out_dim = 512
        self.token_emb_dim = 512
        self.cl_dim = 128  # cl: contrastive learning (persona cohesion loss)
        self.pr_token_proj = MLP(self.clip_out_dim, self.token_emb_dim, dropout=args.drop_out)
        self.cl_head = MLP(self.clip_out_dim, self.cl_dim, dropout=args.drop_out)

        # Persona extractor
        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.clip_out_dim,
            nhead=4,
            dim_feedforward=self.clip_out_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.pr_extractor = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=1
        )

        # Gamma_t
        self.txt_gamma = nn.Parameter(torch.zeros(1))

        # Load pretrained diffusion (MDM)
        self.max_frame = 196
        self.mdm, self.diffusion = self.load_motion_diffusion()
        self.schedule_sampler = create_named_schedule_sampler('uniform', self.diffusion)


    def load_motion_clip(self, cfg):
        print('Loading motion clip...')

        # Init and load motion clip
        clip_cfg = read_config(cfg)
        motion_clip = load_model_from_cfg(clip_cfg, 'last', eval_mode=False, device=self.device)
        token_emb = TokenEmbeddings(modelname='', clip_model=motion_clip.clip_model, device=self.device, 
                                    preload=False, is_train=self.is_train, token_renorm=self.token_renorm)
        
        # Freeze motion clip
        for p in motion_clip.parameters():
            p.requires_grad = False

        return motion_clip, token_emb


    def load_motion_diffusion(self):
        print('Loading motion diffusion...')
        
        if self.is_train == True:
            mdm_args = train_args(self.pass_args)  
        else:
            mdm_args = generate_args(self.mdm_path, self.pass_args)
        mdm_args.dataset = 'humanml'
        mdm_args.diffusion_steps = 50

        setup_dist(self.device)
        mdm, diffusion = create_model_and_diffusion(mdm_args, self.is_train, motion_renorm=self.motion_renorm)
        mdm.to(self.device)

        # Load pretrained params
        state_dict = torch.load(self.mdm_path)
        load_model_wo_clip(mdm, state_dict)

        # Freeze mdm
        for p in mdm.parameters():
            p.requires_grad = False
        
        # Unfreeze adaptive layer in motion diffusion model 
        for layer in mdm.seqTransEncoder.layers:

            for p in layer.norm_ada.parameters():
                p.requires_grad = True

            for p in layer.self_attn_adapt.parameters():
                p.requires_grad = True
                
            layer.gamma.requires_grad = True
            
        # Classifier-Free Guidance 
        if not self.is_train:
            mdm = ClassifierFreeSampleModel(mdm, self.device, self.w_txt, self.guide_motion, self.guide_txt)
        
        return mdm, diffusion
        

    def run_diffusion(self, text_feat, viz_feat, num_batch, gen_frame=None, gen_shape=None, viz_mask=None):

        sample_fn = self.diffusion.p_sample_loop

        if gen_frame == None:
            gen_frame = self.max_frame
        else:
            gen_frame = min(self.max_frame, gen_frame)

        if gen_shape == None:
            gen_shape = (num_batch, self.mdm.njoints, self.mdm.nfeats, gen_frame)
       
        sample = sample_fn(
            self.mdm,
            gen_shape,
            clip_denoised=False,
            model_kwargs={'text':text_feat, 'viz': viz_feat, 'viz_mask': viz_mask},
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        return sample

    def run_diffusion_get_loss(self, motion, mask, text_feat, viz_feat, viz_mask=None):
        
        t, weights = self.schedule_sampler.sample(motion.shape[0], self.device)

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.mdm,
            motion,
            mask,
            t,
            model_kwargs={'text':text_feat, 'viz': viz_feat, 'viz_mask': viz_mask}
        )
        losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()

        return loss


    def forward(self, input_motions, gt_text, gt_motion, gt_motion_mask):

        batch_size = len(input_motions)
        batch_motions = self.merge_batch(input_motions)
        motion_x_dict = collate_x_dict_both(batch_motions, device=self.device)

        # Motion Clip
        with torch.no_grad():
            clip_feats, clip_feats_whole = self.motion_clip.encode(motion_x_dict, sample_mean=True, modality='motion')

        # Persona Extractor
        mask = motion_x_dict["mask"]
        token_mask = torch.ones((len(motion_x_dict["mask"]), 1), dtype=bool, device=self.device)
        aug_mask = torch.cat((token_mask, mask), 1)   
        pr_feat = self.pr_extractor(clip_feats_whole, src_key_padding_mask=~aug_mask)
            # Persona cohesion loss
        cl_feat = self.cl_head(pr_feat[:,0,:])
        pc_loss = contrastive_loss(cl_feat)

        # Project output features into a persona token feature (P*) and a visual persona feature (V*)
        pr_feat = pr_feat[:batch_size]
        pr_viz_feat = pr_feat
        pr_token_feat = self.pr_token_proj(pr_feat[:batch_size,0,:])

        # Personalized Text Encoder
        token_feats = self.token_emb(gt_text, rm_pr_token=False, pr_feat=pr_token_feat, gamma=self.txt_gamma)
        text_feats = self.motion_clip.encode(token_feats, sample_mean=True, modality='text')[0]
     
        # Motion Diffusion
        gt_motion = torch.stack(gt_motion)
        gt_motion = torch.permute(gt_motion, (0, 2, 1))
        gt_motion = gt_motion.unsqueeze(2).to(self.device)
        gt_motion.requires_grad = True
        gt_motion_mask = torch.stack(gt_motion_mask)
        gt_motion_mask = gt_motion_mask.unsqueeze(1).unsqueeze(1).to(self.device)
        loss = self.run_diffusion_get_loss(gt_motion, gt_motion_mask, text_feats, pr_viz_feat)

        return loss + 0.01*pc_loss, pc_loss
 

    def inference(self, input_motions, num_samples, input_texts, gen_frame=None, gen_shape=None):
        
        # Merge several motions in each batch into one batch
        batch_size = len(input_texts)
        batch_motions = self.merge_batch(input_motions)
        motion_x_dict = collate_x_dict(batch_motions, device=self.device)

        # Motion Clip
        with torch.no_grad():
            clip_feats, clip_feats_whole = self.motion_clip.encode(motion_x_dict, sample_mean=True, modality='motion')

        # Persona Extractor
        mask = motion_x_dict["mask"]
        token_mask = torch.ones((len(motion_x_dict["mask"]), 1), dtype=bool, device=self.device)
        aug_mask = torch.cat((token_mask, mask), 1)   
        pr_feat = self.pr_extractor(clip_feats_whole, src_key_padding_mask=~aug_mask)

        # Project output features into a persona token feature (P*) and a visual persona feature (V*)
        pr_viz_feat = pr_feat
        pr_token_feat = self.pr_token_proj(pr_feat[:,0,:])

        pr_token_feats_l = torch.split(pr_token_feat, num_samples)
        pr_viz_feats_l = torch.split(pr_viz_feat, num_samples)
        clip_feats_l = torch.split(clip_feats, num_samples)
        mask_l = torch.split(aug_mask, num_samples)

        # Context Aware Fusion
        pr_token_feat_sums = []
        pr_viz_feat_sums = []
        masks = []
        for i, input_text in enumerate(input_texts):
            # Encode text by CLIP
            input_text_l = [input_text] * num_samples[i]
            token_feat = self.token_emb(input_text_l, rm_pr_token=False, pr_feat=pr_token_feats_l[i], gamma=self.txt_gamma)
            text_feat, _ = self.motion_clip.encode(token_feat, sample_mean=True, modality='text')
            
            # Extract top-k and their wieghts
            even_weight = (torch.ones(text_feat.shape[0])/text_feat.shape[0]).to(self.device)
            pr_token_feat_mean = torch.matmul(even_weight, pr_token_feats_l[i]).unsqueeze(0)
            input_token_feat = self.token_emb([input_text], rm_pr_token=False, pr_feat=pr_token_feat_mean, gamma=self.txt_gamma)
            input_text_feat, _ = self.motion_clip.encode(input_token_feat, sample_mean=True, modality='text')
            
            weight = get_topk_weights(input_text_feat, clip_feats_l[i], k=5)  
 
            # Linear combination
            pr_token_feat_sum = torch.matmul(weight, pr_token_feats_l[i]).unsqueeze(0)
            weighted_tensor = pr_viz_feats_l[i] * weight[:, None, None]
            pr_viz_feat_sum = weighted_tensor.sum(dim=0).unsqueeze(0)
            mask = max(mask_l[i], key=sum).unsqueeze(0)

            pr_token_feat_sums.append(pr_token_feat_sum)
            pr_viz_feat_sums.append(pr_viz_feat_sum)
            masks.append(mask)

        pr_token_feat_sums = torch.cat(pr_token_feat_sums, dim=0)
        pr_viz_feat_sums = torch.cat(pr_viz_feat_sums, dim=0)
        masks = torch.cat(masks, dim=0)

        token_feats_sums = self.token_emb(input_texts, rm_pr_token=False, pr_feat=pr_token_feat_sums, gamma=self.txt_gamma)
        text_feats_sums = self.motion_clip.encode(token_feats_sums, sample_mean=True, modality='text')[0]

        # Motion Diffusion
        gen_motion = self.run_diffusion(text_feats_sums, pr_viz_feat_sums, batch_size, gen_frame, viz_mask=masks)

        return gen_motion


    def merge_batch(self, lst):
        merged_list = []

        for ls in lst:
            for l in ls:
                merged_list.append(l)

        return merged_list


class MLP(nn.Module):
    def __init__(self, intput_dim, output_dim, dropout):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        hidden_size = output_dim
        self.fc1 = nn.Linear(intput_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        preds = self.relu(self.fc3(x))
        return preds


def get_topk_weights(x, y, k=1):

    if y.shape[0] < k:
        k = y.shape[0]

    scaling_factor = 1

    cos_sim = F.cosine_similarity(x, y)
    top_k_values, top_k_indices = torch.topk(cos_sim, k)

    top_k_weights = torch.exp(top_k_values * scaling_factor)
    top_k_weights /= top_k_weights.sum()

    weights = torch.zeros_like(cos_sim)
    weights[top_k_indices] = top_k_weights

    return weights

