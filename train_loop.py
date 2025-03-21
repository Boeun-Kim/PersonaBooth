import torch
import torch.nn as nn
from torch.optim import AdamW
import os

from model.PersonaBooth import Personality
from logger import logger

class TrainPR(nn.Module):

    def __init__(self, args, mdm_args, device='gpu'):
        super().__init__()
        self.model = Personality(args, mdm_args, device=device).to(device)
        self.model.train()

        self.epoch = 0
        self.batch_size = args.batch_size
        self.save_interval = args.save_interval
        self.save_dir = args.checkpoint_dir

        self.model_params = list(self.model.parameters())
        proj_params = (
                    list(self.model.pr_token_proj.parameters()) +
                    list(self.model.pr_extractor.parameters())
                    )
        
        Adapt_params = []
        for layer in self.model.mdm.seqTransEncoder.layers:

            for p in layer.norm_ada.parameters():
                Adapt_params.append(p)

            for p in layer.self_attn_adapt.parameters():
                Adapt_params.append(p)

            Adapt_params.append(layer.gamma)

        Adapt_params.append(self.model.txt_gamma)
        
        self.optimizer = AdamW([
            {'params': proj_params, 'lr': args.lr_proj},
            {'params': Adapt_params, 'lr': args.lr_Adapt}
        ])

        self.layerw_to_save = ['pr_token_proj', 'pr_extractor', 'txt_gamma']
        self.layerw_to_save_mdm = ['self_attn_adapt', 'norm_ada', 'gamma']

        self.step = 0

        self.load_checkpoint(args.checkpoint_path)


    def run_step(self, input_motions, gt_text, gt_motion, gt_motion_mask, epoch):

        self.epoch = epoch
        self.optimizer.zero_grad()
        
        self.loss, self.pc_loss = self.model(input_motions, gt_text, gt_motion, gt_motion_mask)
        self.loss.backward()
        self.optimizer.step()

        self.log_step()
        self.step += 1


    def save_checkpoint(self):

        state_dict = self.model.state_dict()
        save_state_dict = {k: v for k, v in state_dict.items() if any(k.startswith(layer) 
                            for layer in self.layerw_to_save)}
        save_state_dict_mdm = {k: v for k, v in state_dict.items() 
                            if k.startswith("mdm") and any(substring in k for substring in self.layerw_to_save_mdm)}
        save_state_dict.update(save_state_dict_mdm)

        # Save checkpoint
        checkpoint = {
            'param' : save_state_dict,
            'optimizer' : self.optimizer.state_dict(),
            'step' : self.step
        }
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        path = os.path.join(self.save_dir, f"{(self.epoch):04d}.pt")
        torch.save(checkpoint, path)
 

    def load_checkpoint(self, path):
        if os.path.exists(path):
            logger.info("Load saved checkpoint-%s", path)
            print("Load saved checkpoint.", path)
            checkpoint = torch.load(path)
            self.step = checkpoint['step'] + 1
            self.model.load_state_dict(checkpoint['param'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.step = 0
            

    def zero_grad(self, model_params):
        for param in model_params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()


    def log_step(self):
        sample = (self.step + 1) * self.batch_size
        logger.info("epoch %d step %d samples %d - loss : %f, pc_loss : %f", self.epoch, self.step, sample, self.loss, self.pc_loss)
        #print("epoch %d step %d samples %d - loss : %f, pc_loss : %f" % (self.epoch ,self.step, sample, self.loss, self.pc_loss))

