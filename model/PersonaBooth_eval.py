import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import functools
import numpy as np
from tqdm import tqdm
import re
#import spacy
import random

import sys
sys.path.append('./dependency/TMR')
sys.path.append('./dependency/MDM')

from feeder.loader_PerMo import PerMoDataset_eval
from model.PersonaBooth import Personality
from arguments_PerMo import parse_args_test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(seed=123):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed()

def collate(batch):

    input_motions = [item[0] for item in batch]
    num_sample = [item[1] for item in batch]


    return input_motions, num_sample

class PersonalityEval(Dataset):

    def __init__(self, args, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, multi_input, num_input):
        self.args = args
        self.dataloader = dataloader
        self.dataset = dataloader.dataset

        pr_args, _ = parse_args_test()
        pr_dataset = PerMoDataset_eval(pr_args.TMR_stats, pr_args.MDM_stats, pr_args.text_path, 
                                pr_args.guofeat_path, pr_args.crop_ratio, multi_input=multi_input, num_input=num_input)

        if multi_input:
            weights = [1/len(pr_dataset)] * len(pr_dataset)
            sampler = WeightedRandomSampler(weights, num_samples=pr_args.batch_size, replacement=True)

            pr_dataloader = DataLoader(
                dataset=pr_dataset,
                batch_size=pr_args.batch_size,
                pin_memory=True,
                num_workers= 0,
                worker_init_fn=init_seed,
                collate_fn=collate,
                sampler = sampler
            )

        else:
            pr_dataloader = DataLoader(
                dataset=pr_dataset,
                batch_size=pr_args.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers= 0,
                drop_last=True,
                worker_init_fn=init_seed,
                collate_fn=collate
            )
        pr_dataloader_iter = iter(pr_dataloader)

        assert mm_num_samples < len(dataloader.dataset)
        self.max_motion_length = max_motion_length


        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        ## Load model #####################
        args, mdm_args = parse_args_test()
        model = Personality(args, mdm_args, device=device, is_train=False)

        if args.checkpoint_path != '':
            print("Load model from: ", args.checkpoint_path)
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['param'], strict=False)

        model.to(device)
        model.eval()
        ####################################

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)): # HumanML3D data loop. Use only text from HumanML3D data

                try:
                    pr_motions, num_samples  = next(pr_dataloader_iter)
                except StopIteration:  #if end of pr_data_iterator, make it again
                    pr_dataloader_iter = iter(pr_dataloader)
                    pr_motions, num_samples  = next(pr_dataloader_iter)

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]
                texts = model_kwargs['y']['text']

                # Insert persona token to the input prompt for PersonaBooth
                new_texts = []
                for text in texts:
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Pp]erson)\b', 'sks person', text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Pp]ersons)\b', 'sks person', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Ii]ndividual)\b', 'sks individual', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Mm]an)\b', 'sks man', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Ww]man)\b', 'sks woman', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Bb]oy)\b', 'sks boy', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Gg]irl)\b', 'sks girl', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Ff]igure)\b', 'sks', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Tt]oon)\b', 'sks', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Rr]obot)\b', 'sks', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Cc]haracter)\b', 'sks', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Ss]im)\b', 'sks', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Ss]tickman)\b', 'sks', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Uu]ser)\b', 'sks', new_text)
                    new_text = re.sub(r'\b([Aa]|[Tt]he)?\s?([Bb]ody)\b', 'sks', new_text)
                    new_text = re.sub(r'\b([Ss]omeone|[Ss]omebody|[Hh]e|[Ss]he)\b', 'sks', new_text)
                    
                    if 'sks' not in new_text:
                        new_text = 'sks person ' + new_text

                    new_texts.append(new_text)
        
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []

                for t in range(repeat_times):
                    sample = model.inference(pr_motions, num_samples, new_texts, gen_shape=motion.shape)
                    
                    if t == 0:
                        sub_dicts = [{
                            'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                            'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                            'caption': model_kwargs['y']['text'][bs_i],
                            'tokens': tokens[bs_i],
                            # Fixed cap_len calculation, changed from len(tokens[bs_i])
                            # Lead to improved R-precision and Multimodal Dist.
                            # issue: https://github.com/GuyTevet/motion-diffusion-model/issues/182
                            'cap_len': tokens[bs_i].index('eos/OTHER') + 1, 
                            } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],
                                    } for bs_i in range(dataloader.batch_size)]

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer
        

    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)