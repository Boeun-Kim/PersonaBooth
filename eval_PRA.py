import torch
import numpy as np
import random
from torch.utils.data import DataLoader, WeightedRandomSampler
from os.path import join as pjoin
import itertools
import re


from model.PersonaBooth import Personality
from arguments_PerMo import parse_args_test
from feeder.loader_PerMo import PerMoDataset_eval, PR_dict

from dependency.MDM.data_loaders.get_data import get_dataset_loader
from dependency.PRA_classifier.PRA_train import inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(seed=123):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed()

mean_MDM = np.load(pjoin('dependency/MDM/dataset/HumanML3D', 'Mean.npy'))
std_MDM = np.load(pjoin('dependency/MDM/dataset/HumanML3D', 'Std.npy'))

def collate(batch):

    input_motions = [item[0] for item in batch]
    num_sample = [item[1] for item in batch]
    class_names = [item[2] for item in batch]

    return input_motions, num_sample, class_names


def eval_loop(group, multi_input, num_input):
    
    pr_dataset = PerMoDataset_eval(pr_args.TMR_stats, pr_args.MDM_stats, pr_args.text_path, 
                        pr_args.guofeat_path, pr_args.crop_ratio, multi_input=multi_input, num_input=num_input,
                        for_PRA = True, group = group)

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

    humanml_loader = get_dataset_loader(name='humanml', batch_size=pr_args.batch_size, num_frames=None, split='test', hml_mode='eval')

    humanml_iter = itertools.cycle(humanml_loader)
    pr_iter = itertools.cycle(pr_dataloader)

    motion_data = []
    gt_class = []
    iter = 0
    while iter < 100:
        print(iter)
        iter = iter + 1

        motion_humanml, model_kwargs = next(humanml_iter)
        pr_motions, num_samples, pr_class = next(pr_iter)


        texts = model_kwargs['y']['text']
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

            #new_text = "sks person walks forward"
            new_texts.append(new_text)

        with torch.no_grad():
            sample = model.inference(pr_motions, num_samples, new_texts)

        
        for bs_i in range(pr_args.batch_size):
            guo_feat = sample[bs_i].squeeze().permute(1, 0).cpu().numpy()

            motion_data.append(guo_feat)
            gt_class.append(pr_class[bs_i])


    acc = inference(group, motion_data, gt_class)

    return acc, pr_dataset.total_data


if __name__ == '__main__': 

    pr_args, mdm_args = parse_args_test()
    if  pr_args.MI_setting:
        log_file = ('eval_result/PRA_MI.log')
    else:
        log_file = ('eval_result/PRA.log')

    ## Load model #####################
    model = Personality(pr_args, mdm_args, device=device, is_train=False)

    model.eval()
    if pr_args.checkpoint_path != '':
        print("Load model from ", pr_args.checkpoint_path)
        checkpoint = torch.load(pr_args.checkpoint_path)
        model.load_state_dict(checkpoint['param'], strict=False)

    model.to(device)
    model.eval()
    ####################################

    with open(log_file, 'w', encoding="utf-8") as file:

        for group in PR_dict.keys():
            acc, num_data = eval_loop(group, pr_args.MI_setting, pr_args.num_input)
            print("Accuracy: ", str(acc))
            file.write(group + ": "+ str(acc)+"%"+"   num_data="+ str(num_data)+ '\n')
            file.flush()


