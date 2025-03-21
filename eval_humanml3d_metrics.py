# This code as adapted by https://github.com/GuyTevet/motion-diffusion-model

import torch
import numpy as np
import random
from model.PersonaBooth import Personality
from arguments_PerMo import parse_args_test
from dependency.MDM.data_loaders.get_data import get_dataset_loader
from dependency.MDM.data_loaders.humanml.motion_loaders.model_motion_loaders import get_eval_loader
from dependency.MDM.data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from dependency.MDM.eval.eval_humanml import evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(seed=123):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed()

if __name__ == '__main__': 
    
    args, mdm_args = parse_args_test()

    if args.MI_setting:
        log_file = ('eval_result/humanml3d_metrics_MI.log')
    else:
        log_file = ('eval_result/humanml3d_metrics.log')
    
    model = Personality(args, mdm_args, device=device, is_train=False)

    if args.checkpoint_path != '':
        print("Load the checkpoint from: ")
        print(args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['param'], strict=False)
    

    model.to(device)
    model.eval()


    batch_size = 32 # MDM: This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    args.batch_size = 32
    args.eval_mode = 'wo_mm'

    if args.eval_mode == 'debug':
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 5
    elif args.eval_mode == 'wo_mm':
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 10
    elif args.eval_mode == 'mm_short':
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 300
        replication_times = 5 
    else:
        raise ValueError()


    split = 'test'
    gt_loader = get_dataset_loader(name='humanml', batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt')
    gen_loader = get_dataset_loader(name='humanml', batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval')

    eval_motion_loaders = {
        'vald': lambda: get_eval_loader(
            args, batch_size, gen_loader, mm_num_samples, mm_num_repeats, gt_loader.dataset.opt.max_motion_length
            , num_samples_limit, multi_input=args.MI_setting, num_input = args.num_input
        )
    }


    eval_wrapper = EvaluatorMDMWrapper('humanml', device, checkpoints_dir='dependency/MDM/eval')
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)
