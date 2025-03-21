import torch
import numpy as np
import random
import logging
from torch.utils.data import DataLoader

from train_loop import TrainPR
from arguments_PerMo import parse_args_train
from feeder.loader_PerMo import PerMoDataset

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='dummy.log', level=logging.INFO)

def init_seed(seed=123):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate(batch):

    input_motions = [item[0] for item in batch]
    gt_text = [item[1] for item in batch]
    gt_motion = [item[2] for item in batch]
    gt_motion_mask = [item[3] for item in batch]

    return input_motions, gt_text, gt_motion, gt_motion_mask

if __name__ == '__main__': 

    args, mdm_args = parse_args_train()

    trainer = TrainPR(args, mdm_args, device=device)

    train_dataset = PerMoDataset(args.TMR_stats, args.MDM_stats, args.text_path, 
                                args.guofeat_path, args.crop_ratio)

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
        worker_init_fn=init_seed,
        collate_fn=collate
    )

    trainer.save_checkpoint()

    for epoch in range(1, args.max_epoch + 1):
        for data in dataloader:
            trainer.run_step(*data, epoch)

        if trainer.epoch % args.save_interval == 0:
            trainer.save_checkpoint()