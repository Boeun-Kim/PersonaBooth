import argparse

def parse_preprocess():
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('--smpl_folder', type=str, default='smpl/PerMo', help='raw data path')
    parser.add_argument('--pose_folder', type=str, default='pose/PerMo', help='pose feat path')
    parser.add_argument('--guo_folder', type=str, default='guofeat/PerMo', help='guo feat path')
    parser.add_argument('--force_redo', type=int, default=False, help='force preprocessing existing data')

    args = parser.parse_args()
    return args

def parse_args_train():
    parser = argparse.ArgumentParser(description='Training arguments')

    # train hyperparameters
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--drop_out', type=float, default=0.3)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--crop_ratio', type=float, default=0.7)

    # model hyperparameters
    parser.add_argument('--lr_proj', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_Adapt', type=float, default=0.0001, help='learning rate')
    #parser.add_argument('--lr_anneal_steps', type=int, default=0)

    # paths
    parser.add_argument('--text_path', type=str, default='data/dataset/description/PerMo', help='directory path of text annotations')
    parser.add_argument('--guofeat_path', type=str, default='data/dataset/guofeat/PerMo', help='directory path of preprocessed motions')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/test', help='directory path of checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default='', help='checkpoint path for load')

    parser.add_argument('--TMR_config', type=str, default='dependency/TMR/pretrained', help='directory path of TMR cfg and weights')
    parser.add_argument('--TMR_stats', type=str, default='dependency/TMR/stats/humanml3d/guoh3dfeats', help='directory path of TMR stats')
    parser.add_argument('--MDM_stats', type=str, default='dependency/MDM/dataset/HumanML3D', help='directory path of MDM stats')
    parser.add_argument('--MDM_path', type=str, default='dependency/MDM/pretrained/50step/model000750000.pt', help='directory path of MDM weights')

    args = parser.parse_args()
    _, unknown = parser.parse_known_args()

    return args, unknown


def parse_args_test():
    parser = argparse.ArgumentParser(description='Testing arguments')

    # paths
    parser.add_argument('--checkpoint_path', type=str, default='pretrained/0500.pt', help='checkpoint path for load')
    parser.add_argument('--force_redo', type=int, default=True, help='force preprocessing existing data')
    parser.add_argument('--TMR_config', type=str, default='dependency/TMR/pretrained', help='directory path of TMR cfg and weights')
    parser.add_argument('--TMR_stats', type=str, default='dependency/TMR/stats/humanml3d/guoh3dfeats', help='directory path of TMR stats')
    parser.add_argument('--MDM_stats', type=str, default='dependency/MDM/dataset/HumanML3D', help='directory path of MDM stats')
    parser.add_argument('--MDM_path', type=str, default='dependency/MDM/pretrained/50step/model000750000.pt', help='directory path of MDM weights')
    parser.add_argument('--demo_folder', type=str, default='demo/input', help='path for the demo input motions')

    # hyperparameters
    parser.add_argument('--MI_setting', type=bool, default=False, help='eval setting: multiple input(MI)-Ture, single_input (SI)-False')
    parser.add_argument('--num_input', type=int, default=15)
    parser.add_argument('--eval_mode', type=str, default='wo_mm', help='eval mode: debug/wo_mm/mm_short')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--text_path', type=str, default='data/dataset/description/PerMo', help='directory path of text annotations')
    parser.add_argument('--guofeat_path', type=str, default='data/dataset/guofeat/PerMo', help='directory path of preprocessed motions')
    parser.add_argument('--crop_ratio', type=float, default=1.0)
    parser.add_argument('--drop_out', type=float, default=0.3)
    
    # CFG params
    parser.add_argument('--w_txt', type=float, default=0.5)
    parser.add_argument('--guide_motion', type=float, default=15)
    parser.add_argument('--guide_txt', type=float, default=10)
    parser.add_argument('--token_renorm', type=float, default=0.3)
    parser.add_argument('--motion_renorm', type=float, default=0.3)

    # input prompt
    parser.add_argument('--text', type=str, default='sks person is walking forward.', help='input text')

    args = parser.parse_args()
    _, unknown = parser.parse_known_args()

    return args, unknown