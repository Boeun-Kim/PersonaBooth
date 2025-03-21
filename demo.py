# This file is adapted by https://github.com/EricGuo5513/text-to-motion

import torch
import numpy as np
import random
import os
import shutil

from model.PersonaBooth import Personality
from arguments_PerMo import parse_args_test
from data.dataset.pose2guofeat import compute_guoh3dfeats
from dependency.TMR.src.data.motion import Normalizer
from dependency.MDM.utils.parser_util import generate_args
from dependency.MDM.data_loaders.humanml.scripts.motion_process import recover_from_ric
import dependency.MDM.data_loaders.humanml.utils.paramUtil as paramUtil
from dependency.MDM.data_loaders.humanml.utils.plot_script import plot_3d_motion
from dependency.MDM.sample.generate import construct_template_variables, save_multiple_samples, load_dataset
from dependency.MDM.model.rotation2xyz import Rotation2xyz

def init_seed(seed=123):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_frames = 196
fps = 20


def collate(batch):

    input_motions = [item[0] for item in batch]
    num_sample = [item[1] for item in batch]
    gt_text = [item[2] for item in batch]
    gt_motion = [item[3] for item in batch]
    gt_motion_mask = [item[4] for item in batch]

    return input_motions, num_sample, gt_text, gt_motion, gt_motion_mask


def inv_transform(data, std, mean):
    return data * std + mean


def visualize(args, mdm_args, samples, text, out_path, is_outputs=True):
    mdm_args = generate_args(args.MDM_path, mdm_args)

    all_motions = []
    mean = np.load('dependency/MDM/dataset/HumanML3D/Mean.npy')
    std = np.load('dependency/MDM/dataset/HumanML3D/Std.npy')

    for sample in samples:
        # Recover XYZ *positions* from HumanML3D vector representation
        
        n_joints = 22 if sample.shape[1] == 263 else 21
        sample = inv_transform(sample.cpu().permute(0, 2, 3, 1), std, mean).float()
        sample = recover_from_ric(sample, n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' 
        rot2xyz_mask = None 
        
        rot2xyz = Rotation2xyz(device="cuda")
        sample = rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                get_rotations_back=False)

        all_motions.append(sample.cpu().numpy())
    
    all_motions = np.concatenate(all_motions, axis=0)
    

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, {'motion': all_motions,})
    
    skeleton = paramUtil.t2m_kinematic_chain
    all_text = text

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(False)

    num_samples = 1
    num_repetitions = len(samples)
    batch_size = 1
    for sample_i in range(num_samples):
        rep_files = []
        for rep_i in range(num_repetitions):
            caption = all_text[rep_i*batch_size + sample_i]
            motion = all_motions[rep_i*batch_size + sample_i].transpose(2, 0, 1)
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            plot_3d_motion(animation_save_path, skeleton, motion, dataset="humanml", title=caption, fps=fps)
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(mdm_args, out_path,
                                               row_print_template, all_print_template, row_file_template, all_file_template,
                                               caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

if __name__ == '__main__': 

    args, mdm_args = parse_args_test()
    input_path = args.demo_folder
    num_repetition = 3

    input_text = args.text
    #guo_feats = compute_guoh3dfeats(input_path, None, True)  
        # (if you put pose files in the folder, convert them to guofeats first)
    guo_feats = []
    for filename in os.listdir(input_path):
        motion_path = os.path.join(input_path, filename)
        if not motion_path.endswith(".npy"):
            continue
        guo_feats.append(np.load(motion_path))

    normalizer_TMR = Normalizer(base_dir=args.TMR_stats)

    motion_dicts = []
    for motion in guo_feats:
        motion = torch.from_numpy(motion).to(torch.float)
        motion.requires_grad = True
        motion = normalizer_TMR(motion)
        motion_dicts.append({"x": motion, "length": len(motion)})
    
    num_sample = len(motion_dicts)

    model = Personality(args, mdm_args, device=device, is_train=False)

    if args.checkpoint_path != '':
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['param'], strict=False)

    model.to(device)
    model.eval()

    gen_motions = []
    input_texts = []
    for i in range(num_repetition):
        gen_motion = model.inference([motion_dicts], [num_sample], [input_text], max_frames)
        gen_motions.append(gen_motion)
        input_texts.append(input_text)

    visualize(args, mdm_args, gen_motions, input_texts, "demo/result")

    #########################################################
    ## Code for saving guofeats for the generated motion
    #mean_MDM = np.load('dependency/MDM/dataset/HumanML3D/Mean.npy')
    #std_MDM = np.load('dependency/MDM/dataset/HumanML3D/Std.npy')
    #out_guo = gen_motions[0].squeeze(0).squeeze(1) # save the first generated motion
    #out_guo = out_guo.transpose(1,0)
    #out_guo = out_guo[:150] # frame range to save
    #out_guo = out_guo.cpu().numpy() * std_MDM + mean_MDM
    #np.save('demo/result/result1.npy', out_guo)
    #########################################################

    # Save input videos
    inputs = []
    for motion in guo_feats:
        if len(motion) < max_frames:
            motion = np.concatenate([motion,
                                    np.zeros((max_frames - len(motion), motion.shape[1]))
                                    ], axis=0)
        else:
            motion = motion[:max_frames]

        motion = torch.from_numpy(motion).to(torch.float)

        # gt motion preparing for MDM
        mean_MDM = np.load('dependency/MDM/dataset/HumanML3D/Mean.npy')
        std_MDM = np.load('dependency/MDM/dataset/HumanML3D/Std.npy')
        motion = (motion - mean_MDM) / (std_MDM)
        
        guo_feat = motion.transpose(1,0)
        guo_feat = guo_feat.unsqueeze(1)
        guo_feat = guo_feat.unsqueeze(0)
        inputs.append(guo_feat)

    visualize(args, mdm_args, inputs, input_text, "demo/result/inputs", False)