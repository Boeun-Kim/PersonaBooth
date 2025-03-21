# This file is originally from HumanML3D

import sys, os
import torch
import numpy as np
from tqdm import tqdm
import time
import sys
import importlib

sys.path.append('../../')
sys.path.append('../../dependency/HumanML3D')

from arguments_PerMo import parse_preprocess
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from common.quaternion import *
from paramUtil import *


os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

male_bm_path = '../body_models/smplh/male/model.npz'
male_dmpl_path = '../body_models/dmpls/male/model.npz'

female_bm_path = '../body_models/smplh/female/model.npz'
female_dmpl_path = '../body_models/dmpls/female/model.npz'

num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

trans_matrix = np.array([[1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0]])
ex_fps = 20



def amass_to_pose(src_path, save_path):

    bdata = np.load(src_path, allow_pickle=True)
    fps = 0

    try:
        fps = bdata['mocap_framerate']
        frame_number = bdata['trans'].shape[0]
    except:
        return fps

    fId = 0 # frame id of the mocap sequence
    pose_seq = []
    if bdata['gender'] == 'male':
        bm = male_bm
    else:
        bm = female_bm
    down_sample = int(fps / ex_fps)

    
    bdata_poses = bdata['poses'][::down_sample,...]
    bdata_trans = bdata['trans'][::down_sample,...]
    body_parms = {
            'root_orient': torch.Tensor(bdata_poses[:, :3]).to(comp_device),
            'pose_body': torch.Tensor(bdata_poses[:, 3:66]).to(comp_device),
            'pose_hand': torch.Tensor(bdata_poses[:, 66:]).to(comp_device),
            'trans': torch.Tensor(bdata_trans).to(comp_device),
            'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=len(bdata_trans), axis=0)).to(comp_device),
        }
    
    with torch.no_grad():
        body = bm(**body_parms)
    pose_seq_np = body.Jtr.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)
    
    np.save(save_path, pose_seq_np_n)

    return pose_seq_np_n, fps 


if __name__ == '__main__': 
    args = parse_preprocess()

    save_root = args.pose_folder
    os.makedirs(save_root, exist_ok=True)

    male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(comp_device)
    faces = c2c(male_bm.f)

    female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path).to(comp_device)

    base_folder = args.smpl_folder
    print(base_folder)
    paths = []
    folders = []
    dataset_names = []
    
    for root, dirs, files in os.walk(base_folder):
        folders.append(root)
        for name in files:
            dataset_name = root.split('/')[2]
            
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)
            paths.append(os.path.join(root, name))

    save_folders = [folder.replace(base_folder, save_root) for folder in folders]
    for folder in save_folders:
        os.makedirs(folder, exist_ok=True)

    group_path = [[path for path in paths if name in path] for name in dataset_names]

    group_path = group_path
    all_count = sum([len(paths) for paths in group_path])
    cur_count = 0
    
    for paths in group_path:
        dataset_name = paths[0].split('/')[2]
        pbar = tqdm(paths)
        pbar.set_description('Processing: %s'%dataset_name)
        fps = 0
        for path in pbar:
            save_path = path.replace(base_folder, save_root)
            save_path = save_path[:-3] + 'npy'

            pose_seq_np_n, fps = amass_to_pose(path, save_path)


        cur_count += len(paths)


        print('Processed / All (fps %d): %d/%d'% (fps, cur_count, all_count) )
        time.sleep(0.5)
