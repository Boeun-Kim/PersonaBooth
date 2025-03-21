# This file is adapted from TMR and modified.

import os
import numpy as np
import sys
from glob import glob

sys.path.append('../../')
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append('../../dependency/TMR')

from arguments_PerMo import parse_preprocess
from src.guofeats import joints_to_guofeats
from prepare.tools import loop_amass


def extract_h3d(feats):
    from einops import unpack

    root_data, ric_data, rot_data, local_vel, feet_l, feet_r = unpack(
        feats, [[4], [63], [126], [66], [2], [2]], "i *"
    )
    return root_data, ric_data, rot_data, local_vel, feet_l, feet_r


def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def compute_guoh3dfeats(base_folder, output_folder, force_redo):

    print("Get h3d features from Guo et al.")
    
    if output_folder != None:
        print("Output folder ", output_folder)
        os.makedirs(output_folder, exist_ok=True)

        iterator = loop_amass(
            base_folder, output_folder, ext=".npy", newext=".npy", force_redo=force_redo
        )

        for motion_path, new_motion_path in iterator:
            
            joints = np.load(motion_path)

            if "humanact12" not in motion_path:
                # This is because the authors of HumanML3D
                # save the motions by swapping Y and Z (det = -1)
                # which is not a proper rotation (det = 1)
                # so we should invert x, to make it a rotation
                # that is why the authors use "data[..., 0] *= -1" inside the "if"
                # before swapping left/right
                # https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb
                joints[..., 0] *= -1
                # the humanact12 motions are normally saved correctly, no need to swap again
                # (but in fact this may not be true and the orignal H3D features
                # corresponding to HumanAct12 appears to be left/right flipped..)
                # At least we are compatible with previous work :/
            
            joints_m = swap_left_right(joints)

            # apply transformation
            try:
                features = joints_to_guofeats(joints)
                features_m = joints_to_guofeats(joints_m)
            except (IndexError, ValueError):
                assert len(joints) == 1
                continue

            # Save the features
            np.save(new_motion_path, features)

            # Save the mirrored features
            np.save(new_motion_path[:-4]+'_m.npy', features_m)

    else:
        guo_list = []
        ext=".npy"
        for motion_file in glob(f"**/*{ext}", root_dir=base_folder, recursive=True):
            print("motion_file", motion_file)
            motion_path = os.path.join(base_folder, motion_file)

            joints = np.load(motion_path)

            if "humanact12" not in motion_path:
                joints[..., 0] *= -1
            try:
                features = joints_to_guofeats(joints)

            except (IndexError, ValueError):
                assert len(joints) == 1
                continue
        
            guo_list.append(features)

        return guo_list

if __name__ == "__main__":
    args = parse_preprocess()

    compute_guoh3dfeats(args.pose_folder, args.guo_folder, args.force_redo)