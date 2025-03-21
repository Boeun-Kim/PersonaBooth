import os
import numpy as np
from torch.utils.data import Dataset
import random
import torch
from os.path import join as pjoin

from dependency.TMR.src.data.motion import Normalizer

class PerMoDataset(Dataset):
    def __init__(
        self,
        TMR_stats,
        MDM_stats,
        text_path,
        motion_path,
        crop_ratio
    ):
        self.normalizer_TMR = Normalizer(base_dir=TMR_stats)
        self.text_path = text_path
        self.motion_path = motion_path
        self.load_text()
        self.load_motion()
        self.MDM_stats = MDM_stats
        self.max_motion_length = 196
        self.crop_ratio = crop_ratio

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):

        # pick gt motion-text pair
        set_idx = self.set_idx[idx]
        data_idx = self.data_idx[idx]
          # select random text from the text samples
        type = self.motion_type_list[set_idx][data_idx]
        gt_text = random.choice(self.text_dict[type])
        gt_motion = self.motion_list[set_idx][data_idx]
        input_motion = gt_motion
        gt_motion = self.augment(gt_motion)


        input_motion = self.augment(input_motion)
        motion = torch.from_numpy(input_motion).to(torch.float)
        motion.requires_grad = True
        motion = self.normalizer_TMR(motion)

        # pick arbitrary motions (except same type with gt) from the same set
        cand_list = self.motion_list[set_idx]
        cand_type_list = self.motion_type_list[set_idx]
        filtered_list = [val for flag, val in zip(cand_type_list, cand_list) if flag!=type]
        num_in_set = len(filtered_list)
        pick_idx = np.random.choice(np.arange(0, num_in_set), 1, replace=False)
        selected_motion = filtered_list[pick_idx[0]]

        # feed gt and rand motion
        sel_motion = selected_motion.copy()
        sel_motion = self.augment(sel_motion)
        sel_motion = torch.from_numpy(sel_motion).to(torch.float)
        sel_motion.requires_grad = True
        sel_motion = self.normalizer_TMR(sel_motion)


        motion_dicts = [{"x": motion, "length": len(motion),
                        "x_diff": sel_motion, "length_diff": len(sel_motion)}]

        # gt motion preparing for MDM
        mean_MDM = np.load(pjoin(self.MDM_stats, 'Mean.npy'))
        std_MDM = np.load(pjoin(self.MDM_stats, 'Std.npy'))
        gt_motion = (gt_motion - mean_MDM) / (std_MDM)

        if len(gt_motion) < self.max_motion_length:
            gt_motion = np.concatenate([gt_motion,
                                    np.zeros((self.max_motion_length - len(gt_motion), gt_motion.shape[1]))
                                    ], axis=0)

        assert not np.any(np.isnan(gt_motion))
        gt_motion = torch.from_numpy(gt_motion).to(torch.float)

        gt_motion_mask = torch.zeros(self.max_motion_length, dtype=torch.bool)
        gt_motion_mask[:len(gt_motion)] = True

        
        return motion_dicts, gt_text, gt_motion, gt_motion_mask

    def augment(self, motion):
        # Random crop
        seq_len = motion.shape[0]
        if seq_len < self.max_motion_length*0.3:  # do not crop when the motion is too short
            return motion

        if self.crop_ratio > 0 and self.crop_ratio < 1:
            min_len = int(seq_len*self.crop_ratio)
            rand_len = np.random.randint(min_len, seq_len)
            rand_len = np.minimum(rand_len, self.max_motion_length)
            rand_st = np.random.randint(0, seq_len-rand_len)
            motion = motion[rand_st:rand_st+rand_len]
            
        return motion


    def load_text(self):
        file_dict = {}
        for filename in os.listdir(self.text_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.text_path, filename)
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    file_dict[os.path.splitext(filename)[0]] = [line.strip() for line in lines]

        self.text_dict = file_dict


    def load_motion(self):

        data_list = []
        type_list = []
        set_idx_list = []
        data_idx_list = []

        for _, dir_f1_list, _ in os.walk(self.motion_path):
            break

        num_set = 0
        total_data = 0
        for dir_f1 in dir_f1_list:
            subpath = os.path.join(self.motion_path, dir_f1)
            for _, dir_f2_list, _ in os.walk(subpath):
                break

            for dir_f2 in dir_f2_list:
                num_data = 0
                subsubpath = os.path.join(subpath, dir_f2)
                motion_data = []
                motion_types = []
                for filename in os.listdir(subsubpath):
                    if filename.endswith(".npy"):
                        file_path = os.path.join(subsubpath, filename)
                        
                        base_name = os.path.splitext(filename)[0]
                        split_name = base_name.split('_')
                        motion_type = split_name[1]

                        data = np.load(file_path)
                        
                        motion_data.append(data)
                        motion_types.append(motion_type)
                        set_idx_list.append(num_set)
                        data_idx_list.append(num_data)
                        num_data += 1
                        total_data += 1

                num_set += 1
                data_list.append(motion_data)
                type_list.append(motion_types)

        self.motion_list = data_list
        self.motion_type_list = type_list 
        self.set_idx = set_idx_list
        self.data_idx = data_idx_list 
        self.total_data = total_data



class PerMoDataset_eval(Dataset):
    def __init__(
        self,
        TMR_stats,
        MDM_stats,
        text_path,
        motion_path,
        crop_ratio,
        multi_input,
        num_input,
        for_PRA = False,
        group = 0,
    ):
        self.multi_input = multi_input
        self.num_input = num_input
        self.normalizer_TMR = Normalizer(base_dir=TMR_stats)
        self.text_path = text_path
        self.motion_path = motion_path

        if for_PRA: # for eval PRA
            self.group = group
            self.load_motion_PRA() # load personas in the specific sub-category
        else:
            self.load_motion()

        self.MDM_stats = MDM_stats
        self.max_motion_length = 196
        self.crop_ratio = crop_ratio
        self.for_PRA = for_PRA


    def __len__(self):
        if self.multi_input == False:
            return len(self.set_idx)
        else:
            return len(self.motion_num_per_set)

    def __getitem__(self, idx):

        # Single Input (SI) setting : pick rand input for eval
        if self.multi_input == False:
            set_idx = self.set_idx[idx]
            data_idx = self.data_idx[idx]

            input_motion = self.motion_list[set_idx][data_idx]
        
            motion = torch.from_numpy(input_motion).to(torch.float)
            motion.requires_grad = True
            motion = self.normalizer_TMR(motion)
            motion_dicts = [{"x": motion, "length": len(motion)}]

            if self.for_PRA: # evaluate per group
                gt_pr = self.pr_list[set_idx][data_idx]
                return motion_dicts, 1, gt_pr

            return motion_dicts, 1

        # Multiple Input (MI) setting 
        # : pick rand set first and pick [self.num_input] differnt motions from the set
        else:
            picked_motion_idx = []
            set_idx = idx
            cand_list = self.motion_list[set_idx]
            cand_type_list = self.motion_type_list[set_idx]

            while (len(cand_list) > 0) and (len(picked_motion_idx) < self.num_input):

                pick_idx = random.randint(0, len(cand_list)-1)
                picked_motion_idx.append(pick_idx)
                pick_type = cand_type_list[pick_idx]

                filtered_cand_list = [val for flag, val in zip(cand_type_list, cand_list) if flag != pick_type]
                filtered_type_list = [flag for flag in cand_type_list if flag != pick_type]

                cand_list = filtered_cand_list
                cand_type_list = filtered_type_list
            
            motion_dicts = []
            for data_idx in picked_motion_idx:
                input_motion = self.motion_list[set_idx][data_idx]
                
                motion = torch.from_numpy(input_motion).to(torch.float)
                motion.requires_grad = True
                motion = self.normalizer_TMR(motion)
                motion_dicts.append({"x": motion, "length": len(motion)})
        
            if self.for_PRA:
                gt_pr = self.pr_list[set_idx][data_idx]
                return motion_dicts, len(picked_motion_idx), gt_pr

            return motion_dicts, len(picked_motion_idx)


    def augment(self, motion):
        # Random crop
        seq_len = motion.shape[0]
        if seq_len < self.max_motion_length*0.3:  # do not crop when the motion is too short
            return motion

        if self.crop_ratio > 0 and self.crop_ratio < 1:
            min_len = int(seq_len*self.crop_ratio)
            rand_len = np.random.randint(min_len, seq_len)
            rand_len = np.minimum(rand_len, self.max_motion_length)
            rand_st = np.random.randint(0, seq_len-rand_len)
            motion = motion[rand_st:rand_st+rand_len]
            
        return motion


    def load_motion(self):

        data_list = []
        data_num_list = []
        type_list = []
        set_idx_list = []
        data_idx_list = []
        #pr_list = []

        for _, dir_f1_list, _ in os.walk(self.motion_path):
            break

        num_set = 0
        total_data = 0
        for dir_f1 in dir_f1_list:  # Loop for style categories
            subpath = os.path.join(self.motion_path, dir_f1)
            for _, dir_f2_list, _ in os.walk(subpath):
                break

            for dir_f2 in dir_f2_list: # loop for actors

                # Loading data for each persona set
                num_data = 0
                subsubpath = os.path.join(subpath, dir_f2)
                motion_data = []
                motion_types = []  # = content
                style_idx = []
                
                for filename in os.listdir(subsubpath):
                    if filename.endswith(".npy"):
                        file_path = os.path.join(subsubpath, filename)
                        
                        base_name = os.path.splitext(filename)[0]
                        split_name = base_name.split('_')
                        motion_type = split_name[1]
                        motion_idx = split_name[3]
                        mirrored = len(split_name) == 5
                        style_name = split_name[0]

                        load_data = True
                        if self.multi_input:
                            if mirrored:
                                load_data == False

                        if load_data == True:
                            data = np.load(file_path)
                            motion_data.append(data)
                            motion_types.append(motion_type)
                            set_idx_list.append(num_set)
                            data_idx_list.append(num_data)

                            #style_key = next(key for key, value in sty_enumerator.items() if value == style_name)
                            #style_idx.append(style_key)

                            num_data += 1
                            total_data += 1
                            
                data_num_list.append(num_data)
                num_set += 1
                data_list.append(motion_data)
                type_list.append(motion_types)
                #pr_list.append(style_idx)
                

        self.motion_list = data_list
        self.motion_num_per_set = data_num_list
        self.motion_type_list = type_list 
        self.set_idx = set_idx_list
        self.data_idx = data_idx_list 
        self.total_data = total_data
        #self.pr_list = pr_list

        print("Total loaded data: ", self.total_data)

    def load_motion_PRA(self):

        data_list = []
        data_num_list = []
        type_list = []
        set_idx_list = []
        data_idx_list = []
        pr_list = []

        num_set = 0
        total_data = 0
        
        substyle_list = PR_dict[self.group].values()
        
        for substyle in substyle_list:
            subpath = os.path.join(self.motion_path, substyle)
            for _, dir_list, _ in os.walk(subpath):
                break

            for dir in dir_list:
                num_data = 0
                subsubpath = os.path.join(subpath, dir)
                motion_data = []
                motion_types = []
                pr_idx = []
                
                for filename in os.listdir(subsubpath):
                    if filename.endswith(".npy"):
                        file_path = os.path.join(subsubpath, filename)
                        
                        base_name = os.path.splitext(filename)[0]
                        split_name = base_name.split('_')
                        
                        style_name = split_name[0]
                        motion_type = split_name[1]
                        actor_name = split_name[2]
                        motion_idx = split_name[3]
                        mirrored = len(split_name) == 5
                        actor_idx = Actor_enumerator[actor_name]
                        
                        load_data = True
                        if self.multi_input:
                            if mirrored:
                                load_data == False

                        if load_data == True:
                            data = np.load(file_path)
                            motion_data.append(data)
                            motion_types.append(motion_type)
                            set_idx_list.append(num_set)
                            data_idx_list.append(num_data)

                            substyle_idx = next(key for key, value in  PR_dict[self.group].items() if value == style_name)
                            pr_key = substyle_idx * 5 + actor_idx
                            pr_idx.append(pr_key)

                            num_data += 1
                            total_data += 1
                                
                data_num_list.append(num_data)
                num_set += 1
                data_list.append(motion_data)
                type_list.append(motion_types)
                pr_list.append(pr_idx)
                

        self.motion_list = data_list
        self.motion_num_per_set = data_num_list
        self.motion_type_list = type_list 
        self.set_idx = set_idx_list
        self.data_idx = data_idx_list 
        self.total_data = total_data
        self.pr_list = pr_list

        print("Total loaded data: ", self.total_data)


Actor_enumerator = {
    'A01': 0, 
    'A02': 1, 
    'A03': 2,
    'A04': 3, 
    'A05': 4
}

Age = {
    0: "Childish", 
    1: "Old", 
    2: "Teenage",
    3: "Neutral"
}
Character1 = {
    0: "Ballerina", 
    1: "Hulk", 
    2: "Monkey",
    3: "Ninja"
}
Character2 = {
    0: "Penguin", 
    1: "Robot", 
    2: "SWAT",
    3: "Waiter",
    4: "Zombie"
}
Condition1 = {
    0: "Armaching", 
    1: "Drunken", 
    2: "Exhausted",
    3: "Headaching"
}
Condition2 = {
    0: "Healthy", 
    1: "Legaching", 
    2: "Textnecked",
}
Emotion1 = {
    0: "Angry", 
    1: "Fearful", 
    2: "Happy",
}
Emotion2 = {
    0: "Sad", 
    1: "Strained", 
    2: "Surprising",
}
Personality = {
    0: "Elegant", 
    1: "Shy", 
    2: "Silly",
    3: "Uppity"
}
Surroundings = {
    0: "Cold", 
    1: "Crowded", 
    2: "Muddyfloor",
    3: "Unpleasantfloor"
}

PR_dict = {
    "Age": Age,
    "Character1": Character1,
    "Character2": Character2,
    "Condition1": Condition1,
    "Condition2": Condition2,
    "Emotion1": Emotion1,
    "Emotion2": Emotion2,
    "Personality": Personality,
    "Surroundings": Surroundings
}