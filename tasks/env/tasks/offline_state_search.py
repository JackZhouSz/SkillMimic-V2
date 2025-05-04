from enum import Enum
import torch.nn.functional as F
import numpy as np
import torch
import pickle
from typing import Dict
from torch import Tensor
from typing import Tuple
import glob, os, random
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from datetime import datetime

from utils import torch_utils
from utils.motion_data_handler import MotionDataHandlerOfflineNew
from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObject


class OfflineStateSearch(HumanoidWholeBodyWithObject): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        motion_file = cfg['env']['motion_file']
        reward_weights_default = cfg["env"]["rewardWeights"]
        init_vel = cfg['env']['initVel']
        play_dataset = cfg['env']['playdataset']
        graph_save_path = cfg['env']['graph_save_path']

        self.skill_name = motion_file.split('/')[-1] #metric
        self.max_episode_length = 60

        self._motion_data = MotionDataHandlerOfflineNew(motion_file, self.device, self._key_body_ids, self.cfg, self.num_envs, 
                                            self.max_episode_length, reward_weights_default, init_vel, play_dataset)
        motion_classes = set(self._motion_data.motion_class)

        state_search_dict = {cls:{} for cls in motion_classes}
        for source_id in range(self._motion_data.num_motions):
            source_class = self._motion_data.motion_class[source_id]
            state_search_dict[source_class][source_id] = {}
            other_classes = motion_classes - {source_class}
            for source_time in range(self._motion_data.motion_lengths[source_id]):
                # print(f"source_class: {source_class}, source_id: {source_id}, source_time: {source_time}, other_classes: {other_classes}")
                for switch_class in other_classes:
                    switch_id, switch_time, max_sim = self._motion_data._get_switch_time(source_class, source_id, source_time, switch_class)
                    if source_time not in state_search_dict[source_class][source_id]:
                        state_search_dict[source_class][source_id][source_time] = [[switch_class, switch_id, switch_time, max_sim]]
                    else:
                        state_search_dict[source_class][source_id][source_time].append([switch_class, switch_id, switch_time, max_sim])
                # print(switch_id, switch_time, max_sim)
                # exit()
            print(f"Finished processing source_id {source_id}!")
        # print(state_search_dict[13][3]) # run
        
        # save state_search_dict
        root_dir = os.path.dirname(graph_save_path)
        os.makedirs(root_dir, exist_ok=True)
        with open(graph_save_path, "wb") as f:
            pickle.dump(state_search_dict, f)
        print(f"Saving state_search_dict to {graph_save_path}!")
        exit()