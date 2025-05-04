import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict
import glob, os, random, pickle
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from datetime import datetime

from utils import torch_utils
from utils.motion_data_handler import MotionDataHandler

from env.tasks.skillmimic1 import SkillMimicBallPlay


class SkillMimicBallPlayReweight(SkillMimicBallPlay): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        ############## Rewegiht ##############
        self.progress_buf_total = 0
        self.motion_ids_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.motion_times_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        total_frames = sum([self._motion_data.motion_lengths[motion_id] for motion_id in self._motion_data.hoi_data_dict])
        self.reweight_intervel = 10 * total_frames
        self.envs_reward = torch.zeros(self.num_envs, self.max_episode_length, device=self.device)
        self.average_rewards = {}
        self.motion_time_seqreward = {motion_id: torch.zeros(self._motion_data.motion_lengths[motion_id]-3, device=self.device) 
                                   for motion_id in self._motion_data.hoi_data_dict}
        ######################################

    
    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._motion_data.envid2episode_lengths, self.isTest, self.cfg["env"]["episodeLength"],
                                                   )
        # reweight the motion
        reset_env_ids = torch.nonzero(self.reset_buf == 1).squeeze()
        self._reweight_motion(reset_env_ids)
        return

    def after_reset_actors(self, env_ids):
        super().after_reset_actors(env_ids)
        self.motion_ids_total[env_ids] = self.motion_ids.to(self.device) # update motion_ids_total
        self.motion_times_total[env_ids] = self.motion_times.to(self.device) # update motion_times_total
        return
    
    ################ reweight according to the class reward ################
    def _reweight_motion(self, reset_env_ids):
        # record the reward for each motion clip at each time step
        if self.cfg['env']['reweight']: # and not self.isTest:
            self.record_motion_time_reward(reset_env_ids)

        if self.cfg['env']['reweight']: # and not self.isTest:
            if self.progress_buf_total % self.reweight_intervel == 0 and self.progress_buf_total > 0:
                # reweight the motion clip
                if len(self._motion_data.motion_class) > 1:
                    print('##### Reweight the sampling rate #####')
                    unique_ids = self._motion_data.hoi_data_dict.keys()
                    for motion_id in unique_ids:
                        indices = (self.motion_ids_total == motion_id)
                        avg_reward = self.rew_buf[indices].mean().item()
                        self.average_rewards[motion_id] = avg_reward
                    # for motion_id in unique_ids:
                    #     avg_reward = self.motion_time_seqreward[motion_id].mean().item()
                    #     self.average_rewards[motion_id] = avg_reward
                    print('Class Average Reward:', self.average_rewards)
                    self._motion_data._reweight_clip_sampling_rate(self.average_rewards)
                # reweight the motion time
                self._motion_data._reweight_time_sampling_rate(self.motion_time_seqreward)
    #######################################################################

    def record_motion_time_reward(self, reset_env_ids):
        reset_env_ids = reset_env_ids.clone().tolist() if reset_env_ids is not None else []
        for env_id in range(self.num_envs):
            ts = self.progress_buf[env_id]
            motion_id = self.motion_ids_total[env_id].item()
            motion_time = self.motion_times_total[env_id].item() # motion start time
            reset_env_ids = [reset_env_ids] if type(reset_env_ids) == int else reset_env_ids
            if env_id in reset_env_ids:
                self.envs_reward[env_id][ts] = self.rew_buf[env_id]
                non_zero_reward = self.envs_reward[env_id][self.envs_reward[env_id] != 0].mean()
                # 如果有nan，说明这个motion clip在这个时间点没有reward
                if torch.isnan(non_zero_reward).any():
                    self.motion_time_seqreward[motion_id][motion_time-2] = 0
                # 如果这个motion clip在这个时间点已经有reward了，那么取平均
                else:
                    self.motion_time_seqreward[motion_id][motion_time-2] = (self.motion_time_seqreward[motion_id][motion_time-2] + non_zero_reward) / 2
                self.envs_reward[env_id] = torch.zeros(self.max_episode_length, device=self.device)
            else:
                self.envs_reward[env_id][ts] = self.rew_buf[env_id]
        return
    


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights,
                        #    hoi_ref, hoi_obs, 
                           envid2episode_lengths, isTest, maxEpisodeLength, 
                        #    skill_label
                           ):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, bool, int) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rigid_body_pos[:, 0, 2] # root height
        body_fall = body_height < termination_heights# [4096] 
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 1)
        
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
        ########## Modified by Runyi ##########
        # skill_mask = (skill_label == 0)
        # terminated = torch.where(skill_mask, torch.zeros_like(terminated), terminated)
        #######################################

    if isTest and maxEpisodeLength > 0 :
        reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    else:
        reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC

    # reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    # reset = torch.zeros_like(reset_buf) #ZC300

    return reset, terminated
