from enum import Enum
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

from env.tasks.skillmimic1_rand import SkillMimic1BallPlayRand
from utils.history_encoder import HistoryEncoder

class SkillMimic1BallPlayHist(SkillMimic1BallPlayRand): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        if cfg["env"]["histEncoderCkpt"]:
            self.history_length = cfg['env']['historyLength']
            self.hist_vecotr_dim = cfg['env']['histVectorDim']
            self.ref_hoi_data_size = 1 + self._dof_obs_size*2 + 3
            self._hist_obs_batch = torch.zeros([self.num_envs, self.history_length, self.ref_hoi_data_size], device=self.device, dtype=torch.float)
            self.hist_encoder = HistoryEncoder(self.history_length, input_dim=316, final_dim=self.hist_vecotr_dim).to(self.device)
            self.hist_encoder.resume_from_checkpoint(cfg["env"]["histEncoderCkpt"])
            self.hist_encoder.eval()
            for param in self.hist_encoder.parameters():
                param.requires_grad = False


    def get_obs_size(self):
        obs_size = super().get_obs_size()

        if self.cfg["env"]["histEncoderCkpt"]:
            obs_size += self.get_hist_size()
        return obs_size
    
    def get_hist_size(self): 
        return 3 #0 # actually 3, but realzied temporarily by `asset_file == "mjcf/mocap_humanoid_hist.xml"`

    def get_hist(self, env_ids, ts):
        return self.hist_encoder(self._hist_obs_batch[env_ids])
        
    def _compute_observations(self, env_ids=None): # called @ reset & post step
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        obj_obs = self._compute_obj_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)
        
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs)

        textemb_batch = self.hoi_data_label_batch[env_ids]
        obs = torch.cat((obs, textemb_batch), dim=-1)
        
        if self.cfg["env"]["histEncoderCkpt"]:
            hist_vector = self.get_hist(env_ids, self.progress_buf[env_ids])
            obs = torch.cat([obs, hist_vector], dim=-1)

            # [0, 1, 2, a, b, c, d] -> [1, 2, a, b, c, d, currect_obs]
            current_obs = torch.cat([humanoid_obs[..., :157],  self._dof_pos[env_ids], obj_obs[..., :3]], dim=-1) # (envs, 316)
            self._hist_obs_batch[env_ids] = torch.cat([self._hist_obs_batch[env_ids, 1:], current_obs.unsqueeze(1)], dim=1)

        self.obs_buf[env_ids] = obs

        return

    def after_reset_actors(self, env_ids):
        super().after_reset_actors(env_ids)
        
        if self.cfg["env"]["histEncoderCkpt"]:
            ######### Modified by Runyi #########
            # pt data (337 dim): root_pos(3) + root_rot(3) + root_rot(3) + dof_pos(52*3) + body_pos(53*3) 
            #                   + obj_pos(3) + zero_obj_rot(3) + zero_obj_pos_vel(3) + zero_obj_rot_vel(3) + contact_graph(1)
            # initialize the history observation
            self._hist_obs_batch[env_ids] = torch.zeros([env_ids.shape[0], self.history_length, self.ref_hoi_data_size], device=self.device, dtype=torch.float)
            for ind in range(env_ids.shape[0]):
                env_id = env_ids[ind]
                ref_data = self._motion_data.hoi_data_dict[int(self.motion_ids[ind])]
                humanoid_obs = get_humanoid_obs(ref_data['root_pos'], ref_data['root_rot_3d'], ref_data['body_pos'])
                obj_obs = get_obj_obs(ref_data['root_pos'], ref_data['root_rot_3d'], ref_data['obj_pos'])
                ref_data_obs = torch.cat([humanoid_obs, ref_data['dof_pos'].view(-1, 52*3), obj_obs], dim=-1)
                start_frame = self.motion_times[ind] - self.history_length
                end_frame = self.motion_times[ind]
                if start_frame >= 0:
                    self._hist_obs_batch[env_id] = ref_data_obs[start_frame:end_frame]
                else:
                    self._hist_obs_batch[env_id, -end_frame:] = ref_data_obs[:end_frame]
            #####################################
            #Z self.motion_ids, self.motion_times are local variable, only referenced here, so `self.` is not necessary


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def get_humanoid_obs(root_pos, root_rot, body_pos):
    root_h_obs = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    obs = torch.cat((root_h_obs, local_body_pos), dim=-1)
    return obs

@torch.jit.script
def get_obj_obs(
    root_pos: torch.Tensor,  # 参考点位置
    root_rot: torch.Tensor,  # 参考点旋转
    tar_pos: torch.Tensor,   # 目标点位置
) -> torch.Tensor:  # 返回值是 Tensor
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)

    return local_tar_pos