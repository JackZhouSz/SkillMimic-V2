from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
import glob, os, random
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from datetime import datetime

from utils import torch_utils
from utils.motion_data_handler import MotionDataHandler
from utils.history_encoder import HistoryEncoder

from env.tasks.deepmimic import DeepMimicBallPlay

class DeepMimicBallPlayLocalHist(DeepMimicBallPlay): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.history_length = cfg['env']['historyLength']
        self.hist_vecotr_dim = cfg['env']['histVectorDim']

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
                
        self.progress_buf_total = 0
        self.max_epochs = cfg['env']['maxEpochs']
        self.ref_hoi_obs_size = 6 + self._dof_obs_size*2 + 10 + len(self.cfg["env"]["keyBodies"])*3 + 1
        self.ref_hoi_data_size = 1 + self._dof_obs_size*2 + 3
        self.weights = {'root_pos': 1,'root_pos_vel': 1,'root_rot_3d': 1,'root_rot_vel': 1,'dof_pos': 0.25, 'dof_pos_vel': 1,'obj_pos': 2,'obj_pos_vel': 2,'obj_rot': 1,'obj_rot_vel': 1}
        
        self._hist_obs_batch = torch.zeros([self.num_envs, self.history_length, self.ref_hoi_data_size], device=self.device, dtype=torch.float)
        self.hist_encoder = HistoryEncoder(self.history_length, input_dim=316, final_dim=self.hist_vecotr_dim).to(self.device)
        self.hist_encoder.eval()
        self.hist_encoder.resume_from_checkpoint(cfg["env"]["histEncoderCkpt"]) if cfg["env"]["histEncoderCkpt"] else None
        for param in self.hist_encoder.parameters():
            param.requires_grad = False

        self.motion_dict = {}

    def get_obs_size(self):
        obs_size = super().get_obs_size()

        obs_size += self.get_hist_size()
        return obs_size
    
    def get_hist_size(self): 
        return 0 # actually 3, but realzied temporarily by `asset_file == "mjcf/mocap_humanoid_hist.xml"`

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
        
        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids, ts].clone()
        
        ######### Modified by Runyi #########
        hist_vector = self.get_hist(env_ids, ts)
        obs = torch.cat([obs, hist_vector], dim=-1)

        self.obs_buf[env_ids] = obs

        # [0, 1, 2, a, b, c, d] -> [1, 2, a, b, c, d, currect_obs]
        current_obs = torch.cat([humanoid_obs[..., :157],  self._dof_pos[env_ids], obj_obs[..., :3]], dim=-1) # (envs, 316)
        self._hist_obs_batch[env_ids] = torch.cat([self._hist_obs_batch[env_ids, 1:], current_obs.unsqueeze(1)], dim=1)
        #####################################

        return

    def after_reset_actors(self, env_ids):
        super().after_reset_actors(env_ids)
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
    
    def _build_frame_for_blender(self,motion_dict, rootpos, rootrot, dofpos, dofrot, ballpos, ballrot):
        if 'rootpos' not in motion_dict:
            motion_dict['rootpos']=[]
        if 'rootrot' not in motion_dict:
            motion_dict['rootrot']=[]
        if 'dofpos' not in motion_dict:
            motion_dict['dofpos']=[]
        if 'dofrot' not in motion_dict:
            motion_dict['dofrot']=[]
        if 'ballpos' not in motion_dict:
            motion_dict['ballpos']=[]
        if 'ballrot' not in motion_dict:
            motion_dict['ballrot']=[]

        motion_dict['rootpos'].append(rootpos.clone())
        motion_dict['rootrot'].append(rootrot.clone())
        motion_dict['dofpos'].append(dofpos.clone())
        motion_dict['dofrot'].append(dofrot.clone())
        motion_dict['ballpos'].append(ballpos.clone())
        motion_dict['ballrot'].append(ballrot.clone())

        # print("motion_dict['rootpos']",motion_dict['rootpos'])
        # print("rootpos",rootpos)

    def _save_motion_dict(self, motion_dict, filename='motion.pt'):

        motion_dict['rootpos'] = torch.stack(motion_dict['rootpos'])
        motion_dict['rootrot'] = torch.stack(motion_dict['rootrot'])
        motion_dict['dofpos'] = torch.stack(motion_dict['dofpos'])
        motion_dict['dofrot'] = torch.stack(motion_dict['dofrot'])
        motion_dict['ballpos'] = torch.stack(motion_dict['ballpos'])
        motion_dict['ballrot'] = torch.stack(motion_dict['ballrot'])

        torch.save(motion_dict, filename)
        exit()

    def post_physics_step(self):
        super().post_physics_step()
        
        # to save data for blender
        body_ids = list(range(53))
        self._build_frame_for_blender(self.motion_dict,
                        self._rigid_body_pos[0, 0, :],
                        self._rigid_body_rot[0, 0, :],
                        self._rigid_body_pos[0, body_ids, :],
                        #torch.cat((self._rigid_body_rot[0, :1, :], torch_utils.exp_map_to_quat(self._dof_pos[0].reshape(-1,3))),dim=0),
                        self._rigid_body_rot[0, body_ids, :],
                        self._target_states[0, :3],
                        self._target_states[0, 3:7]
                        # self._proj_states[0, :3],
                        # self._proj_states[0, 3:7]
                        )
        # if self.progress_buf[0] == 400:
        #     self._save_motion_dict(self.motion_dict, 'blender_motions/fig1_ballplay_rrun.pt')


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