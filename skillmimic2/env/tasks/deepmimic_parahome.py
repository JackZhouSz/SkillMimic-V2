import torch
import random
import numpy as np
from torch import Tensor
from typing import Dict
from env.tasks.skillmimic_parahome import SkillMimicParahome

class DeepMimicParahome(SkillMimicParahome): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
    def _compute_reward(self):
        self.rew_buf[:] = compute_humanoid_reward(
                                                  self._curr_ref_obs,
                                                  self._curr_obs,
                                                  self._hist_obs,
                                                  self._contact_forces,
                                                  self._tar_contact_forces,
                                                  len(self._key_body_ids),
                                                  self._dof_obs_size,
                                                  self._motion_data.reward_weights,
                                                  )
        return


class DeepMimicParahomeDomRand(DeepMimicParahome): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
    
    def _reset_random_ref_state_init(self, env_ids): #Z11
        motion_ids, motion_times = super()._reset_random_ref_state_init(env_ids)
        self._init_with_domrand_noise(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        motion_ids, motion_times = super()._reset_deterministic_ref_state_init(env_ids)
        self._init_with_domrand_noise(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times
    
    def _init_with_domrand_noise(self, env_ids, motion_ids, motion_times): 
        # Random noise for initial state
        self.state_random_flags = [np.random.rand() < self.cfg['env']['state_init_random_prob'] for _ in env_ids]
        if self.cfg['env']['state_init_random_prob'] > 0:
            for ind, env_id in enumerate(env_ids):
                if self.state_random_flags[ind]:
                    self.init_root_pos[env_id, 2] += random.random() * 0.1
                    self.init_root_rot[env_id] += torch.randn_like(self.init_root_rot[env_id]) * 0.1
                    self.init_root_pos_vel[env_id] += torch.randn_like(self.init_root_pos_vel[env_id]) * 0.1
                    self.init_root_rot_vel[env_id] += torch.randn_like(self.init_root_rot_vel[env_id]) * 0.1
                    self.init_dof_pos[env_id] += torch.randn_like(self.init_dof_pos[env_id]) * 0.1
                    self.init_dof_pos_vel[env_id]  += torch.randn_like(self.init_dof_pos_vel[env_id]) * 0.1
                    self.init_obj_pos[env_id, 2] += random.random() * 0.1
                    self.init_obj_pos_vel[env_id] += torch.randn_like(self.init_obj_pos_vel[env_id]) * 0.1
                    self.init_obj_rot[env_id] += torch.randn_like(self.init_obj_rot[env_id]) * 0.1
                    self.init_obj_rot_vel[env_id] += torch.randn_like(self.init_obj_rot_vel[env_id]) * 0.1
                    if self.isTest:
                        print(f"Random noise added to initial state for env {env_id}")
    
@torch.jit.script
def compute_humanoid_reward(hoi_ref: Tensor, hoi_obs: Tensor, hoi_obs_hist: Tensor, contact_buf: Tensor, tar_contact_forces: Tensor, 
                            len_keypos: int, dof_obs_size: int, w: Dict[str, Tensor]): #ZCr

    ### data preprocess ###

    # simulated states
    root_pos = hoi_obs[:,:3]
    root_rot = hoi_obs[:,3:6]
    start_ind = 6
    dof_pos = hoi_obs[:,start_ind:start_ind+dof_obs_size]
    start_ind += dof_obs_size
    dof_pos_vel = hoi_obs[:,start_ind:start_ind+dof_obs_size]
    start_ind += dof_obs_size
    obj_pos = hoi_obs[:,start_ind:start_ind+3]
    start_ind += 3
    obj_rot = hoi_obs[:,start_ind:start_ind+4]
    start_ind += 4
    obj_pos_vel = hoi_obs[:,start_ind:start_ind+3]
    start_ind += 3
    key_pos = hoi_obs[:,start_ind:start_ind+len_keypos*3]
    start_ind += len_keypos*3
    key_pos = torch.cat((root_pos, key_pos),dim=-1)
    body_rot = torch.cat((root_rot, dof_pos),dim=-1)
    ig = key_pos.view(-1,len_keypos+1,3).transpose(0,1) - obj_pos[:,:3]
    ig = ig.transpose(0,1).view(-1,(len_keypos+1)*3)

    start_ind = 6 + dof_obs_size
    dof_pos_vel_hist = hoi_obs_hist[:,start_ind:start_ind+dof_obs_size] #ZC

    # reference states
    ref_root_pos = hoi_ref[:,:3]
    ref_root_rot = hoi_ref[:,3:6]
    start_ind = 6
    ref_dof_pos = hoi_ref[:,start_ind:start_ind+dof_obs_size]
    start_ind += dof_obs_size
    ref_dof_pos_vel = hoi_ref[:,start_ind:start_ind+dof_obs_size]
    start_ind += dof_obs_size
    ref_obj_pos = hoi_ref[:,start_ind:start_ind+3]
    start_ind += 3
    ref_obj_rot = hoi_ref[:,start_ind:start_ind+4]
    start_ind += 4
    ref_obj_pos_vel = hoi_ref[:,start_ind:start_ind+3]
    start_ind += 3
    ref_key_pos = hoi_ref[:,start_ind:start_ind+len_keypos*3]
    start_ind += len_keypos*3
    ref_key_pos = torch.cat((ref_root_pos, ref_key_pos),dim=-1)
    ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos),dim=-1)
    ref_ig = ref_key_pos.view(-1,len_keypos+1,3).transpose(0,1) - ref_obj_pos[:,:3]
    ref_ig = ref_ig.transpose(0,1).view(-1,(len_keypos+1)*3)


    ### body reward ###

    # body pos reward
    ep = torch.mean((ref_key_pos - key_pos)**2,dim=-1)
    rp = torch.exp(-ep*w['p'])

    # body rot reward
    er = torch.mean((ref_body_rot - body_rot)**2,dim=-1)
    rr = torch.exp(-er*w['r'])

    # body rot vel reward
    erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2,dim=-1)
    rrv = torch.exp(-erv*w['rv'])

    rb = rp + rr + rrv


    ### object reward ###
    # object pos reward
    eop = torch.mean((ref_obj_pos - obj_pos)**2,dim=-1)
    rop = torch.exp(-eop*w['op'])

    ro = rop


    reward = rb + ro
    
    return reward