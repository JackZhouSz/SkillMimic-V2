import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, List
import glob, os, random, pickle
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from datetime import datetime

from utils import torch_utils

from env.tasks.amp import SkillMimicAMPLocomotion


class SkillMimicAMPLocomotion4Eval(SkillMimicAMPLocomotion): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        
    def _hook_post_step(self):
        # extra calc of self._curr_obs, for imitation reward
        self._compute_hoi_observations()
        env_ids = torch.arange(self.num_envs)
        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids, ts].clone()

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()
        return
    
    def _compute_reset(self):
        # print(self.progress_buf)
        # if(self.progress_buf[0]==60):
        #     print("?")
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._motion_data.envid2episode_lengths, self.isTest, self.cfg["env"]["episodeLength"],
                                                   self.cfg["env"]["NR"]
                                                   )
        return
    
    
    def _compute_reward(self):
        self.rew_buf[:] = compute_humanoid_reward(
                                                  self._curr_ref_obs,
                                                  self._curr_obs,
                                                  self._hist_obs,
                                                  len(self._key_body_ids),
                                                  self._motion_data.reward_weights,
                                                  )

        return

    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]

        if (env_ids is None):
            
            self._curr_obs[:] = build_hoi_observations(self._humanoid_root_states[:,0:3], #self._rigid_body_pos[:, 0, :],
                                                               self._humanoid_root_states[:,3:7], #self._rigid_body_rot[:, 0, :],
                                                               self._humanoid_root_states[:,7:10], #self._rigid_body_vel[:, 0, :],
                                                               self._humanoid_root_states[:,10:13], #self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, torch.zeros_like(self._humanoid_root_states),
                                                               self._hist_obs,
                                                               self.progress_buf)
        else:
            self._curr_obs[env_ids] = build_hoi_observations(self._humanoid_root_states[env_ids][:,0:3], #self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._humanoid_root_states[env_ids][:,3:7], #self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._humanoid_root_states[env_ids][:,7:10], #self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._humanoid_root_states[env_ids][:,10:13], #self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, torch.zeros_like(self._humanoid_root_states[env_ids]),
                                                                   self._hist_obs[env_ids],
                                                                   self.progress_buf[env_ids])
        return
#####################################################################
###=========================jit functions=========================###
#####################################################################
# @torch.jit.script
def compute_humanoid_reward(hoi_ref: Tensor, hoi_obs: Tensor, hoi_obs_hist: Tensor, 
                            len_keypos: int, w: Dict[str, Tensor], 
                            ) -> Tensor:

    ### data preprocess ###

    # simulated states
    root_pos = hoi_obs[:,:3]
    root_rot = hoi_obs[:,3:3+3]
    dof_pos = hoi_obs[:,6:6+52*3]
    dof_pos_vel = hoi_obs[:,162:162+52*3]
    obj_pos = hoi_obs[:,318:318+3]
    obj_rot = hoi_obs[:,321:321+4]
    obj_pos_vel = hoi_obs[:,325:325+3]
    key_pos = hoi_obs[:,328:328+len_keypos*3]
    contact = hoi_obs[:,-1:]# fake one
    key_pos = torch.cat((root_pos, key_pos),dim=-1)
    body_rot = torch.cat((root_rot, dof_pos),dim=-1)
    ig = key_pos.view(-1,len_keypos+1,3).transpose(0,1) - obj_pos[:,:3]
    ig_wrist = ig.transpose(0,1)[:,0:7+1,:].view(-1,(7+1)*3) #ZC
    ig = ig.transpose(0,1).view(-1,(len_keypos+1)*3)

    dof_pos_vel_hist = hoi_obs_hist[:,162:162+52*3] #ZC

    # reference states
    ref_root_pos = hoi_ref[:,:3]
    ref_root_rot = hoi_ref[:,3:3+3]
    ref_dof_pos = hoi_ref[:,6:6+52*3]
    ref_dof_pos_vel = hoi_ref[:,162:162+52*3]
    ref_obj_pos = hoi_ref[:,318:318+3]
    ref_obj_rot = hoi_ref[:,321:321+4]
    ref_obj_pos_vel = hoi_ref[:,325:325+3]
    ref_key_pos = hoi_ref[:,328:328+len_keypos*3]
    ref_obj_contact = hoi_ref[:,-1:]
    ref_key_pos = torch.cat((ref_root_pos, ref_key_pos),dim=-1)
    ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos),dim=-1)
    ref_ig = ref_key_pos.view(-1,len_keypos+1,3).transpose(0,1) - ref_obj_pos[:,:3]
    ref_ig_wrist = ref_ig.transpose(0,1)[:,0:7+1,:].view(-1,(7+1)*3) #ZC
    ref_ig = ref_ig.transpose(0,1).view(-1,(len_keypos+1)*3)

    # return torch.exp(-torch.mean((root_pos - ref_root_pos)**2,dim=-1)) \
    # * torch.exp(-torch.mean((ref_dof_pos - dof_pos)**2,dim=-1)) \
    # * torch.exp(-torch.mean((ref_obj_pos - obj_pos)**2,dim=-1))
    # test for 0th hoi reward (failed)(because of forward kinematics not applied to cal body pos in reset)

    ### body reward ###

    # body pos reward
    ep = torch.mean((ref_key_pos - key_pos)**2,dim=-1)
    # ep = torch.mean((ref_key_pos[:,0:(7+1)*3] - key_pos[:,0:(7+1)*3])**2,dim=-1) #ZC
    rp = torch.exp(-ep*w['p'])

    # body rot reward
    er = torch.mean((ref_body_rot - body_rot)**2,dim=-1)
    rr = torch.exp(-er*w['r'])

    # body pos vel reward
    epv = torch.zeros_like(ep)
    rpv = torch.exp(-epv*w['pv'])

    # body rot vel reward
    erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2,dim=-1)
    rrv = torch.exp(-erv*w['rv'])

    # body vel smoothness reward
    # e_vel_diff = torch.mean((dof_pos_vel - dof_pos_vel_hist)**2, dim=-1)
    # r_vel_diff = torch.exp(-e_vel_diff * 0.05) #w['vel_diff']
    e_vel_diff = torch.mean((dof_pos_vel - dof_pos_vel_hist)**2 / (((ref_dof_pos_vel**2) + 1e-12)*1e12), dim=-1)
    r_vel_diff = torch.exp(-e_vel_diff * 0.1) #w['vel_diff']


    rb = rp*rr*rpv*rrv *r_vel_diff #ZC3
    # print(rp, rr, rpv, rrv) 

    reward = rb

    return reward
    
@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, 
                        #    hoi_ref, hoi_obs, 
                           envid2episode_lengths, isTest, maxEpisodeLength, 
                        #    skill_label
                            NR = False
                           ):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, bool, int, bool) -> Tuple[Tensor, Tensor]
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

    if isTest and NR:
        reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC
    # if isTest and maxEpisodeLength > 0 :
    #     reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    # else:
    #     reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC
    else:
        reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    
    # reset = torch.zeros_like(reset_buf) #ZC300

    return reset, terminated

@torch.jit.script
def build_hoi_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, target_states, hist_obs, progress_buf):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, Tensor, Tensor, Tensor) -> Tensor
    ## diffvel, set 0 for the first frame
    # hist_dof_pos = hist_obs[:,6:6+156]
    # dof_diffvel = (dof_pos - hist_dof_pos)*fps
    # dof_diffvel = dof_diffvel*(progress_buf!=1).to(float).unsqueeze(dim=-1)

    dof_vel = dof_vel*(progress_buf!=1).unsqueeze(dim=-1)

    contact = torch.zeros(key_body_pos.shape[0],1,device=dof_vel.device)
    obs = torch.cat((root_pos, torch_utils.quat_to_exp_map(root_rot),
                    dof_pos, dof_vel,
                    target_states[:,:10],
                    key_body_pos.contiguous().view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), 
                    contact,
                    ), dim=-1)
    return obs