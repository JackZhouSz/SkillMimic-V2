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

from env.tasks.skillmimic1_reweight import SkillMimic1BallPlayReweight


class SkillMimic1BallPlayUnified(SkillMimic1BallPlayReweight): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.skill_labels = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._curr_ref_obs, self._curr_obs, self._motion_data.envid2episode_lengths,
                                                   self.isTest, self.cfg["env"]["episodeLength"],
                                                   self.skill_labels, #Z unified
                                                   self.cfg["env"]["NR"]
                                                   )
        # reweight the motion
        reset_env_ids = torch.nonzero(self.reset_buf == 1).squeeze(-1)
        self._reweight_motion(reset_env_ids)

        # if self.reset_buf[:].sum() != 0:
        #     print(self.progress_buf[0])

        return
    
    def _compute_reward(self):
        self.rew_buf[:] = compute_humanoid_reward(
                                                  self._curr_ref_obs,
                                                  self._curr_obs,
                                                  self._hist_obs,
                                                  self._contact_forces,
                                                  self._tar_contact_forces,
                                                  len(self._key_body_ids),
                                                  self._motion_data.reward_weights,
                                                  self.skill_labels, #Z unified
                                                  )
        return
    

    def after_reset_actors(self, env_ids):
        super().after_reset_actors(env_ids)
        self.skill_labels[env_ids] = self._motion_data.motion_class_tensor[self.motion_ids] #torch.tensor(, device=self.device, dtype=torch.long)

    # def _generate_fall_states(self):
    #     """生成摔倒状态：随机初始化root姿态，给随机动作并模拟一段时间，让机器人摔倒，再记录此时的状态。"""
    #     max_steps = 150
        
    #     env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    #     # 随机化root姿态
    #     root_states = self._initial_humanoid_root_states[env_ids].clone()
    #     root_states[..., 3:7] = torch.randn_like(root_states[..., 3:7])
    #     root_states[..., 3:7] = torch.nn.functional.normalize(root_states[..., 3:7], dim=-1)
    #     self._humanoid_root_states[env_ids] = root_states

    #     env_ids_int32 = self._humanoid_actor_ids[env_ids]
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                  gymtorch.unwrap_tensor(self._root_states),
    #                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    #     self.gym.set_dof_state_tensor_indexed(self.sim,
    #                                           gymtorch.unwrap_tensor(self._dof_state),
    #                                           gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    #     # 给随机动作让角色摔倒
    #     rand_actions = np.random.uniform(-0.5, 0.5, size=[self.num_envs, self.get_action_size()])
    #     rand_actions = torch.tensor(rand_actions, dtype=torch.float32, device=self.device)
    #     self.pre_physics_step(rand_actions)

    #     # step physics and render each frame
    #     for i in range(max_steps):
    #         self.render()
    #         self.gym.simulate(self.sim)

    #     self._refresh_sim_tensors()

    #     # 记录摔倒状态
    #     self._fall_root_states = self._humanoid_root_states.clone()
    #     self._fall_root_states[:, 7:13] = 0
    #     self._fall_dof_pos = self._dof_pos.clone()
    #     self._fall_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

    #     return
    
    # def _reset_actors(self, env_ids):
    #     self._reset_fall_episode(env_ids)
            
    # def _reset_fall_episode(self, env_ids):
    #     self._generate_fall_states()
    #     # 从预先生成的摔倒状态中选取一组状态，用于初始化这些env
    #     fall_state_ids = torch.randint_like(env_ids, low=0, high=self._fall_root_states.shape[0])
    #     self._humanoid_root_states[env_ids] = self._fall_root_states[fall_state_ids]
    #     self._dof_pos[env_ids] = self._fall_dof_pos[fall_state_ids]
    #     self._dof_vel[env_ids] = self._fall_dof_vel[fall_state_ids]
    

#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def compute_humanoid_reward(hoi_ref: Tensor, hoi_obs: Tensor, hoi_obs_hist: Tensor, contact_buf: Tensor, tar_contact_forces: Tensor, 
                            len_keypos: int, w: Dict[str, Tensor],  
                            skill_label: Tensor
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


    ### object reward ###

    # object pos reward
    eop = torch.mean((ref_obj_pos - obj_pos)**2,dim=-1)
    rop = torch.exp(-eop*w['op'])

    # object rot reward
    eor = torch.zeros_like(ep) #torch.mean((ref_obj_rot - obj_rot)**2,dim=-1)
    ror = torch.exp(-eor*w['or'])

    # object pos vel reward
    eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2,dim=-1)
    ropv = torch.exp(-eopv*w['opv'])

    # object rot vel reward
    eorv = torch.zeros_like(ep) #torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
    rorv = torch.exp(-eorv*w['orv'])

    ro = rop*ror*ropv*rorv


    ### interaction graph reward ###

    eig = torch.mean((ref_ig - ig)**2,dim=-1) #Zw
    # eig = torch.mean((ref_ig_wrist - ig_wrist)**2,dim=-1)
    rig = torch.exp(-eig*w['ig'])


    ### simplified contact graph reward ###

    # Since Isaac Gym does not yet provide API for detailed collision detection in GPU pipeline, 
    # we use force detection to approximate the contact status.
    # In this case we use the CG node istead of the CG edge for imitation.
    # TODO: update the code once collision detection API is available.

    ## body ids
    # Pelvis, 0 
    # L_Hip, 1 
    # L_Knee, 2
    # L_Ankle, 3
    # L_Toe, 4
    # R_Hip, 5 
    # R_Knee, 6
    # R_Ankle, 7
    # R_Toe, 8
    # Torso, 9
    # Spine, 10 
    # Spine1, 11
    # Chest, 12
    # Neck, 13
    # Head, 14
    # L_Thorax, 15
    # L_Shoulder, 16
    # L_Elbow, 17
    # L_Wrist, 18
    # L_Hand, 19-33
    # R_Thorax, 34 
    # R_Shoulder, 35
    # R_Elbow, 36
    # R_Wrist, 37
    # R_Hand, 38-52

    # body contact
    contact_body_ids = [0,1,2,5,6,9,10,11,12,13,14,15,16,17,34,35,36]
    body_contact_buf = contact_buf[:, contact_body_ids, :].clone()
    body_contact = torch.all(torch.abs(body_contact_buf) < 0.1, dim=-1)
    body_contact = 1. - torch.all(body_contact, dim=-1).to(float) # =0 when no contact happens to the body

    # object contact
    obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(float) # =1 when contact happens to the object

    ref_body_contact = torch.zeros_like(ref_obj_contact) # no body contact for all time
    ecg1 = torch.abs(body_contact - ref_body_contact[:,0])
    rcg1 = torch.exp(-ecg1*w['cg1'])
    ecg2 = torch.abs(obj_contact - ref_obj_contact[:,0])
    rcg2 = torch.exp(-ecg2*w['cg2'])

    rcg = rcg1*rcg2


    ### task-agnostic HOI imitation reward ###
    # reward = rb*ro*rig*rcg
    ########## Modified by Runyi ##########
    reward = torch.where((skill_label!=0) & (skill_label!=10), 
                         rb*ro*rig*rcg, rb)

    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, hoi_ref, hoi_obs, envid2episode_lengths,
                           isTest, maxEpisodeLength, 
                           skill_label,
                           NR = False
                           ):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, Tensor, bool, int, Tensor, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rigid_body_pos[:, 0, 2] # root height
        body_fall = body_height < termination_heights# [4096] 
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 1)
        
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
        ########## Modified by Runyi ##########
        skill_mask = (skill_label == 0)
        terminated = torch.where(skill_mask, torch.zeros_like(terminated), terminated)
        #######################################

    if isTest and NR:
        reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC
    elif isTest and maxEpisodeLength > 0 :
        reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    else:
        reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC

    return reset, terminated