from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict
import glob, os, random
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from datetime import datetime
import pickle

from utils import torch_utils
from utils.motion_data_handler import MotionDataHandler
from utils.history_encoder import HistoryEncoder

from env.tasks.skillmimic_localhist import SkillMimicBallPlayLocalHist

class SkillMimicBallPlayLocalHistUnifiedRIS(SkillMimicBallPlayLocalHist): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        self.history_length = cfg['env']['historyLength']

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.skill_labels = torch.ones(self.num_envs, device=self.device, dtype=torch.long) * 100
        self.state_switch_flags = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if 'stateSearchGraph' in cfg['env']:
            with open(f"{cfg['env']['stateSearchGraph']}", "rb") as f:
                self.state_search_graph = pickle.load(f)
        self.max_sim = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        

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
        reset_env_ids = torch.nonzero(self.reset_buf == 1).squeeze()
        self._reweight_motion(reset_env_ids)
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

    def _reset_random_ref_state_init(self, env_ids): #Z11
        motion_ids, motion_times = super()._reset_random_ref_state_init(env_ids)
        motion_ids, motion_times = self._init_with_random_noise(env_ids, motion_ids, motion_times)
        motion_ids, motion_times = self._init_from_random_skill(env_ids, motion_ids, motion_times)
        skill_label = self._motion_data.motion_class[motion_ids.cpu().numpy()]
        self.skill_labels[env_ids] = torch.from_numpy(skill_label).to(self.device)
        return motion_ids, motion_times

    def _reset_deterministic_ref_state_init(self, env_ids):
        motion_ids, motion_times = super()._reset_deterministic_ref_state_init(env_ids)
        motion_ids, motion_times = self._init_with_random_noise(env_ids, motion_ids, motion_times)
        motion_ids, motion_times = self._init_from_random_skill(env_ids, motion_ids, motion_times)
        skill_label = self._motion_data.motion_class[motion_ids.cpu().numpy()]
        self.skill_labels[env_ids] = torch.from_numpy(skill_label).to(self.device)
        return motion_ids, motion_times

    def _init_with_random_noise(self, env_ids, motion_ids, motion_times): 
        # Random noise for initial state
        self.state_random_flags = [np.random.rand() < self.cfg['env']['state_init_random_prob'] for _ in env_ids]
        
        for ind, env_id in enumerate(env_ids):
            if self.state_random_flags[ind]:
                noise_weight = 0.1 if self.skill_labels[env_id] != 0 else 1.0
                self.init_root_pos[env_id, 2] += random.random() * noise_weight
                self.init_root_rot[env_id] = apply_random_rotation(self.init_root_rot[env_id], noise_weight)
                self.init_root_pos_vel[env_id] += (torch.randn_like(self.init_root_pos_vel[env_id])* 2 - 1) * 0.1
                self.init_root_rot_vel[env_id] += (torch.randn_like(self.init_root_rot_vel[env_id])* 2 - 1) * 0.1
                if self.skill_labels[env_id] != 0:
                    self.init_dof_pos[env_id] += (torch.randn_like(self.init_dof_pos[env_id])* 2 - 1) * 0.1
                else:
                    self.init_dof_pos[env_id] += (torch.randn_like(self.init_dof_pos[env_id])* 2 - 1) * 0.5
                self.init_dof_pos_vel[env_id]  += (torch.randn_like(self.init_dof_pos_vel[env_id])* 2 - 1) * 0.1
                self.init_obj_pos[env_id, 2] += random.random() * 0.1
                self.init_obj_pos_vel[env_id] += (torch.randn_like(self.init_obj_pos_vel[env_id])* 2 - 1) * 0.1
                self.init_obj_rot[env_id] = apply_random_rotation(self.init_obj_rot[env_id], 0.1)
                self.init_obj_rot_vel[env_id] += (torch.randn_like(self.init_obj_rot_vel[env_id])* 2 - 1) * 0.1

                noisy_motion = {
                    'root_pos': self.init_root_pos[env_id],
                    'root_pos_vel': self.init_root_pos_vel[env_id],
                    'root_rot': self.init_root_rot[env_id],
                    'root_rot_vel': self.init_root_rot_vel[env_id],
                    'key_body_pos': self._rigid_body_pos[env_id, self._key_body_ids, :],
                    'key_body_pos_vel': self._rigid_body_vel[env_id, self._key_body_ids, :],
                    'dof_pos': self.init_dof_pos[env_id],
                    'dof_pos_vel': self.init_dof_pos_vel[env_id],
                    'obj_pos': self.init_obj_pos[env_id],
                    'obj_pos_vel': self.init_obj_pos_vel[env_id],
                    'obj_rot': self.init_obj_rot[env_id],
                    'obj_rot_vel': self.init_obj_rot_vel[env_id],
                }
                motion_id = motion_ids[ind:ind+1]
                new_source_motion_time, self.max_sim[env_id] = self._motion_data.noisy_resample_time(noisy_motion, motion_id)
                
                # resample the hoi_data_batch
                self.hoi_data_batch[env_id], _, _, _, _, _, _, _, _, _, _ \
                    = self._motion_data.get_initial_state(env_ids[ind:ind+1], motion_id, new_source_motion_time)
                
                # change motion_times
                motion_times[ind:ind+1] = new_source_motion_time

                # if self.isTest:
                #     print(f"Random noise added to initial state for env {env_id}")
        return motion_ids, motion_times
    

    def _init_from_random_skill(self, env_ids, motion_ids, motion_times): 
        # Random init to other skills
        state_switch_flags = [np.random.rand() < self.cfg['env']['state_switch_prob'] for _ in env_ids]
        if self.cfg['env']['state_switch_prob'] > 0:
            # # # getup skill can't switch to other skills
            # getup_index = np.where(skill_label==0)[0]
            # state_switch_flags = [state_switch_flags[i] if i not in getup_index else False for i in range(len(state_switch_flags))]
            for ind, env_id in enumerate(env_ids):
                if state_switch_flags[ind] and not self.state_random_flags[ind]:
                    switch_motion_class = self._motion_data.motion_class[motion_ids[ind]]
                    switch_motion_id = motion_ids[ind:ind+1]
                    switch_motion_time = motion_times[ind:ind+1]

                    # load source motion info from state_search_graph
                    source_motion_class, source_motion_id, source_motion_time, max_sim = \
                        random.choice(self.state_search_graph[switch_motion_class][switch_motion_id.item()][switch_motion_time.item()])
                    if source_motion_id is None and source_motion_time is None:
                        # print(f"Switch from time {switch_motion_time.item()} of {switch_motion_id.item()} failed")
                        continue
                    else:
                        self.max_sim[env_id] = max_sim
                    source_motion_id = torch.tensor([source_motion_id], device=self.device)
                    source_motion_time = torch.tensor([source_motion_time], device=self.device)
                    
                    # resample the hoi_data_batch
                    self.hoi_data_batch[env_id], _, _,  _, _, _, _, _, _, _, _ = \
                        self._motion_data.get_initial_state(env_ids[ind:ind+1], source_motion_id, source_motion_time)
                    self.hoi_data_batch[env_id] = compute_local_hoi_data(self.hoi_data_batch[env_id], self.init_root_pos[env_id], 
                                                                         self.init_root_rot[env_id], len(self._key_body_ids))
                    # change skill label
                    skill_label = self._motion_data.motion_class[source_motion_id.tolist()]
                    self.hoi_data_label_batch[env_id] = F.one_hot(torch.tensor(skill_label, device=self.device), num_classes=self.condition_size).float()
                    # change motion_ids and motion_times
                    motion_ids[ind:ind+1] = source_motion_id
                    motion_times[ind:ind+1] = source_motion_time

                    if self.isTest:
                        print(f"Switched from skill {switch_motion_class} to {source_motion_class} for env {env_id}")

        return motion_ids, motion_times 

    def _compute_observations(self, env_ids=None): # called @ reset & post step
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        obj_obs = self._compute_obj_obs(env_ids)
        # obj_obs_cond = ((self.skill_labels[env_ids] != 0) & (self.skill_labels[env_ids] != 10)).unsqueeze(-1).squeeze(0)
        # obj_obs = torch.where(obj_obs_cond, obj_obs, torch.zeros_like(obj_obs))
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
    
    
class SkillMimicBallPlayLocalHistUnifiedRISBuffernode(SkillMimicBallPlayLocalHistUnifiedRIS):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                    sim_params=sim_params,
                    physics_engine=physics_engine,
                    device_type=device_type,
                    device_id=device_id,
                    headless=headless)
        self.buffer_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    
    def _compute_buffer_steps(self, env_ids):
        for env_id in env_ids:
            if self.max_sim[env_id] > 0.5:
                self.buffer_steps[env_id] = 0
            elif self.max_sim[env_id] != 0:
                self.buffer_steps[env_id] = min(-int(torch.floor(torch.log10(self.max_sim[env_id]))), 10)
            
    def _reset_state_init(self, env_ids):
        super()._reset_state_init(env_ids)
        if self.progress_buf_total > 0:
            self._compute_buffer_steps(env_ids)
            self.max_sim = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    # def _compute_reward(self):
    #     super()._compute_reward()
    #     if torch.nonzero(self.buffer_steps).numel() != 0:
    #         for buffer_env_id in torch.nonzero(self.buffer_steps).squeeze():
    #             buffer_motion_id = self.motion_ids_total[buffer_env_id].item()
    #             indices = (self.motion_ids_total == buffer_motion_id)
    #             self.rew_buf[buffer_env_id] = self.rew_buf[indices].mean().item()
    #     self.buffer_steps = torch.where(self.buffer_steps > 0, self.buffer_steps - 1, self.buffer_steps)
    #     return

    def _compute_reward(self):
        super()._compute_reward()
        non_zero_indices = torch.nonzero(self.buffer_steps)
        
        if non_zero_indices.numel() != 0:
            # Use view to ensure it's a 1-D tensor
            for buffer_env_id in non_zero_indices.view(-1):
                buffer_motion_id = self.motion_ids_total[buffer_env_id].item()
                indices = (self.motion_ids_total == buffer_motion_id)
                self.rew_buf[buffer_env_id] = self.rew_buf[indices].mean().item()
        
        self.buffer_steps = torch.where(self.buffer_steps > 0, self.buffer_steps - 1, self.buffer_steps)
        return
    

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
    body_contact = 1. - torch.all(body_contact, dim=-1).to(torch.float) # =0 when no contact happens to the body

    # object contact
    obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(torch.float) # =1 when contact happens to the object

    ref_body_contact = torch.zeros_like(ref_obj_contact) # no body contact for all time
    ecg1 = torch.abs(body_contact - ref_body_contact[:,0])
    rcg1 = torch.exp(-ecg1*w['cg1'])
    ecg2 = torch.abs(obj_contact - ref_obj_contact[:,0])
    rcg2 = torch.exp(-ecg2*w['cg2'])

    rcg = rcg1*rcg2


    ### task-agnostic HOI imitation reward ###
    reward = torch.where((skill_label!=0) & (skill_label!=10), rb*ro*rig*rcg, rb)
    
    return reward

# @torch.jit.script
def apply_random_rotation(init_obj_rot, max_radians):
    # 生成在 [-max_radians, max_radians] 之间的随机旋转角度 (X, Y, Z)
    rand_angles = (torch.rand(3) * 2 - 1) * max_radians
    rand_angles = rand_angles.to(init_obj_rot.device)
    roll, pitch, yaw = rand_angles[0], rand_angles[1], rand_angles[2]
    # 获取当前四元数的欧拉角
    current_euler = torch_utils.quat_to_euler(init_obj_rot)  # (N, 3)
    roll_cur, pitch_cur, yaw_cur = current_euler[0], current_euler[1], current_euler[2]
    # 施加随机旋转
    roll_new, pitch_new, yaw_new = roll_cur + roll, pitch_cur + pitch, yaw_cur + yaw
    # 转回四元数
    new_quat = torch_utils.quat_from_euler_xyz(roll_new, pitch_new, yaw_new)
    return new_quat

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, hoi_ref, hoi_obs, envid2episode_lengths,
                           isTest, maxEpisodeLength, 
                           skill_label,
                           NR = False,
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
    
    # reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    # reset = torch.zeros_like(reset_buf) #ZC300

    return reset, terminated

@torch.jit.script
def compute_local_hoi_data(hoi_data_batch: Tensor, switch_root_pos: Tensor, switch_root_rot_quat: Tensor, len_keypos: int) -> Tensor:
    # hoi_data_batch (60, 337)
    # switch_root_rot_quat (1, 4)
    local_hoi_data_batch = hoi_data_batch.clone()
    init_root_pos = hoi_data_batch[0,:3]
    init_root_rot = hoi_data_batch[0,3:3+3]

    root_pos = hoi_data_batch[:,:3]
    root_rot = hoi_data_batch[:,3:3+3]
    dof_pos = hoi_data_batch[:,6:6+52*3]
    dof_pos_vel = hoi_data_batch[:,162:162+52*3]
    obj_pos = hoi_data_batch[:,318:318+3]
    obj_rot = hoi_data_batch[:,321:321+4]
    obj_pos_vel = hoi_data_batch[:,325:325+3]
    key_pos = hoi_data_batch[:,328:328+len_keypos*3]
    contact = hoi_data_batch[:,-1:] # fake one
    nframes = hoi_data_batch.shape[0]

    switch_root_rot_euler_z = torch_utils.quat_to_euler(switch_root_rot_quat)[2] # (1, 1) 
    source_root_rot_euler_z = torch_utils.quat_to_euler(torch_utils.exp_map_to_quat(init_root_rot))[2]  # (1, 1) 
    source_to_switch_euler_z = switch_root_rot_euler_z - source_root_rot_euler_z # (1, 1)
    source_to_switch_euler_z = (source_to_switch_euler_z + torch.pi) % (2 * torch.pi) - torch.pi  # 归一化到 [-pi, pi]
    source_to_switch_euler_z = source_to_switch_euler_z.squeeze()
    zeros = torch.zeros_like(source_to_switch_euler_z)
    source_to_switch = quat_from_euler_xyz(zeros, zeros, source_to_switch_euler_z)
    source_to_switch = source_to_switch.repeat(nframes, 1) # (nframes, 4)

    # referece to the new root
    # local_root_pos
    relative_root_pos = root_pos - init_root_pos
    local_relative_root_pos = torch_utils.quat_rotate(source_to_switch, relative_root_pos)
    local_root_pos = local_relative_root_pos + switch_root_pos
    local_root_pos[:, 2] = root_pos[:, 2]
    # local_root_rot
    root_rot_quat = torch_utils.exp_map_to_quat(root_rot)
    local_root_rot = torch_utils.quat_to_exp_map(torch_utils.quat_multiply(source_to_switch, root_rot_quat))
    # local_obj_pos
    relative_obj_pos = obj_pos - init_root_pos
    local_relative_obj_pos = torch_utils.quat_rotate(source_to_switch, relative_obj_pos)
    local_obj_pos = local_relative_obj_pos + switch_root_pos
    local_obj_pos[:, 2] = obj_pos[:, 2]
    # local_obj_pos_vel
    local_obj_pos_vel = torch_utils.quat_rotate(source_to_switch, obj_pos_vel)
    # local_key_pos
    key_pos = key_pos.reshape(-1, len_keypos, 3)
    relative_key_pos = key_pos - init_root_pos
    local_relative_key_pos = torch.zeros_like(relative_key_pos)
    for i in range(len_keypos):
        local_relative_key_pos[:,i] = torch_utils.quat_rotate(source_to_switch, relative_key_pos[:,i])
    local_key_pos = local_relative_key_pos + switch_root_pos
    local_key_pos[..., 2] = key_pos[..., 2]
    # print('key_pos:', key_pos[20, 8])
    # print('local_key_pos:', local_key_pos[20, 8])

    local_hoi_data_batch[:,:3] =  local_root_pos
    local_hoi_data_batch[:,3:3+3] =  local_root_rot
    local_hoi_data_batch[:,318:318+3] = local_obj_pos
    local_hoi_data_batch[:,325:325+3] = local_obj_pos_vel
    local_hoi_data_batch[:,328:328+len_keypos*3] = local_key_pos.reshape(-1, len_keypos*3)

    return local_hoi_data_batch