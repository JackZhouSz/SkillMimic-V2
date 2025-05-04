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

from env.tasks.skillmimic import SkillMimicBallPlay


class SkillMimicBallPlayDomRand(SkillMimicBallPlay): 
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
        skill_label = self._motion_data.motion_class[motion_ids.tolist()]
        # Random noise for initial state
        self.state_random_flags = [np.random.rand() < self.cfg['env']['state_init_random_prob'] for _ in env_ids]
        if self.cfg['env']['state_init_random_prob'] > 0:
            for ind, env_id in enumerate(env_ids):
                if self.state_random_flags[ind]:
                    noise_weight = [0.1 for _ in range(10)] if skill_label[ind] != 0 else [1.0, 1.0] + [0.1 for _ in range(8)]
                    self.init_root_pos[env_id, 2] += random.random() * noise_weight[0]
                    self.init_root_rot[env_id] += torch.randn_like(self.init_root_rot[env_id]) * noise_weight[1]
                    self.init_root_pos_vel[env_id] += torch.randn_like(self.init_root_pos_vel[env_id]) * noise_weight[2]
                    self.init_root_rot_vel[env_id] += torch.randn_like(self.init_root_rot_vel[env_id]) * noise_weight[3]
                    self.init_dof_pos[env_id] += torch.randn_like(self.init_dof_pos[env_id]) * noise_weight[4]
                    self.init_dof_pos_vel[env_id]  += torch.randn_like(self.init_dof_pos_vel[env_id]) * noise_weight[5]
                    self.init_obj_pos[env_id, 2] += random.random() * noise_weight[6]
                    self.init_obj_pos_vel[env_id] += torch.randn_like(self.init_obj_pos_vel[env_id]) * noise_weight[7]
                    self.init_obj_rot[env_id] += torch.randn_like(self.init_obj_rot[env_id]) * noise_weight[8]
                    self.init_obj_rot_vel[env_id] += torch.randn_like(self.init_obj_rot_vel[env_id]) * noise_weight[9]
                    if self.isTest:
                        print(f"Random noise added to initial state for env {env_id}")


class SkillMimicBallPlayRIS(SkillMimicBallPlay): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        if 'stateSearchGraph' in cfg['env']:
            with open(f"{cfg['env']['stateSearchGraph']}", "rb") as f:
                self.state_search_graph = pickle.load(f)
        self.max_sim = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
    def _reset_random_ref_state_init(self, env_ids): #Z11
        motion_ids, motion_times = super()._reset_random_ref_state_init(env_ids)
        motion_ids, motion_times = self._init_with_risrand_noise(env_ids, motion_ids, motion_times)
        motion_ids, motion_times = self._init_from_risrand_skill(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        motion_ids, motion_times = super()._reset_deterministic_ref_state_init(env_ids)
        motion_ids, motion_times = self._init_with_risrand_noise(env_ids, motion_ids, motion_times)
        motion_ids, motion_times = self._init_from_risrand_skill(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times
    
    def after_reset_actors(self, env_ids):
        super().after_reset_actors(env_ids)

    def _init_with_risrand_noise(self, env_ids, motion_ids, motion_times): 
        skill_label = self._motion_data.motion_class[motion_ids.tolist()]
        # Random noise for initial state
        self.state_random_flags = [np.random.rand() < self.cfg['env']['state_init_random_prob'] for _ in env_ids]
        if self.cfg['env']['state_init_random_prob'] > 0:
            for ind, env_id in enumerate(env_ids):
                if self.state_random_flags[ind]:
                    noise_weight = [0.1 for _ in range(10)] if skill_label[ind] != 0 else [1.0, 1.0] + [0.1 for _ in range(8)]
                    self.init_root_pos[env_id, 2] += random.random() * noise_weight[0]
                    self.init_root_rot[env_id] += torch.randn_like(self.init_root_rot[env_id]) * noise_weight[1]
                    self.init_root_pos_vel[env_id] += torch.randn_like(self.init_root_pos_vel[env_id]) * noise_weight[2]
                    self.init_root_rot_vel[env_id] += torch.randn_like(self.init_root_rot_vel[env_id]) * noise_weight[3]
                    self.init_dof_pos[env_id] += torch.randn_like(self.init_dof_pos[env_id]) * noise_weight[4]
                    self.init_dof_pos_vel[env_id]  += torch.randn_like(self.init_dof_pos_vel[env_id]) * noise_weight[5]
                    self.init_obj_pos[env_id, 2] += random.random() * noise_weight[6]
                    self.init_obj_pos_vel[env_id] += torch.randn_like(self.init_obj_pos_vel[env_id]) * noise_weight[7]
                    self.init_obj_rot[env_id] += torch.randn_like(self.init_obj_rot[env_id]) * noise_weight[8]
                    self.init_obj_rot_vel[env_id] += torch.randn_like(self.init_obj_rot_vel[env_id]) * noise_weight[9]
                    noisy_motion = {
                        'root_pos': self.init_root_pos[env_id],
                        'key_body_pos': self._rigid_body_pos[env_id, self._key_body_ids, :],
                        'key_body_pos_vel': self._rigid_body_vel[env_id, self._key_body_ids, :],
                        'root_rot': self.init_root_rot[env_id],
                        'root_pos_vel': self.init_root_pos_vel[env_id],
                        'root_rot_vel': self.init_root_rot_vel[env_id],
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


    def _init_from_risrand_skill(self, env_ids, motion_ids, motion_times): 
        # Random init to other skills
        state_switch_flags = [np.random.rand() < self.cfg['env']['state_switch_prob'] for _ in env_ids]
        if self.cfg['env']['state_switch_prob'] > 0:
            for ind, env_id in enumerate(env_ids):
                if state_switch_flags[ind] and not self.state_random_flags[ind]:
                    switch_motion_class = self._motion_data.motion_class[motion_ids[ind]]
                    switch_motion_id = motion_ids[ind:ind+1]
                    switch_motion_time = motion_times[ind:ind+1]

                    # load source motion info from state_search_graph
                    source_motion_class, source_motion_id, source_motion_time, max_sim = random.choice(self.state_search_graph[switch_motion_class][switch_motion_id.item()][switch_motion_time.item()])
                    if source_motion_id is None and source_motion_time is None:
                        # print(f"Switch from time {switch_motion_time.item()} of {switch_motion_id.item()} failed")
                        continue
                    else:
                        self.max_sim[env_id] = max_sim
                    source_motion_id = torch.tensor([source_motion_id], device=self.device)
                    source_motion_time = torch.tensor([source_motion_time], device=self.device)
                    
                    # get hoi_data_batch from source motion
                    self.hoi_data_batch[env_id], _, _,  _, _, _, _, _, _, _, _ = \
                        self._motion_data.get_initial_state(env_ids[ind:ind+1], source_motion_id, source_motion_time)
                    self.hoi_data_batch[env_id] = compute_local_hoi_data(self.hoi_data_batch[env_id], self.init_root_pos[env_id], 
                                                                         self.init_root_rot[env_id], len(self._key_body_ids))
                    # change skill label -> source_motion_class
                    skill_label = self._motion_data.motion_class[source_motion_id.tolist()]
                    self.hoi_data_label_batch[env_id] = F.one_hot(torch.tensor(skill_label, device=self.device), num_classes=self.condition_size).float()
                    # change motion_ids and motion_times
                    motion_ids[ind:ind+1] = source_motion_id
                    motion_times[ind:ind+1] = source_motion_time

                    if self.isTest:
                        print(f"Switched from skill {switch_motion_class} to {source_motion_class} for env {env_id}")
        
        return motion_ids, motion_times 


class SkillMimicBallPlayRISBuffernode(SkillMimicBallPlayRIS):
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