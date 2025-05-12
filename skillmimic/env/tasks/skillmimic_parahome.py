from enum import Enum
import torch.nn.functional as F
import numpy as np
import math
import torch
from torch import Tensor
from typing import Tuple, Dict
import glob, os, random
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from datetime import datetime

from utils import torch_utils
from utils.paramotion_data_handler import ParaMotionDataHandler

from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObject
from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObjectParahome


class SkillMimicParahome(HumanoidWholeBodyWithObjectParahome): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = str(cfg["env"]["stateInit"])
        if state_init.lower() == "random":
            self._state_init = -1
            print("Random Reference State Init (RRSI)")
        else:
            self._state_init = int(state_init)
            print(f"Deterministic Reference State Init from {self._state_init}")

        self.motion_file = cfg['env']['motion_file']
        self.play_dataset = cfg['env']['playdataset']
        self.reward_weights_default = cfg["env"]["rewardWeights"]
        self.save_images = cfg['env']['saveImages']
        self.save_images_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.init_vel = cfg['env']['initVel']
        self.isTest = cfg['args'].test

        self.condition_size = 64

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.progress_buf_total = 0
        self.max_epochs = cfg['env']['maxEpochs']
        self.adapt_prob = cfg['env']['adapt_prob']
        self.ref_hoi_obs_size = 6 + self._dof_obs_size*2 + len(self.cfg["env"]["keyBodies"])*3 + 1 + 10
        
        ########################## Rewegiht ##########################
        self.reweight = cfg['env']['reweight']
        self.reweight_alpha = cfg['env']['reweight_alpha']
        self._load_motion(self.motion_file, self._dof_obs_size) #ZC1
        self.motion_ids_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.motion_times_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        total_frames = sum([self._motion_data.motion_lengths[motion_id] for motion_id in self._motion_data.hoi_data_dict])
        self.reweight_intervel = 5 * total_frames
        self.average_rewards = {}
        self.envs_reward = torch.zeros(self.num_envs, self.max_episode_length, device=self.device)
        self.motion_time_seqreward = {motion_id: torch.zeros(self._motion_data.motion_lengths[motion_id]-3, device=self.device) 
                                      for motion_id in self._motion_data.hoi_data_dict}
        ##############################################################

        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        
        # get the label of the skill
        # skill_number = int(os.listdir(self.motion_file)[0].split('_')[0])
        # self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(skill_number), num_classes=self.condition_size).repeat(self.num_envs,1).to(self.device)
        self.hoi_data_label_batch = torch.zeros([self.num_envs, self.condition_size], device=self.device, dtype=torch.float)

        self._subscribe_events_for_change_condition()

        self.envid2motid = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) #{}

        self.show_motion_test = False
        self.motion_id_test = 0
        self.succ_pos = []
        self.fail_pos = []
        self.reached_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.int) #metric torch.bool

        self.show_abnorm = [0] * self.num_envs #V1
        self.timestep = 0

        return
    
    def get_state_for_metric(self):
        # 提供 Metric 计算所需的状态
        wrist_index = [10, 33] # Right Wrist, Left Wrist
        threshold = 0.2 if self.skill_name != 'place_pan' else 0.35
        return {
            'ball_pos': self._target_states[..., 0:3],
            'root_pos': self._humanoid_root_states[..., 0:3],
            'root_pos_vel': self._humanoid_root_states[..., 7:10],
            'wrist_pos': self._rigid_body_pos[..., wrist_index, :],
            'threshold': threshold,
            # 'progress': self.progress_buf,
            # 根据需要添加其他状态
        }

    def post_physics_step(self):
        self._update_condition()
        
        # extra calc of self._curr_obs, for imitation reward
        self._compute_hoi_observations()

        super().post_physics_step()

        self._update_hist_hoi_obs()

        return

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()
        return

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        
        obs_size += self.condition_size
        return obs_size

    def get_task_obs_size(self):
        return 0
    
    def _norm_quat(self, quat):
        norm = torch.norm(quat)
        return quat / norm

    def _compute_observations(self, env_ids=None): # called @ reset & post step
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        obj_obs = self._compute_obj_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)
        
        ######### Modified by Runyi #########
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs)

        textemb_batch = self.hoi_data_label_batch[env_ids]
        obs = torch.cat((obs, textemb_batch),dim=-1)
        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids,ts].clone()

        global_mean = obs[~torch.isnan(obs)].mean()
        nan_indices = torch.isnan(obs).nonzero(as_tuple=True)
        if nan_indices[0].numel() > 0:
            for dim in range(obs.shape[-1]):
                valid_values = obs[~torch.isnan(obs[:, dim]), dim]  # 获取非 NaN 值
                mean_value = valid_values.mean() if valid_values.numel() > 0 else global_mean
                obs[torch.isnan(obs[:, dim]), dim] = mean_value
                if valid_values.numel() < obs.shape[0]:
                    nan_envs = nan_indices[0][nan_indices[1] == dim] # 获取 NaN 所在的环境索引
                    print(f"NaN observation in Env: {nan_envs.tolist()}, Dimension {dim}")

        self.obs_buf[env_ids] = obs
        #####################################

        return

    def get_state_init_random_prob(self):
        epoch = int(self.progress_buf_total // 40 / self.max_epochs)
        state_init_random_prob = 0.2 * (math.exp(3*epoch) - 1) / (math.exp(3) - 1) # 0 -> 0.2
        return state_init_random_prob

    def _reset_state_init(self, env_ids):
        if self._state_init == -1:
            self.motion_ids, self.motion_times = self._reset_random_ref_state_init(env_ids) #V1 Random Ref State Init (RRSI)
        elif self._state_init >= 2:
            self.motion_ids, self.motion_times = self._reset_deterministic_ref_state_init(env_ids)
        else:
            assert(False), f"Unsupported state initialization from: {self._state_init}"

    def _reset_actors(self, env_ids):
        self._reset_state_init(env_ids)

        super()._reset_actors(env_ids)

        self.after_reset_actors(env_ids)
        return

    def after_reset_actors(self, env_ids):
        # if self.switch_skill_name is not None: 
        #     skill_dict = {'run': 13, 'lrun': 11, 'rrun': 12, 'layup': 31, 'shot': 9, 'run_no_object': 10, 'getup': 0, 'pickup': 1}
        #     self.hoi_data_label_batch = F.one_hot(torch.tensor(skill_dict[self.skill_name], device=self.device), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1).float()
        self.motion_ids_total[env_ids] = self.motion_ids.to(self.device) # update motion_ids_total
        self.motion_times_total[env_ids] = self.motion_times.to(self.device) # update motion_times_total
        pass

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
    
    ################ reweight according to the class reward ################
    def _reweight_motion(self, reset_env_ids):
        # record the reward for each motion clip at each time step
        if self.reweight: # and not self.isTest:
            self.record_motion_time_reward(reset_env_ids)

        if self.reweight: # and not self.isTest:
            if self.progress_buf_total % self.reweight_intervel == 0 and self.progress_buf_total > 0:
                # reweight the motion clip
                if len(self._motion_data.motion_class) > 1:
                    print('##### Reweight the sampling rate #####')
                    unique_ids = self._motion_data.hoi_data_dict.keys()
                    # for motion_id in unique_ids:
                    #     indices = (self.motion_ids_total == motion_id)
                    #     avg_reward = self.rew_buf[indices].mean().item()
                    #     self.average_rewards[motion_id] = avg_reward
                    for motion_id in unique_ids:
                        avg_reward = self.motion_time_seqreward[motion_id].mean().item()
                        self.average_rewards[motion_id] = avg_reward
                    print('Class Average Reward:', self.average_rewards)
                    self._motion_data._reweight_clip_sampling_rate(self.average_rewards)
                # reweight the motion time
                self._motion_data._reweight_time_sampling_rate(self.motion_time_seqreward)
    #######################################################################

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._curr_ref_obs, self._curr_obs, self._motion_data.envid2episode_lengths,
                                                   self.isTest, self.cfg["env"]["episodeLength"],
                                                   self.cfg["env"]["NR"],
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
                                                  self._dof_obs_size,
                                                  self._motion_data.reward_weights
                                                  )
        
        ######### Modified by Runyi #########
        # to save data for blender
        # body_ids = list(range(61))
        # self.save_frame(self.motion_dict,
        #                  self._rigid_body_pos[0, 0, :],
        #                  self._rigid_body_rot[0, 0, :],
        #                  self._rigid_body_pos[0, body_ids, :],
        #                  #torch.cat((self._rigid_body_rot[0, :1, :], torch_utils.exp_map_to_quat(self._dof_pos[0].reshape(-1,3))),dim=0),
        #                  self._rigid_body_rot[0, body_ids, :],
        #                  self._target_states[0, :3],
        #                  self._target_states[0, 3:7]
        #                  )
        # if self.progress_buf_total == 400:
        #     self.save_motion_dict(self.motion_dict, '/home/runyi/blender_for_SkillMimic/RIS_blender_motions/drink_cup.pt')
        #####################################
        return
    

    def _load_motion(self, motion_file, dof_obs_size):
        self.skill_name = os.path.basename(os.path.normpath(motion_file)) #metric
        self.max_episode_length = 60
        if self.cfg["env"]["episodeLength"] > 0:
            self.max_episode_length =  self.cfg["env"]["episodeLength"]
        
        self._motion_data = ParaMotionDataHandler(motion_file, dof_obs_size, self.device, self._key_body_ids, self.cfg,
                                                   self.num_envs, self.max_episode_length, self.reward_weights_default, 
                                                   self.init_vel, self.play_dataset, reweight_alpha=self.reweight_alpha)
        
        if self.play_dataset:
            self.max_episode_length = self._motion_data.max_episode_length
        self.hoi_data_batch = torch.zeros([self.num_envs, self.max_episode_length, self.ref_hoi_obs_size], device=self.device, dtype=torch.float)
        
        self.motion_dict = {}

        return
    


    def _subscribe_events_for_change_condition(self):
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_1, "001") # pick & place
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_2, "002") # pull
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_3, "003") # drink
        return
    

    def _reset_envs(self, env_ids):
        if(len(env_ids)>0): #metric
            self.reached_target[env_ids] = 0
        
        super()._reset_envs(env_ids)

        return
    
    def _reset_humanoid(self, env_ids):
        self._humanoid_root_states[env_ids, 0:3] = self.init_root_pos[env_ids]
        self._humanoid_root_states[env_ids, 3:7] = self.init_root_rot[env_ids]
        self._humanoid_root_states[env_ids, 7:10] = self.init_root_pos_vel[env_ids]
        self._humanoid_root_states[env_ids, 10:13] = self.init_root_rot_vel[env_ids]
        
        self._dof_pos[env_ids] = self.init_dof_pos[env_ids]
        self._dof_vel[env_ids] = self.init_dof_pos_vel[env_ids]
        return


    def _reset_random_ref_state_init(self, env_ids): #Z11
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = self._motion_data.sample_time(motion_ids)
        skill_label = self._motion_data.motion_class[motion_ids.tolist()]
        self.hoi_data_label_batch[env_ids] = F.one_hot(torch.tensor(skill_label, device=self.device), num_classes=self.condition_size).float()

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return motion_ids, motion_times
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)
        skill_label = self._motion_data.motion_class[motion_ids.tolist()]
        self.hoi_data_label_batch[env_ids] = F.one_hot(torch.tensor(skill_label, device=self.device), num_classes=self.condition_size).float()

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return motion_ids, motion_times

    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]

        if (env_ids is None):
            self._curr_obs[:] = build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._target_states,
                                                               self._hist_obs,
                                                               self.progress_buf)
        else:
            self._curr_obs[env_ids] = build_hoi_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._target_states[env_ids],
                                                                   self._hist_obs[env_ids],
                                                                   self.progress_buf[env_ids])
        
        return
    
    ######### Modified by Runyi #########
    def save_frame(self,motion_dict, rootpos, rootrot, dofpos, dofrot, ballpos, ballrot):
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

    def save_motion_dict(self, motion_dict, filename='motion.pt'):

        motion_dict['rootpos'] = torch.stack(motion_dict['rootpos'])
        motion_dict['rootrot'] = torch.stack(motion_dict['rootrot'])
        motion_dict['dofpos'] = torch.stack(motion_dict['dofpos'])
        motion_dict['dofrot'] = torch.stack(motion_dict['dofrot'])
        motion_dict['ballpos'] = torch.stack(motion_dict['ballpos'])
        motion_dict['ballrot'] = torch.stack(motion_dict['ballrot'])

        torch.save(motion_dict, filename)
        print(f'Successfully save the motion_dict to {filename}!')
        exit()
    
    #####################################
    # def _reset_target(self, env_ids):
    #     super()._reset_target(env_ids)
    #     if self.isTest:
    #         theta = torch.rand(len(env_ids)).to("cuda")*2*np.pi - np.pi
    #         self._target_states[env_ids, 0] += 0.05 * torch.cos(theta)
    #         self._target_states[env_ids, 1] += 0.05 * torch.sin(theta)
    #         # add random z rotation
    #         from scipy.spatial.transform import Rotation as R
    #         original_quats = self._target_states[env_ids, 3:7].cpu()
    #         z_rotations = R.from_euler('z', (theta/4).cpu()).as_quat()
    #         new_quats = (R.from_quat(original_quats) * R.from_quat(z_rotations)).as_quat()
    #         self._target_states[env_ids, 3:7] = torch.tensor(new_quats, dtype=torch.float, device='cuda')
    #####################################


    def _update_condition(self):
        for evt in self.evts:
            if evt.action.isdigit() and evt.value > 0:
                self.hoi_data_label_batch = F.one_hot(torch.tensor(int(evt.action), device=self.device), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1).float()
                print(evt.action)
    
    def play_dataset_step(self, time): #Z12

        t = time

        for env_id, env_ptr in enumerate(self.envs):

            ### update object ###
            # motid = self.envid2motid[env_id].item()
            motid = self.motion_ids_total[env_id].item()
            self._target_states[env_id, :3] = self._motion_data.hoi_data_dict[motid]['obj_pos'][t,:].clone()
            self._target_states[env_id, 3:7] = self._motion_data.hoi_data_dict[motid]['obj_rot'][t,:].clone()
            self._target_states[env_id, 7:10] = self._motion_data.hoi_data_dict[motid]['obj_pos_vel'][t,:].clone()
            self._target_states[env_id, 10:13] = self._motion_data.hoi_data_dict[motid]['obj_rot_vel'][t,:].clone()

            ### update subject ###               
            _humanoid_root_pos = self._motion_data.hoi_data_dict[motid]['root_pos'][t,:].clone()
            _humanoid_root_rot = self._motion_data.hoi_data_dict[motid]['root_rot'][t,:].clone()
            self._humanoid_root_states[env_id, 0:3] = _humanoid_root_pos
            self._humanoid_root_states[env_id, 3:7] = _humanoid_root_rot
            self._humanoid_root_states[env_id, 7:10] = self._motion_data.hoi_data_dict[motid]['root_pos_vel'][t,:].clone()
            self._humanoid_root_states[env_id, 10:13] = self._motion_data.hoi_data_dict[motid]['root_rot_vel'][t,:].clone()

            self._dof_pos[env_id] = self._motion_data.hoi_data_dict[motid]['dof_pos'][t,:].clone()
            self._dof_vel[env_id] = self._motion_data.hoi_data_dict[motid]['dof_pos_vel'][t,:].clone()

            contact = self._motion_data.hoi_data_dict[motid]['contact'][t,:]
            obj_contact = torch.any(contact > 0.1, dim=-1)
            root_rot_vel = self._motion_data.hoi_data_dict[motid]['root_rot_vel'][t,:]
            # angle, _ = torch_utils.exp_map_to_angle_axis(root_rot_vel)
            angle = torch.norm(root_rot_vel)
            abnormal = torch.any(torch.abs(angle) > 5.) #Z

            if abnormal == True:
                print("frame:", t, "abnormal:", abnormal, "angle", angle)
                self.show_abnorm[env_id] = 10

            handle = self._target_handles[env_id]
            if obj_contact == True:
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(1., 0., 0.))
            else:
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(0., 1., 0.))
            
            if abnormal == True or self.show_abnorm[env_id] > 0: #Z
                for j in range(self.num_bodies): #Z humanoid_handle == 0
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 0., 1.)) 
                self.show_abnorm[env_id] -= 1
            else:
                for j in range(self.num_bodies): #Z humanoid_handle == 0
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 1., 0.)) 
        
        # env_ids = torch.arange(len(self.envs), dtype=torch.long, device=self.device)
        # env_ids_int32 = self._humanoid_actor_ids[env_ids]
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self._root_states),
        #                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self._dof_state),
        #                                       gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
                                              
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_pos.contiguous()))
        self._refresh_sim_tensors()
        
        self.render(t=time)
        self.gym.simulate(self.sim)
        
        self._compute_observations()

        ######### Modified by Runyi #########
        # to save data for blender
        body_ids = list(range(61))
        self.save_frame(self.motion_dict,
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
        # print(self._motion_data.motion_lengths)
        # exit()
        self.progress_buf_total += 1
        if self.progress_buf_total == 145:
            self.save_motion_dict(self.motion_dict, '/home/runyi/blender_for_SkillMimic/RIS_blender_motions/video6_Ref_place_book.pt')
        #####################################

        return self.obs_buf
    

    def _draw_task_play(self,t):
        
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32) # color

        self.gym.clear_lines(self.viewer)

        starts = self._motion_data.hoi_data_dict[0]['hoi_data'][t, :3]

        for i, env_ptr in enumerate(self.envs):
            for j in range(len(self._key_body_ids)):
                vec = self._motion_data.hoi_data_dict[0]['key_body_pos'][t, j*3:j*3+3]
                vec = torch.cat([starts, vec], dim=-1).cpu().numpy().reshape([1, 6])
                self.gym.add_lines(self.viewer, env_ptr, 1, vec, cols)

        return

    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
            self.play_dataset
            if self.save_images:
                env_ids = 0
                # frame_id = t if self.play_dataset else self.progress_buf[env_ids]
                os.makedirs("skillmimic/data/images/" + self.save_images_timestamp, exist_ok=True)
                frame_id = len(os.listdir("skillmimic/data/images/" + self.save_images_timestamp))
                rgb_filename = "skillmimic/data/images/" + self.save_images_timestamp + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                self.gym.write_viewer_image_to_file(self.viewer,rgb_filename)
        return
    
    def _draw_task(self):

        # # draw obj contact
        # obj_contact = torch.any(torch.abs(self._tar_contact_forces[..., 0:2]) > 0.1, dim=-1)
        # for env_id, env_ptr in enumerate(self.envs):
        #     env_ptr = self.envs[env_id]
        #     handle = self._target_handles[env_id]

        #     if obj_contact[env_id] == True:
        #         self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
        #                                     gymapi.Vec3(1., 0., 0.))
        #     else:
        #         self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
        #                                     gymapi.Vec3(0., 1., 0.))

        return

    def get_num_amp_obs(self):
        return self.ref_hoi_obs_size


class SkillMimicParahomeDomRand(SkillMimicParahome):
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


class SkillMimicParahomeRefobj(SkillMimicParahome):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

    def _compute_observations(self, env_ids=None):
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        obj_obs = self._compute_obj_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        if (env_ids is None):
            env_ids = torch.arange(self.num_envs)

        textemb_batch = self.hoi_data_label_batch[env_ids]
        obs = torch.cat((obs, textemb_batch),dim=-1)
        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids,ts].clone()

        ######### Modified by Runyi #########
        ref_obj_obs = self._curr_ref_obs[env_ids, 402:408]
        obs = torch.cat([obs, ref_obj_obs], dim=-1)
        
        global_mean = obs[~torch.isnan(obs)].mean()
        nan_indices = torch.isnan(obs).nonzero(as_tuple=True)
        if nan_indices[0].numel() > 0:
            for dim in range(obs.shape[-1]):
                valid_values = obs[~torch.isnan(obs[:, dim]), dim]  # 获取非 NaN 值
                mean_value = valid_values.mean() if valid_values.numel() > 0 else global_mean
                obs[torch.isnan(obs[:, dim]), dim] = mean_value
                if valid_values.numel() < obs.shape[0]:
                    nan_envs = nan_indices[0][nan_indices[1] == dim] # 获取 NaN 所在的环境索引
                    print(f"NaN observation in Env: {nan_envs.tolist()}, Dimension {dim}")

        self.obs_buf[env_ids] = obs
        #####################################
        return


class SkillMimicParahomeRefobjNoisyinit(SkillMimicParahomeRefobj):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.weights = {'root_pos': 1,'root_pos_vel': 1,'root_rot_3d': 1,'root_rot_vel': 1,'dof_pos': 0.25, 'dof_pos_vel': 1,'obj_pos': 2,'obj_pos_vel': 2,'obj_rot': 1,'obj_rot_vel': 1}
    
    def _reset_random_ref_state_init(self, env_ids): #Z11
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = self._motion_data.sample_time(motion_ids)
        self._init_with_random_noise(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)
        self._init_with_random_noise(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times

    def _init_with_random_noise(self, env_ids, motion_ids, motion_times):
        skill_label = self._motion_data.motion_class[motion_ids.tolist()]
        self.hoi_data_label_batch[env_ids] = F.one_hot(torch.tensor(skill_label, device=self.device), num_classes=self.condition_size).float()
        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)
        # Random noise for initial state
        state_init_random_prob = self.cfg['env']['state_init_random_prob'] if self.adapt_prob is False else self.get_state_init_random_prob() 
        state_random_flags = [np.random.rand() < state_init_random_prob for _ in env_ids]
        if self.cfg['env']['state_init_random_prob'] > 0:
            for ind, env_id in enumerate(env_ids):
                if state_random_flags[ind]:
                    noise_weight = [0.15 for _ in range(10)] if skill_label[ind] != 0 else [1.0, 1.0] + [0.1 for _ in range(8)]
                    self.init_root_pos[env_id, 2] += random.random() * 0.1 # * noise_weight[0]
                    self.init_root_rot[env_id] += torch.randn_like(self.init_root_rot[env_id]) * noise_weight[1]
                    self.init_root_pos_vel[env_id] += torch.randn_like(self.init_root_pos_vel[env_id]) * noise_weight[2]
                    self.init_root_rot_vel[env_id] += torch.randn_like(self.init_root_rot_vel[env_id]) * noise_weight[3]
                    self.init_dof_pos[env_id] += torch.randn_like(self.init_dof_pos[env_id]) * noise_weight[4]
                    self.init_dof_pos_vel[env_id]  += torch.randn_like(self.init_dof_pos_vel[env_id]) * noise_weight[5]
                    # self.init_obj_pos[env_id, 2] += random.random() * noise_weight[6]
                    # self.init_obj_pos_vel[env_id] += torch.randn_like(self.init_obj_pos_vel[env_id]) * noise_weight[7]
                    # self.init_obj_rot[env_id] += torch.randn_like(self.init_obj_rot[env_id]) * noise_weight[8]
                    # self.init_obj_rot_vel[env_id] += torch.randn_like(self.init_obj_rot_vel[env_id]) * noise_weight[9]
                    # Normalize quaternion
                    self.init_root_rot[env_id] = self._norm_quat(self.init_root_rot[env_id])
                    self.init_obj_rot[env_id] = self._norm_quat(self.init_obj_rot[env_id])
                    noisy_motion = {
                        'root_pos': self.init_root_pos[env_id],
                        'root_rot': self.init_root_rot[env_id],
                        'root_pos_vel': self.init_root_pos_vel[env_id],
                        'root_rot_vel': self.init_root_rot_vel[env_id],
                        'dof_pos': self.init_dof_pos[env_id],
                        'dof_pos_vel': self.init_dof_pos_vel[env_id],
                        'obj_pos': self.init_obj_pos[env_id],
                        'obj_pos_vel': self.init_obj_pos_vel[env_id],
                        'obj_rot': self.init_obj_rot[env_id],
                        'obj_rot_vel': self.init_obj_rot_vel[env_id],
                        'key_body_pos': self._rigid_body_pos[env_id, self._key_body_ids, :],
                        'key_body_pos_vel': self._rigid_body_vel[env_id, self._key_body_ids, :],
                    }

                    motion_id = motion_ids[ind:ind+1]
                    # new_source_motion_time = self._motion_data.noisy_resample_time(noisy_motion, motion_id)
                    # motion_times[ind:ind+1] = new_source_motion_time
                    # # resample the hoi_data_batch
                    # self.hoi_data_batch[env_id], _, _, _, _, _, _, _, _, _, _ \
                    #     = self._motion_data.get_initial_state(env_ids[ind:ind+1], motion_id, new_source_motion_time)
                    if self.isTest:
                        print(f"Random noise added to initial state for env {env_id}")


class SkillMimicParahomeRIS(SkillMimicParahome):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

    def _reset_random_ref_state_init(self, env_ids): #Z11
        motion_ids, motion_times = super()._reset_random_ref_state_init(env_ids)
        motion_ids, motion_times = self._init_with_risrand_noise(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        motion_ids, motion_times = super()._reset_deterministic_ref_state_init(env_ids)
        motion_ids, motion_times = self._init_with_risrand_noise(env_ids, motion_ids, motion_times)
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
                    new_source_motion_time, _ = self._motion_data.noisy_resample_time(noisy_motion, motion_id)
                    
                    # resample the hoi_data_batch
                    self.hoi_data_batch[env_id], _, _, _, _, _, _, _, _, _, _ \
                        = self._motion_data.get_initial_state(env_ids[ind:ind+1], motion_id, new_source_motion_time)
                    # change motion_times
                    motion_times[ind:ind+1] = new_source_motion_time

                    # if self.isTest:
                    #     print(f"Random noise added to initial state for env {env_id}")

        return motion_ids, motion_times 

class SkillMimicParahomePhase(SkillMimicParahome):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

    def _compute_observations(self, env_ids=None):
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        obj_obs = self._compute_obj_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        if (env_ids is None):
            env_ids = torch.arange(self.num_envs)

        textemb_batch = self.hoi_data_label_batch[env_ids]
        obs = torch.cat((obs, textemb_batch),dim=-1)
        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids,ts].clone()

        ######### Modified by Runyi #########
        motion_length = self._motion_data.motion_lengths[self._motion_data.envid2motid[env_ids]]
        phase = ((ts + self._motion_data.envid2sframe[env_ids]) / motion_length).unsqueeze(-1).repeat(1,6)
        obs = torch.cat([obs, phase], dim=-1)

        global_mean = obs[~torch.isnan(obs)].mean()
        nan_indices = torch.isnan(obs).nonzero(as_tuple=True)
        if nan_indices[0].numel() > 0:
            for dim in range(obs.shape[-1]):
                valid_values = obs[~torch.isnan(obs[:, dim]), dim]  # 获取非 NaN 值
                mean_value = valid_values.mean() if valid_values.numel() > 0 else global_mean
                obs[torch.isnan(obs[:, dim]), dim] = mean_value
                if valid_values.numel() < obs.shape[0]:
                    nan_envs = nan_indices[0][nan_indices[1] == dim] # 获取 NaN 所在的环境索引
                    print(f"NaN observation in Env: {nan_envs.tolist()}, Dimension {dim}")

        self.obs_buf[env_ids] = obs
        #####################################
        return

class SkillMimicParahomePhaseNoisyinit(SkillMimicParahomeRefobjNoisyinit):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.weights = {'root_pos': 1,'root_pos_vel': 1,'root_rot_3d': 1,'root_rot_vel': 1,'dof_pos': 0.25, 'dof_pos_vel': 1,'obj_pos': 2,'obj_pos_vel': 2,'obj_rot': 1,'obj_rot_vel': 1}
    
    def _compute_observations(self, env_ids=None):
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        obj_obs = self._compute_obj_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        if (env_ids is None):
            env_ids = torch.arange(self.num_envs)

        textemb_batch = self.hoi_data_label_batch[env_ids]
        obs = torch.cat((obs, textemb_batch),dim=-1)
        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids,ts].clone()

        ######### Modified by Runyi #########
        motion_length = self._motion_data.motion_lengths[self._motion_data.envid2motid[env_ids]]
        phase = ((ts + self._motion_data.envid2sframe[env_ids]) / motion_length).unsqueeze(-1).repeat(1,6)
        obs = torch.cat([obs, phase], dim=-1)

        global_mean = obs[~torch.isnan(obs)].mean()
        nan_indices = torch.isnan(obs).nonzero(as_tuple=True)
        if nan_indices[0].numel() > 0:
            for dim in range(obs.shape[-1]):
                valid_values = obs[~torch.isnan(obs[:, dim]), dim]  # 获取非 NaN 值
                mean_value = valid_values.mean() if valid_values.numel() > 0 else global_mean
                obs[torch.isnan(obs[:, dim]), dim] = mean_value
                if valid_values.numel() < obs.shape[0]:
                    nan_envs = nan_indices[0][nan_indices[1] == dim] # 获取 NaN 所在的环境索引
                    print(f"NaN observation in Env: {nan_envs.tolist()}, Dimension {dim}")

        self.obs_buf[env_ids] = obs
        #####################################

        return

    
class SkillMimicParahomePhaseDomainrand(SkillMimicParahomePhase):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.weights = {'root_pos': 1,'root_pos_vel': 1,'root_rot_3d': 1,'root_rot_vel': 1,'dof_pos': 0.25, 'dof_pos_vel': 1,'obj_pos': 2,'obj_pos_vel': 2,'obj_rot': 1,'obj_rot_vel': 1}
    
    def _reset_random_ref_state_init(self, env_ids): #Z11
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = self._motion_data.sample_time(motion_ids)
        self._domain_rand(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)
        self._domain_rand(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times
    
    def _norm_quat(self, quat):
        norm = torch.norm(quat)
        return quat / norm

    def _domain_rand(self, env_ids, motion_ids, motion_times):
        skill_label = self._motion_data.motion_class[motion_ids.tolist()]
        self.hoi_data_label_batch[env_ids] = F.one_hot(torch.tensor(skill_label, device=self.device), num_classes=self.condition_size).float()
        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)
        # Random noise for initial state
        state_init_random_prob = self.cfg['env']['state_init_random_prob'] if self.adapt_prob is False else self.get_state_init_random_prob() 
        state_random_flags = [np.random.rand() < state_init_random_prob for _ in env_ids]
        if self.cfg['env']['state_init_random_prob'] > 0:
            for ind, env_id in enumerate(env_ids):
                if state_random_flags[ind]:
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
                    # Normalize quaternion
                    self.init_root_rot[env_id] = self._norm_quat(self.init_root_rot[env_id])
                    self.init_obj_rot[env_id] = self._norm_quat(self.init_obj_rot[env_id])
                    if self.isTest:
                        print(f"Domain Randomization added to initial state for env {env_id}")

class SkillMimicParahomePhaseNoisyinitRandskill(SkillMimicParahomePhaseNoisyinit):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.weights = {'root_pos': 1,'root_pos_vel': 1,'root_rot_3d': 1,'root_rot_vel': 1,'dof_pos': 0.25, 'dof_pos_vel': 1,'obj_pos': 2,'obj_pos_vel': 2,'obj_rot': 1,'obj_rot_vel': 1}
        self.switch_dict = {1:2, 2:3, 3:4, 4:1}

    def _reset_random_ref_state_init(self, env_ids): #Z11
        motion_ids, motion_times = super()._reset_random_ref_state_init(env_ids)
        self._init_with_random_noise(env_ids, motion_ids, motion_times)
        self._init_from_random_skill(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        motion_ids, motion_times = super()._reset_deterministic_ref_state_init(env_ids)
        self._init_with_random_noise(env_ids, motion_ids, motion_times)
        self._init_from_random_skill(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times

    def _init_from_random_skill(self, env_ids, motion_ids, motion_times): 
        # Random init to other skills
        state_switch_flags = [np.random.rand() < self.cfg['env']['state_switch_prob'] for _ in env_ids]
        motion_phase = motion_times / self._motion_data.motion_lengths[motion_ids]
        if self.cfg['env']['state_switch_prob'] > 0:
            for ind, env_id in enumerate(env_ids):
                # only switch if motion_time is in last 1/5 of the motion
                if state_switch_flags[ind] and motion_phase[ind]>0.8:
                    switch_motion_class = self._motion_data.motion_class[motion_ids[ind]]
                    switch_motion_id = motion_ids[ind:ind+1]
                    switch_motion_time = motion_times[ind:ind+1]

                    # load source motion info from state_search_graph
                    source_motion_class = self.switch_dict[switch_motion_class]
                    source_motion_id = torch.tensor([source_motion_class-1], device=self.device)
                    source_motion_time = torch.tensor([0], device=self.device)
                    
                    # state_switch changes the root and object states
                    _, self.init_root_pos[env_id], self.init_root_rot[env_id],  self.init_root_pos_vel[env_id], self.init_root_rot_vel[env_id], \
                    self.init_dof_pos[env_id], self.init_dof_pos_vel[env_id], \
                    self.init_obj_pos[env_id], self.init_obj_pos_vel[env_id], self.init_obj_rot[env_id], self.init_obj_rot_vel[env_id] \
                        = self._motion_data.get_initial_state(env_ids[ind:ind+1], switch_motion_id, switch_motion_time)

                    # resample the hoi_data_batch
                    self.hoi_data_batch[env_id], _, _,  _, _, _, _, _, _, _, _ = \
                        self._motion_data.get_initial_state(env_ids[ind:ind+1], source_motion_id, source_motion_time)
                    self.hoi_data_batch[env_id] = compute_local_hoi_data(self.hoi_data_batch[env_id], self.init_root_pos[env_id], 
                                                                         self.init_root_rot[env_id], len(self._key_body_ids))
                    
                    # change skill label
                    skill_label = self._motion_data.motion_class[source_motion_id.tolist()]
                    self.hoi_data_label_batch[env_id] = F.one_hot(torch.tensor(skill_label, device=self.device), num_classes=self.condition_size).float()
                    
                    if self.isTest:
                        print(f"Switched from skill {switch_motion_class} to {source_motion_class} for env {env_id}")

       
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_hoi_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, local_root_obs, 
                           root_height_obs, dof_obs_size, target_states, hist_obs, progress_buf):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, Tensor, Tensor, Tensor) -> Tensor
    
    ## diffvel, set 0 for the first frame
    # hist_dof_pos = hist_obs[:,6:6+156]
    # dof_diffvel = (dof_pos - hist_dof_pos)*fps
    # dof_diffvel = dof_diffvel*(progress_buf!=1).to(float).unsqueeze(dim=-1)

    dof_vel = dof_vel*(progress_buf!=1).unsqueeze(dim=-1)

    contact = torch.zeros(key_body_pos.shape[0],1,device=dof_vel.device)
    obs = torch.cat((root_pos, torch_utils.quat_to_exp_map(root_rot), 
                     dof_pos, dof_vel, 
                     target_states[:,:10], # object pos, rot, pos_vel
                     key_body_pos.contiguous().view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), 
                     contact,
                     ), dim=-1)
    return obs

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
    contact = hoi_obs[:,-1:]# fake one
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
    ref_obj_contact = hoi_ref[:,-1:]
    ref_key_pos = torch.cat((ref_root_pos, ref_key_pos),dim=-1)
    ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos),dim=-1)
    ref_ig = ref_key_pos.view(-1,len_keypos+1,3).transpose(0,1) - ref_obj_pos[:,:3]
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
    e_vel_diff = torch.mean((dof_pos_vel - dof_pos_vel_hist)**2 / (((ref_dof_pos_vel**2) + 1e-12)*1e12), dim=-1)
    r_vel_diff = torch.exp(-e_vel_diff * 0.1) #w['vel_diff']

    rb = rp*rr*rpv*rrv*r_vel_diff


    ### object reward ###

    # object pos reward
    eop = torch.mean((ref_obj_pos - obj_pos)**2,dim=-1)
    rop = torch.exp(-eop*w['op'])

    # object rot reward
    # eor = torch.zeros_like(ep) 
    eor = torch.mean((ref_obj_rot - obj_rot)**2,dim=-1)
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
    # {'pHipOrigin': 0, 'jL5S1': 1, 'jL4L3': 2, 'jL1T12': 3, 'jT9T8': 4, 'jT1C7': 5, 'jC1Head': 6, 
    # 'jRightT4Shoulder': 7, 'jRightShoulder': 8, 'jRightElbow': 9, 'jRightWrist': 10, 
    # 'jRightFirstCMC': 11, 'jRightFirstMCP': 12, 'jRightIP': 13, 
    # 'jRightSecondCMC': 14, 'jRightSecondMCP': 15, 'jRightSecondPIP': 16, 'jRightSecondDIP': 17, 
    # 'jRightThirdCMC': 18, 'jRightThirdMCP': 19, 'jRightThirdPIP': 20, 'jRightThirdDIP': 21, 
    # 'jRightFourthCMC': 22, 'jRightFourthMCP': 23, 'jRightFourthPIP': 24, 'jRightFourthDIP': 25, 
    # 'jRightFifthCMC': 26, 'jRightFifthMCP': 27, 'jRightFifthPIP': 28, 'jRightFifthDIP': 29, 
    # 'jLeftT4Shoulder': 30, 'jLeftShoulder': 31, 'jLeftElbow': 32, 'jLeftWrist': 33, 
    # 'jLeftFirstCMC': 34, 'jLeftFirstMCP': 35, 'jLeftIP': 36, 
    # 'jLeftSecondCMC': 37, 'jLeftSecondMCP': 38, 'jLeftSecondPIP': 39, 'jLeftSecondDIP': 40, 
    # 'jLeftThirdCMC': 41, 'jLeftThirdMCP': 42, 'jLeftThirdPIP': 43, 'jLeftThirdDIP': 44, 
    # 'jLeftFourthCMC': 45, 'jLeftFourthMCP': 46, 'jLeftFourthPIP': 47, 'jLeftFourthDIP': 48, 
    # 'jLeftFifthCMC': 49, 'jLeftFifthMCP': 50, 'jLeftFifthPIP': 51, 'jLeftFifthDIP': 52, 
    # 'jRightHip': 53, 'jRightKnee': 54, 'jRightAnkle': 55, 'jRightBallFoot': 56, 
    # 'jLeftHip': 57, 'jLeftKnee': 58, 'jLeftAnkle': 59, 'jLeftBallFoot': 60}

    # body contact
    # ["pHipOrigin", "jL5S1", "jL4L3", "jL1T12", "jT9T8", "jT1C7", "jC1Head", "jRightShoulder", "jRightElbow", "jLeftShoulder", "jLeftElbow", "jRightHip". "jRightKnee", "jLeftHip", "jLeftKnee"]
    # contact_body_ids = [0, 1, 2, 3, 4, 5, 6, 8, 9, 36, 37, 63, 64, 67, 68]
    # if dof_obs_size == 180:
    #     contact_body_ids = [0, 1, 2, 3, 4, 5, 6, 8, 9, 31, 32, 53, 54, 57, 58]
    # elif dof_obs_size == 210:
    #     contact_body_ids = [0, 1, 2, 3, 4, 5, 6, 8, 9, 36, 37, 63, 64, 67, 68]
    # body_contact_buf = contact_buf[:, contact_body_ids, :].clone()
    # body_contact = torch.all(torch.abs(body_contact_buf) < 0.1, dim=-1)
    # body_contact = 1. - torch.all(body_contact, dim=-1).to(float) # no contact: 0; contact: 1
    
    # object contact
    # obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(float) # =1 when contact happens to the object

    # ref_body_contact = torch.zeros_like(ref_obj_contact) # no body contact for all time
    # ecg1 = torch.abs(body_contact - ref_body_contact[:,0])
    # rcg1 = torch.exp(-ecg1*w['cg1'])
    # ecg2 = torch.abs(obj_contact - ref_obj_contact[:,0])
    # rcg2 = torch.exp(-ecg2*w['cg2'])
    
    # rcg = rcg1*rcg2

    # desk-cup contact
    reward = rb*ro*rig#*rcg
    
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, hoi_ref, hoi_obs, envid2episode_lengths,
                           isTest, maxEpisodeLength, NR = False):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, Tensor, bool, int, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rigid_body_pos[:, 0, 2] # root height
        body_fall = body_height < termination_heights# [4096] 
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 1)
        
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

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

    # print('init_root_pos:', init_root_pos)
    # print('local_root_pos:', local_root_pos[0])
    # print('switch_root_pos:', switch_root_pos)
    # print('local_root_rot:', torch_utils.quat_to_euler(torch_utils.exp_map_to_quat(local_root_rot[0])))
    # print('switch_root_rot:', torch_utils.quat_to_euler(switch_root_rot_quat))
    # exit()

    return local_hoi_data_batch