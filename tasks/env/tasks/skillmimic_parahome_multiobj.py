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
from utils.paramotion_data_handler_multiobj import ParaMotionDataHandlerMultiobj
from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObjectParahomeMultiobj


class SkillMimicParahomeMultiobj(HumanoidWholeBodyWithObjectParahomeMultiobj): 
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
        ######### Modified by Runyi #########
        self.progress_buf_total = 0
        self.max_epochs = cfg['env']['maxEpochs']
        self.adapt_prob = cfg['env']['adapt_prob']
        self.ref_hoi_obs_size = 6 + self._dof_obs_size*2 + len(self.cfg["env"]["keyBodies"])*3 + 1 + 20
        #####################################

        self._load_motion(self.motion_file, self._dof_obs_size) #ZC1

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
        # self.envid2episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.show_motion_test = False
        # self.init_from_frame_test = 0 #2 #ZC3
        self.motion_id_test = 0
        # self.options = [i for i in range(6) if i != 2]
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
            'ball0_pos': self._target0_states[..., 0:3],
            'ball1_pos': self._target1_states[..., 0:3],
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

        # ref_obj_obs = self._curr_ref_obs[env_ids, 366:376]
        # obs = torch.cat([obs, ref_obj_obs], dim=-1)

        self.obs_buf[env_ids] = obs
        #####################################
        
        ######### Modified by Runyi #########
        # # to save data for blender
        # body_ids = list(range(61))
        # self.save_frame(self.motion_dict,
        #                  self._rigid_body_pos[0, 0, :],
        #                  self._rigid_body_rot[0, 0, :],
        #                  self._rigid_body_pos[0, body_ids, :],
        #                  #torch.cat((self._rigid_body_rot[0, :1, :], torch_utils.exp_map_to_quat(self._dof_pos[0].reshape(-1,3))),dim=0),
        #                  self._rigid_body_rot[0, body_ids, :],
        #                  self._target_states[0, :3],
        #                  self._target_states[0, 3:7]
        #                 # self._proj_states[0, :3],
        #                 # self._proj_states[0, 3:7]
        #                  )
        # self.timestep += 1
        # if self.timestep == 100:
        #     self.save_motion_dict(self.motion_dict, '/home/runyi/SkillMimic2_yry/blender_motions/fig2_parahome_drink_cup.pt')
        #####################################

        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._curr_ref_obs, self._curr_obs, self._motion_data.envid2episode_lengths,
                                                   self.isTest, self.cfg["env"]["episodeLength"],
                                                   self.cfg["env"]["NR"],
                                                   )
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
        #                 self._rigid_body_pos[0, 0, :],
        #                 self._rigid_body_rot[0, 0, :],
        #                 self._rigid_body_pos[0, body_ids, :],
        #                 #torch.cat((self._rigid_body_rot[0, :1, :], torch_utils.exp_map_to_quat(self._dof_pos[0].reshape(-1,3))),dim=0),
        #                 self._rigid_body_rot[0, body_ids, :],
        #                 self._target0_states[0, :3],
        #                 self._target0_states[0, 3:7],
        #                 self._target1_states[0, :3],
        #                 self._target1_states[0, 3:7]
        #                 # self._proj_states[0, :3],
        #                 # self._proj_states[0, 3:7]
        #                 )
        # if self.progress_buf_total == 590:
        #     self.save_motion_dict(self.motion_dict, '/home/runyi/blender_for_SkillMimic/RIS_blender_motions/pour_kettle_cup.pt')
        # #####################################
        return
    

    def _load_motion(self, motion_file, dof_obs_size):
        self.skill_name = motion_file.split('/')[-1] #metric
        self.max_episode_length = 60
        if self.cfg["env"]["episodeLength"] > 0:
            self.max_episode_length =  self.cfg["env"]["episodeLength"]
        
        self._motion_data = ParaMotionDataHandlerMultiobj(motion_file, dof_obs_size, self.device, self._key_body_ids, self.cfg, self.num_envs, 
                                                  self.max_episode_length, self.reward_weights_default, self.init_vel, self.play_dataset)
        
        if self.play_dataset:
            self.max_episode_length = self._motion_data.max_episode_length
        self.hoi_data_batch = torch.zeros([self.num_envs, self.max_episode_length, self.ref_hoi_obs_size], device=self.device, dtype=torch.float)
        
        self.motion_dict = {}

        return
    

    def _subscribe_events_for_change_condition(self):
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "011") # dribble left
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "012") # dribble right
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "013") # dribble forward
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "001") # pick up
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "002") # shot
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "031") # layup
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "032") #
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "033") #
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "034") # turnaround layup
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "035") #
        
        return
    

    def _reset_envs(self, env_ids):
        if(len(env_ids)>0): #metric
            self.reached_target[env_ids] = 0
        
        super()._reset_envs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if self._state_init == -1:
            self._reset_random_ref_state_init(env_ids) #V1 Random Ref State Init (RRSI)
        elif self._state_init >= 2:
            self._reset_deterministic_ref_state_init(env_ids)
        else:
            assert(False), f"Unsupported state initialization from: {self._state_init}"

        super()._reset_actors(env_ids)

        return

    def _reset_humanoid(self, env_ids):
        self._humanoid_root_states[env_ids, 0:3] = self.init_root_pos[env_ids]
        self._humanoid_root_states[env_ids, 3:7] = self.init_root_rot[env_ids]
        self._humanoid_root_states[env_ids, 7:10] = self.init_root_pos_vel[env_ids]
        self._humanoid_root_states[env_ids, 10:13] = self.init_root_rot_vel[env_ids]
        
        self._dof_pos[env_ids] = self.init_dof_pos[env_ids]
        self._dof_vel[env_ids] = self.init_dof_pos_vel[env_ids]
        return

    def get_state_init_random_prob(self):
        epoch = self.progress_buf_total // 40 / self.max_epochs
        state_init_random_prob = 0.2 * (math.exp(3*epoch) - 1) / (math.exp(3) - 1) # 0 -> 0.2
        return state_init_random_prob
    
    def _reset_random_ref_state_init(self, env_ids): #Z11
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = self._motion_data.sample_time(motion_ids)
        skill_label = self._motion_data.motion_class[motion_ids.tolist()]
        self.hoi_data_label_batch[env_ids] = F.one_hot(torch.tensor(skill_label, device=self.device), num_classes=self.condition_size).float()

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj0_pos[env_ids], self.init_obj0_pos_vel[env_ids], self.init_obj0_rot[env_ids], self.init_obj0_rot_vel[env_ids], \
        self.init_obj1_pos[env_ids], self.init_obj1_pos_vel[env_ids], self.init_obj1_rot[env_ids], self.init_obj1_rot_vel[env_ids] \
                    = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)
        skill_label = self._motion_data.motion_class[motion_ids.tolist()]
        self.hoi_data_label_batch[env_ids] = F.one_hot(torch.tensor(skill_label, device=self.device), num_classes=self.condition_size).float()

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj0_pos[env_ids], self.init_obj0_pos_vel[env_ids], self.init_obj0_rot[env_ids], self.init_obj0_rot_vel[env_ids], \
        self.init_obj1_pos[env_ids], self.init_obj1_pos_vel[env_ids], self.init_obj1_rot[env_ids], self.init_obj1_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return

    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]

        if (env_ids is None):
            self._curr_obs[:] = build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._target0_states, self._target1_states,
                                                               self._hist_obs,
                                                               self.progress_buf)
        else:
            self._curr_obs[env_ids] = build_hoi_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._target0_states[env_ids], self._target1_states[env_ids],
                                                                   self._hist_obs[env_ids],
                                                                   self.progress_buf[env_ids])
        
        return
    
    ######### Modified by Runyi #########
    def save_frame(self,motion_dict, rootpos, rootrot, dofpos, dofrot, ball0pos, ball0rot, ball1pos, ball1rot):
        if 'rootpos' not in motion_dict:
            motion_dict['rootpos']=[]
        if 'rootrot' not in motion_dict:
            motion_dict['rootrot']=[]
        if 'dofpos' not in motion_dict:
            motion_dict['dofpos']=[]
        if 'dofrot' not in motion_dict:
            motion_dict['dofrot']=[]
        if 'ball0pos' not in motion_dict:
            motion_dict['ball0pos']=[]
        if 'ball0rot' not in motion_dict:
            motion_dict['ball0rot']=[]
        if 'ball1pos' not in motion_dict:
            motion_dict['ball1pos']=[]
        if 'ball1rot' not in motion_dict:
            motion_dict['ball1rot']=[]

        motion_dict['rootpos'].append(rootpos.clone())
        motion_dict['rootrot'].append(rootrot.clone())
        motion_dict['dofpos'].append(dofpos.clone())
        motion_dict['dofrot'].append(dofrot.clone())
        motion_dict['ball0pos'].append(ball0pos.clone())
        motion_dict['ball0rot'].append(ball0rot.clone())
        motion_dict['ball1pos'].append(ball1pos.clone())
        motion_dict['ball1rot'].append(ball1rot.clone())

        # print("motion_dict['rootpos']",motion_dict['rootpos'])
        # print("rootpos",rootpos)

    def save_motion_dict(self, motion_dict, filename='motion.pt'):

        motion_dict['rootpos'] = torch.stack(motion_dict['rootpos'])
        motion_dict['rootrot'] = torch.stack(motion_dict['rootrot'])
        motion_dict['dofpos'] = torch.stack(motion_dict['dofpos'])
        motion_dict['dofrot'] = torch.stack(motion_dict['dofrot'])
        motion_dict['ball0pos'] = torch.stack(motion_dict['ball0pos'])
        motion_dict['ball0rot'] = torch.stack(motion_dict['ball0rot'])
        motion_dict['ball1pos'] = torch.stack(motion_dict['ball1pos'])
        motion_dict['ball1rot'] = torch.stack(motion_dict['ball1rot'])

        torch.save(motion_dict, filename)
        exit()
    
    #####################################
    # def _reset_target(self, env_ids):
    #     super()._reset_target(env_ids)
    #     if self.isTest:
    #         theta = torch.rand(len(env_ids)).to("cuda")*2*np.pi - np.pi
    #         from scipy.spatial.transform import Rotation as R
    #         z_rotations = R.from_euler('z', (theta/4).cpu()).as_quat()
    #         obj_ind = 1
    #         if obj_ind == 0:
    #             self._target0_states[env_ids, 0] += 0.05 * torch.cos(theta)
    #             self._target0_states[env_ids, 1] += 0.05 * torch.sin(theta)
    #             # add random z rotation
    #             original_quats = self._target0_states[env_ids, 3:7].cpu()
    #             new_quats = (R.from_quat(original_quats) * R.from_quat(z_rotations)).as_quat()
    #             self._target0_states[env_ids, 3:7] = torch.tensor(new_quats, dtype=torch.float, device='cuda')
    #         else:
    #             self._target1_states[env_ids, 0] += 0.05 * torch.cos(theta)
    #             self._target1_states[env_ids, 1] += 0.05 * torch.sin(theta)
    #             # add random z rotation
    #             original_quats = self._target1_states[env_ids, 3:7].cpu()
    #             new_quats = (R.from_quat(original_quats) * R.from_quat(z_rotations)).as_quat()
    #             self._target1_states[env_ids, 3:7] = torch.tensor(new_quats, dtype=torch.float, device='cuda')
            # print(f'target{obj_ind}, x bias:{0.05 * torch.cos(theta[:10]), }, y bias:{0.05 * torch.sin(theta[:10])}, rotate bias:{theta[:10]/4}')
    #####################################


    def _update_condition(self):
        for evt in self.evts:
            if evt.action.isdigit() and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(int(evt.action)).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)
    
    
    def play_dataset_step(self, time): #Z12

        t = time

        for env_id, env_ptr in enumerate(self.envs):

            ### update object ###
            motid = self.envid2motid[env_id].item()
            self._target0_states[env_id, :3] = self._motion_data.hoi_data_dict[motid]['obj0_pos'][t,:].clone()
            self._target0_states[env_id, 3:7] = self._motion_data.hoi_data_dict[motid]['obj0_rot'][t,:].clone()
            self._target0_states[env_id, 7:10] = self._motion_data.hoi_data_dict[motid]['obj0_pos_vel'][t,:].clone()
            self._target0_states[env_id, 10:13] = self._motion_data.hoi_data_dict[motid]['obj0_rot_vel'][t,:].clone()
            self._target1_states[env_id, :3] = self._motion_data.hoi_data_dict[motid]['obj1_pos'][t,:].clone()
            self._target1_states[env_id, 3:7] = self._motion_data.hoi_data_dict[motid]['obj1_rot'][t,:].clone()
            self._target1_states[env_id, 7:10] = self._motion_data.hoi_data_dict[motid]['obj1_pos_vel'][t,:].clone()
            self._target1_states[env_id, 10:13] = self._motion_data.hoi_data_dict[motid]['obj1_rot_vel'][t,:].clone()

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
        
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        self._refresh_sim_tensors()
        
        self.render(t=time)
        self.gym.simulate(self.sim)
        
        self._compute_observations()

        ######### Modified by Runyi #########
        # to save data for blender
        # body_ids = list(range(61))
        # self.save_frame(self.motion_dict,
        #                  self._rigid_body_pos[0, 0, :],
        #                  self._rigid_body_rot[0, 0, :],
        #                  self._rigid_body_pos[0, body_ids, :],
        #                  #torch.cat((self._rigid_body_rot[0, :1, :], torch_utils.exp_map_to_quat(self._dof_pos[0].reshape(-1,3))),dim=0),
        #                  self._rigid_body_rot[0, body_ids, :],
        #                  self._target0_states[0, :3],
        #                  self._target0_states[0, 3:7],
        #                  self._target1_states[0, :3],
        #                  self._target1_states[0, 3:7]
        #                 # self._proj_states[0, :3],
        #                 # self._proj_states[0, 3:7]
        #                  )
        # self.progress_buf[0] += 1
        # if self.progress_buf[0] == 200:
        #     self.save_motion_dict(self.motion_dict, '/home/runyi/blender_for_SkillMimic/RIS_blender_motions/fig5-2_ours_pour_tea.pt')
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
    
    def get_num_amp_obs(self):
        return self.ref_hoi_obs_size


class SkillMimicParahomePhaseMultiobj(SkillMimicParahomeMultiobj):
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
        #####################################

        self.obs_buf[env_ids] = obs
        return

class SkillMimicParahomePhaseNoisyinitMultiobj(SkillMimicParahomePhaseMultiobj):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
    
    def _reset_random_ref_state_init(self, env_ids): #Z11
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = self._motion_data.sample_time(motion_ids)
        skill_label = self._motion_data.motion_class[motion_ids.tolist()]
        self.hoi_data_label_batch[env_ids] = F.one_hot(torch.tensor(skill_label, device=self.device), num_classes=self.condition_size).float()

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj0_pos[env_ids], self.init_obj0_pos_vel[env_ids], self.init_obj0_rot[env_ids], self.init_obj0_rot_vel[env_ids], \
        self.init_obj1_pos[env_ids], self.init_obj1_pos_vel[env_ids], self.init_obj1_rot[env_ids], self.init_obj1_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        # Random noise for initial state
        state_init_random_prob = self.cfg['env']['state_init_random_prob'] if self.adapt_prob is False else self.get_state_init_random_prob() 
        state_random_flags = [np.random.rand() < state_init_random_prob for _ in env_ids]
        if state_init_random_prob > 0:
            for ind, env_id in enumerate(env_ids):
                if state_random_flags[ind]:
                    noise_weight = [0.1 for _ in range(10)] if skill_label[ind] != 0 else [1.0, 1.0] + [0.1 for _ in range(8)]
                    self.init_root_pos[env_id, 2] += random.random() * noise_weight[0]
                    self.init_root_rot[env_id] += torch.randn_like(self.init_root_rot[env_id]) * noise_weight[1]
                    self.init_root_pos_vel[env_id] += torch.randn_like(self.init_root_pos_vel[env_id]) * noise_weight[2]
                    self.init_root_rot_vel[env_id] += torch.randn_like(self.init_root_rot_vel[env_id]) * noise_weight[3]
                    self.init_dof_pos[env_id] += torch.randn_like(self.init_dof_pos[env_id]) * noise_weight[4]
                    self.init_dof_pos_vel[env_id]  += torch.randn_like(self.init_dof_pos_vel[env_id]) * noise_weight[5]
                    self.init_obj0_pos[env_id, 2] += random.random() * noise_weight[6]
                    self.init_obj0_pos_vel[env_id] += torch.randn_like(self.init_obj0_pos_vel[env_id]) * noise_weight[7]
                    self.init_obj0_rot[env_id] += torch.randn_like(self.init_obj0_rot[env_id]) * noise_weight[8]
                    self.init_obj0_rot_vel[env_id] += torch.randn_like(self.init_obj0_rot_vel[env_id]) * noise_weight[9]
                    self.init_obj1_pos[env_id, 2] += random.random() * noise_weight[6]
                    self.init_obj1_pos_vel[env_id] += torch.randn_like(self.init_obj1_pos_vel[env_id]) * noise_weight[7]
                    self.init_obj1_rot[env_id] += torch.randn_like(self.init_obj1_rot[env_id]) * noise_weight[8]
                    self.init_obj1_rot_vel[env_id] += torch.randn_like(self.init_obj1_rot_vel[env_id]) * noise_weight[9]
                    noisy_motion = {
                        'root_pos': self.init_root_pos[env_id],
                        'root_pos_vel': self.init_root_pos_vel[env_id],
                        'root_rot': self.init_root_rot[env_id],
                        'root_rot_vel': self.init_root_rot_vel[env_id],
                        'key_body_pos': self._rigid_body_pos[env_id, self._key_body_ids, :],
                        'key_body_pos_vel': self._rigid_body_vel[env_id, self._key_body_ids, :],
                        'dof_pos': self.init_dof_pos[env_id],
                        'dof_pos_vel': self.init_dof_pos_vel[env_id],
                        'obj0_pos': self.init_obj0_pos[env_id],
                        'obj0_pos_vel': self.init_obj0_pos_vel[env_id],
                        'obj0_rot': self.init_obj0_rot[env_id],
                        'obj0_rot_vel': self.init_obj0_rot_vel[env_id],
                        'obj1_pos': self.init_obj1_pos[env_id],
                        'obj1_pos_vel': self.init_obj1_pos_vel[env_id],
                        'obj1_rot': self.init_obj1_rot[env_id],
                        'obj1_rot_vel': self.init_obj1_rot_vel[env_id],
                    }

                    motion_id = motion_ids[ind:ind+1]
                    new_source_motion_time, max_sim = self._motion_data.noisy_resample_time(noisy_motion, motion_id)
                    motion_times[ind:ind+1] = new_source_motion_time
                    # resample the hoi_data_batch
                    self.hoi_data_batch[env_id], _, _, _, _, _, _, _, _, _, _, _, _, _, _ \
                        = self._motion_data.get_initial_state(env_ids[ind:ind+1], motion_id, new_source_motion_time)
                    if self.isTest:
                        print(f"Random noise added to initial state for env {env_id}")
        return 
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)
        skill_label = self._motion_data.motion_class[motion_ids.tolist()]
        self.hoi_data_label_batch[env_ids] = F.one_hot(torch.tensor(skill_label, device=self.device), num_classes=self.condition_size).float()

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj0_pos[env_ids], self.init_obj0_pos_vel[env_ids], self.init_obj0_rot[env_ids], self.init_obj0_rot_vel[env_ids], \
        self.init_obj1_pos[env_ids], self.init_obj1_pos_vel[env_ids], self.init_obj1_rot[env_ids], self.init_obj1_rot_vel[env_ids], \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        # Random noise for initial state
        state_init_random_prob = self.cfg['env']['state_init_random_prob'] if self.adapt_prob is False else self.get_state_init_random_prob() 
        state_random_flags = [np.random.rand() < state_init_random_prob for _ in env_ids]
        if state_init_random_prob > 0:
            for ind, env_id in enumerate(env_ids):
                if state_random_flags[ind]:
                    noise_weight = [0.1 for _ in range(10)] if skill_label[ind] != 0 else [1.0, 1.0] + [0.1 for _ in range(8)]
                    self.init_root_pos[env_id, 2] += random.random() * noise_weight[0]
                    self.init_root_rot[env_id] += torch.randn_like(self.init_root_rot[env_id]) * noise_weight[1]
                    self.init_root_pos_vel[env_id] += torch.randn_like(self.init_root_pos_vel[env_id]) * noise_weight[2]
                    self.init_root_rot_vel[env_id] += torch.randn_like(self.init_root_rot_vel[env_id]) * noise_weight[3]
                    self.init_dof_pos[env_id] += torch.randn_like(self.init_dof_pos[env_id]) * noise_weight[4]
                    self.init_dof_pos_vel[env_id]  += torch.randn_like(self.init_dof_pos_vel[env_id]) * noise_weight[5]
                    self.init_obj0_pos[env_id, 2] += random.random() * noise_weight[6]
                    self.init_obj0_pos_vel[env_id] += torch.randn_like(self.init_obj0_pos_vel[env_id]) * noise_weight[7]
                    self.init_obj0_rot[env_id] += torch.randn_like(self.init_obj0_rot[env_id]) * noise_weight[8]
                    self.init_obj0_rot_vel[env_id] += torch.randn_like(self.init_obj0_rot_vel[env_id]) * noise_weight[9]
                    self.init_obj1_pos[env_id, 2] += random.random() * noise_weight[6]
                    self.init_obj1_pos_vel[env_id] += torch.randn_like(self.init_obj1_pos_vel[env_id]) * noise_weight[7]
                    self.init_obj1_rot[env_id] += torch.randn_like(self.init_obj1_rot[env_id]) * noise_weight[8]
                    self.init_obj1_rot_vel[env_id] += torch.randn_like(self.init_obj1_rot_vel[env_id]) * noise_weight[9]
                    noisy_motion = {
                        'root_pos': self.init_root_pos[env_id],
                        'root_pos_vel': self.init_root_pos_vel[env_id],
                        'root_rot': self.init_root_rot[env_id],
                        'root_rot_vel': self.init_root_rot_vel[env_id],
                        'key_body_pos': self._rigid_body_pos[env_id, self._key_body_ids, :],
                        'key_body_pos_vel': self._rigid_body_vel[env_id, self._key_body_ids, :],
                        'dof_pos': self.init_dof_pos[env_id],
                        'dof_pos_vel': self.init_dof_pos_vel[env_id],
                        'obj0_pos': self.init_obj0_pos[env_id],
                        'obj0_pos_vel': self.init_obj0_pos_vel[env_id],
                        'obj0_rot': self.init_obj0_rot[env_id],
                        'obj0_rot_vel': self.init_obj0_rot_vel[env_id],
                        'obj1_pos': self.init_obj1_pos[env_id],
                        'obj1_pos_vel': self.init_obj1_pos_vel[env_id],
                        'obj1_rot': self.init_obj1_rot[env_id],
                        'obj1_rot_vel': self.init_obj1_rot_vel[env_id],
                    }

                    motion_id = motion_ids[ind:ind+1]
                    new_source_motion_time, max_sim = self._motion_data.noisy_resample_time(noisy_motion, motion_id)
                    motion_times[ind:ind+1] = new_source_motion_time
                    # resample the hoi_data_batch
                    self.hoi_data_batch[env_id], _, _, _, _, _, _, _, _, _, _, _, _, _, _ \
                        = self._motion_data.get_initial_state(env_ids[ind:ind+1], motion_id, new_source_motion_time)
                    if self.isTest:
                        print(f"Random noise added to initial state for env {env_id}")
        return 
    
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_hoi_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, local_root_obs, 
                           root_height_obs, dof_obs_size, target0_states, target1_states, hist_obs, progress_buf):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, Tensor, Tensor, Tensor, Tensor) -> Tensor
    
    ## diffvel, set 0 for the first frame
    # hist_dof_pos = hist_obs[:,6:6+156]
    # dof_diffvel = (dof_pos - hist_dof_pos)*fps
    # dof_diffvel = dof_diffvel*(progress_buf!=1).to(float).unsqueeze(dim=-1)

    dof_vel = dof_vel*(progress_buf!=1).unsqueeze(dim=-1)

    contact = torch.zeros(key_body_pos.shape[0],1,device=dof_vel.device)
    obs = torch.cat((root_pos, torch_utils.quat_to_exp_map(root_rot), 
                     dof_pos, dof_vel, 
                     target0_states[:,:10], # object0 pos, rot, pos_vel
                     target1_states[:,:10], # object1 pos, rot, pos_vel
                     key_body_pos.contiguous().view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), 
                     contact,
                     ), dim=-1)
    return obs

# @torch.jit.script
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
    obj0_pos = hoi_obs[:,start_ind:start_ind+3]
    start_ind += 3
    obj0_rot = hoi_obs[:,start_ind:start_ind+4]
    start_ind += 4
    obj0_pos_vel = hoi_obs[:,start_ind:start_ind+3]
    start_ind += 3
    obj1_pos = hoi_obs[:,start_ind:start_ind+3]
    start_ind += 3
    obj1_rot = hoi_obs[:,start_ind:start_ind+4]
    start_ind += 4
    obj1_pos_vel = hoi_obs[:,start_ind:start_ind+3]
    start_ind += 3
    key_pos = hoi_obs[:,start_ind:start_ind+len_keypos*3]
    start_ind += len_keypos*3

    key_pos = torch.cat((root_pos, key_pos),dim=-1)
    body_rot = torch.cat((root_rot, dof_pos),dim=-1)
    ig_obj0 = key_pos.view(-1,len_keypos+1,3).transpose(0,1) - obj0_pos[:,:3]
    ig_obj1 = key_pos.view(-1,len_keypos+1,3).transpose(0,1) - obj1_pos[:,:3]
    ig = torch.cat((ig_obj0, ig_obj1),dim=-1)
    ig = ig.transpose(0,1).reshape(-1,(len_keypos+1)*6)

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
    ref_obj0_pos = hoi_ref[:,start_ind:start_ind+3]
    start_ind += 3
    ref_obj0_rot = hoi_ref[:,start_ind:start_ind+4]
    start_ind += 4
    ref_obj0_pos_vel = hoi_ref[:,start_ind:start_ind+3]
    start_ind += 3
    ref_obj1_pos = hoi_ref[:,start_ind:start_ind+3]
    start_ind += 3
    ref_obj1_rot = hoi_ref[:,start_ind:start_ind+4]
    start_ind += 4
    ref_obj1_pos_vel = hoi_ref[:,start_ind:start_ind+3]
    start_ind += 3
    ref_key_pos = hoi_ref[:,start_ind:start_ind+len_keypos*3]
    start_ind += len_keypos*3
    ref_obj_contact = hoi_ref[:,-1:]
    ref_key_pos = torch.cat((ref_root_pos, ref_key_pos),dim=-1)
    ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos),dim=-1)
    ref_ig_obj0 = ref_key_pos.view(-1,len_keypos+1,3).transpose(0,1) - ref_obj0_pos[:,:3]
    ref_ig_obj1 = ref_key_pos.view(-1,len_keypos+1,3).transpose(0,1) - ref_obj1_pos[:,:3]
    ref_ig = torch.cat((ref_ig_obj0, ref_ig_obj1),dim=-1)
    ref_ig = ref_ig.transpose(0,1).reshape(-1,(len_keypos+1)*6)


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
    ref_obj_pos = torch.cat((ref_obj0_pos, ref_obj1_pos),dim=-1)
    obj_pos = torch.cat((obj0_pos, obj1_pos),dim=-1)
    eop = torch.mean((ref_obj_pos - obj_pos)**2,dim=-1)
    rop = torch.exp(-eop*w['op'])

    ro = rop


    ### interaction graph reward ###

    eig = torch.mean((ref_ig - ig)**2,dim=-1) #Zw
    # eig = torch.mean((ref_ig_wrist - ig_wrist)**2,dim=-1)
    rig = torch.exp(-eig*w['ig'])

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
