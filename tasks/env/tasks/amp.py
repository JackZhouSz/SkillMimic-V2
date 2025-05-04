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
from utils.motion_data_handler import MotionDataHandler4AMP

from env.tasks.humanoid_task import HumanoidWholeBody
from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObject


class SkillMimicAMPLocomotion(HumanoidWholeBody): 
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
        self.robot_type = cfg["env"]["asset"]["assetFileName"]
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
        
        self.progress_buf_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ref_hoi_obs_size = 323 + len(self.cfg["env"]["keyBodies"])*3 + 6 #V1
        self.hoi_data_label_batch = torch.zeros([self.num_envs, self.condition_size], device=self.device, dtype=torch.float)

        self._subscribe_events_for_change_condition()

        self.show_motion_test = False
        self.motion_id_test = 0
        self.succ_pos = []
        self.fail_pos = []

        self.show_abnorm = [0] * self.num_envs #V1

        self.switch_motion_file = cfg['env']['switch_motion_file'] if 'switch_motion_file' in cfg['env'] else None
        self._load_motion(self.motion_file, self.switch_motion_file) #ZC1

        self._dof_offsets = list(range(0, self.num_dof, 3)) + [self.num_dof]

        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        self._amp_obs_demo_buf = None
        self._reset_ref_env_ids = []

        self.state_search_to_align_reward = cfg['env']['state_search_to_align_reward']
        self.max_sim = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/mocap_humanoid.xml"):
            self._dof_obs_size = (52)*6
            self._num_actions = (52)*3
            self._num_obs = 1 + (53) * (3 + 6 + 3 + 3) - 3 + 10*3
            obj_amp_obs_size = 12 #Z ball
            self._num_amp_obs_per_step = ( 1 + 6 + 3 + 3 + self._dof_obs_size + self._dof_obs_size//2 + num_key_bodies * 3 
                                        #   + obj_amp_obs_size
                                        )

        elif (asset_file == "mjcf/mocap_humanoid_boxhand.xml"):
            self._dof_obs_size = (52)*6
            self._num_actions = (52)*3
            self._num_obs = 1 + (53) * (3 + 6 + 3 + 3) - 3 + 10*3
            obj_amp_obs_size = 12
            self._num_amp_obs_per_step = ( 1 + 6 + 3 + 3 + self._dof_obs_size + self._dof_obs_size//2 + num_key_bodies * 3
                                        #   + obj_amp_obs_size
                                        )
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

    def get_state_for_metric(self):
        # 提供 Metric 计算所需的状态
        return {
            # 'ball_pos': self._target_states[..., 0:3], #Z ball
            'root_pos': self._humanoid_root_states[..., 0:3],
            'root_pos_vel': self._humanoid_root_states[..., 7:10],
            # 'progress': self.progress_buf,
            # 根据需要添加其他状态
        }
    
    def post_physics_step(self):
        self._update_condition()

        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations()
        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            self._amp_obs_buf[:, 1:] = self._amp_obs_buf[:, :-1].clone()
        else:
            self._amp_obs_buf[env_ids, 1:] = self._amp_obs_buf[env_ids, :-1].clone()
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

        # obj_obs = self._compute_obj_obs(env_ids) #Z ball
        # obs = torch.cat([obs, obj_obs], dim=-1)

        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)
        
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs)

        textemb_batch = self.hoi_data_label_batch[env_ids]
        obs = torch.cat((obs, textemb_batch), dim=-1)
        self.obs_buf[env_ids] = obs
        # ts = self.progress_buf[env_ids].clone()
        # self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids, ts].clone()


        return obs
    
    def _compute_reset(self):
        # print(self.progress_buf)
        # if(self.progress_buf[0]==60):
        #     print("?")
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._motion_data.envid2episode_lengths, self.isTest, self.cfg["env"]["episodeLength"],
                                                   )
        return
    
    def _compute_reward(self):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)

        return
    
    def _load_motion(self, motion_file, switch_motion_file=None):
        self.skill_name = os.path.basename(motion_file) #motion_file.split('/')[-1] #metric
        # self.skill_name = "Chestpass"
        self.max_episode_length = 60
        if self.cfg["env"]["episodeLength"] > 0:
            self.max_episode_length =  self.cfg["env"]["episodeLength"]
        
        self.hoi_data_batch = torch.zeros([self.num_envs, self.max_episode_length, self.ref_hoi_obs_size], device=self.device, dtype=torch.float)
        
        self._motion_data = MotionDataHandler4AMP(motion_file, self.device, self._key_body_ids, self.cfg, self.num_envs, 
                                            self.max_episode_length, self.reward_weights_default, self.init_vel, self.play_dataset)
        
        if self.play_dataset:
            self.max_episode_length = self._motion_data.max_episode_length
            print("--------------------------------------------", self.max_episode_length)

        if self.skill_name in ['layup', 'shot', 'Setshot', 'Jumpshot', 'Chestpass'] and self.isTest:
            layup_target_ind = torch.argmax(self._motion_data.hoi_data_dict[0]['obj_pos'][:,2])
            self.layup_target = self._motion_data.hoi_data_dict[0]['obj_pos'][layup_target_ind]
            
        self.switch_skill_name = os.path.basename(switch_motion_file) if switch_motion_file is not None else None # switch_skill -> skill
        if switch_motion_file is not None:
            self._motion_data = MotionDataHandler4AMP(switch_motion_file, self.device, self._key_body_ids, self.cfg,
                                                         self.num_envs, self.max_episode_length, self.reward_weights_default, self.init_vel, self.play_dataset)

        if self.play_dataset:
            self.max_episode_length = self._motion_data.max_episode_length
            print("--------------------------------------------", self.max_episode_length)
        
        return

    def _subscribe_events_for_change_condition(self):
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "011") # dribble left
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "012") # dribble right
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "013") # dribble forward
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "001") # pick up
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "009") # shot
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "031") # layup
        #############################################################################################
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_0, "000") # getup
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_1, "010") # run
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "034") # turnaround layup
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "035") #
        
        return
    
    def _reset_state_init(self, env_ids):
        if self._state_init == -1:
            self.motion_ids, self.motion_times = self._reset_random_ref_state_init(env_ids) #V1 Random Ref State Init (RRSI)
        elif self._state_init >= 2:
            self.motion_ids, self.motion_times = self._reset_deterministic_ref_state_init(env_ids)
        else:
            assert(False), f"Unsupported state initialization from: {self._state_init}"
        
        self.state_random_flags = [False for _ in env_ids]
        if self.cfg['env']['state_init_random_prob'] > 0:
            self.motion_ids, self.motion_times = self._init_with_random_noise(env_ids, self.motion_ids, self.motion_times)
        if self.cfg['env']['state_switch_prob'] > 0:
            self.motion_ids, self.motion_times = self._init_from_random_skill(env_ids, self.motion_ids, self.motion_times)

    def _init_with_random_noise(self, env_ids, motion_ids, motion_times): 
        # Random noise for initial state
        self.state_random_flags = [np.random.rand() < self.cfg['env']['state_init_random_prob'] for _ in env_ids]
        
        skill_label = self._motion_data.motion_class_tensor[motion_ids]
        for ind, env_id in enumerate(env_ids):
            if self.state_random_flags[ind]:
                noise_weight = [0.1 for _ in range(10)] if skill_label[ind] != 0 else [1.0, 1.0] + [0.1 for _ in range(8)]
                self.init_root_pos[env_id, 2] += random.random() * noise_weight[0]
                self.init_root_rot[env_id] += torch.randn_like(self.init_root_rot[env_id]) * noise_weight[1]
                self.init_root_pos_vel[env_id] += torch.randn_like(self.init_root_pos_vel[env_id]) * noise_weight[2]
                self.init_root_rot_vel[env_id] += torch.randn_like(self.init_root_rot_vel[env_id]) * noise_weight[3]
                self.init_dof_pos[env_id] += torch.randn_like(self.init_dof_pos[env_id]) * noise_weight[4]
                self.init_dof_pos_vel[env_id]  += torch.randn_like(self.init_dof_pos_vel[env_id]) * noise_weight[5]
                # self.init_obj_pos[env_id, 2] += random.random() * noise_weight[6]
                # self.init_obj_pos_vel[env_id] += torch.randn_like(self.init_obj_pos_vel[env_id]) * noise_weight[7]
                # self.init_obj_rot[env_id] += torch.randn_like(self.init_obj_rot[env_id]) * noise_weight[8]
                # self.init_obj_rot_vel[env_id] += torch.randn_like(self.init_obj_rot_vel[env_id]) * noise_weight[9]
                if self.isTest:
                    pass #print(f"Random noise added to initial state for env {env_id}")

                if self.state_search_to_align_reward:
                    noisy_motion = {
                        'root_pos': self.init_root_pos[env_id],
                        'key_body_pos': self._rigid_body_pos[env_id, self._key_body_ids, :],
                        'key_body_pos_vel': self._rigid_body_vel[env_id, self._key_body_ids, :],
                        'root_rot': self.init_root_rot[env_id],
                        'root_pos_vel': self.init_root_pos_vel[env_id],
                        'root_rot_vel': self.init_root_rot_vel[env_id],
                        'dof_pos': self.init_dof_pos[env_id],
                        'dof_pos_vel': self.init_dof_pos_vel[env_id],
                        # 'obj_pos': self.init_obj_pos[env_id],
                        # 'obj_pos_vel': self.init_obj_pos_vel[env_id],
                        # 'obj_rot': self.init_obj_rot[env_id],
                        # 'obj_rot_vel': self.init_obj_rot_vel[env_id],
                    }
                    motion_id = motion_ids[ind:ind+1]
                    new_source_motion_time, self.max_sim[env_id] = self._motion_data.noisy_resample_time(noisy_motion, motion_id)
                    motion_times[ind:ind+1] = new_source_motion_time
                    # resample the hoi_data_batch
                    self.hoi_data_batch[env_id], _, _, _, _, _, _, _, _, _, _ \
                        = self._motion_data.get_initial_state(env_ids[ind:ind+1], motion_id, new_source_motion_time)

        return motion_ids, motion_times
        
    def _init_from_random_skill(self, env_ids, motion_ids, motion_times): 
        # Random init from other skills
        state_switch_flags = [np.random.rand() < self.cfg['env']['state_switch_prob'] for _ in env_ids]
        for ind, env_id in enumerate(env_ids):
            if state_switch_flags[ind] and not self.state_random_flags[ind]: 
                source_motion_class = self._motion_data.motion_class[motion_ids[ind]]
                source_motion_id = motion_ids[ind:ind+1]
                source_motion_time = motion_times[ind:ind+1]

                if self.state_search_to_align_reward:
                    switch_motion_class, switch_motion_id, switch_motion_time, max_sim = \
                        random.choice(self.state_search_graph[source_motion_class][source_motion_id.item()][source_motion_time.item()])
                    if switch_motion_id is None and switch_motion_time is None:
                        continue
                    else:
                        self.max_sim[env_id] = max_sim
                    switch_motion_id = torch.tensor([switch_motion_id], device=self.device)
                    switch_motion_time = torch.tensor([switch_motion_time], device=self.device)

                    # switch_motion_time, new_source_motion_time = self._motion_data.resample_time(source_motion_id, switch_motion_id, weights=self.similarity_weights)
                    # motion_times[ind:ind+1] = new_source_motion_time
                    # resample the hoi_data_batch
                    # self.hoi_data_batch[env_id], init_root_pos_source, init_root_rot_source,  _, _, _, _, init_obj_pos_source , _, _, _ = \
                    #     self._motion_data.get_initial_state(env_ids[ind:ind+1], source_motion_id, new_source_motion_time)
                else:
                    switch_motion_id = self._motion_data.sample_switch_motions(source_motion_id)
                    motion_len = self._motion_data.motion_lengths[switch_motion_id].item()
                    switch_motion_time = torch.randint(2, motion_len - 2, (1,), device=self.device)

                    # resample the hoi_data_batch
                    # self.hoi_data_batch[env_id], init_root_pos_source, init_root_rot_source,  _, _, _, _, init_obj_pos_source, _, _, _ = \
                    #     self._motion_data.get_initial_state(env_ids[ind:ind+1], source_motion_id, source_motion_time)
                 
                # 从switch中获取待对齐的初始状态
                _, init_root_pos_switch, init_root_rot_switch, init_root_pos_vel_switch, init_root_rot_vel_switch, \
                init_dof_pos_switch, init_dof_pos_vel_switch, \
                _, _, _, _  \
                    = self._motion_data.get_initial_state(env_ids[ind:ind+1], switch_motion_id, switch_motion_time)
                #init_obj_pos_switch, init_obj_pos_vel_switch, init_obj_rot_switch, init_obj_rot_vel_switch \
                
                # 从source中获取初始状态的对齐目标
                self.hoi_data_batch[env_id], init_root_pos_source, init_root_rot_source,  _, _, _, _, init_obj_pos_source , _, _, _ = \
                        self._motion_data.get_initial_state(env_ids[ind:ind+1], source_motion_id, source_motion_time)


                # 计算 yaw 差异: 我们要从switch参考系转到source参考系
                yaw_source = torch_utils.quat_to_euler(init_root_rot_source)[0][2]
                yaw_switch = torch_utils.quat_to_euler(init_root_rot_switch)[0][2]
                yaw_diff = yaw_source - yaw_switch
                yaw_diff = (yaw_diff + torch.pi) % (2*torch.pi) - torch.pi

                # 对齐 root_pos 
                self.init_root_pos[env_id] = self.rotate_xy(init_root_pos_switch[0], init_root_pos_switch[0], yaw_diff, init_root_pos_source[0])
                #从 get_initial_state 返回的 init_root_pos_source、init_root_pos_switch 通常是 (1,3)
                # equal to: init_root_pos_switch[:2] = init_root_pos_source[:2]

                # 对齐 root_rot
                yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw_diff), torch.zeros_like(yaw_diff), yaw_diff)
                self.init_root_rot[env_id] = torch_utils.quat_multiply(yaw_quat, init_root_rot_switch)

                # # 对齐 obj_pos #Z ball
                # self.init_obj_pos[env_id] = self.rotate_xy(init_obj_pos_switch[0], init_root_pos_switch[0], yaw_diff, init_root_pos_source[0])

                # # 对齐 obj_rot # 因为是ball，所以不需要变换
                # # self.init_obj_rot[env_id] # 因为球的旋转不计算奖励，所以不需要更新

                # 速度和dof不需要坐标对齐
                self.init_root_pos_vel[env_id] = init_root_pos_vel_switch
                self.init_root_rot_vel[env_id] = init_root_rot_vel_switch
                self.init_dof_pos[env_id] = init_dof_pos_switch
                self.init_dof_pos_vel[env_id] = init_dof_pos_vel_switch
                # self.init_obj_pos_vel[env_id] = init_obj_pos_vel_switch
                # self.init_obj_rot_vel[env_id] = init_obj_rot_vel_switch

                if self.isTest:
                    print(f"Switched from skill {switch_motion_class} to {source_motion_class} for env {env_id}")
        
        return motion_ids, motion_times

    def rotate_xy(self, pos, center, angle, target_root_pos):
        # pos, center都是(3,)的Tensor, angle为标量，target_root_pos为(3,)
        x_rel = pos[0] - center[0]
        y_rel = pos[1] - center[1]
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        x_new = x_rel * cos_a - y_rel * sin_a + target_root_pos[0]
        y_new = x_rel * sin_a + y_rel * cos_a + target_root_pos[1]
        # Z保持原样
        z_new = pos[2]
        return torch.tensor([x_new, y_new, z_new], device=pos.device)
    
    def after_reset_actors(self, env_ids):
        self._reset_ref_env_ids = env_ids
        # self._reset_ref_motion_ids = self._motion_data.envid2motid[env_ids] # self.motion_ids
        # self._reset_ref_motion_times = self._motion_data.envid2sframe[env_ids] # self.motion_times
        
        if self.switch_skill_name is not None: 
            skill_dict = {'run': 13, 'lrun': 11, 'rrun': 12, 'layup': 31, 'shot': 9, 'run_no_object': 10, 'getup': 0, 'pickup': 1}
            self.hoi_data_label_batch = F.one_hot(torch.tensor(skill_dict[self.skill_name], device=self.device), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1).float()

        return
    
    def _reset_actors(self, env_ids):
        self._reset_state_init(env_ids)

        super()._reset_actors(env_ids)

        self.after_reset_actors(env_ids)

        return

    def _reset_humanoid(self, env_ids):
        if self.isTest and self.switch_skill_name is None and self.skill_name == 'getup':
            self._humanoid_root_states[env_ids, 0:3] = self.init_root_pos[env_ids, 0:3]
            self._humanoid_root_states[env_ids, 3] += torch.rand(1).to("cuda")
            self._humanoid_root_states[env_ids, 3:7] = self.init_root_rot[env_ids]
            self._humanoid_root_states[env_ids, 7:10] = torch.rand(3).to("cuda")
            self._humanoid_root_states[env_ids, 10:13] = torch.rand(3).to("cuda")

            self._dof_pos[env_ids] = torch.rand_like(self.init_dof_pos[env_ids])
            self._dof_vel[env_ids] = torch.rand_like(self.init_dof_pos_vel[env_ids])
        else:
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
        _, _, _, _ = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)
        # self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \ #Z abll

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
        _, _, _, _ = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)
        # self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \ #Z abll

        return motion_ids, motion_times
    
    
    def _update_condition(self):

        for evt in self.evts:
            if evt.action.isdigit() and evt.value > 0:
                self.hoi_data_label_batch = F.one_hot(torch.tensor(int(evt.action), device=self.device), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1).float()
                print(evt.action)

    def play_dataset_step(self, time): #Z12

        t = time

        for env_id, env_ptr in enumerate(self.envs):
            motid = self._motion_data.envid2motid[env_id].item() # if not self.play_dataset_switch[env_id] else 1
            t = t % self._motion_data.motion_lengths[motid]

            ### update object ###
            self._target_states[env_id, :3] = self._motion_data.hoi_data_dict[motid]['obj_pos'][t,:]
            self._target_states[env_id, 3:7] = self._motion_data.hoi_data_dict[motid]['obj_rot'][t,:]
            self._target_states[env_id, 7:10] = torch.zeros_like(self._target_states[env_id, 7:10])
            self._target_states[env_id, 10:13] = torch.zeros_like(self._target_states[env_id, 10:13])

            ### update subject ###
            _humanoid_root_pos = self._motion_data.hoi_data_dict[motid]['root_pos'][t,:].clone()
            _humanoid_root_rot = self._motion_data.hoi_data_dict[motid]['root_rot'][t,:].clone()
            self._humanoid_root_states[env_id, 0:3] = _humanoid_root_pos
            self._humanoid_root_states[env_id, 3:7] = _humanoid_root_rot
            self._humanoid_root_states[env_id, 7:10] = torch.zeros_like(self._humanoid_root_states[env_id, 7:10])
            self._humanoid_root_states[env_id, 10:13] = torch.zeros_like(self._humanoid_root_states[env_id, 10:13])
            
            self._dof_pos[env_id] = self._motion_data.hoi_data_dict[motid]['dof_pos'][t,:].clone()
            self._dof_vel[env_id] = torch.zeros_like(self._dof_vel[env_id])

            # env_id_int32 = self._humanoid_actor_ids[env_id].unsqueeze(0)

            contact = self._motion_data.hoi_data_dict[motid]['contact'][t,:]
            obj_contact = torch.any(contact > 0.1, dim=-1)
            root_rot_vel = self._motion_data.hoi_data_dict[motid]['root_rot_vel'][t,:]
            angle = torch.norm(root_rot_vel)
            abnormal = torch.any(torch.abs(angle) > 5.) #Z

            if abnormal == True:
                # print("frame:", t, "abnormal:", abnormal, "angle", angle)
                # print(" ", self._motion_data.hoi_data_dict[motid]['root_rot_vel'][t])
                # print(" ", angle)
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
                os.makedirs("skillmimic/data/images/" + self.save_images_timestamp, exist_ok=True)
                frame_id = t if self.play_dataset else self.progress_buf[env_ids]
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
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        self._init_amp_obs(env_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self.motion_ids, self.motion_times)
        
        return
    
    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        # dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -(torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1) # * dt
        motion_times = motion_times + time_steps #(n_envs,1) + (1,n_steps) broadcasting
        motion_times = torch.clip(motion_times, min=2)

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        root_pos = self._motion_data.hoi_data_for_amp_obs['root_pos'][motion_ids, motion_times]
        root_rot = self._motion_data.hoi_data_for_amp_obs['root_rot'][motion_ids, motion_times]
        dof_pos = self._motion_data.hoi_data_for_amp_obs['dof_pos'][motion_ids, motion_times]
        root_vel = self._motion_data.hoi_data_for_amp_obs['root_pos_vel'][motion_ids, motion_times]
        root_ang_vel = self._motion_data.hoi_data_for_amp_obs['root_rot_vel'][motion_ids, motion_times]
        dof_vel = self._motion_data.hoi_data_for_amp_obs['dof_pos_vel'][motion_ids, motion_times]
        key_pos = self._motion_data.hoi_data_for_amp_obs['key_body_pos'][motion_ids, motion_times].view(root_pos.shape[0],len(self._key_body_ids),3)
        # target_states = self._motion_data.hoi_data_for_amp_obs['object_data'][motion_ids, motion_times] #Z ball

        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets,
                                            #   target_states #Z ball
                                              )
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def fetch_amp_obs_demo(self, num_samples):

        if (self._amp_obs_demo_buf is None):
            self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        # motion_ids = self._motion_lib.sample_motions(num_samples)
        motion_ids = np.random.choice(
            np.arange(self._motion_data.num_motions),
            size=num_samples,
            p=self._motion_data._motion_weights / self._motion_data._motion_weights.sum()
        )
        motion_ids = torch.tensor(motion_ids, dtype=torch.long)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        # truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        # motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        # motion_times0 += truncate_time

        truncate_step = (self._num_amp_obs_steps - 1)
        motion_lengths = self._motion_data.motion_lengths[motion_ids] - truncate_step
        phase = torch.rand(motion_ids.shape, device=self.device)
        motion_timestep0 = (phase * motion_lengths).long()
        motion_timestep0 += truncate_step

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_timestep0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = - torch.arange(0, self._num_amp_obs_steps, device=self.device) # dt *
        motion_times = motion_times + time_steps
        motion_times = torch.clip(motion_times, min=2)

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        # root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
        #        = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_pos = self._motion_data.hoi_data_for_amp_obs['root_pos'][motion_ids, motion_times]
        root_rot = self._motion_data.hoi_data_for_amp_obs['root_rot'][motion_ids, motion_times]
        dof_pos = self._motion_data.hoi_data_for_amp_obs['dof_pos'][motion_ids, motion_times]
        root_vel = self._motion_data.hoi_data_for_amp_obs['root_pos_vel'][motion_ids, motion_times]
        root_ang_vel = self._motion_data.hoi_data_for_amp_obs['root_rot_vel'][motion_ids, motion_times]
        dof_vel = self._motion_data.hoi_data_for_amp_obs['dof_pos_vel'][motion_ids, motion_times]
        key_pos = self._motion_data.hoi_data_for_amp_obs['key_body_pos'][motion_ids, motion_times].view(root_pos.shape[0],len(self._key_body_ids),3)
        target_states = self._motion_data.hoi_data_for_amp_obs['object_data'][motion_ids, motion_times]

        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets,
                                            #   target_states #Z ball
                                              )
        return amp_obs_demo

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets,
                                                            #    self._target_states #Z ball
                                                               )
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets,
                                                                #    self._target_states[env_ids] #Z ball
                                                                   )
        return
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

    
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

    # if isTest and maxEpisodeLength > 0 :
    #     reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    # else:
    #     reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC

    reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    # reset = torch.zeros_like(reset_buf) #ZC300

    return reset, terminated


@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif (dof_size == 1):
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device)
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert(False), "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert((num_joints * joint_obs_size) == dof_obs_size)

    return dof_obs

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets,
                        #    tar_states
                           ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    # #Z ball
    # tar_pos = tar_states[:, 0:3]
    # tar_rot = tar_states[:, 3:7]
    # tar_vel = tar_states[:, 7:10]
    # # tar_ang_vel = tar_states[:, 10:13]

    # local_tar_pos = tar_pos - root_pos
    # local_tar_pos[..., -1] = tar_pos[..., -1]
    # local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    # local_tar_vel = quat_rotate(heading_rot, tar_vel)
    # # local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    # local_tar_rot = quat_mul(heading_rot, tar_rot)
    # local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos,
                    #  local_tar_pos, local_tar_rot_obs, local_tar_vel, #local_tar_ang_vel #Z ball
                     ), dim=-1)
    return obs