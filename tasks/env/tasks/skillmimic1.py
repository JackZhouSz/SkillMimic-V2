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

from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObject


class SkillMimic1BallPlay(HumanoidWholeBodyWithObject): 
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
        self.test_random_pick = cfg['env']['testRandomPick']

        if not hasattr(self, 'condition_size'):
            self.condition_size = 64

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.condition_size = 64
        
        self.ref_hoi_obs_size = 323 + len(self.cfg["env"]["keyBodies"])*3 + 6 #V1

        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        # self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        
        # skill_number = int(os.listdir(self.motion_file)[0].split('_')[0])
        # self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(skill_number), num_classes=self.condition_size).repeat(self.num_envs,1).to(self.device)
        self.hoi_data_label_batch = torch.zeros([self.num_envs, self.condition_size], device=self.device, dtype=torch.float)

        self._subscribe_events_for_change_condition()

        self.show_motion_test = False
        self.motion_id_test = 0
        self.succ_pos = []
        self.fail_pos = []

        self.show_abnorm = [0] * self.num_envs #V1
        
        self.switch_motion_file = cfg['env']['switch_motion_file'] if 'switch_motion_file' in cfg['env'] else None
        self._load_motion(self.motion_file, self.switch_motion_file) #ZC1

        self.progress_buf_total = 0 # +1 操作应该放到reweight自己的task里

    # self.switch_skill_name
    def get_state_for_metric(self):
        # 提供 Metric 计算所需的状态
        return {
            'ball_pos': self._target_states[..., 0:3],
            'ball_pos_vel': self._target_states[..., 7:10],
            'root_pos': self._humanoid_root_states[..., 0:3],
            'root_pos_vel': self._humanoid_root_states[..., 7:10],
            # 'progress': self.progress_buf,
            # 根据需要添加其他状态
        }
    
    def post_physics_step(self):
        self._update_condition()
        super().post_physics_step()
        self._update_hist_hoi_obs()

        return

    def _hook_post_step(self):
        # extra calc of self._curr_obs, for imitation reward
        self._compute_hoi_observations()
        env_ids = torch.arange(self.num_envs)
        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids, ts].clone()
     
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
        
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs)

        textemb_batch = self.hoi_data_label_batch[env_ids]
        obs = torch.cat((obs, textemb_batch), dim=-1)
        self.obs_buf[env_ids] = obs

        return obs
    
    def _compute_reset(self):
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
                                                  self._contact_forces,
                                                  self._tar_contact_forces,
                                                  len(self._key_body_ids),
                                                  self._motion_data.reward_weights,
                                                  )
        
        return
    
    def _load_motion(self, motion_file, switch_motion_file=None):
        self.skill_name = os.path.basename(motion_file) #motion_file.split('/')[-1] #metric
        self.max_episode_length = 60
        if self.cfg["env"]["episodeLength"] > 0:
            self.max_episode_length =  self.cfg["env"]["episodeLength"]

        self.hoi_data_batch = torch.zeros([self.num_envs, self.max_episode_length, self.ref_hoi_obs_size], device=self.device, dtype=torch.float)
        
        self._motion_data = MotionDataHandler(motion_file, self.device, self._key_body_ids, self.cfg, self.num_envs, 
                                            self.max_episode_length, self.reward_weights_default, self.init_vel, self.play_dataset,
                                            reweight=self.cfg['env']['reweight'], reweight_alpha=self.cfg['env']['reweight_alpha'])
        
        if self.play_dataset:
            self.max_episode_length = self._motion_data.max_episode_length
            print("--------------------------------------------", self.max_episode_length)

        if self.skill_name in ['layup', 'shot', 'Setshot', 'Jumpshot', 'Chestpass', 'turnlayup'] and self.isTest:
            layup_target_ind = torch.argmax(self._motion_data.hoi_data_dict[0]['obj_pos'][:,2])
            self.layup_target = self._motion_data.hoi_data_dict[0]['obj_pos'][layup_target_ind]
            
        self.switch_skill_name = os.path.basename(switch_motion_file) if switch_motion_file is not None else None # switch_skill -> skill
        if switch_motion_file is not None:
            self._motion_data = MotionDataHandler(switch_motion_file, self.device, self._key_body_ids, self.cfg,
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
    
    def _reset_target(self, env_ids):
        super()._reset_target(env_ids)
        if self.isTest and self.skill_name == 'pickup' and self.test_random_pick:
            radius = torch.rand(len(env_ids), device=self.device)*2.5 + 0.5 # [0.5, 3]
            theta = torch.rand(len(env_ids), device=self.device)*2*np.pi
            self._target_states[env_ids, 0] = self.init_root_pos[env_ids, 0] + radius * torch.cos(theta)
            self._target_states[env_ids, 1] = self.init_root_pos[env_ids, 1] + radius * torch.sin(theta)
            # self._target_states[env_ids, :2] = self.init_root_pos[env_ids, :2] + torch.rand(len(env_ids),2, device=self.device)*8 - 4
            self._target_states[env_ids, 2:3] = torch.rand(len(env_ids),1, device=self.device) + 0.1 # max 1.1
            self._target_states[env_ids, 3:7] = torch.tensor([0, 0, 0, 1], device=self.device, dtype=torch.float)
            self._target_states[env_ids, 7:10] = torch.rand(len(env_ids),3, device=self.device) - 0.5 # max 0.5
    
    def _reset_state_init(self, env_ids):
        if self._state_init == -1:
            self.motion_ids, self.motion_times = self._reset_random_ref_state_init(env_ids) #V1 Random Ref State Init (RRSI)
        elif self._state_init >= 2:
            self.motion_ids, self.motion_times = self._reset_deterministic_ref_state_init(env_ids)
        else:
            assert(False), f"Unsupported state initialization from: {self._state_init}"
    
    def after_reset_actors(self, env_ids):
        skill_label = self._motion_data.motion_class_tensor[self.motion_ids]
        self.hoi_data_label_batch[env_ids] = F.one_hot(skill_label, num_classes=self.condition_size).float() #torch.tensor(skill_label, device=self.device)
        
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
        if self.isTest and not self.cfg["env"]["NR"] and self.switch_skill_name is None and self.skill_name == 'getup':
            self._humanoid_root_states[env_ids, 0:3] = self.init_root_pos[env_ids, 0:3]
            self._humanoid_root_states[env_ids, 3] += torch.rand(1, device=self.device)
            self._humanoid_root_states[env_ids, 3:7] = self.init_root_rot[env_ids]
            self._humanoid_root_states[env_ids, 7:10] = torch.rand(3, device=self.device)
            self._humanoid_root_states[env_ids, 10:13] = torch.rand(3, device=self.device)

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

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return motion_ids, motion_times
    
    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]

        if (env_ids is None):
            self._curr_obs[:] = build_hoi_observations(self._humanoid_root_states[:,0:3], #self._rigid_body_pos[:, 0, :],
                                                               self._humanoid_root_states[:,3:7], #self._rigid_body_rot[:, 0, :],
                                                               self._humanoid_root_states[:,7:10], #self._rigid_body_vel[:, 0, :],
                                                               self._humanoid_root_states[:,10:13], #self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._target_states,
                                                               self._hist_obs,
                                                               self.progress_buf)
        else:
            self._curr_obs[env_ids] = build_hoi_observations(self._humanoid_root_states[env_ids][:,0:3], #self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._humanoid_root_states[env_ids][:,3:7], #self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._humanoid_root_states[env_ids][:,7:10], #self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._humanoid_root_states[env_ids][:,10:13], #self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._target_states[env_ids],
                                                                   self._hist_obs[env_ids],
                                                                   self.progress_buf[env_ids])
        return
    
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
        return self.ref_hoi_obs_size


#####################################################################
###=========================jit functions=========================###
#####################################################################

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

@torch.jit.script
def compute_humanoid_reward(hoi_ref: Tensor, hoi_obs: Tensor, hoi_obs_hist: Tensor, contact_buf: Tensor, tar_contact_forces: Tensor, 
                            len_keypos: int, w: Dict[str, Tensor]) -> Tensor:

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
    reward = rb*ro*rig*rcg

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
    elif isTest and maxEpisodeLength > 0 :
        reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    else:
        reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC

    # reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    # reset = torch.zeros_like(reset_buf) #ZC300

    return reset, terminated
