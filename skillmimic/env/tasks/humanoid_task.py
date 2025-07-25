from enum import Enum
import numpy as np
import torch
from torch import Tensor
from typing import Tuple
import glob, os, random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils

from env.tasks.base_task import BaseTask


PERTURB_OBJS = [
    ["small", 60],
    # ["large", 60],
]

class HumanoidWholeBody(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.projtype = cfg['env']['projtype']

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"] #V1
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        
        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
         
        super().__init__(cfg=self.cfg)
        
        self.dt = self.control_freq_inv * sim_params.dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim) #V1
        # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim) # Although the performance impact is usually quite small, it is best to only enable the sensors when needed.
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        # sensors_per_env = 2
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        # self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()
        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 3] = 1.0    # w 
        self._initial_humanoid_root_states[:, 4:7] = 0.0  # x, y, z
        self._initial_humanoid_root_states[:, 7:13] = 0.0
        self.init_root_pos = self._initial_humanoid_root_states[:, 0:3]     
        self.init_root_rot = self._initial_humanoid_root_states[:, 3:7]     
        self.init_root_pos_vel = self._initial_humanoid_root_states[:, 7:10]  
        self.init_root_rot_vel = self._initial_humanoid_root_states[:, 10:13] 
        
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
        self.init_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self.init_dof_pos_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]
                
        self._build_termination_heights()
        
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(self.cfg["env"]["contactBodies"])
        
        if self.viewer != None:
            self._init_camera()

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long) # in  extras/info
        return

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/smplx_capsule.xml"):
            self._dof_obs_size = (51)*3
            self._num_actions = (51)*3
            self._num_obs = 1 + (52) * (3 + 6 + 3 + 3) - 3 + 10*3 #V1 + self.condition_size + obj_obs_size + self.get_task_obs_size()
        elif (asset_file == "mjcf/mocap_humanoid.xml"):
            self._dof_obs_size = (52)*3
            self._num_actions = (52)*3
            obj_obs_size = 15
            self._num_obs = 1 + (53) * (3 + 6 + 3 + 3) - 3 + 10*3
        elif (asset_file == "mjcf/mocap_humanoid_boxhand.xml"):
            self._dof_obs_size = (52)*3
            self._num_actions = (52)*3
            obj_obs_size = 15
            self._num_obs = 1 + (53) * (3 + 6 + 3 + 3) - 3 + 10*3
        ######### parahome #########
        elif (asset_file in ["mjcf/mocap_parahome_boxhand.xml", "mjcf/mocap_parahome_boxhand_s22.xml"]):
            self._dof_obs_size = (60)*3
            self._num_actions = (60)*3
            obj_obs_size = 15
            self._num_obs = 1 + (61) * (3 + 6 + 3 + 3) - 3 + 10*3
        elif ("mocap_parahome_boxhand_refobj" in asset_file):
            self._dof_obs_size = (60)*3
            self._num_actions = (60)*3
            obj_obs_size = 15
            self._num_obs = 1 + (61) * (3 + 6 + 3 + 3) - 3 + 10*3 + 6 # reference object info
        elif ("mocap_parahome_boxhand_multiobj" in asset_file):
            self._dof_obs_size = (60)*3
            self._num_actions = (60)*3
            obj_obs_size = 15
            self._num_obs = 1 + (61) * (3 + 6 + 3 + 3) - 3 + 10*3 + 15 + 6 # reference object info
        elif ("mocap_parahome_boxhand_multirefobj" in asset_file):
            self._dof_obs_size = (60)*3
            self._num_actions = (60)*3
            obj_obs_size = 15
            self._num_obs = 1 + (61) * (3 + 6 + 3 + 3) - 3 + 10*3 + 21 + 6 # reference object info
        elif ("mocap_parahome_boxhand_hist" in asset_file):
            self._dof_obs_size = (60)*3
            self._num_actions = (60)*3
            obj_obs_size = 15
            self._num_obs = 1 + (61) * (3 + 6 + 3 + 3) - 3 + 10*3 + 3 # hist vector
        #####################################
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return
    
    def get_obs_size(self):
        obs_size = 0
        humanoid_obs_size = self._num_obs
        obs_size += humanoid_obs_size
        if self._enable_task_obs:
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors
    
    def get_task_obs_size(self):
        return 0

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # asset_options.fix_base_link = True
        # asset_options.disable_gravity = True
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_humanoid_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_humanoid_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # compute aggregate size
        max_agg_bodies = self.num_humanoid_bodies + 2
        max_agg_shapes = self.num_humanoid_shapes + 2
        
        ######### Modified by Runyi #########
        if "mocap_parahome_boxhand" in self.cfg["env"]["asset"]["assetFileName"]:
            if self.cfg['env']['in_scene_obj_dynamic'][0] in ['kettle', 'chair']:
                max_agg_bodies += 50
                max_agg_shapes += 50
            elif len(self.cfg['env']['in_scene_obj_dynamic']) == 1 and 'sink' in self.cfg['env']['in_scene_obj_static']:
                max_agg_bodies += 140
                max_agg_shapes += 140
            elif len(self.cfg['env']['in_scene_obj_dynamic']) == 1:
                max_agg_bodies += 20
                max_agg_shapes += 20
            elif len(self.cfg['env']['in_scene_obj_dynamic']) == 2:
                max_agg_bodies += 50
                max_agg_shapes += 50
        if self.cfg["env"]["asset"]["assetFileName"] == "mocap_parahome_boxhand_multirefobj":
            max_agg_bodies += 30
            max_agg_shapes += 30
        #####################################
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            self._build_env(i, env_ptr, humanoid_asset)
            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id #0 #Z dual
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        char_h = 0.89

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, segmentation_id)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self.humanoid_handles.append(humanoid_handle)

        return

    def _build_pd_action_offset_scale(self):
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _get_humanoid_collision_filter(self):
        return 0

    def _build_termination_heights(self):
        self._termination_heights = 0.3
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return
    


    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
        return

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
        return

    def _reset_env_tensors(self, env_ids): #Z10
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return
    
    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim) 
        self.gym.refresh_net_contact_force_tensor(self.sim)

        return
    
    def _reset_actors(self, env_ids):
        self._reset_humanoid(env_ids)
        return
    
    def _reset_humanoid(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids] 
        self._dof_pos[env_ids] = self.init_dof_pos[env_ids] 
        self._dof_vel[env_ids] = self.init_dof_pos_vel[env_ids]



    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if (self._pd_control): #ZC99
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        self.evts = list(self.gym.query_viewer_action_events(self.viewer))
        return
    
    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def post_physics_step(self):
        self._hook_post_step()

        self._refresh_sim_tensors()

        self._compute_observations() # for policy
        self._compute_reward()
        # self._compute_metrics() #metric zqh
        self._compute_reset()

        # print(f'step: {int(self.progress_buf[0])}, reward: {float(self.rew_buf[0]):.10f}')
        
        self.extras["terminate"] = self._terminate_buf 
        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        self.progress_buf += 1
        self.progress_buf_total += 1
        # print(int(self.progress_buf[0]), "   ", end=' ')

        return

    def _hook_post_step(self):
        pass

    def _compute_reward(self):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf) #V1

        return
    
    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights
                                                   )
        return


    def _compute_observations(self, env_ids=None): # called @ reset & post step
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)

        if (env_ids is None): 
            self.obs_buf[:] = obs

        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            contact_forces = self._contact_forces
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            contact_forces = self._contact_forces[env_ids]
        
        obs = compute_humanoid_observations(body_pos, body_rot, body_vel, body_ang_vel, 
                                                self._local_root_obs, self._root_height_obs,
                                                contact_forces, self._contact_body_ids)

        return obs



    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()
            self._draw_task()

        super().render(sync_frame_time)
        return

    def _draw_task(self):
        return
    
    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
    
    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.50)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        # cam_pos = gymapi.Vec3(0, 
        #                       - 3.0, 
        #                       4.0)
        # cam_target = gymapi.Vec3(0,
        #                          0,
        #                          1.3)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos

        # # # fixed camera
        # new_cam_target = gymapi.Vec3(0, 0.5, 1.0)
        # new_cam_pos = gymapi.Vec3(1, -1, 1.6)
        # self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)
        return
    

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
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
def compute_humanoid_observations(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
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

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    body_contact_buf = contact_forces[:, contact_body_ids, :].clone().view(contact_forces.shape[0],-1)
    
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, body_contact_buf), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rigid_body_pos[:, 0, 2] # root height
        body_fall = body_height < termination_heights # [4096] 
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 1)
        
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated