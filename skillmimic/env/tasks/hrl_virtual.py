import torch
import torch.nn.functional as F
import numpy as np

from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

# from env.tasks.skillmimic2_hist import SkillMimic2BallPlayHist
from env.tasks.skillmimic2 import SkillMimic2BallPlay


class HRLVirtual(SkillMimic2BallPlay):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        # self._enable_task_obs = cfg["env"]["enableTaskObs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._termination_heights = torch.tensor(self.cfg["env"]["terminationHeight"], device=self.device, dtype=torch.float)
        self.reached_target = False
        self.motion_dict = {}

        if cfg["env"]["histEncoderCkpt"]:
            import copy
            self.hist_encoder1 = copy.deepcopy(self.hist_encoder)# deepcopy will copy the entire Module
            self.hist_encoder1.resume_from_checkpoint("hist_encoder/Locomotion/hist_model.ckpt")# Then load weights from another checkpoint
                
            self.hist_encoder1.eval()
            for p in self.hist_encoder1.parameters():
                p.requires_grad = False

        return

    def get_hist(self, env_ids, ts):
        # Support 1 or 2 env_ids
        # When env_id=0 use hist_encoder, when env_id=1 use hist_encoder2
        # Assume the output dimensions of the two encoders are the same
        # Create output tensor
        batch_size = env_ids.numel() if isinstance(env_ids, torch.Tensor) else 1
        out_dim = self.hist_vecotr_dim  # Assume hist_vector_dim == hist_vector_dim2
        hist_vec = torch.zeros(batch_size, out_dim, device=self.device)

        # Extract the corresponding historical observation
        hist_batch = self._hist_obs_batch[env_ids]

        # Handle env_id == 0
        mask0 = (env_ids == 0)
        if mask0.any():
            data0 = hist_batch[mask0]
            hist_vec[mask0] = self.hist_encoder(data0)

        # Handle env_id == 1
        mask1 = (env_ids == 1)
        if mask1.any():
            data1 = hist_batch[mask1]
            hist_vec[mask1] = self.hist_encoder1(data1)

        return hist_vec

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = 0 #env_id #Z dual
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

        self._build_target(env_id, env_ptr)
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj(env_id, env_ptr)
            
        return

    def _reset_deterministic_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        # for i in range(6):
        #     print(self._motion_data.hoi_data_dict[i]['hoi_data_text'])
        motion_ids[0] = 2
        motion_ids[1:] = 4
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return motion_ids, motion_times

    def _reset_humanoid(self, env_ids):
        self._humanoid_root_states[env_ids, 0:3] = self.init_root_pos[env_ids]
        self._humanoid_root_states[env_ids, 3:7] = self.init_root_rot[env_ids]
        self._humanoid_root_states[env_ids, 7:10] = self.init_root_pos_vel[env_ids]
        self._humanoid_root_states[env_ids, 10:13] = self.init_root_rot_vel[env_ids]
        self._dof_pos[env_ids] = self.init_dof_pos[env_ids]
        self._dof_vel[env_ids] = self.init_dof_pos_vel[env_ids]
        self._goal_position = self._humanoid_root_states[0, 0:3] + torch.tensor([16, 0, 2.7], device=self.device, dtype=torch.float)
        
        # For locomotion
        ## Root Pos
        root_base_pos = self._humanoid_root_states[0, 0:2]
        rand_offset = torch.tensor([[7, 2.8], [8, -0.5], [5, -8]],
                                    # [6, 7], [10, 8], [4, 8], [14, 5], [14, -5], [0, 5],  [0, -3],   [10, -8], [7, 2.8], 
                                   device=self.device, dtype=torch.float)
        self._humanoid_root_states[1:-2, 0:2] = root_base_pos + rand_offset
        self._humanoid_root_states[-2, 0:2] = root_base_pos + torch.tensor([15, 0.2], device=self.device)
        self._humanoid_root_states[-1, 0:2] = root_base_pos + torch.tensor([15, -0.2], device=self.device)
        ## Root Rot
        # euler_env0 = torch_utils.quat_to_euler(self._humanoid_root_states[0:1, 3:7])[:, 2]
        # euler_envs = torch_utils.quat_to_euler(self._humanoid_root_states[1:, 3:7])
        # eulerz_x, eulerz_y, eulerz_z = euler_envs[:, 0], euler_envs[:, 1], euler_envs[:, 2]
        # envs_to_env0 = euler_env0 - eulerz_z
        # envs_to_env0 = (envs_to_env0 + torch.pi) % (2 * torch.pi) - torch.pi
        # self._humanoid_root_states[1:, 3:7] = quat_from_euler_xyz(eulerz_x, eulerz_y, envs_to_env0) # (num_envs, 4)
        ## Root Rot
        root_base_rot = self._humanoid_root_states[0, 3:7]
        delta_pos = root_base_pos[0:2] - self._humanoid_root_states[1:, 0:2]  # [num_envs-1, 2]
        yaw_angles = torch.atan2(delta_pos[:, 1], delta_pos[:, 0]) + torch.pi/2  # [num_envs-1]
        yaw_angles[1] += 0.4
        euler_root = torch_utils.quat_to_euler(root_base_rot)
        roll = euler_root[0].expand_as(yaw_angles)
        pitch = euler_root[1].expand_as(yaw_angles)
        new_rot = torch_utils.quat_from_euler_xyz(roll, pitch, yaw_angles)
        self._humanoid_root_states[1:, 3:7] = new_rot
        return

    def _reset_target(self, env_ids):
        super()._reset_target(env_ids)
        self._target_states[1:, 0:3] = torch.tensor([100, 100, 0.5], device=self.device, dtype=torch.float)

    def _update_condition(self):
        actions = {"011", "012", "013", "009", "031"}
        actions1 = {"000", "010"}

        if self.progress_buf[0] == 0:
            self.hoi_data_label_batch[0] = F.one_hot(torch.tensor(13, device=self.device),num_classes=self.condition_size).float()
            self.hoi_data_label_batch[1:] = F.one_hot(torch.tensor(10, device=self.device),num_classes=self.condition_size).float()
            self.hoi_data_label_batch[-2:] = F.one_hot(torch.tensor(0, device=self.device),num_classes=self.condition_size).float()
        # if self.progress_buf[0] == 120:
        #     self.hoi_data_label_batch[0] = F.one_hot(torch.tensor(12, device=self.device),num_classes=self.condition_size).float()
        # if self.progress_buf[0] == 240:
        #     self.hoi_data_label_batch[0] = F.one_hot(torch.tensor(11, device=self.device),num_classes=self.condition_size).float()
        # if self.progress_buf[0] == 340:
        #     self.hoi_data_label_batch[0] = F.one_hot(torch.tensor(13, device=self.device),num_classes=self.condition_size).float()
        # if self.progress_buf[0] == 380:
        #     self.hoi_data_label_batch[0] = F.one_hot(torch.tensor(12, device=self.device),num_classes=self.condition_size).float()
        # if self.progress_buf[0] == 430:
        #     self.hoi_data_label_batch[0] = F.one_hot(torch.tensor(13, device=self.device),num_classes=self.condition_size).float()
        # if self.progress_buf[0] == 500:
        #     self.hoi_data_label_batch[0] = F.one_hot(torch.tensor(31, device=self.device),num_classes=self.condition_size).float()
        
        run_to_stand = torch.norm(self._humanoid_root_states[1:, 0:3] - self._humanoid_root_states[0, 0:3], dim=-1) < 2.0
        stand_lable = F.one_hot(torch.tensor(0, device=self.device),num_classes=self.condition_size).float()
        self.hoi_data_label_batch[1:] = torch.where(run_to_stand.unsqueeze(1), stand_lable, self.hoi_data_label_batch[1:])
        
        for evt in self.evts:
            if evt.action.isdigit() and evt.value > 0:
                idx = int(evt.action)
                one_hot = F.one_hot(torch.tensor(idx, device=self.device),num_classes=self.condition_size).float()
                self.hoi_data_label_batch[0] = one_hot
                # if evt.action in actions:
                #     self.hoi_data_label_batch[0] = one_hot
                # elif evt.action in actions1:
                #     self.hoi_data_label_batch[1] = one_hot
                print(evt.action)
        self.reached_target = torch.norm(self._target_states[0, 0:3] - self._goal_position, dim=-1) < 0.3

    def get_task_obs_size(self):
        obs_size = 0
        # if (self._enable_task_obs):
        #     obs_size = self.goal_size
        return obs_size


    def _compute_reset(self):
        root_pos = self._humanoid_root_states[..., 0:3]
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   root_pos,
                                                   self.max_episode_length, self._enable_early_termination, self._termination_heights
                                                   )
        return
    
    def _compute_reward(self): #, actions

        self.rew_buf[:] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        return

    
    def _subscribe_events_for_change_condition(self):
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "011") # dribble left
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "012") # dribble right
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "013") # dribble forward
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "009") # shot
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "031") # layup
        # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "011") # dribble left
        # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "012") # dribble right
        # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "013") # dribble forward
        # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "009") # shot
        # self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "031") # layup
        #############################################################################################
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "000") # getup
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "010") # run


    def _draw_task(self):
        point_color = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # Red for goal position
        self.gym.clear_lines(self.viewer)
        goal_pos = self._goal_position.cpu().numpy()
        
        # Draw goal position as a small line segment (point)
        point_color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32) if self.reached_target else point_color
        goal_verts = np.array([goal_pos[0]-0.25, goal_pos[1]-0.25, 2.6, goal_pos[0] + 0.25, goal_pos[1] + 0.25, 2.6], dtype=np.float32)
        goal_verts = goal_verts.reshape([1, 6])
        self.gym.add_lines(self.viewer, self.envs[0], goal_verts.shape[0], goal_verts, point_color)
        goal_verts = np.array([goal_pos[0]-0.25, goal_pos[1]+0.25, 2.6, goal_pos[0] + 0.25, goal_pos[1] - 0.25, 2.6], dtype=np.float32)
        goal_verts = goal_verts.reshape([1, 6])
        self.gym.add_lines(self.viewer, self.envs[0], goal_verts.shape[0], goal_verts, point_color)
        return
    
    def _build_frame_for_blender(self, motion_dict, rootpos, rootrot, dofpos, dofrot, ballpos, ballrot, boxpos=None, boxrot=None):
        for key in ['rootpos', 'rootrot', 'dofpos', 'dofrot', 'ballpos', 'ballrot', 'boxpos', 'boxrot']:
            if key not in motion_dict:
                if (key == 'boxpos' and boxpos is None) or (key == 'boxrot' and boxrot is None):
                    continue
                motion_dict[key]=[]

        motion_dict['rootpos'].append(rootpos.clone())
        motion_dict['rootrot'].append(rootrot.clone())
        motion_dict['dofpos'].append(dofpos.clone())
        motion_dict['dofrot'].append(dofrot.clone())
        motion_dict['ballpos'].append(ballpos.clone())
        motion_dict['ballrot'].append(ballrot.clone())
        if boxpos is not None:
            motion_dict['boxpos'].append(boxpos.clone())
        if boxrot is not None:
            motion_dict['boxrot'].append(boxrot.clone())

    def _save_motion_dict(self, motion_dict, filename='motion.pt'):
        for key in motion_dict:
            if len(motion_dict[key]) > 0:
                motion_dict[key] = torch.stack(motion_dict[key])

        torch.save(motion_dict, filename)
        print(f'Successfully save the motion_dict to {filename}!')
        exit()

    def post_physics_step(self):
        super().post_physics_step()
        
        # # to save data for blender
        # body_ids = list(range(53))
        # self._build_frame_for_blender(self.motion_dict,
        #                 self._rigid_body_pos[:, 0, :],
        #                 self._rigid_body_rot[:, 0, :],
        #                 self._rigid_body_pos[:, body_ids, :],
        #                 #torch.cat((self._rigid_body_rot[0, :1, :], torch_utils.exp_map_to_quat(self._dof_pos[0].reshape(-1,3))),dim=0),
        #                 self._rigid_body_rot[:, body_ids, :],
        #                 self._target_states[0, :3],
        #                 self._target_states[0, 3:7]
        #                 )
        # if self.progress_buf_total == 690:
        #     self._save_motion_dict(self.motion_dict, '/home/admin1/runyi/SkillMimic2_yry/blender_motion/teaser_demo.pt')
    
#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, root_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        has_fallen = root_pos[..., 2] < termination_heights
        has_fallen *= (progress_buf > 1) # This is essentially an AND operation
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    # reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC

    reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    # reset = torch.zeros_like(reset_buf) #ZC300

    return reset, terminated