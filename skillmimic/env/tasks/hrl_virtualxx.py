import torch
import torch.nn.functional as F

from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from env.tasks.skillmimic1_hist import SkillMimic1BallPlayHist


class HRLVirtual(SkillMimic1BallPlayHist):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        # self._enable_task_obs = cfg["env"]["enableTaskObs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._termination_heights = torch.tensor(self.cfg["env"]["terminationHeight"], device=self.device, dtype=torch.float)

        if cfg["env"]["histEncoderCkpt"]:
            import copy
            self.hist_encoder1 = copy.deepcopy(self.hist_encoder)# deepcopy 会把整个 Module 复制一份
            self.hist_encoder1.resume_from_checkpoint("models/camready/locomotion/version_4/checkpoints/epoch=999-step=9000.ckpt")# 然后再从另一个 checkpoint 载入权重
                
            self.hist_encoder1.eval()
            for p in self.hist_encoder1.parameters():
                p.requires_grad = False

        return

    def get_hist(self, env_ids, ts):
        # 支持1个 env_id 或者2个 env_ids
        # env_id=0 时用 hist_encoder，env_id=1 时用 hist_encoder2
        # 假设两个 encoder 输出的维度相同
        # 创建输出 tensor
        batch_size = env_ids.numel() if isinstance(env_ids, torch.Tensor) else 1
        out_dim = self.hist_vecotr_dim  # 假设 hist_vector_dim == hist_vector_dim2
        hist_vec = torch.zeros(batch_size, out_dim, device=self.device)

        # 取出对应的历史观测
        hist_batch = self._hist_obs_batch[env_ids]

        # 处理 env_id == 0
        mask0 = (env_ids == 0)
        if mask0.any():
            data0 = hist_batch[mask0]
            hist_vec[mask0] = self.hist_encoder(data0)

        # 处理 env_id == 1
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
        
        root_base = self._humanoid_root_states[0, 0:2]
        # For Env1,2 locomotion
        ## Root Position
        self._humanoid_root_states[1, 0:2] = root_base + torch.tensor([9.0, 2], device=self.device, dtype=torch.float)
        self._humanoid_root_states[2, 0:2] = root_base + torch.tensor([9.0, -2], device=self.device, dtype=torch.float)
        ## Root Rotation
        euler = torch_utils.quat_to_euler(self._humanoid_root_states[1, 3:7])
        eulerz_x, eulerz_y, eulerz_z = euler[0], euler[1], euler[2]
        euler_z_env1 = torch.tensor(1.9*torch.pi, device=self.device, dtype=torch.float)
        self._humanoid_root_states[1:2, 3:7] = quat_from_euler_xyz(eulerz_x, eulerz_y, euler_z_env1)
        euler_z_env2 = torch.tensor(1.6*torch.pi, device=self.device, dtype=torch.float)
        self._humanoid_root_states[2:3, 3:7] = quat_from_euler_xyz(eulerz_x, eulerz_y, euler_z_env2)
        
        # For Env3，4 locomotion
        ## Root Position
        self._humanoid_root_states[3, 0:2] = root_base + torch.tensor([16.0, 1.4], device=self.device, dtype=torch.float)
        self._humanoid_root_states[4, 0:2] = root_base + torch.tensor([16.0, -4], device=self.device, dtype=torch.float)
        ## Root Rotation
        euler = torch_utils.quat_to_euler(self._humanoid_root_states[1, 3:7])
        eulerz_x, eulerz_y, eulerz_z = euler[0], euler[1], euler[2]
        euler_z_env3 = torch.tensor(2*torch.pi, device=self.device, dtype=torch.float)
        self._humanoid_root_states[3:4, 3:7] = quat_from_euler_xyz(eulerz_x, eulerz_y, euler_z_env3)
        euler_z_env4 = torch.tensor(1.2*torch.pi, device=self.device, dtype=torch.float)
        self._humanoid_root_states[4:5, 3:7] = quat_from_euler_xyz(eulerz_x, eulerz_y, euler_z_env4)
        return

    def _reset_target(self, env_ids):
        super()._reset_target(env_ids)
        self._target_states[1:, 0:3] = torch.tensor([100, 100, 0.5], device=self.device, dtype=torch.float)

    def _update_condition(self):
        actions = {"011", "012", "013", "009", "031"}
        actions1 = {"000", "010"}

        if self.progress_buf[0] == 0:
            self.hoi_data_label_batch[1:3] = F.one_hot(torch.tensor(10, device=self.device),num_classes=self.condition_size).float()
            self.hoi_data_label_batch[3:5] = F.one_hot(torch.tensor(0, device=self.device),num_classes=self.condition_size).float()
        if self.progress_buf[0] == 140:
            self.hoi_data_label_batch[1:3] = F.one_hot(torch.tensor(0, device=self.device),num_classes=self.condition_size).float()
        if self.progress_buf[0] == 230:
            self.hoi_data_label_batch[0] = F.one_hot(torch.tensor(13, device=self.device),num_classes=self.condition_size).float()
            self.hoi_data_label_batch[3:5] = F.one_hot(torch.tensor(10, device=self.device),num_classes=self.condition_size).float()
        if self.progress_buf[0] == 400:
            self.hoi_data_label_batch[0] = F.one_hot(torch.tensor(11, device=self.device),num_classes=self.condition_size).float()
            self.hoi_data_label_batch[3:5] = F.one_hot(torch.tensor(0, device=self.device),num_classes=self.condition_size).float()

        for evt in self.evts:
            if evt.action.isdigit() and evt.value > 0:
                idx = int(evt.action)
                one_hot = F.one_hot(torch.tensor(idx, device=self.device),num_classes=self.condition_size).float()
                if evt.action in actions:
                    self.hoi_data_label_batch[0] = one_hot
                elif evt.action in actions1:
                    self.hoi_data_label_batch[1] = one_hot

                print(evt.action)


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
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "011") # dribble left
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "012") # dribble right
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "013") # dribble forward
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "009") # shot
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "031") # layup
        #############################################################################################
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "000") # getup
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "010") # run


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
        has_fallen *= (progress_buf > 1) # 本质就是 与
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    # reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC

    reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    # reset = torch.zeros_like(reset_buf) #ZC300

    return reset, terminated