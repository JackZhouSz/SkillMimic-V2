import torch
import numpy as np
import random
import pickle
import torch.nn.functional as F
from isaacgym.torch_utils import *
from torch import Tensor

from utils import torch_utils
from utils.history_encoder import HistoryEncoder

from env.tasks.skillmimic_parahome import SkillMimicParahome


class SkillMimicParahomeLocalHist(SkillMimicParahome): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.at_target = torch.zeros(self.num_envs, self.max_episode_length, device=self.device, dtype=torch.bool)
        self.ref_hoi_data_size = 1 + self._dof_obs_size*2 + 3 + 30
        
        self.history_length = cfg['env']['historyLength']
        self._hist_obs_batch = torch.zeros([self.num_envs, self.history_length, self.ref_hoi_data_size], device=self.device, dtype=torch.float)
        self.hist_encoder = HistoryEncoder(self.history_length, input_dim=394).to(self.device)
        self.hist_encoder.eval()
        self.hist_encoder.resume_from_checkpoint(cfg["env"]["histEncoderCkpt"])
        for param in self.hist_encoder.parameters():
            param.requires_grad = False

    def get_hist(self, env_ids, ts):
        return self.hist_encoder(self._hist_obs_batch[env_ids])

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

        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids,ts].clone()

        ######### Modified by Runyi #########
        hist_vector = self.get_hist(env_ids, ts)
        obs = torch.cat([obs, hist_vector], dim=-1)

        self.obs_buf[env_ids] = obs

        #### NaN detection and handling ####
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

        # [0, 1, 2, a, b, c, d] -> [1, 2, a, b, c, d, currect_obs]
        current_obs = torch.cat([humanoid_obs[..., :211],  self._dof_pos[env_ids], obj_obs[..., :3]], dim=-1) # (envs, 394)
        self._hist_obs_batch[env_ids] = torch.cat([self._hist_obs_batch[env_ids, 1:], current_obs.unsqueeze(1)], dim=1)
        #####################################
        
        return

    def after_reset_actors(self, env_ids):
        super().after_reset_actors(env_ids)

        ######### Modified by Runyi #########
        # pt data (409 dim): root_pos(3) + root_rot(3) + root_rot(3) + dof_pos(60*3) + body_pos(71*3) 
        #                   + obj_pos(3) + zero_obj_rot(3) + contact_graph(1)
        # initialize the history observation
        self._hist_obs_batch[env_ids] = torch.zeros([env_ids.shape[0], self.history_length, self.ref_hoi_data_size], device=self.device, dtype=torch.float)
        for ind in range(env_ids.shape[0]):
            env_id = env_ids[ind]
            ref_data = self._motion_data.hoi_data_dict[int(self.motion_ids[ind])]
            humanoid_obs = get_humanoid_obs(ref_data['root_pos'], ref_data['root_rot_3d'], ref_data['body_pos'])
            obj_obs = get_obj_obs(ref_data['root_pos'], ref_data['root_rot_3d'], ref_data['obj_pos'])
            ref_data_obs = torch.cat([humanoid_obs, ref_data['dof_pos'].view(-1, 60*3), obj_obs], dim=-1)
            start_frame = self.motion_times[ind] - self.history_length
            end_frame = self.motion_times[ind]
            if start_frame >= 0:
                self._hist_obs_batch[env_id] = ref_data_obs[start_frame:end_frame]
            else:
                self._hist_obs_batch[env_id, -end_frame:] = ref_data_obs[:end_frame]
        #####################################

        return


class SkillMimicParahomeLocalHistRIS(SkillMimicParahomeLocalHist):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.state_switch_flags = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if 'stateSearchGraph' in cfg['env']:
            with open(f"{cfg['env']['stateSearchGraph']}", "rb") as f:
                self.state_search_graph = pickle.load(f)
        self.max_sim = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def _reset_random_ref_state_init(self, env_ids): #Z11
        motion_ids, motion_times = super()._reset_random_ref_state_init(env_ids)
        motion_ids, motion_times = self._init_with_random_noise(env_ids, motion_ids, motion_times)
        motion_ids, motion_times = self._init_from_random_skill(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        motion_ids, motion_times = super()._reset_deterministic_ref_state_init(env_ids)
        motion_ids, motion_times = self._init_with_random_noise(env_ids, motion_ids, motion_times)
        motion_ids, motion_times = self._init_from_random_skill(env_ids, motion_ids, motion_times)
        return motion_ids, motion_times

    def _init_with_random_noise(self, env_ids, motion_ids, motion_times): 
        # Random noise for initial state
        # self.state_random_flags = [np.random.rand() < self.cfg['env']['state_init_random_prob'] for _ in env_ids]
        state_init_random_prob = self.cfg['env']['state_init_random_prob'] if self.adapt_prob is False else self.get_state_init_random_prob() 
        self.state_random_flags = [np.random.rand() < state_init_random_prob for _ in env_ids]
        
        for ind, env_id in enumerate(env_ids):
            if self.state_random_flags[ind]:
                self.init_root_pos[env_id, 2] += random.random() * 0.1
                self.init_root_rot[env_id] += torch.randn_like(self.init_root_rot[env_id]) * 0.1
                self.init_root_pos_vel[env_id] += torch.randn_like(self.init_root_pos_vel[env_id]) * 0.015
                self.init_root_rot_vel[env_id] += torch.randn_like(self.init_root_rot_vel[env_id]) * 0.1
                self.init_dof_pos[env_id] += torch.randn_like(self.init_dof_pos[env_id]) * 0.1
                self.init_dof_pos_vel[env_id]  += torch.randn_like(self.init_dof_pos_vel[env_id]) * 0.1
                self.init_obj_pos[env_id, 2] += random.random() * 0.1
                self.init_obj_pos_vel[env_id] += torch.randn_like(self.init_obj_pos_vel[env_id]) * 0.015
                self.init_obj_rot[env_id] += torch.randn_like(self.init_obj_rot[env_id]) * 0.1
                self.init_obj_rot_vel[env_id] += torch.randn_like(self.init_obj_rot_vel[env_id]) * 0.1

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
        
class SkillMimicParahomeLocalHistRISBuffernode(SkillMimicParahomeLocalHistRIS):
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
def get_humanoid_obs(root_pos, root_rot, body_pos):
    root_h_obs = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
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

    obs = torch.cat((root_h_obs, local_body_pos), dim=-1)
    return obs

@torch.jit.script
def get_obj_obs(root_pos, root_rot, tar_pos):
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)

    return local_tar_pos


@torch.jit.script
def compute_local_hoi_data(hoi_data_batch: Tensor, switch_root_pos: Tensor, switch_root_rot_quat: Tensor, len_keypos: int) -> Tensor:
    # hoi_data_batch (60, 428)
    # switch_root_rot_quat (1, 4)
    local_hoi_data_batch = hoi_data_batch.clone()
    init_root_pos = hoi_data_batch[0,:3]
    init_root_rot = hoi_data_batch[0,3:6]

    root_pos = hoi_data_batch[:,:3]
    root_rot = hoi_data_batch[:,3:6]
    start_ind = 6
    dof_obs_size = 180
    dof_pos = hoi_data_batch[:,start_ind:start_ind+dof_obs_size]
    start_ind += dof_obs_size
    dof_pos_vel = hoi_data_batch[:,start_ind:start_ind+dof_obs_size]
    start_ind += dof_obs_size
    obj_pos = hoi_data_batch[:,start_ind:start_ind+3]
    start_ind += 3
    obj_rot = hoi_data_batch[:,start_ind:start_ind+4]
    start_ind += 4
    obj_pos_vel = hoi_data_batch[:,start_ind:start_ind+3]
    start_ind += 3
    key_pos = hoi_data_batch[:,start_ind:start_ind+len_keypos*3]
    start_ind += len_keypos*3
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