import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
import re
import copy
from collections import Counter
from utils import torch_utils
from isaacgym.torch_utils import *


class MotionDataHandler:
    def __init__(self, motion_file, device, key_body_ids, cfg, num_envs, max_episode_length, reward_weights_default, 
                init_vel=False, play_dataset=False, reweight=False, reweight_alpha=0, use_old_reweight=False):
        self.device = device
        self._key_body_ids = key_body_ids
        self.cfg = cfg
        self.init_vel = init_vel
        self.play_dataset = play_dataset #V1
        self.max_episode_length = max_episode_length
        self.use_old_reweight = use_old_reweight
        
        self.hoi_data_dict = {}
        self.hoi_data_label_batch = None
        self.motion_lengths = None
        self.load_motion(motion_file)
        
        self.num_envs = num_envs
        self.envid2motid = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.envid2sframe = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.envid2episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reweight = reweight
        self.reweight_alpha = reweight_alpha
        # remove the start 2 frames and end 1 frames
        self.time_sample_rate = {
            motion_id: torch.ones(self.motion_lengths[motion_id].item() - 3) / (self.motion_lengths[motion_id].item() - 3)
            for motion_id in self.hoi_data_dict
            }
        self.reward_weights_default = reward_weights_default
        self.reward_weights = {}
        self.reward_weights["p"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["p"])
        self.reward_weights["r"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["r"])
        self.reward_weights["op"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["op"])
        self.reward_weights["ig"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["ig"])
        self.reward_weights["cg1"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["cg1"])
        self.reward_weights["cg2"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["cg2"])
        self.reward_weights["pv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["pv"])
        self.reward_weights["rv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["rv"])
        self.reward_weights["or"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["or"])
        self.reward_weights["opv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["opv"])
        self.reward_weights["orv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["orv"])

        self._init_vectorized_buffers() #ZQH

    def _init_vectorized_buffers(self):
        #ZQH 新增函数: 初始化 GPU 上的 _motion_weights (clip-level) 与 time_sample_rate_tensor (time-level)

        # # 1) clip-level采样权重，先全部置为1，稍后由 _compute_motion_weights(...) 或 reweight函数修正
        # self._motion_weights_tensor = torch.ones(
        #     self.num_motions, device=self.device, dtype=torch.float32
        # ) 
        # #已在 _compute_motion_weights 中求得
        
        # 2) time-level采样权重
        max_len_minus3 = int(self.motion_lengths.max().item() - 3)
        if max_len_minus3 <= 0:
            max_len_minus3 = 1  # 以防万一
        
        self.time_sample_rate_tensor = torch.zeros(
            (self.num_motions, max_len_minus3),
            device=self.device, dtype=torch.float32
        )
        
        # 将每条 motion clip 的 [2, length-1) 段设为均匀分布，但排除缺失帧
        for m_id in range(self.num_motions):
            cur_len = self.motion_lengths[m_id].item()
            cur_len_minus3 = int(cur_len - 3)
            if cur_len_minus3 <= 0:
                continue
            
            # 找到 [2, length-1) 范围内有效帧
            valid_mask = self.hoi_data_dict[m_id]['valid_frame_mask']
            valid_mask_slice = valid_mask[2: 2 + cur_len_minus3]  # shape [cur_len_minus3], bool

            # 若全部帧都无效，则退化为均匀分布
            if valid_mask_slice.sum() == 0:
                self.time_sample_rate_tensor[m_id, :cur_len_minus3] = 1.0 / float(cur_len_minus3)
            else:
                # 在有效帧上均匀分布
                dist = torch.zeros(cur_len_minus3, device=self.device)
                valid_indices = valid_mask_slice.nonzero(as_tuple=True)[0]  # 这些是 [0, cur_len_minus3) 范围
                dist[valid_indices] = 1.0 / valid_indices.numel()
                self.time_sample_rate_tensor[m_id, :cur_len_minus3] = dist
        return
    
    def load_motion(self, motion_file):
        self.skill_name = os.path.basename(motion_file) 
        all_seqs = [motion_file] if os.path.isfile(motion_file) \
            else glob.glob(os.path.join(motion_file, '**', '*.pt'), recursive=True)
        self.num_motions = len(all_seqs)
        self.motion_lengths = torch.zeros(len(all_seqs), device=self.device, dtype=torch.long)
        self.motion_class = np.zeros(len(all_seqs), dtype=int)
        self.layup_target = torch.zeros((len(all_seqs), 3), device=self.device, dtype=torch.float)
        self.root_target = torch.zeros((len(all_seqs), 3), device=self.device, dtype=torch.float)

        all_seqs.sort(key=self._sort_key)
        for i, seq_path in enumerate(all_seqs):
            loaded_dict = self._process_sequence(seq_path)
            self.hoi_data_dict[i] = loaded_dict
            self.motion_lengths[i] = loaded_dict['hoi_data'].shape[0]
            self.motion_class[i] = int(loaded_dict['hoi_data_text'])
            if self.skill_name in ['layup', "SHOT_up"]: #metric
                layup_target_ind = torch.argmax(loaded_dict['obj_pos'][:, 2])
                self.layup_target[i] = loaded_dict['obj_pos'][layup_target_ind]
                self.root_target[i] = loaded_dict['root_pos'][layup_target_ind]
        self._compute_motion_weights(self.motion_class)

        self.motion_class_tensor = torch.tensor(self.motion_class, dtype=torch.long, device=self.device) #ZQH
        # if self.play_dataset:
        #     self.max_episode_length = self.motion_lengths.min()
        print(f"--------Having loaded {len(all_seqs)} motions--------")
    
    def _sort_key(self, filename):
        match = re.search(r'\d+.pt$', filename)
        return int(match.group().replace('.pt', '')) if match else -1

    def _process_sequence(self, seq_path):
        loaded_dict = {}
        hoi_data = torch.load(seq_path)

        #ZQH -------- 新增：找出缺失帧（全零帧）并记录 --------
        # 注意：如果您只想判断关键列是否为零，请自行改写条件
        missing_mask = torch.all(hoi_data == 0, dim=1)  # True 代表该帧是全零
        valid_mask = ~missing_mask
        loaded_dict['valid_frame_mask'] = valid_mask    # True 代表该帧有效

        loaded_dict['hoi_data_text'] = os.path.basename(seq_path)[0:3]
        loaded_dict['hoi_data'] = hoi_data.detach().to(self.device)
        data_frames_scale = self.cfg["env"]["dataFramesScale"]
        fps_data = self.cfg["env"]["dataFPS"] * data_frames_scale
        # step_intervel = int( 1. / self.cfg["env"]["dataFramesScale"] ) 
        #Z This value is needed for subsequent slicing operations, 
        # e.g. loaded_dict['root_pos'] = loaded_dict['hoi_data'][::step_intervel, 0:3].clone()
        # but can be omitted for now since it is currently 1.

        loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
        loaded_dict['root_pos_vel'] = self._compute_velocity(loaded_dict['root_pos'], fps_data, valid_mask)

        loaded_dict['root_rot_3d'] = loaded_dict['hoi_data'][:, 3:6].clone()
        root_quat = torch_utils.exp_map_to_quat(loaded_dict['root_rot_3d']).clone()
        self.smooth_quat_seq(root_quat)  
        loaded_dict['root_rot'] = root_quat.clone()

        # root rot vel
        loaded_dict['root_rot_vel'] = self._compute_root_rot_vel(root_quat, fps_data, valid_mask)
        # q_diff = torch_utils.quat_multiply(
        #     torch_utils.quat_conjugate(loaded_dict['root_rot'][:-1, :].clone()), 
        #     loaded_dict['root_rot'][1:, :].clone()
        # )
        # angle, axis = torch_utils.quat_to_angle_axis(q_diff)
        # exp_map = torch_utils.angle_axis_to_exp_map(angle, axis)
        # loaded_dict['root_rot_vel'] = exp_map*fps_data
        # loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])).to(self.device),loaded_dict['root_rot_vel']),dim=0)

        loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+156].clone()
        loaded_dict['dof_pos_vel'] = self._compute_velocity(loaded_dict['dof_pos'], fps_data, valid_mask)

        data_length = loaded_dict['hoi_data'].shape[0]
        loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 165: 165+53*3].clone().view(data_length, 53, 3)
        loaded_dict['key_body_pos'] = loaded_dict['body_pos'][:, self._key_body_ids, :].view(data_length, -1).clone()
        loaded_dict['key_body_pos_vel'] = self._compute_velocity(loaded_dict['key_body_pos'], fps_data, valid_mask)

        loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 318+6:321+6].clone()
        loaded_dict['obj_pos_vel'] = self._compute_velocity(loaded_dict['obj_pos'], fps_data, valid_mask)

        loaded_dict['obj_rot'] = -loaded_dict['hoi_data'][:, 321+6:324+6].clone()
        loaded_dict['obj_rot_vel'] = self._compute_velocity(loaded_dict['obj_rot'], fps_data, valid_mask)
        if self.init_vel:
            loaded_dict['obj_pos_vel'][0] = loaded_dict['obj_pos_vel'][1]
            # loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1],loaded_dict['obj_pos_vel']),dim=0)
        loaded_dict['obj_rot'] = torch_utils.exp_map_to_quat(-loaded_dict['hoi_data'][:, 327:330]).clone()

        loaded_dict['contact'] = torch.round(loaded_dict['hoi_data'][:, 330+6:331+6].clone())

        loaded_dict['hoi_data'] = torch.cat((
            loaded_dict['root_pos'],
            loaded_dict['root_rot_3d'],
            loaded_dict['dof_pos'],
            loaded_dict['dof_pos_vel'],
            loaded_dict['obj_pos'],
            loaded_dict['obj_rot'],
            loaded_dict['obj_pos_vel'],
            loaded_dict['key_body_pos'],
            loaded_dict['contact']
        ), dim=-1)

        return loaded_dict

    # def _compute_velocity(self, positions, fps):
    #     velocity = (positions[1:, :].clone() - positions[:-1, :].clone()) * fps
    #     velocity = torch.cat((torch.zeros((1, positions.shape[-1])).to(self.device), velocity), dim=0)
    #     return velocity
    def _compute_velocity(self, positions: torch.Tensor, fps: float, valid_mask: torch.Tensor):
        """
        不做插值，先用 (pos[i] - pos[i-1])*fps。
        计算完后，若 i 帧 或 i-1 帧是缺失帧，则 velocity[i] = velocity[i-1] (或者 0)。
        """
        T = positions.shape[0]
        velocity = torch.zeros_like(positions)

        if T == 0:
            return velocity
        
        for i in range(1, T):
            velocity[i] = (positions[i] - positions[i - 1]) * fps

        # post-process
        for i in range(1, T):
            if (not valid_mask[i]) or (not valid_mask[i-1]):
                # 你可以改成 velocity[i] = 0
                velocity[i] = velocity[i - 1]

        return velocity

    def _compute_root_rot_vel(self, root_rot_quat: torch.Tensor, fps: float, valid_mask: torch.Tensor):
        T = root_rot_quat.size(0)
        rot_vel = torch.zeros((T, 3), device=root_rot_quat.device, dtype=root_rot_quat.dtype)

        for i in range(1, T):
            q_prev = root_rot_quat[i - 1]
            q_curr = root_rot_quat[i]
            q_diff = torch_utils.quat_multiply(quat_conjugate(q_prev), q_curr)
            angle, axis = torch_utils.quat_to_angle_axis(q_diff)
            exp_map = torch_utils.angle_axis_to_exp_map(angle, axis)
            rot_vel[i] = exp_map * fps

        # post-process
        for i in range(1, T):
            if (not valid_mask[i]) or (not valid_mask[i-1]):
                rot_vel[i] = rot_vel[i-1]

        return rot_vel
    
    def smooth_quat_seq(self, quat_seq):
        n = quat_seq.size(0)

        for i in range(1, n):
            dot_product = torch.dot(quat_seq[i-1], quat_seq[i])
            if dot_product < 0:
                quat_seq[i] *=-1

        return quat_seq

    def _compute_motion_weights(self, motion_class):
        unique_classes, counts = np.unique(motion_class, return_counts=True)
        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
        # class_weights = counts / counts.sum()
        class_weights = 1.0 / counts
        if 1 in class_to_index: # raise sampling probs of skill pick
            class_weights[class_to_index[1]]*=2
        self.indexed_classes = np.array([class_to_index[int(cls)] for cls in motion_class], dtype=int)
        
        if self.use_old_reweight:
            self._motion_weights = class_weights[self.indexed_classes] #ZQH
        else:
            w_array = class_weights[self.indexed_classes]  # shape=[num_motions]
            self._motion_weights_tensor = torch.tensor(w_array, device=self.device, dtype=torch.float32)
    
    # def _reweight_clip_sampling_rate(self, average_rewards):
    #     counts = Counter(self.motion_class)
    #     rewards_tensor = torch.tensor(list(average_rewards.values()), dtype=torch.float32)
    #     for idx, motion_class in enumerate(self.motion_class):
    #         rewards_tensor[idx] /= counts[motion_class]
    #     self._motion_weights = (1 - self.reweight_alpha) / len(counts) + \
    #         self.reweight_alpha * (torch.exp(-5*rewards_tensor) / torch.exp(-5*rewards_tensor).sum())
    #     print('#########################', self._motion_weights)
    #     exit()

    def _reweight_clip_sampling_rate(self, average_rewards):
        counts = Counter(self.motion_class)
        rewards_tensor = torch.tensor([average_rewards[idx] for idx in range(len(self.motion_class))], dtype=torch.float32)
        # 计算类别的平均奖励
        class_rewards = {cls: 0.0 for cls in counts}
        for idx, motion_class in enumerate(self.motion_class):
            class_rewards[motion_class] += rewards_tensor[idx]
        for cls in class_rewards:
            class_rewards[cls] /= counts[cls]
        # 计算类别间的权重
        class_weights = torch.tensor([class_rewards[cls] for cls in self.motion_class], dtype=torch.float32)
        class_weights = torch.exp(-5 * class_weights)
        class_weights /= class_weights.sum()
        
        # # 计算每个clip的权重
        # clip_weights = torch.zeros_like(rewards_tensor)
        # for idx, motion_class in enumerate(self.motion_class):
        #     clip_weights[idx] = class_weights[idx] / counts[motion_class]
        # # 归一化clip权重
        # clip_weights /= clip_weights.sum()
        # # 结合reweight_alpha调整最终权重
        # self._motion_weights = (1 - self.reweight_alpha) / len(self.motion_class) + \
        #     self.reweight_alpha * clip_weights

        # 类内权重：每个clip的奖励的指数衰减（奖励越高，权重越低）
        intra_class_weights = torch.zeros_like(rewards_tensor)
        for cls in counts:
            cls_indices = [idx for idx, c in enumerate(self.motion_class) if c == cls]
            cls_rewards = rewards_tensor[cls_indices]
            # 类内权重计算
            intra_weights = torch.exp(-5 * cls_rewards)
            intra_weights /= intra_weights.sum()  # 归一化
            intra_class_weights[cls_indices] = intra_weights
        
        # 综合权重：类间权重 × 类内权重
        combined_weights = class_weights * intra_class_weights
        # 归一化clip权重
        combined_weights /= combined_weights.sum()

        # 结合reweight_alpha调整最终权重
        self._motion_weights = (1 - self.reweight_alpha) / len(self.motion_class) + \
            self.reweight_alpha * combined_weights
        print('##### Reweight clip sampling rate #####', self._motion_weights)

    # def _reweight_clip_sampling_rate_vectorized(self, average_rewards_tensor: torch.Tensor):
    #     #ZQH 替换函数：用向量化方式更新 clip-level 采样权重, average_rewards_tensor: shape=[num_motions], 在GPU
    #     if self.num_motions < 1:
    #         return
        
    #     alpha = self.reweight_alpha
    #     negative_exp = torch.exp(-5.0 * average_rewards_tensor)  # [num_motions]
    #     sum_neg = negative_exp.sum() + 1e-8
        
    #     baseline = (1.0 - alpha) / float(self.num_motions)
    #     self._motion_weights_tensor = baseline + alpha * (negative_exp / sum_neg)

    def _reweight_clip_sampling_rate_vectorized(self, average_rewards_tensor: torch.Tensor):
        """
         功能：
           1) 先对同一类别内所有 clip 的 reward 做平均，得到 class_avg_rewards；
           2) 对 class_avg_rewards 做 exp(-5 * class_avg) 后 softmax，得到类别级别的 weight；
           3) 类别内部，再根据各 clip 的 reward 做 exp(-5 * reward_clip) 归一化，得到 clip 在类别内的相对权重；
           4) 最终 clip i 的 weight = baseline + alpha * [class_weight(class_i) * clip_intra_class_weight(i)]，
              其中 baseline = (1 - alpha) / num_motions。

         参数:
            average_rewards_tensor: shape=[num_motions], 每条 motion clip 的平均奖励 (GPU 上)

         输出:
            无显式返回值，但更新 self._motion_weights_tensor：shape=[num_motions]
        
         Tip:
           - 避免了手动 for i in range(self.num_motions)。
           - 使用 PyTorch 的 index_add_、unique + return_inverse=True 等操作完成向量化。
        """
        if self.num_motions < 1:
            return

        alpha = self.reweight_alpha  # reweight 系数
        device = self.device

        # -- 1) 计算每个类别的平均奖励 (class-level) ---------------------------------

        # 取得所有 clip 所属的类别（整型），并找到 unique class
        # cls_idx 的形状与 motion_class_tensor 相同，表示每个 clip 属于 unique_classes 中的哪个下标
        unique_classes, cls_idx = torch.unique(self.motion_class_tensor, return_inverse=True)
        # unique_classes.shape = [C], cls_idx.shape = [num_motions]

        # 用 index_add_ 做聚合：将 average_rewards_tensor 根据 cls_idx 加到 class_sum_rewards
        class_sum_rewards = torch.zeros_like(unique_classes, dtype=average_rewards_tensor.dtype, device=device)
        class_counts = torch.zeros_like(unique_classes, dtype=average_rewards_tensor.dtype, device=device)

        # 例如：对属于同一类别 c 的 clip，其奖励会加到 class_sum_rewards[c]
        class_sum_rewards.index_add_(0, cls_idx, average_rewards_tensor)
        class_counts.index_add_(0, cls_idx, torch.ones_like(average_rewards_tensor))

        # 避免除零
        class_avg_rewards = class_sum_rewards / (class_counts + 1e-8)  # shape=[C]

        # -- 2) 计算类别级别的 exp(-5*avg_reward)，再 softmax -------------------------
        negative_exp_class = torch.exp(-5.0 * class_avg_rewards)  # shape=[C]
        sum_neg_class = negative_exp_class.sum() + 1e-8
        class_weights = negative_exp_class / sum_neg_class        # shape=[C]

        # -- 3) 对各 clip 的 reward 同样做 exp(-5*reward)，并在类别内部进行归一化 ----
        negative_exp_motion = torch.exp(-5.0 * average_rewards_tensor)  # shape=[num_motions]
        sum_neg_motion_per_class = torch.zeros_like(unique_classes, dtype=average_rewards_tensor.dtype, device=device)

        # 将每个 clip 的 negative_exp_motion 累加到所属类别
        sum_neg_motion_per_class.index_add_(0, cls_idx, negative_exp_motion)
        # sum_neg_motion_per_class[k] 表示第 k 个类别内部所有 clip 的 exp(-5*reward) 之和

        # 对单条 clip 而言，其在类别内的相对权重 = negative_exp_motion[i] / sum_neg_motion_per_class[ cls_idx[i] ]
        clip_intra_class = negative_exp_motion / (sum_neg_motion_per_class[cls_idx] + 1e-8)

        # -- 4) 最终 clip 权重 = baseline + alpha * [class_weights[cls_idx] * clip_intra_class]
        baseline = (1.0 - alpha) / float(self.num_motions)
        motion_weights = class_weights[cls_idx] * clip_intra_class  # shape=[num_motions]
        motion_weights = baseline + alpha * motion_weights

        # 存储到 self._motion_weights_tensor 以便后续采样时使用
        self._motion_weights_tensor = motion_weights
        # print(self.motion_class_tensor, cls_idx, unique_classes)
        # print(average_rewards_tensor,motion_weights)


    def sample_motions(self, n):
        # motion_ids = torch.multinomial(torch.tensor(self._motion_weights), num_samples=n, replacement=True) #ZQH
        if self.use_old_reweight:
            # print('#####################', self._motion_weights)
            motion_ids = torch.multinomial(
                torch.tensor(self._motion_weights), 
                num_samples=n, 
                replacement=True
            )
        else:
            motion_ids = torch.multinomial(
                self._motion_weights_tensor, 
                num_samples=n, 
                replacement=True
            )
        # print(self._motion_weights_tensor)
        return motion_ids
    

    ################# Modified by Runyi #################
    def sample_switch_motions(self, motion_ids):
        if self.use_old_reweight:
            new_motion_ids = torch.multinomial(torch.tensor(self._motion_weights), num_samples=1, replacement=True)
            while torch.any(torch.eq(new_motion_ids, motion_ids)):
                new_motion_ids = torch.multinomial(torch.tensor(self._motion_weights), num_samples=1, replacement=True)
        else:
            new_motion_ids = torch.multinomial(torch.tensor(self._motion_weights_tensor), num_samples=1, replacement=True)
            while torch.any(torch.eq(new_motion_ids, motion_ids)):
                new_motion_ids = torch.multinomial(torch.tensor(self._motion_weights_tensor), num_samples=1, replacement=True)
        # print(f'#########Switched from {motion_ids} to {new_motion_ids}')
        return new_motion_ids

    def randskill_find_most_similarity_state(self, q_motion, q_motion_time, t_motion, weights):
        num_frames = t_motion['root_pos'].shape[0]
        total_diff = torch.zeros(num_frames, device=q_motion['hoi_data'].device)
        
        # z rotation from t to q
        q_root_rot = torch_utils.exp_map_to_quat(q_motion['root_rot_3d'][q_motion_time]).unsqueeze(0) # (1, 4)
        q_root_rot_euler_z = torch_utils.quat_to_euler(q_root_rot)[:, 2] # (1,)
        t_root_rot = torch_utils.exp_map_to_quat(t_motion['root_rot_3d']) # (num_frames, 4) 
        t_root_rot_euler_z = torch_utils.quat_to_euler(t_root_rot)[:, 2] # (num_frames,)
        t2q_root_rot_euler_z = q_root_rot_euler_z - t_root_rot_euler_z # (num_frames,)
        t2q_root_rot_euler_z = (t2q_root_rot_euler_z + torch.pi) % (2 * torch.pi) - torch.pi  # 归一化到 [-pi, pi]
        zeros = torch.zeros_like(t2q_root_rot_euler_z)
        t2q_root_rot = quat_from_euler_xyz(zeros, zeros, t2q_root_rot_euler_z)


        for key, weight in weights.items():
            if key == 'root_pos':
                diff = torch.abs(q_motion['root_pos'][q_motion_time, 2] - t_motion['root_pos'][:, 2])  # (num_frames,)
            elif key == 'obj_pos':
                q_relative_obj_pos = q_motion['obj_pos'][q_motion_time] - q_motion['root_pos'][q_motion_time]
                t_relative_obj_pos = t_motion['obj_pos'] - t_motion['root_pos']
                changed_t_relative_obj_pos = torch_utils.quat_rotate(t2q_root_rot, t_relative_obj_pos)
                diff = torch.norm(q_relative_obj_pos - changed_t_relative_obj_pos, dim=-1) # (num_frames,)
            elif key == 'obj_pos_vel':
                changed_t_obj_pos_vel = torch_utils.quat_rotate(t2q_root_rot, t_motion['obj_pos_vel'])
                diff = torch.norm(q_motion['obj_pos_vel'][q_motion_time] - changed_t_obj_pos_vel, dim=-1)
            # elif key == 'root_pos_vel':
            #     changed_t_root_pos_vel = torch_utils.quat_rotate(t2q_root_rot, t_motion['root_pos_vel'])
            #     diff = torch.norm(q_motion['root_pos_vel'][q_motion_time] - changed_t_root_pos_vel, dim=-1)
            elif key in ['root_rot_3d', 'obj_rot', 'root_rot_vel', 'dof_pos_vel','obj_rot_vel']:
                diff = 0
            elif key == 'dof_pos':
                q_dof_pos = q_motion[key][q_motion_time].reshape(52,3) if key == 'dof_pos' else q_motion[key][q_motion_time]
                t_dof_pos = t_motion[key].reshape(num_frames, 52, 3) if key == 'dof_pos' else t_motion[key]
                # 计算旋转差异
                q_dof_pos_quat = torch_utils.exp_map_to_quat(q_dof_pos).unsqueeze(0) # (num_joints, 4)
                t_dof_pos_quat = torch_utils.exp_map_to_quat(t_dof_pos) # (num_frames, num_joints, 4)
                # 计算相对四元数差异
                q_diff = torch_utils.quat_multiply(torch_utils.quat_conjugate(q_dof_pos_quat),t_dof_pos_quat)  # (num_frames, num_joints, 4)
                # 计算角度差异
                diff, _ = torch_utils.quat_to_angle_axis(q_diff)  # (num_frames,) 
                diff = torch.sum(torch.abs(diff), dim=-1)
            diff_frame = diff[8] if key not in ['root_rot_3d', 'obj_rot', 'root_rot_vel', 'dof_pos_vel','obj_rot_vel'] else 0
            # print(f'key: {key}, diff: {diff}, diff[84]: {diff_frame}')
            total_diff += diff * weight
        
        # 返回最小差异的帧索引
        total_diff = total_diff[2:-2] # (num_frames-4,)
        # print(f'total_diff: {total_diff}')
        # print(torch.argmin(total_diff, dim=0) + 2)
        return torch.argmin(total_diff, dim=0) + 2

    def resample_time(self, source_motion_id, switch_motion_id, weights=None, switch_motion_time=None):
        if weights is None:
            weights = {'root_pos': 1,'root_pos_vel': 1,'root_rot_3d': 1,'root_rot_vel': 1,
                       'dof_pos': 1,'dof_pos_vel': 1,
                       'obj_pos': 1,'obj_pos_vel': 1,'obj_rot': 1,'obj_rot_vel': 1}
        #########debug###########
        # new_motion_id = torch.tensor(0, device=self.device, dtype=torch.long)
        # source_motion_id = torch.tensor(2, device=self.device, dtype=torch.long)
        #########################
        source_motion = self.hoi_data_dict[source_motion_id.item()]
        switch_motion = self.hoi_data_dict[switch_motion_id.item()]
        if switch_motion_time is None:
            switch_motion_time = np.random.randint(2, switch_motion['hoi_data'].shape[0]-2)
        new_source_motion_time = self.randskill_find_most_similarity_state(switch_motion, switch_motion_time, source_motion, weights)
        switch_motion_time = torch.tensor(switch_motion_time, device=self.device, dtype=torch.int).unsqueeze(0)
        new_source_motion_time = new_source_motion_time.unsqueeze(0)
        
        # print(f'#########Switched from {source_motion_id} to {new_motion_id} at frame {switch_motion_time} to {new_source_motion_time}')
        # exit()
        return switch_motion_time, new_source_motion_time
    
    def noisyinit_find_most_similarity_state(self, noisy_motion, motion):
        if motion['hoi_data_text'] not in ['000', '010']:
            w = {'pos_diff': 2, 'pos_vel_diff': 0.1, 'root_rot_diff': 2, 'rot_diff': 2, 'obj_pos_diff': 1, 'obj_pos_vel_diff': 0.1, 'rel_pos_diff': 2}
        else:
            # w = {'pos_diff': 2, 'pos_vel_diff': 0.1, 'root_rot_diff': 2, 'root_height_diff': 2, 'rot_diff': 2, 'obj_pos_diff': 0, 'obj_pos_vel_diff': 0, 'rel_pos_diff': 0}
            w = {'pos_diff': 1, 'pos_vel_diff': 0.1, 'root_rot_diff': 1, 'root_height_diff': 1, 'rot_diff': 1, 'obj_pos_diff': 0, 'obj_pos_vel_diff': 0, 'rel_pos_diff': 0}
                
        # humanoid pose similarity
        noisy_motion['key_body_pos'] = noisy_motion['key_body_pos'].reshape(-1, len(self._key_body_ids)*3)
        noisy_motion['key_body_pos_vel'] = noisy_motion['key_body_pos_vel'].reshape(-1, len(self._key_body_ids)*3)
        pos_diff = torch.norm(noisy_motion['key_body_pos']- motion['key_body_pos']**2,dim=-1)
        pos_vel_diff = torch.mean((noisy_motion['key_body_pos_vel'] - motion['key_body_pos_vel'])**2,dim=-1)
        noisy_motion['root_rot_3d'] = torch_utils.quat_to_exp_map(noisy_motion['root_rot'])
        root_rot_diff = get_dof_pos_diff(noisy_motion['root_rot_3d'], motion['root_rot_3d'])
        rot_diff = get_dof_pos_diff(noisy_motion['dof_pos'], motion['dof_pos'])
        sim_pose = torch.exp(-w['pos_diff'] * pos_diff) * torch.exp(-w['pos_vel_diff'] * pos_vel_diff) * \
                   torch.exp(-w['root_rot_diff'] * root_rot_diff) * torch.exp(-w['rot_diff'] * rot_diff)
        if motion['hoi_data_text'] in ['000', '010']:
            root_height_diff = (noisy_motion['root_pos'][2] - motion['root_pos'][:, 2])**2
            sim_pose *= torch.exp(-w['root_height_diff'] * root_height_diff)

        # object configuration similarity
        obj_pos_diff = torch.mean((noisy_motion['obj_pos'] - motion['obj_pos'])**2,dim=-1)
        obj_pos_vel_diff = torch.mean((noisy_motion['obj_pos_vel'] - motion['obj_pos_vel'])**2,dim=-1)
        sim_obj = torch.exp(-w['obj_pos_diff'] * obj_pos_diff) * torch.exp(-w['obj_pos_vel_diff'] * obj_pos_vel_diff)

        # relative spatial relationship similarity
        noisy_ig = noisy_motion['key_body_pos'].reshape(len(self._key_body_ids), 3) - noisy_motion['obj_pos'].unsqueeze(0)
        motion_ig = motion['key_body_pos'].reshape(-1, len(self._key_body_ids), 3) - motion['obj_pos'].unsqueeze(1)
        rel_pos_diff = torch.mean((noisy_ig.reshape(-1, len(self._key_body_ids)*3) - motion_ig.reshape(-1, len(self._key_body_ids)*3))**2,dim=-1)
        sim_rel = torch.exp(-w['rel_pos_diff'] * rel_pos_diff)
        
        # No need to calculate sim_obj & sim_rel for getup and run
        sim = sim_pose * sim_obj * sim_rel if motion['hoi_data_text'] not in ['000', '010'] else sim_pose

        # 找到第一个大于2,且不等于-1的索引
        sorted_indices = torch.argsort(sim, descending=True)
        max_ind = next(
            (ind.item() for ind in sorted_indices if ind.item() not in [0, 1, len(sim)-1]),
            sorted_indices[0].item()
        )

        return torch.tensor(max_ind, device=sim.device), max(sim)

        # for key, weight in weights.items():
        #     if key in ['root_pos', 'root_pos_vel', 'obj_pos', 'obj_pos_vel']:
        #         # No need to calculate obj differences for getup and run
        #         if motion['hoi_data_text'] in ['000', '010'] and key in ['obj_pos', 'obj_pos_vel']:
        #             continue
        #         diff = torch.norm(noisy_motion[key]- motion[key])  # (num_frames,)
        #     elif key in ['root_rot_vel', 'dof_pos_vel', 'obj_rot_vel']:
        #         diff = 0
        #     elif key in ['root_rot_3d', 'dof_pos', 'obj_rot']:
        #         # No need to calculate obj differences for getup and run
        #         if motion['hoi_data_text'] in ['000', '010'] and key == 'obj_rot':
        #             continue
        #         q1 = noisy_motion[key].reshape(52,3) if key == 'dof_pos' else noisy_motion[key]
        #         q2 = motion[key].reshape(num_frames, 52, 3) if key == 'dof_pos' else motion[key]
        #         # 计算旋转差异
        #         q1 = torch_utils.exp_map_to_quat(q1).unsqueeze(0)  # (num_joints, 4)
        #         q2 = torch_utils.exp_map_to_quat(q2)  # (num_frames, num_joints, 4)
        #         # 计算相对四元数差异
        #         q_diff = torch_utils.quat_multiply(torch_utils.quat_conjugate(q1),q2)  # (num_frames, num_joints, 4)
        #         # 计算角度差异
        #         diff, _ = torch_utils.quat_to_angle_axis(q_diff)  # (num_frames,) 
        #         diff = torch.sum(torch.abs(diff), dim=-1) if key == 'dof_pos' else torch.abs(diff)
        #         if diff.shape[0] == num_frames-1:
        #             diff = torch.cat((torch.zeros(1, device=diff.device), diff), dim=0)
        #     total_diff += diff * weight
        
        # # 返回最小差异的帧索引
        # total_diff = total_diff[2:-2] # (num_frames-4,)
        # return torch.argmin(total_diff, dim=0) + 2

    def noisy_resample_time(self, noisy_motion, motion_id):
        motion = self.hoi_data_dict[motion_id.item()]
        new_source_motion_time, max_sim = self.noisyinit_find_most_similarity_state(noisy_motion, motion)
        new_source_motion_time = new_source_motion_time.unsqueeze(0)
        return new_source_motion_time, max_sim
    
    def _reweight_time_sampling_rate(self, motion_time_seqreward):
        # motion_time_seqreward: {motion_id: [reward1, reward1, reward1, ...]}
        # self.time_sample_rate: {motion_id: [p1, p2, p3, ...]}
        for motion_id, reward in motion_time_seqreward.items():
            reward = reward.cpu().detach().numpy()
            lengths = self.motion_lengths[motion_id].cpu().numpy() 
            self.time_sample_rate[motion_id] = (1 - self.reweight_alpha) / (lengths - 3) + \
                self.reweight_alpha * (torch.exp(-10*torch.tensor(reward)) / torch.exp(-10*torch.tensor(reward)).sum())
        print('motion_time_seqreward:', motion_time_seqreward)
        print('Reweighted time sampling rate:', self.time_sample_rate)

    def _reweight_time_sampling_rate_vectorized(self, motion_time_seqreward_tensor: torch.Tensor):
        #ZQH 新增：用向量化方式更新 time-level 采样权重 motion_time_seqreward_tensor: shape=[num_motions, max_len_minus3], 在GPU
        alpha = self.reweight_alpha
        
        # 若没有可用列, 直接返回
        if motion_time_seqreward_tensor.size(1) == 0:
            return

        #----------------1) 计算 baseline + alpha * exp(-10 * R)----------------#
        baseline = (1.0 - alpha) / float(motion_time_seqreward_tensor.size(1))
        negative_exp = torch.exp(-10.0 * motion_time_seqreward_tensor)  # same shape
        sum_per_clip = negative_exp.sum(dim=1, keepdim=True) + 1e-8
        
        new_dist = baseline + alpha * (negative_exp / sum_per_clip)

        #----------------2) 重新对无效帧设为0，再归一化----------------#
        for m_id in range(self.num_motions):
            cur_len_minus3 = int(self.motion_lengths[m_id].item() - 3)
            if cur_len_minus3 <= 0:
                continue
            
            # 从 hoi_data_dict 里取出这一 clip 的 valid_frame_mask[2 : 2+cur_len_minus3]
            valid_mask_slice = self.hoi_data_dict[m_id]['valid_frame_mask'][2 : 2 + cur_len_minus3]
            
            # 先将该 clip 的无效帧概率设为 0
            # 注意: new_dist[m_id, :cur_len_minus3] 覆盖区间 [0, cur_len_minus3)
            new_dist[m_id, :cur_len_minus3][~valid_mask_slice] = 0.0  # [修改处1] 清零无效帧

            # # 对有效帧再次归一化, 防止 sum < 1.0
            # sum_valid = new_dist[m_id, :cur_len_minus3].sum() + 1e-8
            # new_dist[m_id, :cur_len_minus3] /= sum_valid            # [修改处2] 对有效帧归一化
        
        #----------------3) 写回 time_sample_rate_tensor----------------#
        self.time_sample_rate_tensor = new_dist

    def sample_time_old(self, motion_ids, truncate_time=None):
        lengths = self.motion_lengths[motion_ids].cpu().numpy() 

        start = np.full_like(lengths, 2) #2 #ZQH
        end = lengths - 2

        assert np.all(end > start) # Maybe some motions are too short to sample time properly.

        ###ZQH
        # motion_times = np.random.randint(start, end + 1)  # +1  Because the upper limit of np.random.randint is an open interval
        
        #####################################################
        if not self.reweight:
            motion_times = np.random.randint(start, end + 1)
        else:
            possible_times = [np.arange(s, e + 1) for s, e in zip(start, end)] # 计算每个 motion 的可能时间点
            motion_times = np.zeros_like(motion_ids)
            for i in range(len(motion_ids)):
                sample_rate = self.time_sample_rate[motion_ids[i].item()].numpy()
                motion_times[i] = np.random.choice(possible_times[i], p=sample_rate)
        #####################################################

        motion_times = torch.tensor(motion_times, device=self.device, dtype=torch.int)

        if truncate_time is not None:
            assert truncate_time >= 0
            motion_times = torch.min(motion_times, self.motion_lengths[motion_ids] - truncate_time)

        if self.play_dataset:
            motion_times = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        return motion_times

    def sample_time_new(self, motion_ids, truncate_time=None):
        """#ZQH
        改动后版本：使用 GPU 上的 self.time_sample_rate_tensor 做采样
        （假设 self.time_sample_rate_tensor[m_id, :L] 存储该 motion_id
        在 [2, L+2) 范围内的 time 概率分布）
        """
        # motion_times shape与 motion_ids 一致
        motion_times = torch.zeros_like(motion_ids, dtype=torch.int32, device=self.device)

        if not self.reweight: # 若没有 reweight，可直接走均匀分布(可以在外部把 time_sample_rate_tensor 初始化成均匀即可)
            for i in range(len(motion_ids)):
                m_id = motion_ids[i].item()
                L = self.motion_lengths[m_id] - 3
                dist = torch.ones(L, device=self.device)/(L+1e-6)
                idx = torch.multinomial(dist, 1)
                motion_times[i] = idx.item() + 2
        else:
            for i in range(len(motion_ids)):
                m_id = motion_ids[i].item()
                length_minus3 = int(self.motion_lengths[m_id].item() - 3)
                if length_minus3 <= 0:
                    # motion太短，强行置0或别的逻辑
                    motion_times[i] = 0
                    continue
                
                # dist shape=[length_minus3], GPU
                dist = self.time_sample_rate_tensor[m_id, :length_minus3]

                # 在 dist 上多项式采样1个time index, 结果 shape=[1]
                t_idx = torch.multinomial(dist, 1)
                # +2 抵消因为我们只存了 [2, length-1) 这段的概率
                motion_times[i] = t_idx.item() + 2

        # 截断逻辑
        if truncate_time is not None:
            assert truncate_time >= 0
            # clamp
            # (self.motion_lengths[motion_ids] - truncate_time) shape=[len(motion_ids)]
            max_allowed = self.motion_lengths[motion_ids] - truncate_time
            # 需要确保 motion_times和 max_allowed在同一device
            motion_times = torch.min(motion_times, max_allowed.to(self.device).long())

        if self.play_dataset:
            motion_times = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)

        return motion_times

    def sample_time(self, motion_ids, truncate_time=None):
        if self.use_old_reweight:
            return self.sample_time_old(motion_ids, truncate_time)
        else:
            return self.sample_time_new(motion_ids, truncate_time)

    def get_initial_state(self, env_ids, motion_ids, start_frames):
        """
        Get the initial state for given motion_ids and start_frames.
        
        Parameters:
        motion_ids (Tensor): A tensor containing the motion id for each environment.
        start_frames (Tensor): A tensor containing the starting frame number for each environment.
        
        Returns:
        Tuple: A tuple containing the initial state
        """
        assert len(motion_ids) == len(env_ids)
        valid_lengths = self.motion_lengths[motion_ids] - start_frames # if not self.play_dataset else self.motion_lengths[motion_ids]
        self.envid2episode_lengths[env_ids] = torch.where(valid_lengths < self.max_episode_length,
                                    valid_lengths, self.max_episode_length)

        # reward_weights_list = []
        hoi_data_list = []
        root_pos_list = []
        root_rot_list = []
        root_vel_list = []
        root_ang_vel_list = []
        dof_pos_list = []
        dof_vel_list = []
        obj_pos_list = []
        obj_pos_vel_list = []
        obj_rot_list = []
        obj_rot_vel_list = []

        for i, env_id in enumerate(env_ids):
            motion_id = motion_ids[i].item()
            start_frame = start_frames[i].item()

            self.envid2motid[env_id] = motion_id #V1
            self.envid2sframe[env_id] = start_frame 
            episode_length = self.envid2episode_lengths[env_id].item()

            if self.hoi_data_dict[motion_id]['hoi_data_text'] in ['000', '010']:
                state = self._get_special_case_initial_state(motion_id, start_frame, episode_length)
            else:
                state = self._get_general_case_initial_state(motion_id, start_frame, episode_length)

            # reward_weights_list.append(state['reward_weights'])
            for k in self.reward_weights_default:
                self.reward_weights[k][env_id] =  torch.tensor(state['reward_weights'][k], dtype=torch.float32, device=self.device)
            hoi_data_list.append(state["hoi_data"])
            root_pos_list.append(state['init_root_pos'])
            root_rot_list.append(state['init_root_rot'])
            root_vel_list.append(state['init_root_pos_vel'])
            root_ang_vel_list.append(state['init_root_rot_vel'])
            dof_pos_list.append(state['init_dof_pos'])
            dof_vel_list.append(state['init_dof_pos_vel'])
            obj_pos_list.append(state["init_obj_pos"])
            obj_pos_vel_list.append(state["init_obj_pos_vel"])
            obj_rot_list.append(state["init_obj_rot"])
            obj_rot_vel_list.append(state["init_obj_rot_vel"])

        hoi_data = torch.stack(hoi_data_list, dim=0)
        root_pos = torch.stack(root_pos_list, dim=0)
        root_rot = torch.stack(root_rot_list, dim=0)
        root_vel = torch.stack(root_vel_list, dim=0)
        root_ang_vel = torch.stack(root_ang_vel_list, dim=0)
        dof_pos = torch.stack(dof_pos_list, dim=0)
        dof_vel = torch.stack(dof_vel_list, dim=0)
        obj_pos = torch.stack(obj_pos_list, dim =0)
        obj_pos_vel = torch.stack(obj_pos_vel_list, dim =0)
        obj_rot = torch.stack(obj_rot_list, dim =0)
        obj_rot_vel = torch.stack(obj_rot_vel_list, dim =0)

        return hoi_data, \
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, \
                obj_pos, obj_pos_vel, obj_rot, obj_rot_vel
                

    def _get_special_case_initial_state(self, motion_id, start_frame, episode_length):
        hoi_data = F.pad(
            self.hoi_data_dict[motion_id]['hoi_data'][start_frame:start_frame + episode_length],
            (0, 0, 0, self.max_episode_length - episode_length)
        )

        return {
            "reward_weights": self._get_special_case_reward_weights(),
            "hoi_data": hoi_data,
            "init_root_pos": self.hoi_data_dict[motion_id]['root_pos'][start_frame, :],
            "init_root_rot": self.hoi_data_dict[motion_id]['root_rot'][start_frame, :],
            "init_root_pos_vel": self.hoi_data_dict[motion_id]['root_pos_vel'][start_frame, :],
            "init_root_rot_vel": self.hoi_data_dict[motion_id]['root_rot_vel'][start_frame, :],
            "init_dof_pos": self.hoi_data_dict[motion_id]['dof_pos'][start_frame, :],
            "init_dof_pos_vel": self.hoi_data_dict[motion_id]['dof_pos_vel'][start_frame, :],
            "init_obj_pos": (torch.rand(3, device=self.device) * 10 - 5),
            "init_obj_pos_vel": torch.rand(3, device=self.device) * 5,
            "init_obj_rot": torch.rand(4, device=self.device),
            "init_obj_rot_vel": torch.rand(3, device=self.device) * 0.1
        }

    def _get_general_case_initial_state(self, motion_id, start_frame, episode_length):
        hoi_data = F.pad(
            self.hoi_data_dict[motion_id]['hoi_data'][start_frame:start_frame + episode_length],
            (0, 0, 0, self.max_episode_length - episode_length)
        )

        return {
            "reward_weights": self._get_general_case_reward_weights(),
            "hoi_data": hoi_data,
            "init_root_pos": self.hoi_data_dict[motion_id]['root_pos'][start_frame, :],
            "init_root_rot": self.hoi_data_dict[motion_id]['root_rot'][start_frame, :],
            "init_root_pos_vel": self.hoi_data_dict[motion_id]['root_pos_vel'][start_frame, :],
            "init_root_rot_vel": self.hoi_data_dict[motion_id]['root_rot_vel'][start_frame, :],
            "init_dof_pos": self.hoi_data_dict[motion_id]['dof_pos'][start_frame, :],
            "init_dof_pos_vel": self.hoi_data_dict[motion_id]['dof_pos_vel'][start_frame, :],
            "init_obj_pos": self.hoi_data_dict[motion_id]['obj_pos'][start_frame, :],
            "init_obj_pos_vel": self.hoi_data_dict[motion_id]['obj_pos_vel'][start_frame, :],
            "init_obj_rot": self.hoi_data_dict[motion_id]['obj_rot'][start_frame, :],
            "init_obj_rot_vel": self.hoi_data_dict[motion_id]['obj_rot_vel'][start_frame, :]
        }

    def _get_special_case_reward_weights(self):
        reward_weights = self.reward_weights_default
        return {
            "p": reward_weights["p"],
            "r": reward_weights["r"],
            "op": reward_weights["op"] * 0.,
            "ig": reward_weights["ig"] * 0.,
            "cg1": reward_weights["cg1"] * 0.,
            "cg2": reward_weights["cg2"] * 0.,
            "pv": reward_weights["pv"],
            "rv": reward_weights["rv"],
            "or": reward_weights["or"],
            "opv": reward_weights["opv"],
            "orv": reward_weights["orv"],
        }

    def _get_general_case_reward_weights(self):
        reward_weights = self.reward_weights_default
        return {
            "p": reward_weights["p"],
            "r": reward_weights["r"],
            "op": reward_weights["op"],
            "ig": reward_weights["ig"],
            "cg1": reward_weights["cg1"],
            "cg2": reward_weights["cg2"],
            "pv": reward_weights["pv"],
            "rv": reward_weights["rv"],
            "or": reward_weights["or"],
            "opv": reward_weights["opv"],
            "orv": reward_weights["orv"],
        }


class MotionDataHandlerOfflineNew(MotionDataHandler):
    def __init__(self, motion_file, device, key_body_ids, cfg, num_envs, max_episode_length, reward_weights_default, 
                init_vel=False, play_dataset=False):
        super().__init__(motion_file=motion_file, device=device, key_body_ids=key_body_ids, cfg=cfg, num_envs=num_envs, 
                        max_episode_length=max_episode_length, reward_weights_default=reward_weights_default, 
                        init_vel=init_vel, play_dataset=play_dataset)
    
    def _get_switch_time(self, source_class, source_id, source_time, switch_class):
        source_motion = self.hoi_data_dict[source_id]
        switch_motion_ids = self._get_class_motion_ids(switch_class)
        switch_id, switch_time, max_sim = self._randskill_find_class_most_similarity_state(source_class, source_id, source_motion, source_time, switch_motion_ids)
        
        return switch_id, switch_time, max_sim
    
    def _randskill_find_class_most_similarity_state(self, source_class, source_id, source_motion, source_time, switch_motion_ids):
        if source_class not in [0, 10]:
            w = {'pos_diff': 20, 'pos_vel_diff': 0.1, 'rot_diff': 20, 'obj_pos_diff': 1, 'obj_pos_vel_diff': 0.1, 'rel_pos_diff': 20}
        else:
            # w = {'pos_diff': 15, 'pos_vel_diff': 0.1, 'rot_diff': 15, 'root_height_diff': 30, 'obj_pos_diff': 0, 'obj_pos_vel_diff': 0, 'rel_pos_diff': 0}
            w = {'pos_diff': 10, 'pos_vel_diff': 0.1, 'rot_diff': 10, 'root_height_diff': 5, 'obj_pos_diff': 0, 'obj_pos_vel_diff': 0, 'rel_pos_diff': 0}
        max_sim = []
        max_info = []
        for switch_motion_id in switch_motion_ids:
            switch_motion = self.hoi_data_dict[switch_motion_id]
            switch_motion_aligned = get_local_motion(source_motion, source_time, switch_motion, len(self._key_body_ids))
            # humanoid pose similarity
            pos_diff = torch.mean((source_motion['key_body_pos'][source_time] - switch_motion_aligned['key_body_pos'])**2,dim=-1) # (nframes)
            pos_vel_diff = torch.mean((source_motion['key_body_pos_vel'][source_time] - switch_motion_aligned['key_body_pos_vel'])**2,dim=-1)
            rot_diff = get_dof_pos_diff(source_motion['dof_pos'][source_time], switch_motion_aligned['dof_pos'])
            sim_pose = torch.exp(-w['pos_diff'] * pos_diff) * torch.exp(-w['pos_vel_diff'] * pos_vel_diff) * torch.exp(-w['rot_diff'] * rot_diff)
            if source_class in [0, 10]:
                root_height_diff = (source_motion['root_pos'][source_time][2] - switch_motion_aligned['root_pos'][:, 2])**2
                sim_pose *= torch.exp(-w['root_height_diff'] * root_height_diff)

            # object configuration similarity
            obj_pos_diff = torch.mean((source_motion['obj_pos'][source_time] - switch_motion_aligned['obj_pos'])**2,dim=-1)
            obj_pos_vel_diff = torch.mean((source_motion['obj_pos_vel'][source_time] - switch_motion_aligned['obj_pos_vel'])**2,dim=-1)
            sim_obj = torch.exp(-w['obj_pos_diff'] * obj_pos_diff) * torch.exp(-w['obj_pos_vel_diff'] * obj_pos_vel_diff)

            # relative spatial relationship similarity
            switch_ig = switch_motion_aligned['key_body_pos'].reshape(-1, len(self._key_body_ids), 3) - switch_motion_aligned['obj_pos'].unsqueeze(1)
            source_ig = source_motion['key_body_pos'][source_time].reshape(len(self._key_body_ids), 3) - source_motion['obj_pos'][source_time].unsqueeze(0)
            rel_pos_diff = torch.mean((source_ig.reshape(-1, len(self._key_body_ids)*3) - switch_ig.reshape(-1, len(self._key_body_ids)*3))**2,dim=-1)
            sim_rel = torch.exp(-w['rel_pos_diff'] * rel_pos_diff)

            # total similarity
            sim = sim_pose * sim_obj * sim_rel
            # if source_class == 0 and source_time in [129]:
                # print(f'Frame {source_time}, sim_pose: {sim_pose}')
                # print(f'sim_obj: {sim_obj}')
                # print(f'sim_rel: {sim_rel}')
                # print(f'sim: {sim[0]}')

            # 找到第一个大于2, 且不为-1的索引
            sorted_indices = torch.argsort(sim, descending=True)
            max_ind = next((ind.item() for ind in sorted_indices if ind.item() not in [0, 1, len(sim)-1]),
                            sorted_indices[0].item())
            max_sim.append(sim[max_ind])
            max_info.append((switch_motion_id, max_ind))
        
        max_max_sim = max(max_sim)
        if max_max_sim < 1e-10:
            return None, None, 0
        max_max_ind = max_sim.index(max_max_sim)
        max_switch_id, max_switch_time = max_info[max_max_ind]
        return max_switch_id, max_switch_time, max_max_sim

    def _get_class_motion_ids(self, source_class):
        return [i for i in range(self.num_motions) if self.motion_class[i] == source_class]

    
class MotionDataHandler4AMP(MotionDataHandler):
    def __init__(self, motion_file, device, key_body_ids, cfg, num_envs, max_episode_length, reward_weights_default, 
                init_vel=False, play_dataset=False, use_old_reweight=True):
        super().__init__(motion_file=motion_file, device=device, key_body_ids=key_body_ids, cfg=cfg, num_envs=num_envs, 
                        max_episode_length=max_episode_length, reward_weights_default=reward_weights_default, 
                        init_vel=init_vel, play_dataset=play_dataset, use_old_reweight=use_old_reweight)

    def _process_sequence(self, seq_path):
        loaded_dict = super()._process_sequence(seq_path)

        loaded_dict['object_data'] = torch.cat((loaded_dict['obj_pos'].clone(),
                                                loaded_dict['obj_rot'].clone(),
                                                loaded_dict['obj_pos_vel'].clone(),),dim=-1)
        
        return loaded_dict
    
    def load_motion(self, motion_file):
        super().load_motion(motion_file)
        self.hoi_data_for_amp_obs = {}
        keys = ['root_pos', 'root_rot', 'dof_pos', 'root_pos_vel', 'root_rot_vel', 'dof_pos_vel', 'key_body_pos', \
                'object_data' ]
        for key in keys:
            # 创建一个列表来存储每个 motion_id 对应的张量
            tensor_list = []
            max_timesteps = 0
            
            # 找到最大的 timesteps
            for motion_id in self.hoi_data_dict:
                max_timesteps = max(max_timesteps, self.hoi_data_dict[motion_id][key].shape[0])
            
            # 填充张量以确保形状一致
            for motion_id in self.hoi_data_dict:
                tensor = self.hoi_data_dict[motion_id][key]
                timesteps, _ = tensor.shape
                if timesteps < max_timesteps:
                    # 填充张量，使其形状一致
                    padding = (0, 0, 0, max_timesteps - timesteps)
                    tensor = torch.nn.functional.pad(tensor, padding)
                tensor_list.append(tensor)
            
            # 将列表转换为三维张量（num_motion_ids, max_timesteps, dim）
            self.hoi_data_for_amp_obs[key] = torch.stack(tensor_list)
            
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def get_local_motion(source_motion, source_time, switch_motion, len_keypos):
    # z rotation from switch to source
    sc_root_rot = torch_utils.exp_map_to_quat(source_motion['root_rot_3d'][source_time]).unsqueeze(0)
    sc_root_rot_euler_z = torch_utils.quat_to_euler(sc_root_rot)[:, 2] # (1,)
    sw_root_rot = torch_utils.exp_map_to_quat(switch_motion['root_rot_3d']) # (num_frames, 4)
    sw_root_rot_euler_z = torch_utils.quat_to_euler(sw_root_rot)[:, 2] # (num_frames,)
    sw2sc_root_rot_euler_z = sc_root_rot_euler_z - sw_root_rot_euler_z # (num_frames,)
    sw2sc_root_rot_euler_z = (sw2sc_root_rot_euler_z + torch.pi) % (2 * torch.pi) - torch.pi  # 归一化到 [-pi, pi]
    zeros = torch.zeros_like(sw2sc_root_rot_euler_z)
    switch_to_source = quat_from_euler_xyz(zeros, zeros, sw2sc_root_rot_euler_z) # (num_frames, 4)

    # switch motion aligned with source motion
    switch_motion_aligned = copy.deepcopy(switch_motion)

    sw2sc_root_trans = source_motion['root_pos'][source_time] - switch_motion['root_pos'] # (num_frames, 3)
    sw2sc_root_trans[..., 2] = 0
    switch_motion_aligned['key_body_pos'] = switch_motion_aligned['key_body_pos'].reshape(-1, len_keypos, 3) # (num_frames, 17, 3)
    switch_body_pos_delta = switch_motion_aligned['key_body_pos'] - switch_motion['root_pos'].unsqueeze(1)
    for i in range(len_keypos):
        switch_body_pos_delta[:,i] = torch_utils.quat_rotate(switch_to_source, switch_body_pos_delta[:,i])
    switch_motion_aligned['key_body_pos'] = source_motion['root_pos'][source_time] + switch_body_pos_delta
    switch_motion_aligned['key_body_pos'][..., 2] = switch_motion['key_body_pos'].reshape(-1, len_keypos, 3)[..., 2]
    switch_motion_aligned['key_body_pos'] = switch_motion_aligned['key_body_pos'].reshape(-1, len_keypos*3) # (num_frames, 51)
    switch_motion_aligned['key_body_pos_vel'] = switch_motion_aligned['key_body_pos_vel'].reshape(-1, len_keypos, 3)
    for i in range(len_keypos):
        switch_motion_aligned['key_body_pos_vel'][:,i] = torch_utils.quat_rotate(switch_to_source, switch_motion_aligned['key_body_pos_vel'][:,i])
    switch_motion_aligned['key_body_pos_vel'] = switch_motion_aligned['key_body_pos_vel'].reshape(-1, len_keypos*3)
    switch_motion_aligned['root_pos'] += sw2sc_root_trans
    # print(switch_motion_aligned['root_pos'])
    # switch_motion_aligned['dof_pos'] no change
    switch_obj_pos_delta = switch_motion_aligned['obj_pos'] - switch_motion['root_pos']
    switch_obj_pos_delta = torch_utils.quat_rotate(switch_to_source, switch_obj_pos_delta)
    switch_motion_aligned['obj_pos'] = source_motion['root_pos'][source_time] + switch_obj_pos_delta
    switch_motion_aligned['obj_pos'][..., 2] = switch_motion['obj_pos'][..., 2]
    switch_motion_aligned['obj_pos_vel'] = torch_utils.quat_rotate(switch_to_source, switch_motion_aligned['obj_pos_vel'])

    return switch_motion_aligned

@torch.jit.script
def get_dof_pos_diff(source_dof_pos, switch_dof_pos):
    num_frames = switch_dof_pos.shape[0]
    source_dof_pos = source_dof_pos.reshape(-1,3)
    switch_dof_pos = switch_dof_pos.reshape(num_frames, -1, 3)
    source_dof_pos_quat = torch_utils.exp_map_to_quat(source_dof_pos).unsqueeze(0)
    switch_dof_pos_quat = torch_utils.exp_map_to_quat(switch_dof_pos)
    q_diff = torch_utils.quat_multiply(torch_utils.quat_conjugate(source_dof_pos_quat), switch_dof_pos_quat)
    rot_diff, _ = torch_utils.quat_to_angle_axis(q_diff)  # (num_frames,)
    rot_diff = torch.mean(rot_diff**2, dim=-1)
    return rot_diff