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

class ParaMotionDataHandler:
    def __init__(self, motion_file, dof_obs_size, device, key_body_ids, cfg, num_envs, max_episode_length, reward_weights_default, 
                init_vel=False, play_dataset=False, reweight=False, reweight_alpha=0):
        self.device = device
        self._key_body_ids = key_body_ids
        self.cfg = cfg
        self.init_vel = init_vel
        self.play_dataset = play_dataset #V1
        self.max_episode_length = max_episode_length
        
        self.hoi_data_dict = {}
        self.hoi_data_label_batch = None
        self.motion_lengths = None
        self.load_motion(motion_file, dof_obs_size)

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
        

    def load_motion(self, motion_file, dof_obs_size):
        self.skill_name = os.path.basename(motion_file)
        # all_seqs = [motion_file] if os.path.isfile(motion_file) \
        #     else glob.glob(os.path.join(motion_file, '*.pt'), recursive=True)
        all_seqs = [motion_file] if os.path.isfile(motion_file) else [ \
            os.path.join(root, f) 
            for root, dirs, filenames in os.walk(motion_file) 
            for f in filenames 
            if f.endswith('.pt')
        ]
        self.num_motions = len(all_seqs)
        self.motion_lengths = torch.zeros(len(all_seqs), device=self.device, dtype=torch.long)
        self.motion_class = np.zeros(len(all_seqs), dtype=int)
        self.layup_target = torch.zeros((len(all_seqs), 3), device=self.device, dtype=torch.float)
        self.root_target = torch.zeros((len(all_seqs), 3), device=self.device, dtype=torch.float)
        
        all_seqs.sort(key=self._sort_key)
        for i, seq_path in enumerate(all_seqs):
            loaded_dict = self._process_sequence(seq_path, dof_obs_size)
            self.hoi_data_dict[i] = loaded_dict
            self.motion_lengths[i] = loaded_dict['hoi_data'].shape[0]
            self.motion_class[i] = int(loaded_dict['hoi_data_text'])
            if self.skill_name in ['layup', "SHOT_up"]:
                layup_target_ind = torch.argmax(loaded_dict['obj_pos'][:, 2])
                self.layup_target[i] = loaded_dict['obj_pos'][layup_target_ind]
                self.root_target[i] = loaded_dict['root_pos'][layup_target_ind]
        self._compute_motion_weights(self.motion_class)
        if self.play_dataset:
            self.max_episode_length = self.motion_lengths.min() - 2
        print(f"--------Having loaded {len(all_seqs)} motions--------")
    
    def _sort_key(self, filename):
        match = re.search(r'\d+.pt$', filename)
        return int(match.group().replace('.pt', '')) if match else -1

    def _process_sequence(self, seq_path, dof_obs_size):
        loaded_dict = {}
        hoi_data = torch.load(seq_path)
        loaded_dict['hoi_data_text'] = os.path.basename(seq_path)[0:3]
        loaded_dict['hoi_data'] = hoi_data.detach().to(self.device) #  
        data_frames_scale = self.cfg["env"]["dataFramesScale"]
        fps_data = self.cfg["env"]["dataFPS"] * data_frames_scale

        loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
        loaded_dict['root_pos_vel'] = self._compute_velocity(loaded_dict['root_pos'], fps_data)

        loaded_dict['root_rot_3d'] = loaded_dict['hoi_data'][:, 3:6].clone()
        loaded_dict['root_rot'] = torch_utils.exp_map_to_quat(loaded_dict['root_rot_3d']).clone()
        self.smooth_quat_seq(loaded_dict['root_rot'])

        q_diff = torch_utils.quat_multiply(
            torch_utils.quat_conjugate(loaded_dict['root_rot'][:-1, :].clone()), 
            loaded_dict['root_rot'][1:, :].clone()
        )
        angle, axis = torch_utils.quat_to_angle_axis(q_diff)
        exp_map = torch_utils.angle_axis_to_exp_map(angle, axis)
        loaded_dict['root_rot_vel'] = self._compute_velocity(exp_map, fps_data)

        ########### Modified by Runyi ###########
        start_ind = 9
        #########################################
        loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, start_ind:start_ind+dof_obs_size].clone() # dof rotation
        loaded_dict['dof_pos_vel'] = self._compute_velocity(loaded_dict['dof_pos'], fps_data)
        
        data_length = loaded_dict['hoi_data'].shape[0]
        start_ind += dof_obs_size
        loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, start_ind:start_ind+71*3].clone().view(data_length, 71, 3) # joint position
        loaded_dict['key_body_pos'] = loaded_dict['body_pos'][:, self._key_body_ids, :].view(data_length, -1).clone()
        loaded_dict['key_body_pos_vel'] = self._compute_velocity(loaded_dict['key_body_pos'], fps_data)
        
        start_ind += 71*3
        loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, start_ind:start_ind+3].clone()
        # loaded_dict['obj_pos'] += torch.tensor([-0.1, 0.1, 0], device=self.device)
        loaded_dict['obj_pos_vel'] = self._compute_velocity(loaded_dict['obj_pos'], fps_data)

        start_ind += 3
        loaded_dict['obj_rot'] = loaded_dict['hoi_data'][:, start_ind:start_ind+3].clone()
        loaded_dict['obj_rot_vel'] = self._compute_velocity(loaded_dict['obj_rot'], fps_data)
        if self.init_vel:
            loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1],loaded_dict['obj_pos_vel']),dim=0)
        loaded_dict['obj_rot'] = torch_utils.exp_map_to_quat(loaded_dict['obj_rot']).clone()

        start_ind += 3
        loaded_dict['contact'] = torch.round(loaded_dict['hoi_data'][:, start_ind:start_ind+1].clone())

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

    def _compute_velocity(self, positions, fps):
        velocity = (positions[1:, :].clone() - positions[:-1, :].clone()) * fps
        velocity = torch.cat((torch.zeros((1, positions.shape[-1])).to(self.device), velocity), dim=0)
        return velocity

    def smooth_quat_seq(self, quat_seq):
        n = quat_seq.size(0)

        for i in range(1, n):
            dot_product = torch.dot(quat_seq[i-1], quat_seq[i])
            if dot_product < 0:
                quat_seq[i] *=-1

        return quat_seq

    def _compute_motion_weights(self, motion_class):
        unique_classes, counts = np.unique(motion_class, return_counts=True)
        class_to_index = {k: v for v, k in enumerate(unique_classes)}
        class_weights = 1 / counts
        indexed_classes = np.array([class_to_index[int(cls)] for cls in motion_class], dtype=int)
        self._motion_weights = class_weights[indexed_classes]

    def _reweight_clip_sampling_rate(self, average_rewards):
        counts = Counter(self.motion_class)
        rewards_tensor = torch.tensor(list(average_rewards.values()), dtype=torch.float32)
        for idx, motion_class in enumerate(self.motion_class):
            rewards_tensor[idx] /= counts[motion_class]
        self._motion_weights = (1 - self.reweight_alpha) / len(counts) + \
            self.reweight_alpha * (torch.exp(-5*rewards_tensor) / torch.exp(-5*rewards_tensor).sum())

    def sample_motions(self, n):
        motion_ids = torch.multinomial(torch.tensor(self._motion_weights), num_samples=n, replacement=True)
        return motion_ids


    ################# Modified by Runyi #################
    def noisyinit_find_most_similarity_state(self, noisy_motion, motion):
        w = {'pos_diff': 5, 'pos_vel_diff': 0.1, 'root_rot_diff': 2, 'rot_diff': 2, 
             'obj_pos_diff': 1, 'obj_pos_vel_diff': 0.1,
             'rel_pos_diff': 2}
                
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

        # Find the first index that is greater than 2 and not -1
        sorted_indices = torch.argsort(sim, descending=True)
        max_ind = next((ind.item() for ind in sorted_indices if ind.item() not in [0, 1, len(sim)-1]),
                            sorted_indices[0].item())
        return torch.tensor(max_ind, device=sim.device), max(sim)
    
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
                self.reweight_alpha * (torch.exp(-5*torch.tensor(reward)) / torch.exp(-5*torch.tensor(reward)).sum())
        print('motion_time_seqreward:', motion_time_seqreward)
        print('Reweighted time sampling rate:', self.time_sample_rate)


    def sample_time(self, motion_ids, truncate_time=None):
        lengths = self.motion_lengths[motion_ids].cpu().numpy()

        start = 2
        end = lengths - 2

        assert np.all(end > start) # Maybe some motions are too short to sample time properly.

        # motion_times = np.random.randint(start, end + 1)  # +1  Because the upper limit of np.random.randint is an open interval
        #####################################################
        if not self.reweight:
            motion_times = np.random.randint(start, end + 1)
        else:
            possible_times = [np.arange(s, e + 1) for s, e in zip(start, end)] # Calculate possible time points for each motion
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
            motion_times = torch.ones((1), device=self.device, dtype=torch.int32)
        return motion_times


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
        valid_lengths = self.motion_lengths[motion_ids] - start_frames if not self.play_dataset else self.motion_lengths[motion_ids]
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

            if self.hoi_data_dict[motion_id]['hoi_data_text'] == '000':
                state = self._get_special_case_initial_state(motion_id, start_frame, episode_length)
            else:
                state = self._get_general_case_initial_state(motion_id, start_frame, episode_length)

            # reward_weights_list.apget_initial_statepend(state['reward_weights'])
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
            "init_obj_rot_vel": torch.rand(4, device=self.device) * 0.1
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

class ParaMotionDataHandlerOfflineNew(ParaMotionDataHandler):
    def __init__(self, motion_file, dof_obs_size, device, key_body_ids, cfg, num_envs, max_episode_length, reward_weights_default, 
                init_vel=False, play_dataset=False):
        super().__init__(motion_file=motion_file, dof_obs_size=dof_obs_size, device=device, key_body_ids=key_body_ids, 
                        cfg=cfg, num_envs=num_envs, max_episode_length=max_episode_length, reward_weights_default=reward_weights_default, 
                        init_vel=init_vel, play_dataset=play_dataset)
    
    def _get_switch_time(self, source_class, source_id, source_time, switch_class):
        source_motion = self.hoi_data_dict[source_id]
        switch_motion_ids = self._get_class_motion_ids(switch_class)
        switch_id, switch_time, max_sim = self._randskill_find_class_most_similarity_state(source_class, source_id, source_motion, source_time, switch_motion_ids)
        
        return switch_id, switch_time, max_sim
    
    def _randskill_find_class_most_similarity_state(self, source_class, source_id, source_motion, source_time, switch_motion_ids):
        w = {'pos_diff': 30, 'pos_vel_diff': 0.1, 'rot_diff': 30, 'obj_pos_diff': 1, 'obj_pos_vel_diff': 0.1, 'rel_pos_diff': 30}
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
            #     print(f'Frame {source_time}, sim_pose: {sim_pose}')
            #     print(f'sim_obj: {sim_obj}')
            #     print(f'sim_rel: {sim_rel}')
            #     print(f'sim: {sim[0]}')

            # Find the first index that is greater than 2 and not -1
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
    sw2sc_root_rot_euler_z = (sw2sc_root_rot_euler_z + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize to [-pi, pi]
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