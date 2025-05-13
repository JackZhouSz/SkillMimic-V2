import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
import re
from utils import torch_utils
from utils.paramotion_data_handler import ParaMotionDataHandler

class ParaMotionDataHandlerMultiobj(ParaMotionDataHandler):
    def __init__(self, motion_file, dof_obs_size, device, key_body_ids, cfg, num_envs, max_episode_length, reward_weights_default, 
                init_vel=False, play_dataset=False):
        super().__init__(motion_file, dof_obs_size, device, key_body_ids, cfg, num_envs, max_episode_length, reward_weights_default, 
                init_vel=False, play_dataset=False)
        
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
        
        # object 1 (kettle)
        start_ind += 71*3
        loaded_dict['obj0_pos'] = loaded_dict['hoi_data'][:, start_ind:start_ind+3].clone()
        loaded_dict['obj0_pos_vel'] = self._compute_velocity(loaded_dict['obj0_pos'], fps_data)
        start_ind += 3
        loaded_dict['obj0_rot'] = loaded_dict['hoi_data'][:, start_ind:start_ind+3].clone()
        loaded_dict['obj0_rot_vel'] = self._compute_velocity(loaded_dict['obj0_rot'], fps_data)
        if self.init_vel:
            loaded_dict['obj0_pos_vel'] = torch.cat((loaded_dict['obj0_pos_vel'][:1],loaded_dict['obj0_pos_vel']),dim=0)
        loaded_dict['obj0_rot'] = torch_utils.exp_map_to_quat(loaded_dict['obj0_rot']).clone()

        # object 2 (cup)
        start_ind += 3
        loaded_dict['obj1_pos'] = loaded_dict['hoi_data'][:, start_ind:start_ind+3].clone()
        loaded_dict['obj1_pos_vel'] = self._compute_velocity(loaded_dict['obj1_pos'], fps_data)
        start_ind += 3
        loaded_dict['obj1_rot'] = loaded_dict['hoi_data'][:, start_ind:start_ind+3].clone()
        loaded_dict['obj1_rot_vel'] = self._compute_velocity(loaded_dict['obj1_rot'], fps_data)
        if self.init_vel:
            loaded_dict['obj1_pos_vel'] = torch.cat((loaded_dict['obj1_pos_vel'][:1],loaded_dict['obj1_pos_vel']),dim=0)
        loaded_dict['obj1_rot'] = torch_utils.exp_map_to_quat(loaded_dict['obj1_rot']).clone()

        # contact
        start_ind += 3
        loaded_dict['contact'] = torch.round(loaded_dict['hoi_data'][:, start_ind:start_ind+1].clone())

        loaded_dict['hoi_data'] = torch.cat((
            loaded_dict['root_pos'],
            loaded_dict['root_rot_3d'],
            loaded_dict['dof_pos'],
            loaded_dict['dof_pos_vel'],
            loaded_dict['obj0_pos'],
            loaded_dict['obj0_rot'],
            loaded_dict['obj0_pos_vel'],
            loaded_dict['obj1_pos'],
            loaded_dict['obj1_rot'],
            loaded_dict['obj1_pos_vel'],
            loaded_dict['key_body_pos'],
            loaded_dict['contact']
        ), dim=-1)
        
        return loaded_dict

    ################# Modified by Runyi #################
    def noisyinit_find_most_similarity_state(self, noisy_motion, motion,):
        w = {'pos_diff': 2, 'pos_vel_diff': 0.1, 'root_rot_diff': 2, 'rot_diff': 2, 
             'obj0_pos_diff': 1, 'obj0_pos_vel_diff': 0.1, 'obj0_rot_diff': 1,
             'obj1_pos_diff': 1, 'obj1_pos_vel_diff': 0.1, 'obj1_rot_diff': 1,
             'rel0_pos_diff': 1, 'rel1_pos_diff': 1}
                
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
        obj0_pos_diff = torch.mean((noisy_motion['obj0_pos'] - motion['obj0_pos'])**2,dim=-1)
        obj0_pos_vel_diff = torch.mean((noisy_motion['obj0_pos_vel'] - motion['obj0_pos_vel'])**2,dim=-1)
        obj0_rot_diff = get_dof_pos_diff(noisy_motion['obj0_rot'], motion['obj0_rot'])
        sim_obj0 = torch.exp(-w['obj0_pos_diff'] * obj0_pos_diff) * torch.exp(-w['obj0_pos_vel_diff'] * obj0_pos_vel_diff) * \
                   torch.exp(-w['obj0_rot_diff'] * obj0_rot_diff)

        # object configuration similarity
        obj1_pos_diff = torch.mean((noisy_motion['obj1_pos'] - motion['obj1_pos'])**2,dim=-1)
        obj1_pos_vel_diff = torch.mean((noisy_motion['obj1_pos_vel'] - motion['obj1_pos_vel'])**2,dim=-1)
        obj1_rot_diff = get_dof_pos_diff(noisy_motion['obj1_rot'], motion['obj1_rot'])
        sim_obj1 = torch.exp(-w['obj1_pos_diff'] * obj1_pos_diff) * torch.exp(-w['obj1_pos_vel_diff'] * obj1_pos_vel_diff) * \
                   torch.exp(-w['obj1_rot_diff'] * obj1_rot_diff)

        # relative spatial relationship similarity
        noisy_ig0 = noisy_motion['key_body_pos'].reshape(len(self._key_body_ids), 3) - noisy_motion['obj0_pos'].unsqueeze(0)
        motion_ig0 = motion['key_body_pos'].reshape(-1, len(self._key_body_ids), 3) - motion['obj0_pos'].unsqueeze(1)
        rel0_pos_diff = torch.mean((noisy_ig0.reshape(-1, len(self._key_body_ids)*3) - motion_ig0.reshape(-1, len(self._key_body_ids)*3))**2,dim=-1)
        sim_rel0 = torch.exp(-w['rel0_pos_diff'] * rel0_pos_diff)
        
        noisy_ig1 = noisy_motion['key_body_pos'].reshape(len(self._key_body_ids), 3) - noisy_motion['obj1_pos'].unsqueeze(0)
        motion_ig1 = motion['key_body_pos'].reshape(-1, len(self._key_body_ids), 3) - motion['obj1_pos'].unsqueeze(1)
        rel1_pos_diff = torch.mean((noisy_ig1.reshape(-1, len(self._key_body_ids)*3) - motion_ig1.reshape(-1, len(self._key_body_ids)*3))**2,dim=-1)
        sim_rel1 = torch.exp(-w['rel1_pos_diff'] * rel1_pos_diff)

        # No need to calculate sim_obj & sim_rel for getup and run
        sim = sim_pose * sim_obj0 * sim_obj1 * sim_rel0 * sim_rel1

        # Find the first index that is greater than 2 and not -1
        sorted_indices = torch.argsort(sim, descending=True)
        max_ind = next((ind.item() for ind in sorted_indices if ind.item() not in [0, 1, len(sim)-1]),
                            sorted_indices[0].item())
        return torch.tensor(max_ind, device=sim.device), max(sim)
        


    
    def noisy_resample_time(self, noisy_motion, motion_id, weights=None):
        motion = self.hoi_data_dict[motion_id.item()]
        new_source_motion_time, max_sim = self.noisyinit_find_most_similarity_state(noisy_motion, motion)
        new_source_motion_time = new_source_motion_time.unsqueeze(0)
        return new_source_motion_time, max_sim
    #####################################################

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
        obj0_pos_list = []
        obj0_pos_vel_list = []
        obj0_rot_list = []
        obj0_rot_vel_list = []
        obj1_pos_list = []
        obj1_pos_vel_list = []
        obj1_rot_list = []
        obj1_rot_vel_list = []

        for i, env_id in enumerate(env_ids):
            motion_id = motion_ids[i].item()
            start_frame = start_frames[i].item()

            self.envid2motid[env_id] = motion_id #V1
            self.envid2sframe[env_id] = start_frame
            episode_length = self.envid2episode_lengths[env_id].item()

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
            obj0_pos_list.append(state["init_obj0_pos"])
            obj0_pos_vel_list.append(state["init_obj0_pos_vel"])
            obj0_rot_list.append(state["init_obj0_rot"])
            obj0_rot_vel_list.append(state["init_obj0_rot_vel"])
            obj1_pos_list.append(state["init_obj1_pos"])
            obj1_pos_vel_list.append(state["init_obj1_pos_vel"])
            obj1_rot_list.append(state["init_obj1_rot"])
            obj1_rot_vel_list.append(state["init_obj1_rot_vel"])

        hoi_data = torch.stack(hoi_data_list, dim=0)
        root_pos = torch.stack(root_pos_list, dim=0)
        root_rot = torch.stack(root_rot_list, dim=0)
        root_vel = torch.stack(root_vel_list, dim=0)
        root_ang_vel = torch.stack(root_ang_vel_list, dim=0)
        dof_pos = torch.stack(dof_pos_list, dim=0)
        dof_vel = torch.stack(dof_vel_list, dim=0)
        obj0_pos = torch.stack(obj0_pos_list, dim =0)
        obj0_pos_vel = torch.stack(obj0_pos_vel_list, dim =0)
        obj0_rot = torch.stack(obj0_rot_list, dim =0)
        obj0_rot_vel = torch.stack(obj0_rot_vel_list, dim =0)
        obj1_pos = torch.stack(obj1_pos_list, dim =0)
        obj1_pos_vel = torch.stack(obj1_pos_vel_list, dim =0)
        obj1_rot = torch.stack(obj1_rot_list, dim =0)
        obj1_rot_vel = torch.stack(obj1_rot_vel_list, dim =0)

        return hoi_data, \
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, \
                obj0_pos, obj0_pos_vel, obj0_rot, obj0_rot_vel, \
                obj1_pos, obj1_pos_vel, obj1_rot, obj1_rot_vel
                

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
            "init_obj0_pos": self.hoi_data_dict[motion_id]['obj0_pos'][start_frame, :],
            "init_obj0_pos_vel": self.hoi_data_dict[motion_id]['obj0_pos_vel'][start_frame, :],
            "init_obj0_rot": self.hoi_data_dict[motion_id]['obj0_rot'][start_frame, :],
            "init_obj0_rot_vel": self.hoi_data_dict[motion_id]['obj0_rot_vel'][start_frame, :],
            "init_obj1_pos": self.hoi_data_dict[motion_id]['obj1_pos'][start_frame, :],
            "init_obj1_pos_vel": self.hoi_data_dict[motion_id]['obj1_pos_vel'][start_frame, :],
            "init_obj1_rot": self.hoi_data_dict[motion_id]['obj1_rot'][start_frame, :],
            "init_obj1_rot_vel": self.hoi_data_dict[motion_id]['obj1_rot_vel'][start_frame, :]
        }
    
    

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def get_dof_pos_diff(source_dof_pos, switch_dof_pos):
    num_frames, dim = switch_dof_pos.shape[0], switch_dof_pos.shape[1]
    if dim != 4:
        source_dof_pos = source_dof_pos.reshape(-1,3)
        switch_dof_pos = switch_dof_pos.reshape(num_frames, -1, 3)
        source_dof_pos_quat = torch_utils.exp_map_to_quat(source_dof_pos).unsqueeze(0)
        switch_dof_pos_quat = torch_utils.exp_map_to_quat(switch_dof_pos)
    else:
        source_dof_pos_quat = source_dof_pos.unsqueeze(0)
        switch_dof_pos_quat = switch_dof_pos
    q_diff = torch_utils.quat_multiply(torch_utils.quat_conjugate(source_dof_pos_quat), switch_dof_pos_quat)
    rot_diff, _ = torch_utils.quat_to_angle_axis(q_diff)  # (num_frames,)
    rot_diff = torch.mean(rot_diff**2, dim=-1)
    return rot_diff