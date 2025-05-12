import torch
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
from collections import defaultdict
import copy

from env.tasks.skillmimic1 import SkillMimic1BallPlay

class SkillMimic1BallPlayReweight(SkillMimic1BallPlay): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        ############## Reweighting Mechanism Initialization ##############
        self.progress_buf_total = 0  # Tracks total progress across all environments
        
        # Initialize tensors to track motion IDs and times for all environments
        self.motion_ids_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.motion_times_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        # Calculate total frames across all motions for setting reweighting intervals
        total_frames = sum([self._motion_data.motion_lengths[motion_id] for motion_id in self._motion_data.hoi_data_dict])
        self.reweight_interval = 10 * total_frames  # Determines how often reweighting occurs
        
        # Initialize reward tracking tensors
        self.envs_reward = torch.zeros(self.num_envs, self.max_episode_length, device=self.device)
        self.average_rewards = torch.zeros(self._motion_data.num_motions, device=self.device, dtype=torch.float32)
        
        # Initialize motion time sequence rewards using a tensor for efficient indexing
        self.motion_time_seqreward = torch.zeros(
            (self._motion_data.num_motions, self._motion_data.motion_lengths.max() - 3),
            device=self.device,
            dtype=torch.float32
        )
        #######################################################################

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._motion_data.envid2episode_lengths, self.isTest, self.cfg["env"]["episodeLength"],
                                                   )
        
        # Identify environments that need reweighting
        reset_env_ids = torch.nonzero(self.reset_buf == 1, as_tuple=False).squeeze(-1)
        
        # Perform reweighting on the identified environments
        self._reweight_motion(reset_env_ids)
        return

    def after_reset_actors(self, env_ids):
        super().after_reset_actors(env_ids)
        
        # Batch update motion_ids_total and motion_times_total for reset environments
        self.motion_ids_total[env_ids] = self.motion_ids.to(self.device)
        self.motion_times_total[env_ids] = self.motion_times.to(self.device)
        return

    ################### Optimized Reweighting Methods ###################
    def _reweight_motion(self, reset_env_ids):
        """
        Reweights motion sampling probabilities based on accumulated rewards.
        This method is optimized to utilize vectorized tensor operations for efficiency.
        """
        if not self.cfg['env']['reweight']:
            return  # Reweighting is disabled
        
        # # Debug
        # old_time = self._motion_data.time_sample_rate_tensor.clone()
        # old_clip = self._motion_data._motion_weights_tensor.clone()

        # Record rewards for reset environments
        self.record_motion_time_reward(reset_env_ids)
        
        # Perform reweighting at specified intervals
        if (self.progress_buf_total % self.reweight_interval == 0) and (self.progress_buf_total > 0): #256*80
            # print("old_clip",old_clip) #Debug
            # Reweight motion clips based on average rewards
            if self._motion_data.num_motions > 1:
                # Compute average rewards per motion using vectorized operations
                # Create a mask for each motion_id
                motion_ids = self.motion_ids_total  # Shape: (num_envs,)
                # Ensure motion_ids are within valid range
                valid_mask = (motion_ids >= 0) & (motion_ids < self._motion_data.num_motions)
                valid_motion_ids = motion_ids[valid_mask]
                
                # Compute sum of rewards per motion
                reward_sum = torch.bincount(valid_motion_ids, weights=self.rew_buf[valid_mask].float(), minlength=self._motion_data.num_motions)
                
                # Compute counts per motion
                motion_counts = torch.bincount(valid_motion_ids, minlength=self._motion_data.num_motions).float()
                
                # Avoid division by zero
                motion_counts = torch.where(motion_counts > 0, motion_counts, torch.ones_like(motion_counts))
                
                # Compute average rewards
                avg_rewards = reward_sum / motion_counts  # Shape: (num_motions,)
                
                # Update average_rewards tensor
                self.average_rewards = avg_rewards  # Already a tensor
                
                # Debugging information
                print('##### Reweighting Motion Sampling Rates #####')
                print('Class Average Reward:', self.average_rewards.cpu().numpy())
                
                # Update motion sampling weights based on average_rewards
                # 调用修改后的 clip 级别 reweight 函数（内含「类先分配 -> clip 再分配」逻辑）
                self._motion_data._reweight_clip_sampling_rate_vectorized(self.average_rewards)
            
            # Reweight motion time sampling rates
            if not self.cfg['env']['disable_time_reweight']:
                self._motion_data._reweight_time_sampling_rate_vectorized(self.motion_time_seqreward)

            # # Debug
            new_time = self._motion_data.time_sample_rate_tensor
            new_clip = self._motion_data._motion_weights_tensor
            print("new_clip",new_clip)
            # print("******************************Time",(old_time!=new_time).sum())
            # print("******************************Clip",(old_clip!=new_clip).sum())

        # self.progress_buf_total += 1  # Increment total progress
        return

    def record_motion_time_reward(self, reset_env_ids):
        """
        Records rewards for each motion clip at each time step.
        Optimized to use batch processing and vectorized tensor operations.
        """
        if reset_env_ids.numel() == 0:
            return  # No environments to process
        
        # Gather relevant data for reset environments
        ts_reset = self.progress_buf[reset_env_ids]  # Shape: (num_reset_envs,)
        motion_ids_reset = self.motion_ids_total[reset_env_ids]  # Shape: (num_reset_envs,)
        motion_times_reset = self.motion_times_total[reset_env_ids]  # Shape: (num_reset_envs,)
        rew_reset = self.rew_buf[reset_env_ids].float()  # Shape: (num_reset_envs,)
        
        # Update envs_reward for reset environments at their respective time steps
        # self.envs_reward[reset_env_ids, ts_reset] = rew_reset
        self.envs_reward[torch.arange(self.num_envs), self.progress_buf] = self.rew_buf
        
        # Compute non-zero mean rewards for each reset environment
        # Create a mask where rewards are non-zero
        non_zero_mask = self.envs_reward[reset_env_ids] != 0  # Shape: (num_reset_envs, max_episode_length)
        
        # Compute the sum of non-zero rewards per environment
        non_zero_sum = torch.sum(self.envs_reward[reset_env_ids] * non_zero_mask.float(), dim=1)  # Shape: (num_reset_envs,)
        
        # Compute the count of non-zero rewards per environment, avoiding division by zero
        non_zero_count = torch.clamp(torch.sum(non_zero_mask, dim=1), min=1.0)  # Shape: (num_reset_envs,)
        
        # Compute the mean of non-zero rewards
        non_zero_mean = non_zero_sum / non_zero_count  # Shape: (num_reset_envs,)
        
        # Identify environments where mean reward is NaN (all rewards were zero)
        nan_mask = torch.isnan(non_zero_mean)  # Shape: (num_reset_envs,)
        
        # Valid environments where mean reward is not NaN
        valid_envs = ~nan_mask  # Shape: (num_reset_envs,)
        
        # Extract valid motion IDs and times
        valid_motion_ids = motion_ids_reset[valid_envs]  # Shape: (num_valid_envs,)
        valid_motion_times = motion_times_reset[valid_envs] - 2  # Adjust time index as per original logic
        
        # Clamp motion_times to ensure they are within valid range
        # (1) clamp lower bound
        valid_motion_times = torch.clamp(valid_motion_times, min=0)
        # (2) clamp upper bound
        max_tensor = (self._motion_data.motion_lengths[valid_motion_ids] - 3).long()
        valid_motion_times = torch.min(valid_motion_times, max_tensor)

        
        # Extract valid rewards
        valid_rewards = non_zero_mean[valid_envs]  # Shape: (num_valid_envs,)
        
        # Compute the indices for updating motion_time_seqreward
        motion_indices = valid_motion_ids.long()  # Shape: (num_valid_envs,)
        time_indices = valid_motion_times.long()  # Shape: (num_valid_envs,)
        
        # Update motion_time_seqreward in a vectorized manner
        # Formula: (existing_reward + new_reward) / 2
        self.motion_time_seqreward[motion_indices, time_indices] = (
            self.motion_time_seqreward[motion_indices, time_indices] + valid_rewards
        ) / 2.0
        
        # Reset envs_reward for the reset environments
        self.envs_reward[reset_env_ids] = 0.0
        
        return
    #######################################################################

#####################################################################
###=========================jit functions=========================###
#####################################################################
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

    if isTest and maxEpisodeLength > 0 :
        reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    else:
        reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC

    # reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    # reset = torch.zeros_like(reset_buf) #ZC300

    return reset, terminated