# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import time

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import learning.common_player as common_player
from metric.metric_factory import create_metric
from metric.metric_manager import MetricManager
# from utils import fid

class AMPPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._disc_reward_scale = config['disc_reward_scale']
        
        super().__init__(config)

        ######### Modified by Qihan @ 241214 #########
        metric_kwargs = {}
        if hasattr(self.env.task, 'layup_target'):
            metric_kwargs.update({"layup_target":self.env.task.layup_target})  # 如果需要额外的参数，可以从 cfg 中获取
        if hasattr(self.env.task, 'switch_skill_name'):
            metric_kwargs.update({"switch_skill_name":self.env.task.switch_skill_name})  # 如果需要额外的参数，可以从 cfg 中获取
        metric = create_metric(self.env.task.skill_name, self.env.task.num_envs, self.env.task.device, **metric_kwargs) # 使用工厂函数创建 Metric
        # 初始化 Metric 管理器
        if metric:
            self.metric_manager = MetricManager([metric])
        else:
            self.metric_manager = None
        return

    def restore(self, fn):
        if (fn != 'Base'):
            super().restore(fn)
            if self._normalize_amp_input:
                checkpoint = torch_ext.load_checkpoint(fn)
                self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return
    
    def _build_net(self, config):
        super()._build_net(config)
        
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()  
        
        return

    def _post_step(self, info):
        super()._post_step(info)
        # if (self.env.task.viewer):
        #     self._amp_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
        return config

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)

        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r
    
    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        '''
        all_obs = [] #fid
        for t in range(self.env.task.motion_played_length): 
            obs_buf = self.env.task.play_dataset_step(t)
            all_obs.append(obs_buf.clone())
        all_obs_tensor = torch.stack(all_obs)
        '''

        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            hidden_sim = [] #fid
            hidden_ref = []

            print_game_res = False

            done_indices = []

            for n in range(self.env.task.max_episode_length): #fid self.max_steps
                obs_dict = self.env_reset(done_indices)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)

                '''
                a_out = fid.g_a_out #fid
                hidden_sim.append(a_out)
                '''
                # print(action)
                obs_dict, r, done, info =  self.env_step(self.env, action)
                # disc_rewards = self._calc_amp_rewards(info['amp_obs'])['disc_rewards'] #ase
                cr += r #disc_rewards.squeeze(-1) #Z
                steps += 1
  
                self._post_step(info)

                if render:
                    self.env.render(mode = 'human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count
                ######### Modified by Qihan @ 241214 #########
                if self.metric_manager:
                    state = self.env.task.get_state_for_metric()
                    state['progress'] = n
                    self.metric_manager.update(state)
                #########
                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)
                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break
                
                done_indices = done_indices[:, 0]
            ######### Modified by Qihan @ 241214 #########
            # 在模拟结束时计算并输出 Metric
            if self.metric_manager:
                results = self.metric_manager.compute()
                for metric_name, value in results.items():
                    print(f"{metric_name}: {value}")
                self.metric_manager.reset(done_indices)
            #########

            '''
            hidden_dim = hidden_sim[0].shape[-1]
            hidden_sim_3d = torch.stack(hidden_sim, dim=0) #(timesteps, n_envs, dim)
            t1 = hidden_sim_3d[:-1] #(timesteps-1, n_envs, dim)
            t2 = hidden_sim_3d[1:] #(timesteps-1, n_envs, dim)
            paired_samples = torch.cat((t1, t2), dim=2).view(-1, hidden_dim * 2) #((timesteps-1) * n_envs , hidden_dim * 2)
            # n_paired_sample = paired_samples.shape[0]
            # hidden_paired_dim = paired_samples.shape[1]
            mu1 = paired_samples.mean(dim=0) #(hidden_paired_dim,)
            sigma1 = torch.cov(paired_samples.T) #(hidden_paired_dim, hidden_paired_dim)

            ref_obs = all_obs_tensor #torch.load("physhoi/data/obs/layup.pt")
            for frame in range(ref_obs.shape[0]):
                ref_obs_dict = {}
                ref_obs_dict['obs'] = ref_obs[frame]
                _ = self.get_action(ref_obs_dict, is_determenistic) # where network is called #, a_out is added due to fid  
                a_out = fid.g_a_out #fid
                hidden_ref.append(a_out)
            hidden_dim = hidden_ref[0].shape[-1]
            hidden_ref_3d = torch.stack(hidden_ref, dim=0) #(timesteps, n_envs, dim)
            t1 = hidden_ref_3d[:-1] #(timesteps-1, n_envs, dim)
            t2 = hidden_ref_3d[1:] #(timesteps-1, n_envs, dim)
            paired_ref = torch.cat((t1, t2), dim=2).view(-1, hidden_dim * 2) #((timesteps-1) * n_envs , hidden_dim * 2)
            # n_paired_sample = paired_ref.shape[0]
            # hidden_paired_dim = paired_ref.shape[1]
            mu2 = paired_ref.mean(dim=0) #(hidden_paired_dim,)
            sigma2 = torch.cov(paired_ref.T) #(hidden_paired_dim, hidden_paired_dim)

            FID = fid.calculate_frechet_distance(mu1,sigma1,mu2,sigma2)
            print("FID: ", FID)
            '''

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        return
