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

import copy
from gym import spaces
import numpy as np
import os
import torch 
import yaml

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import learning.common_player_discrete as common_player_discrete
import learning.skillmimic_models as skillmimic_models #ZC0
import learning.skillmimic_network_builder as skillmimic_network_builder
import learning.skillmimic_players as skillmimic_players

class HRLPlayerDiscreteLLCs(common_player_discrete.CommonPlayerDiscrete):
    def __init__(self, config):
        with open(os.path.join(os.getcwd(), config['llc_config']), 'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']
        self._latent_dim = config['latent_dim'] #Z0 #V1
        # self._control_mapping = config['control_mapping']
        
        self.delta_action = config['delta_action'] #ZC0

        super().__init__(config)
        
        self._task_size = self.env.task.get_task_obs_size()
        
        self._llc_steps = config['llc_steps']
        llc_checkpoint = config['llc_checkpoint']
        assert(llc_checkpoint != "")

        self._llc_agents = []
        self._build_llc(llc_config_params, llc_checkpoint)
        for i in range(5):
            self._build_llc(llc_config_params, "models/Locomotion/model.pth")

        return
    
    def get_action(self, obs_dict, is_determenistic = False):
        # obs = obs_dict['obs']

        # if len(obs.size()) == len(self.obs_shape):
        #     obs = obs.unsqueeze(0)
        # proc_obs = self._preproc_obs(obs)
        # input_dict = {
        #     'is_train': False,
        #     'prev_actions': None, 
        #     'obs' : proc_obs,
        #     'rnn_states' : self.states
        # }
        # with torch.no_grad():
        #     res_dict = self.model(input_dict)
        # # mu = res_dict['mus']
        # action = res_dict['actions']
        # self.states = res_dict['rnn_states']
        # if is_determenistic:
        #     logits = res_dict['logits']
        #     action_masks = input_dict.get('action_masks', None)
        #     if action_masks is not None:
        #         self.device = action_masks.device
        #         logits = torch.where(action_masks, logits, torch.tensor(-1e+8).to(self.device)) 
        #     current_action = torch.argmax(logits, dim=1)
        # else:
        #     current_action = action
        
        # current_action = current_action.detach()
        # clamped_actions = torch.clamp(current_action, -1.0, 1.0)

        # logits = self.hlmodel_bc(obs_dict['obs'])
        # current_action = torch.argmax(logits, dim=1)
        current_action = torch.zeros(obs_dict['obs'].shape[0], dtype=int, device=self.device)
        # current_action.fill_(self.env.task.control_signal)
        

        return current_action

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
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()
            batch_size = 1
            if len(obs_dict['obs'].size()) > len(self.obs_shape):
                batch_size = obs_dict['obs'].size()[0]
            self.batch_size = batch_size

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            done_indices = []

            for n in range(self.max_steps):
                obs_dict = self.env_reset(done_indices)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info = self.env_step(self.env, obs_dict, action)
                cr += r
                steps += 1
  
                self._post_step(info)

                if render:
                    self.env.render(mode = 'human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

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

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        return

    def env_step(self, env, obs_dict, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
            
        obs = obs_dict['obs']
        rewards = 0.0
        done_count = 0.0
        # disc_rewards = 0.0 #Z0
        for t in range(self._llc_steps):
            llc_actions = self._compute_llc_action(obs, actions)
                        
            obs, curr_rewards, curr_dones, infos = env.step(llc_actions)

            rewards += curr_rewards
            done_count += curr_dones

            # amp_obs = infos['amp_obs']
            # curr_disc_reward = self._calc_disc_reward(amp_obs)
            # curr_disc_reward = curr_disc_reward[0, 0].cpu().numpy()
            # disc_rewards += curr_disc_reward

        rewards /= self._llc_steps
        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0

        # disc_rewards /= self._llc_steps
        #print("disc_reward", disc_rewards)

        if isinstance(obs, dict):
            obs = obs['obs']
        if obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return torch.from_numpy(obs).to(self.device), torch.from_numpy(rewards), torch.from_numpy(dones), infos
    
    def _build_llc(self, config_params, checkpoint_file):
        network_params = config_params['network']

        # network_builder = ase_network_builder.ASEBuilder() 
        network_builder = skillmimic_network_builder.SkillMimicBuilder() #ZC0
        network_builder.load(network_params)

        # network = ase_models.ModelASEContinuous(network_builder)
        network = skillmimic_models.SkillMimicModelContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network)

        # self._llc_agent = ase_players.ASEPlayer(llc_agent_config)
        llc_agent = skillmimic_players.SkillMimicPlayerContinuous(llc_agent_config)

        llc_agent.restore(checkpoint_file)
        print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))

        self._llc_agents.append(llc_agent)
        return

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)
        obs_space = llc_env_info['observation_space']
        obs_size = obs_space.shape[0]
        # obs_size += 64 #Z
        # obs_size += 3 #Z
        obs_size -= self._task_size
        llc_env_info['observation_space'] = spaces.Box(obs_space.low[0], obs_space.high[0], shape=(obs_size,))
        llc_env_info['amp_observation_space'] = self.env.amp_observation_space.shape
        llc_env_info['num_envs'] = self.env.task.num_envs

        config = config_params['config']
        config['network'] = network
        config['env_info'] = llc_env_info

        return config

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim # + self.delta_action #Z0
        return

    def _compute_llc_action(self, obs, actions):
        # Assume the 0th dimension of obs and actions is env_id
        batch_size = obs.shape[0]
        llc_actions = []
        for i in range(batch_size):
            obs_i = obs[i : i+1] # Extract the data for the corresponding env

            # Select the corresponding lowlevel controller
            agent = self._llc_agents[i]   # i==0 uses agent0, i==1 uses agent1

            llc_obs_i = self._extract_llc_obs(obs_i) # Construct llc observation
            processed_obs = agent._preproc_obs(llc_obs_i)

            # Forward pass to get mu
            mu, _ = agent.model.a2c_network.eval_actor(obs=processed_obs)

            # Denormalize to the actual action space
            out_i = players.rescale_actions(
                # torch.full_like(self.actions_low, -1.56), torch.full_like(self.actions_high, +1.56),
                self.actions_low, self.actions_high,
                torch.clamp(mu, -1.0, 1.0))
            llc_actions.append(out_i)

        # Concatenate the actions of the two envs back together
        return torch.cat(llc_actions, dim=0)

    def _extract_llc_obs(self, obs):
        obs_size = obs.shape[-1]
        llc_obs = obs[..., :obs_size - self._task_size]
        return llc_obs
    
    # def _calc_disc_reward(self, amp_obs): #ZC0
    #     disc_reward = self._llc_agent._calc_disc_rewards(amp_obs)
    #     return disc_reward
