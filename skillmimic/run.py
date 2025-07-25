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

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

import numpy as np
import copy
import torch

from learning import skillmimic_agent #, skillmimic_agent_regloss
from learning import skillmimic_players
from learning import skillmimic_models #, skillmimic_models_objhist_reduce
from learning import skillmimic_network_builder #, skillmimic_network_builder_hist_reduce, skillmimic_network_builder_dropout, skillmimic_network_builder_histGate
# from learning import skillmimic_network_builder_objhist_reduce
# from learning import physhoi_network_builder_dual #ZC9 #V1

from learning import hrl_agent_discrete #, hrl_agent, #ZC0
from learning import hrl_players_discrete, hrl_players_discrete_llcs #, hrl_players
from learning import hrl_models_discrete #, hrl_models, 
from learning import hrl_network_builder 

from learning import amp_agent
from learning import amp_players
from learning import amp_models
from learning import amp_network_builder

# from learning import ase_agent
# from learning import ase_players
# from learning import ase_models
# from learning import ase_network_builder

args = None
cfg = None
cfg_train = None

def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))
    
    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        # pdb.set_trace()
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})

def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)

    runner.algo_factory.register_builder('skillmimic', lambda **kwargs : skillmimic_agent.SkillMimicAgent(**kwargs))
    # runner.algo_factory.register_builder('skillmimic_regloss', lambda **kwargs : skillmimic_agent_regloss.SkillMimicAgent(**kwargs))
    runner.player_factory.register_builder('skillmimic', lambda **kwargs : skillmimic_players.SkillMimicPlayerContinuous(**kwargs))
    # runner.player_factory.register_builder('skillmimic_regloss', lambda **kwargs : skillmimic_players.SkillMimicPlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder('skillmimic', lambda network, **kwargs : skillmimic_models.SkillMimicModelContinuous(network)) 
    # runner.model_builder.model_factory.register_builder('skillmimic_objhist_reduce', lambda network, **kwargs : skillmimic_models_objhist_reduce.SkillMimicModelContinuous(network)) 
    runner.model_builder.network_factory.register_builder('skillmimic', lambda **kwargs : skillmimic_network_builder.SkillMimicBuilder())
    # runner.model_builder.network_factory.register_builder('skillmimic_hist_reduce', lambda **kwargs : skillmimic_network_builder_hist_reduce.SkillMimicBuilderHistReduce())
    # runner.model_builder.network_factory.register_builder('skillmimic_objhist_reduce', lambda **kwargs : skillmimic_network_builder_objhist_reduce.SkillMimicBuilder())
    # runner.model_builder.network_factory.register_builder('skillmimic_hist_GateFuse', lambda **kwargs : skillmimic_network_builder_histGate.SkillMimicBuilder())
    # runner.model_builder.network_factory.register_builder('physhoi_dual', lambda **kwargs : physhoi_network_builder_dual.PhysHOIBuilderDual()) #ZC9 #V1

    # runner.algo_factory.register_builder('hrl', lambda **kwargs : hrl_agent.HRLAgent(**kwargs)) #ZC0
    runner.algo_factory.register_builder('hrl_discrete', lambda **kwargs : hrl_agent_discrete.HRLAgentDiscrete(**kwargs)) #ZC0
    # runner.player_factory.register_builder('hrl', lambda **kwargs : hrl_players.HRLPlayer(**kwargs))
    runner.player_factory.register_builder('hrl_discrete', lambda **kwargs : hrl_players_discrete.HRLPlayerDiscrete(**kwargs))
    runner.player_factory.register_builder('hrl_discrete_llcs', lambda **kwargs : hrl_players_discrete_llcs.HRLPlayerDiscreteLLCs(**kwargs))
    # runner.model_builder.model_factory.register_builder('hrl', lambda network, **kwargs : hrl_models.ModelHRLContinuous(network))  
    runner.model_builder.model_factory.register_builder('hrl_discrete', lambda network, **kwargs : hrl_models_discrete.ModelHRLDiscrete(network))  
    runner.model_builder.network_factory.register_builder('hrl', lambda **kwargs : hrl_network_builder.HRLBuilder())

    runner.algo_factory.register_builder('amp', lambda **kwargs : amp_agent.AMPAgent(**kwargs))
    runner.player_factory.register_builder('amp', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder('amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))  
    runner.model_builder.network_factory.register_builder('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

    # runner.algo_factory.register_builder('ase', lambda **kwargs : ase_agent.ASEAgent(**kwargs))
    # runner.player_factory.register_builder('ase', lambda **kwargs : ase_players.ASEPlayer(**kwargs))
    # runner.model_builder.model_factory.register_builder('ase', lambda network, **kwargs : ase_models.ModelASEContinuous(network))  
    # runner.model_builder.network_factory.register_builder('ase', lambda **kwargs : ase_network_builder.ASEBuilder())
 
    return runner

def main():
    global args
    global cfg
    global cfg_train

    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)

    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    if args.horovod:
        cfg_train['params']['config']['multi_gpu'] = args.horovod

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
        
    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file
    
    if args.switch_motion_file:
        cfg['env']['switch_motion_file'] = args.switch_motion_file
    
    if args.graph_file:
        cfg['env']['stateSearchGraph'] = args.graph_file
    
    cfg['env']['graph_save_path'] = args.graph_save_path

    if args.asset_file_name:
        cfg['env']['asset']['assetFileName'] = args.asset_file_name
    
    if args.state_init_random_prob:
        cfg['env']['state_init_random_prob'] = args.state_init_random_prob

    if args.state_switch_prob:
        cfg['env']['state_switch_prob'] = args.state_switch_prob

    if args.play_dataset:
        cfg['env']['playdataset'] = True

    if args.projtype:
        cfg['env']['projtype'] = args.projtype

    if args.cg1 != -1.:
        cfg['env']['rewardWeights']['cg1'] = args.cg1

    if args.cg2 != -1.:
        cfg['env']['rewardWeights']['cg2'] = args.cg2

    if args.ig != -1.:
        cfg['env']['rewardWeights']['ig'] = args.ig

    if args.op != -1.:
        cfg['env']['rewardWeights']['op'] = args.op

    if args.save_images:
        cfg['env']['saveImages'] = args.save_images #True
    
    if args.init_vel:
        cfg['env']['initVel'] = True

    if args.frames_scale!= 0.:
        cfg['env']['dataFramesScale'] = args.frames_scale

    if args.ball_size!= 0.:
        cfg['env']['ballSize'] = args.ball_size

    if args.resume_from:
        cfg_train['params']['config']['resume_from'] = args.resume_from

    # if args.state_init.lower() != "random":
    cfg["env"]["stateInit"] = args.state_init

    if args.history_embedding_size:
        cfg_train['params']['network']['mlp']['history_embedding_size'] = args.history_embedding_size
    
    if args.regloss_weight:
        cfg_train['params']['config']['regloss_weight'] = args.regloss_weight

    # Create default directories for weights and statistics
    cfg_train['params']['config']['train_dir'] = args.output_path
    cfg['env']['localReward'] = args.local_reward
    cfg['env']['localRewardRandSkill'] = args.local_reward_randskill
    cfg['env']['eval_randskill'] = args.eval_randskill
    cfg['env']['state_search_to_align_reward'] = args.state_search_to_align_reward
    cfg['env']['testRandomPick'] = args.test_random_pick
    
    vargs = vars(args)

    algo_observer = RLGPUAlgoObserver()

    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)

    return

if __name__ == '__main__':
    main()
