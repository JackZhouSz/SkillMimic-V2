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

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

from .tcn import TemporalConvNet

DISC_LOGIT_INIT_SCALE = 1.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkillMimicBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):

            kwargs['input_shape'] = (1024,) #ZQH

            super().__init__(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)

            ###ZQH
            self.fc1 = nn.Linear(902, 1024)
            input_dim = 6  # 每个时间步的特征维度 #ZQH
            num_channels = [16, 32, 64]  # 每层的隐藏通道数
            kernel_size = 2
            dropout = 0.2
            self.tcn = TemporalConvNet(num_inputs=input_dim, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
            self.fc2 = nn.Linear(64, 1024)
            # self.dropout = nn.Dropout(0.2)  # 定义 Dropout #ZQH
            # 门控网络
            self.gate_mlp = nn.Sequential(
                nn.Linear(1024 * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output

        def eval_actor(self, obs, cls_latents=None): #ZC0
            if  cls_latents is not None:
                _, indices = torch.max(cls_latents, dim=-1)
                obs[torch.arange(obs.size(0)), -64 + indices] = 1.
            a_out = self.actor_cnn(obs)
            if(type(a_out) == dict): #ZC9
                a_out = a_out['obs']
            a_out = a_out.contiguous().view(a_out.size(0), -1)


            a_out = self.gate_fuse(a_out)#ZQH

            a_out = self.actor_mlp(a_out)


                     
            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            if(type(c_out) == dict): #ZC9
                c_out = c_out['obs']
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.gate_fuse(c_out) #ZQH
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

        def gate_fuse(self, out):
            ###ZQH
            curr, objhist = out[:,:902], out[:,902:]
            curr_hidden = self.fc1(curr)

            objhist = objhist.reshape(out.size(0), 60, 6).permute(0,2,1) #ZQH
            tcn_features = self.tcn(objhist)  # 输出为 (batch_size, num_channels[-1], length)
            # tcn_features = self.dropout(tcn_features) #ZQH
            pooled_features = tcn_features.mean(dim=-1) # 全局平均池化，形状为 (batch_size, num_channels[-1])
            objhist_hidden = self.fc2(pooled_features)
            # objhist_hidden = self.dropout(objhist_hidden) #ZQH

            # 计算门控权重
            combined_input = torch.cat([curr_hidden, objhist_hidden], dim=-1)
            g = self.gate_mlp(combined_input)  # 门控值 g ∈ [0, 1]
            
            return g * curr_hidden + (1 - g) * objhist_hidden

    def build(self, name, **kwargs):
        net = SkillMimicBuilder.Network(self.params, **kwargs)
        return net



