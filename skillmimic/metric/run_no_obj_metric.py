# metric/run_no_object_metric.py
from .base_metric import BaseMetric
import torch

class RunNoObjectMetric(BaseMetric):
    def __init__(self, num_envs, device, switch_skill_name=None):
        super().__init__(num_envs, device)
        self.switch_skill_name = switch_skill_name
        # self.reached_target = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.reached_target = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.at_target_count = torch.zeros(num_envs, device=device, dtype=torch.float)

    def reset(self, env_ids):
        # self.reached_target[env_ids] = True
        self.reached_target[env_ids] = False
        self.at_target_count[env_ids] = 0

    def update(self, state):
        root_pos = state['root_pos']
        root_pos_vel = state['root_pos_vel']
        at_target = (root_pos[:, 2] > 0.7) & (torch.norm(root_pos_vel, dim=-1) > 1.)
        # print(state['progress'], at_target.sum())
        # print(torch.norm(root_pos_vel, dim=-1))

        if self.switch_skill_name is None:
            self.at_target_count += at_target.float()
            fraction = self.at_target_count / state['progress']
            self.reached_target = fraction > 0.8
        else:
            if state['progress'] > 150:
                self.at_target_count += at_target.float()
                fraction = self.at_target_count / (state['progress'] - 150)
                self.reached_target = fraction > 0.8
        

        # if self.switch_skill_name is None:
        #     if state['progress'] < 500:
        #         self.reached_target = self.reached_target & at_target
        #     # else:
        #     #     self.reached_target = self.reached_target | (root_pos[:, 2] > 0.5)
        # else:
        #     if state['progress'] < 150:
        #         self.reached_target = self.reached_target | at_target
        #     else:
        #         self.reached_target = self.reached_target & at_target
        #     # elif state['progress'] < 300:
        #     #     self.reached_target = self.reached_target & at_target
        #     # else:
        #     #     self.reached_target = self.reached_target | (root_pos[:, 2] > 0.5)

        # print(state['progress'], self.reached_target.sum())

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
