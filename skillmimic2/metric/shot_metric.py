# metric/shot_metric.py
from .base_metric import BaseMetric
import torch

class ShotMetric(BaseMetric):
    def __init__(self, num_envs, device, layup_target):
        super().__init__(num_envs, device)
        self.layup_target = layup_target
        self.reached_target = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids):
        self.reached_target[env_ids] = False

    def update(self, state):
        ball_pos = state['ball_pos']
        root_pos = state['root_pos']
        distance_ball2targ = torch.abs(ball_pos[..., 2] - self.layup_target[2])
        at_target = (root_pos[:, 2] > 0.5) & (distance_ball2targ < 0.2)

        if state['progress'] < 300:
            self.reached_target = self.reached_target | at_target
        else:
            self.reached_target = self.reached_target & (root_pos[:, 2] > 0.5)

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
