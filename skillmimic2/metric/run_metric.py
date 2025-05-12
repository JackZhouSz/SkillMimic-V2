# metric/run_metric.py
from .base_metric import BaseMetric
import torch

class RunMetric(BaseMetric):
    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.reached_target = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids):
        self.reached_target[env_ids] = False

    def update(self, state):
        ball_pos = state['ball_pos']
        root_pos = state['root_pos']
        ball_pos_vel = state['ball_pos_vel']
        distance_ball2root = torch.norm(ball_pos - root_pos, dim=-1)
        ball_pos_vel_z = torch.abs(ball_pos_vel[:, 2])
        at_target = (root_pos[:, 2] > 0.5) & (distance_ball2root < 1.5) & (ball_pos_vel_z > 1e-5)

        if state['progress'] < 100:
            self.reached_target = self.reached_target | at_target
        else:
            self.reached_target = self.reached_target & at_target

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
