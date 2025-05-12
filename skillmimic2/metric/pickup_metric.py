# metric/pickup_metric.py
from .base_metric import BaseMetric
import torch

class PickupMetric(BaseMetric):
    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.at_target = torch.zeros((num_envs, 500), device=device, dtype=torch.bool)  # Assuming max progress_buf = 500
        self.reached_target = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids):
        self.at_target[env_ids, :] = False
        self.reached_target[env_ids] = False

    def update(self, state):
        ball_pos = state['ball_pos']
        root_pos = state['root_pos']
        distance_ball2root = torch.norm(ball_pos - root_pos, dim=-1)
        at_target = (root_pos[:, 2] > 0.5) & (distance_ball2root < 0.5)
        
        current_step = state['progress']
        self.at_target[:, current_step] = at_target
        self.reached_target = torch.sum(self.at_target, dim=-1) > 60

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
