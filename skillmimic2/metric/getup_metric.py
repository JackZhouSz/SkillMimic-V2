# metric/getup_metric.py
from .base_metric import BaseMetric
import torch

class GetupMetric(BaseMetric):
    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.at_target = torch.zeros((num_envs, 1000), device=device, dtype=torch.bool)  # Assuming max progress_buf = 350
        self.reached_target = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids):
        self.at_target[env_ids, :] = False
        self.reached_target[env_ids] = False

    def update(self, state):
        root_pos = state['root_pos']
        root_pos_vel = state['root_pos_vel']
        at_target = (root_pos[:, 2] > 0.7) & (torch.norm(root_pos_vel, dim=-1) < 1.0)

        current_step = state['progress']
        self.at_target[:, current_step] = at_target
        self.reached_target = torch.sum(self.at_target, dim=-1) > 60

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
