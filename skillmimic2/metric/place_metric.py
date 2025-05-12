# metric/shot_metric.py
from .base_metric import BaseMetric
import torch

class PlaceMetric(BaseMetric):
    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        self.at_target = torch.zeros((num_envs, 200), device=device, dtype=torch.bool)  # Assuming max progress_buf = 500
        self.reached_target = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids):
        self.at_target[env_ids, :] = False
        self.reached_target[env_ids] = False

    def update(self, state):
        obj_pos = state['ball_pos']
        root_pos = state['root_pos']
        wrist_pos = state['wrist_pos']
        distance_obj2wrist = torch.norm(obj_pos.unsqueeze(-2) - wrist_pos, dim=-1)
        distance_obj2wrist, _ = torch.min(distance_obj2wrist, dim=-1)
        at_target = (root_pos[:, 2] > 0.5) & (distance_obj2wrist < state['threshold']) & (obj_pos[:, 2] > 0.9)
        
        current_step = state['progress']
        self.at_target[:, current_step] = at_target
        self.reached_target = torch.sum(self.at_target, dim=-1) > 60

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
