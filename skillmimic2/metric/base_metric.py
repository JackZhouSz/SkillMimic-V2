from abc import ABC, abstractmethod

class BaseMetric(ABC):
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device

    @abstractmethod
    def reset(self, env_ids):
        pass

    @abstractmethod
    def update(self, state):
        pass

    @abstractmethod
    def compute(self):
        pass