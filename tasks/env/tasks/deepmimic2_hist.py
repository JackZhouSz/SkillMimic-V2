import torch
import random
import numpy as np
import pickle
from torch import Tensor
from typing import Dict

from utils import torch_utils
from isaacgym.torch_utils import *
import torch.nn.functional as F

from env.tasks.deepmimic2_unified import DeepMimic2BallPlayUnified
from env.tasks.skillmimic2_hist import SkillMimic2BallPlayHist

class DeepMimic2BallPlayHist(SkillMimic2BallPlayHist, DeepMimic2BallPlayUnified): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)