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

from env.tasks.vec_task_wrappers import VecTaskPythonWrapper


from env.tasks.offline_state_search import OfflineStateSearch
from env.tasks.offline_state_search_parahome import OfflineStateSearchParahome

from env.tasks.skillmimic_parahome import SkillMimicParahome
from env.tasks.skillmimic_parahome import SkillMimicParahomePhaseNoisyinit
from env.tasks.skillmimic_parahome_multiobj import SkillMimicParahomePhaseNoisyinitMultiobj
from env.tasks.skillmimic_parahome import SkillMimicParahomeRIS
from env.tasks.skillmimic_parahome import SkillMimicParahomeDomRand
from env.tasks.skillmimic_parahome_localhist import SkillMimicParahomeLocalHist
from env.tasks.skillmimic_parahome_localhist import SkillMimicParahomeLocalHistRIS
from env.tasks.skillmimic_parahome_localhist import SkillMimicParahomeLocalHistRISBuffernode
from env.tasks.skillmimic_parahome_multiobj import SkillMimicParahomeMultiobj, SkillMimicParahomePhaseMultiobj

from env.tasks.hrl_virtual import HRLVirtual

from env.tasks.skillmimic1 import SkillMimic1BallPlay

from env.tasks.skillmimic1_unified import SkillMimic1BallPlayUnified
from env.tasks.skillmimic1_rand import SkillMimic1BallPlayRand
from env.tasks.skillmimic1_hist import SkillMimic1BallPlayHist
from env.tasks.skillmimic1_reweight import SkillMimic1BallPlayReweight
from env.tasks.skillmimic2 import SkillMimic2BallPlay


from env.tasks.deepmimic1 import DeepMimic1BallPlay
from env.tasks.deepmimic1_unified import DeepMimic1BallPlayUnified
from env.tasks.deepmimic2 import DeepMimic2BallPlay

from env.tasks.deepmimic_parahome import DeepMimicParahome
from env.tasks.deepmimic_parahome import DeepMimicParahomeDomRand
from env.tasks.deepmimic_parahome_localhist import DeepMimicParahomeLocalHist
from env.tasks.deepmimic_parahome_localhist import DeepMimicParahomeLocalHistRIS
from env.tasks.deepmimic_parahome_localhist import DeepMimicParahomeLocalHistRISBuffernode

from isaacgym import rlgpu

import json
import numpy as np


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [SkillMimicBallPlay, SkillMimicBallPlayDomRand, SkillMimicBallPlayRIS, \
            SkillMimicBallPlayLocalHistPhase, SkillMimicBallPlay60Frame, SkillMimicBallPlayRefobj, SkillMimicBallPlayHist, \
            SkillMimicBallPlayLocalHist, SkillMimicBallPlayLocalHistOnehist, SkillMimicBallPlayLocalHistUnified, SkillMimicBallPlayLocalHistUnifiedBuffernode \
            SkillMimicParahome, SkillMimicParahomeLocalHist, SkillMimicParahomeMultiobj,\
            HRLCircling, HRLHeadingEasy, HRLThrowing, HRLScoringLayup, \
            SkillMimic2BallPlay, SkillMimic2BallPlayUnified, SkillMimicAMPLocomotion, SkillMimicAMPGetup, DeepMimicBallPlay]")


def parse_task(args, cfg, cfg_train, sim_params):
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]
    
    try:
        task = eval(args.task)( # to HumanoidLocation(), obs defined here!
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless)
    except NameError as e:
        print(e)
        warn_task_name()
    env = VecTaskPythonWrapper(task, rl_device, cfg_train.get("clip_observations", np.inf), cfg_train.get("clip_actions", 1.0))

    return task, env
