from env.tasks.deepmimic1_unified import DeepMimic1BallPlayUnified
from env.tasks.skillmimic1_hist import SkillMimic1BallPlayHist

class DeepMimic2BallPlay(SkillMimic1BallPlayHist, DeepMimic1BallPlayUnified): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)