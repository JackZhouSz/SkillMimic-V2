from env.tasks.skillmimic_localhist import SkillMimicBallPlayLocalHist

class SkillMimicBallPlayLocalHistOnehist(SkillMimicBallPlayLocalHist): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
    def get_hist(self, env_ids, ts):
        return self._hist_obs_batch[env_ids,-60,...]

    def get_hist_size(self): 
        return -3 + 316
    