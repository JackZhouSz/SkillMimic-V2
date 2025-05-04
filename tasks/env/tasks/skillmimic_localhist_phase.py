from env.tasks.skillmimic_localhist import SkillMimicBallPlayLocalHist

class SkillMimicBallPlayLocalHistPhase(SkillMimicBallPlayLocalHist): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
    def get_hist(self, env_ids, ts):
        l = self._motion_data.motion_lengths[self._motion_data.envid2motid[env_ids]]
        return (ts + self._motion_data.envid2sframe[env_ids]).unsqueeze(1) #.repeat(1, 3) #Phase(Absolute)
        return ((ts + self._motion_data.envid2sframe[env_ids]) / l).unsqueeze(1) #.repeat(1, 3) #Phase(Relative)
    
    def get_hist_size(self): 
        return -3 + 1
    