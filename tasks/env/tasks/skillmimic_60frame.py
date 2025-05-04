from env.tasks.skillmimic import SkillMimicBallPlay
import torch
import math

class SkillMimicBallPlay60Frame(SkillMimicBallPlay): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        self.history_length = cfg['env']['historyLength']

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._obs_hist_buffer = torch.zeros([self.num_envs, self.history_length, self.get_obs_size()//self.history_length], device=self.device, dtype=torch.float)
        self._time_emb = self.get_time_embeddings(self.history_length, self.get_obs_size()//self.history_length)

    def get_obs_size(self):
        # return super().get_obs_size()
        return (super().get_obs_size() - 3) * self.history_length 
    
    def get_time_embeddings(self, max_len, d_model):
        modified = False
        if d_model %2 == 1:
            d_model += 1
            modified = True 
        position = torch.arange(-(max_len-1), 1, dtype=torch.float).unsqueeze(1).to(self.device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(self.device)

        embeddings = torch.zeros(max_len, d_model, device=self.device)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)

        if modified:
            embeddings = embeddings[:,:-1]
        return embeddings
    
    def _compute_observations(self, env_ids=None):
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        obj_obs = self._compute_obj_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)
        
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs)

        textemb_batch = self.hoi_data_label_batch[env_ids]
        obs = torch.cat((obs, textemb_batch),dim=-1)
        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids,ts].clone()
        

        self._obs_hist_buffer[env_ids] = torch.cat([self._obs_hist_buffer[env_ids, 1:], obs.unsqueeze(1)], dim=1)
        
        self.obs_buf[env_ids] = (self._obs_hist_buffer + self._time_emb)[env_ids].flatten(start_dim=1) 


        return
    