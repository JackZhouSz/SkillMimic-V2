import torch
import numpy as np
from env.tasks.amp import SkillMimicAMPLocomotion
from isaacgym import gymtorch

class SkillMimicAMPGetup(SkillMimicAMPLocomotion):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # 读取配置中的恢复与摔倒参数
        self._recovery_episode_prob = cfg["env"].get("recoveryEpisodeProb", 0.2)
        self._recovery_steps = cfg["env"].get("recoverySteps", 60)
        self._fall_init_prob = cfg["env"].get("fallInitProb", 0.1)
        
        # 增加变量，用于记录从摔倒状态重置的环境ID，用于后续AMP观测初始化
        self._reset_fall_env_ids = []

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        # 初始化恢复计数器
        self._recovery_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        # 用于存储摔倒状态的buffer
        self._fall_root_states = None
        self._fall_dof_pos = None
        self._fall_dof_vel = None
        # 生成摔倒状态样本
        self._generate_fall_states() #train getup

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_recovery_count()

    def _generate_fall_states(self):
        """生成摔倒状态：随机初始化root姿态，给随机动作并模拟一段时间，让机器人摔倒，再记录此时的状态。"""
        max_steps = 150
        
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # 随机化root姿态
        root_states = self._initial_humanoid_root_states[env_ids].clone()
        root_states[..., 3:7] = torch.randn_like(root_states[..., 3:7])
        root_states[..., 3:7] = torch.nn.functional.normalize(root_states[..., 3:7], dim=-1)
        self._humanoid_root_states[env_ids] = root_states

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # 给随机动作让角色摔倒
        rand_actions = np.random.uniform(-0.5, 0.5, size=[self.num_envs, self.get_action_size()])
        rand_actions = torch.tensor(rand_actions, dtype=torch.float32, device=self.device)
        self.pre_physics_step(rand_actions)

        # step physics and render each frame
        for i in range(max_steps):
            self.render()
            self.gym.simulate(self.sim)

        self._refresh_sim_tensors()

        # 记录摔倒状态
        self._fall_root_states = self._humanoid_root_states.clone()
        self._fall_root_states[:, 7:13] = 0
        self._fall_dof_pos = self._dof_pos.clone()
        self._fall_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        return

    def _reset_actors(self, env_ids):
        num_envs = env_ids.shape[0]
        # 根据recoveryEpisodeProb决定是否进入恢复期
        recovery_probs = torch.tensor([self._recovery_episode_prob] * num_envs, device=self.device)
        recovery_mask = torch.bernoulli(recovery_probs) == 1.0
        terminated_mask = (self._terminate_buf[env_ids] == 1)
        recovery_mask = torch.logical_and(recovery_mask, terminated_mask)

        recovery_ids = env_ids[recovery_mask]
        if len(recovery_ids) > 0:
            self._reset_recovery_episode(recovery_ids)
        
        nonrecovery_ids = env_ids[torch.logical_not(recovery_mask)]
        # 根据fallInitProb决定是否从摔倒状态开始
        fall_probs = torch.tensor([self._fall_init_prob] * nonrecovery_ids.shape[0], device=self.device)
        fall_mask = torch.bernoulli(fall_probs) == 1.0
        fall_ids = nonrecovery_ids[fall_mask]
        if len(fall_ids) > 0:
            self._reset_fall_episode(fall_ids)

        nonfall_ids = nonrecovery_ids[torch.logical_not(fall_mask)]
        if len(nonfall_ids) > 0:
            # 调用父类中正常重置逻辑
            super()._reset_actors(nonfall_ids)
            self._recovery_counter[nonfall_ids] = 0

    def _reset_recovery_episode(self, env_ids):
        # 给恢复期赋值，使这些环境在一定步数内不被终止，以便智能体尝试从倒地起身
        self._recovery_counter[env_ids] = self._recovery_steps

    def _reset_fall_episode(self, env_ids):
        # self._generate_fall_states() # set fallInitProb=1.0 to test getup
        # 从预先生成的摔倒状态中选取一组状态，用于初始化这些env
        fall_state_ids = torch.randint_like(env_ids, low=0, high=self._fall_root_states.shape[0])
        self._humanoid_root_states[env_ids] = self._fall_root_states[fall_state_ids]
        self._dof_pos[env_ids] = self._fall_dof_pos[fall_state_ids]
        self._dof_vel[env_ids] = self._fall_dof_vel[fall_state_ids]
        self._recovery_counter[env_ids] = self._recovery_steps

        # 将这些从摔倒状态重置的环境记录下来
        self._reset_fall_env_ids = env_ids

    def _reset_envs(self, env_ids):
        # 每次reset_envs时清空 _reset_fall_env_ids，以免影响后续episode的逻辑
        self._reset_fall_env_ids = []
        super()._reset_envs(env_ids)

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)

        # 如果有刚从摔倒状态初始化的环境，将其历史观测进行默认初始化
        if (len(self._reset_fall_env_ids) > 0):
            self._init_amp_obs_default(self._reset_fall_env_ids)

    def _update_recovery_count(self):
        # 每步减少恢复计数，直到为0
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)

    def _compute_reset(self):
        # 调用父类逻辑计算reset和terminate
        super()._compute_reset()

        # 恢复期内不终止
        is_recovery = self._recovery_counter > 0
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
    
    def _init_amp_obs_default(self, env_ids):
        # 用当前观测填充历史观测缓冲区
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs