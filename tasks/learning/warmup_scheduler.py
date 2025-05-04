from rl_games.common.schedulers import RLScheduler

class WarmupLinearScheduler(RLScheduler):
    """
    两段式：先 warm‑up 线性升到 peak，再线性衰减到 min
    """
    def __init__(self,
                 start_lr,
                 peak_lr,
                 min_lr,
                 warmup_steps,
                 max_steps,
                 use_epochs=True,
                 apply_to_entropy=False,
                 start_entropy_coef=0.0,
                 peak_entropy_coef=0.02,
                 min_entropy_coef=0.0001):
        super().__init__()
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.use_epochs = use_epochs
        self.apply_to_entropy = apply_to_entropy
        self.start_beta = start_entropy_coef
        self.peak_beta = peak_entropy_coef
        self.min_beta = min_entropy_coef

    def _interp(self, x, x0, x1, y0, y1):
        """线性插值"""
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def update(self, current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
        steps = epoch if self.use_epochs else frames
        # --- 1) warm‑up ---
        if steps < self.warmup_steps:
            lr = self._interp(steps, 0, self.warmup_steps,
                              self.start_lr, self.peak_lr)
            if self.apply_to_entropy:
                entropy_coef = self._interp(steps, 0, self.warmup_steps,
                                            self.start_beta, self.peak_beta)
        # --- 2) decay ---
        else:
            t = min(steps - self.warmup_steps,
                    self.max_steps - self.warmup_steps)
            lr = self._interp(t, 0, self.max_steps - self.warmup_steps,
                              self.peak_lr, self.min_lr)
            if self.apply_to_entropy:
                entropy_coef = self._interp(t, 0, self.max_steps - self.warmup_steps,
                                            self.peak_beta, self.min_beta)
        return lr, entropy_coef
