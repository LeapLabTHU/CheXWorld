import numpy as np
import torch


class Mixup:
    def __init__(self, mixup_alpha=1.,prob=1.0,):
        self.mixup_alpha = mixup_alpha
        self.mix_prob = prob
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _params_per_batch(self):
        lam = 1.
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = float(lam_mix)
        return lam

    def _mix_batch(self, x):
        lam = self._params_per_batch()
        if lam == 1.:
            return 1.
        x_flipped = x.flip(0).mul_(1. - lam)
        x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        lam = self._mix_batch(x)
        target = target * lam + target.flip(0) * (1. - lam)
        return x, target

