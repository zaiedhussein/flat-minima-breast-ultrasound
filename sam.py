"""
Sharpness-Aware Minimization (SAM) Optimizer.

Reference:
    Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B. (2021).
    Sharpness-Aware Minimization for Efficiently Improving Generalization.
    In International Conference on Learning Representations (ICLR 2021).
    https://arxiv.org/abs/2010.01412

PyTorch implementation adapted from:
    https://github.com/davda54/sam (MIT License)
"""

import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.

    SAM seeks parameters that lie in neighbourhoods having uniformly low loss,
    rather than simply minimising the loss at the current point. This is
    achieved by a two-step update:

        1. **Ascent step** (`first_step`):  perturb weights in the direction
           that maximises the loss within an ℓ₂-ball of radius ρ.
        2. **Descent step** (`second_step`): compute the gradient at the
           perturbed point and take a standard optimiser step from the
           *original* weights.

    Args:
        params:          Iterable of parameters or parameter groups.
        base_optimizer:  Underlying optimiser class (e.g. ``torch.optim.SGD``).
        rho (float):     Perturbation radius (default: 0.05).
        adaptive (bool): If True, scale perturbation by parameter magnitude.
        **kwargs:        Forwarded to the base optimiser constructor.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False,
                 **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Ascent step: perturb weights to worst-case point within ρ-ball."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0
                       ) * p.grad * scale.to(p)
                p.add_(e_w)                 # climb to the local maximum
                self.state[p]["e_w"] = e_w   # store for restoration

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Descent step: restore original weights, then apply base update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # restore to original point

        self.base_optimizer.step()           # standard optimiser update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Full SAM step (for compatibility).

        Requires a closure that re-evaluates the model and returns the loss.
        """
        assert closure is not None, (
            "SAM requires closure, but it was not provided. "
            "Use first_step / second_step for explicit control."
        )
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0)
                 * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
