"""
Loss-landscape sharpness analysis.

Computes a post-hoc sharpness metric by evaluating the maximum loss increase
within a weight-perturbation neighbourhood, following:

    sharpness = max_{‖ε‖ ≤ ρ}  L(w + ε) − L(w)

This is approximated using random perturbation directions in weight space,
consistent with the theoretical framework of Foret et al. (2021).
"""

import numpy as np
import torch


def compute_sharpness(model, loader, criterion, device,
                      rho=0.05, n_directions=10):
    """
    Compute loss-landscape sharpness for the current model weights.

    Args:
        model:          Trained nn.Module (will not be modified).
        loader:         Validation DataLoader.
        criterion:      Loss function.
        device:         torch.device.
        rho (float):    Perturbation radius (should match the SAM ρ used
                        during training for a fair comparison).
        n_directions:   Number of random perturbation directions to sample.

    Returns:
        dict with keys:
            - 'base_loss':            L(w)
            - 'max_perturbed_loss':   max L(w + ε)
            - 'sharpness':            max increase = max L(w+ε) − L(w)
            - 'avg_sensitivity':      mean increase across directions
            - 'all_perturbed_losses': list of perturbed losses
    """
    model.eval()

    # --- Base loss L(w) ---
    base_loss = 0.0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            base_loss += criterion(model(x), y).item() * y.size(0)
            total += y.size(0)
    base_loss /= total

    # --- Save original weights ---
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    max_perturbed_loss = base_loss
    perturbed_losses = []

    for _ in range(n_directions):
        # Random direction in weight space, normalised to ‖d‖ = ρ
        direction = {}
        total_norm_sq = 0.0
        for name, param in model.named_parameters():
            d = torch.randn_like(param)
            direction[name] = d
            total_norm_sq += d.norm() ** 2
        scale = rho / (total_norm_sq.sqrt() + 1e-12)

        # Perturb: w → w + ρ · d / ‖d‖
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.add_(direction[name] * scale)

        # Perturbed loss L(w + ε)
        p_loss = 0.0
        total_p = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                p_loss += criterion(model(x), y).item() * y.size(0)
                total_p += y.size(0)
        p_loss /= total_p
        perturbed_losses.append(p_loss)

        if p_loss > max_perturbed_loss:
            max_perturbed_loss = p_loss

        # Restore original weights
        model.load_state_dict(original_state)

    sharpness = max_perturbed_loss - base_loss
    avg_sensitivity = np.mean(perturbed_losses) - base_loss

    return {
        'base_loss': base_loss,
        'max_perturbed_loss': max_perturbed_loss,
        'sharpness': sharpness,
        'avg_sensitivity': avg_sensitivity,
        'all_perturbed_losses': perturbed_losses,
    }
