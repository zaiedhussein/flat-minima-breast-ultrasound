"""
Training utilities — epoch runners, validation, and computational profiling.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Training: one epoch with SAM (two-step)
# ---------------------------------------------------------------------------

def train_one_epoch_sam(model, loader, criterion, optimizer, device,
                        grad_clip=None):
    """
    One training epoch using the SAM optimiser.

    Performs two forward–backward passes per mini-batch:
        1. Ascent to worst-case perturbation within the ρ-ball.
        2. Descent from the perturbed point.

    Args:
        model:      nn.Module.
        loader:     DataLoader.
        criterion:  Loss function.
        optimizer:  SAM optimizer instance.
        device:     torch.device.
        grad_clip:  If not None, clip gradient norms to this value.

    Returns:
        (accuracy, avg_loss)
    """
    model.train()
    correct = total = 0
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Step 1 — ascent to worst-case perturbation
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.first_step(zero_grad=True)

        # Step 2 — descent from perturbed point
        out2 = model(x)
        loss2 = criterion(out2, y)
        loss2.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.second_step(zero_grad=True)

        running_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total, running_loss / total


# ---------------------------------------------------------------------------
# Training: one epoch with standard optimiser (Adam / SGD)
# ---------------------------------------------------------------------------

def train_one_epoch_standard(model, loader, criterion, optimizer, device,
                             grad_clip=None):
    """
    One training epoch using a standard (non-SAM) optimiser.

    Args:
        model, loader, criterion, optimizer, device: as above.
        grad_clip: optional gradient clipping norm.

    Returns:
        (accuracy, avg_loss)
    """
    model.train()
    correct = total = 0
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total, running_loss / total


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, loader, device, criterion=None):
    """
    Evaluate model on the validation set.

    Returns:
        (accuracy, y_pred, y_true, avg_loss)
    """
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    correct = total = 0
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * y.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    return (
        correct / total,
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        running_loss / total,
    )


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model):
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

def get_gpu_memory_mb():
    """Return current peak GPU memory allocated (MB)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


# ---------------------------------------------------------------------------
# Inference benchmarking
# ---------------------------------------------------------------------------

def benchmark_inference(model, device, input_size=(1, 3, 224, 224),
                        n_iters=100):
    """
    Benchmark inference speed.

    Returns:
        float — average milliseconds per image.
    """
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return (time.time() - t0) / n_iters * 1000  # ms
