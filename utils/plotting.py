"""
Plotting utilities for training curves, comparisons, and analysis figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ---------------------------------------------------------------------------
# Per-model learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(history, model_name, opt_label, best_epoch,
                         warmup_epochs, out_dir):
    """
    Plot training/validation accuracy and loss curves side by side.

    Args:
        history:       dict with 'train_acc', 'val_acc', 'train_loss', 'val_loss'.
        model_name:    Architecture name (str).
        opt_label:     Optimizer name (e.g. "SAM+Adam").
        best_epoch:    Epoch index of best validation accuracy.
        warmup_epochs: Number of warm-up epochs (vertical line marker).
        out_dir:       Output directory for saving PNGs.
    """
    epochs = list(range(1, len(history['train_acc']) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy ---
    ax1.plot(epochs, history['train_acc'], 'b-', lw=1.5, label='Train Acc')
    ax1.plot(epochs, history['val_acc'],   'r-', lw=1.5, label='Val Acc')
    ax1.axvline(x=warmup_epochs, color='gray', ls='--', alpha=0.6,
                label=f'Unfreeze (epoch {warmup_epochs})')
    ax1.axvline(x=best_epoch, color='green', ls=':', alpha=0.8,
                label=f'Best (epoch {best_epoch})')
    ax1.set(xlabel='Epoch', ylabel='Accuracy',
            title=f'{model_name} ({opt_label}) — Accuracy')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # --- Loss ---
    ax2.plot(epochs, history['train_loss'], 'b-', lw=1.5, label='Train Loss')
    ax2.plot(epochs, history['val_loss'],   'r-', lw=1.5, label='Val Loss')
    ax2.axvline(x=warmup_epochs, color='gray', ls='--', alpha=0.6,
                label=f'Unfreeze (epoch {warmup_epochs})')
    ax2.axvline(x=best_epoch, color='green', ls=':', alpha=0.8,
                label=f'Best (epoch {best_epoch})')
    ax2.set(xlabel='Epoch', ylabel='Loss',
            title=f'{model_name} ({opt_label}) — Loss')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Learning Curves — {model_name} ({opt_label})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    safe = model_name.replace(' ', '_')
    safe_opt = opt_label.replace('+', '_')
    fname = f"{out_dir}/{safe}_{safe_opt}_curves.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fname}")


# ---------------------------------------------------------------------------
# Overlay: baseline vs SAM training trajectories
# ---------------------------------------------------------------------------

def plot_overlay_curves(hist_base, hist_sam, model_name,
                        base_label, sam_label, warmup_epochs, out_dir):
    """
    Overlay baseline vs SAM optimiser: accuracy and loss on same axes (2×2).
    """
    epochs = list(range(1, len(hist_base['train_acc']) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        (0, 0, 'train_acc',  'Training Accuracy',   'Accuracy'),
        (0, 1, 'val_acc',    'Validation Accuracy',  None),
        (1, 0, 'train_loss', 'Training Loss',        'Loss'),
        (1, 1, 'val_loss',   'Validation Loss',      None),
    ]

    for r, c, key, title, ylabel in panels:
        axes[r, c].plot(epochs, hist_base[key], 'b-', label=base_label)
        axes[r, c].plot(epochs, hist_sam[key],  'r-', label=sam_label)
        axes[r, c].set_title(title)
        if ylabel:
            axes[r, c].set_ylabel(ylabel)
        if r == 1:
            axes[r, c].set_xlabel('Epoch')
        axes[r, c].legend()
        axes[r, c].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.axvline(x=warmup_epochs, color='gray', ls='--', alpha=0.4)

    fig.suptitle(f'{model_name} — {base_label} vs {sam_label} '
                 f'Loss Trajectories', fontsize=14, fontweight='bold')
    plt.tight_layout()
    safe = model_name.replace(' ', '_')
    fname = f"{out_dir}/{safe}_{base_label.lower()}_vs_{sam_label.lower().replace('+','_')}_overlay.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fname}")


# ---------------------------------------------------------------------------
# Summary: performance comparison bar chart
# ---------------------------------------------------------------------------

def plot_performance_comparison(base_df, sam_df, models,
                                base_label, sam_label, out_dir):
    """Grouped horizontal bar charts for accuracy, precision, recall, F1."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 7))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(models))
    w = 0.35

    for ax, metric in zip(axes, metrics):
        ax.barh(x - w / 2,
                [base_df.loc[m, metric] for m in models],
                w, label=base_label, color='#5B9BD5')
        ax.barh(x + w / 2,
                [sam_df.loc[m, metric] for m in models],
                w, label=sam_label, color='#ED7D31')
        ax.set_yticks(x)
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlim(0.70, 1.0)
        ax.set_title(metric, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle(f'Performance — {base_label} vs {sam_label}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f"{out_dir}/performance_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {fname}")


# ---------------------------------------------------------------------------
# Improvement heatmap
# ---------------------------------------------------------------------------

def plot_improvement_heatmap(base_df, sam_df, base_label, sam_label, out_dir):
    """Heatmap showing per-model, per-metric improvement (SAM − baseline)."""
    improvement = pd.DataFrame({
        'Acc Δ':  sam_df['Accuracy']  - base_df['Accuracy'],
        'Prec Δ': sam_df['Precision'] - base_df['Precision'],
        'Rec Δ':  sam_df['Recall']    - base_df['Recall'],
        'F1 Δ':   sam_df['F1 Score']  - base_df['F1 Score'],
    })

    plt.figure(figsize=(8, 10))
    sns.heatmap(improvement, annot=True, fmt='.4f', cmap='RdYlGn',
                center=0, linewidths=0.5)
    plt.title(f'{sam_label} Improvement over {base_label}\n'
              f'(positive = SAM better)', fontweight='bold', fontsize=13)
    plt.tight_layout()
    fname = f"{out_dir}/improvement_heatmap.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {fname}")


# ---------------------------------------------------------------------------
# Sharpness comparison
# ---------------------------------------------------------------------------

def plot_sharpness_comparison(cost_df, models, base_label, sam_label,
                              out_dir):
    """Bar chart: sharpness and avg sensitivity for baseline vs SAM."""
    base = cost_df[cost_df['Optimizer'] == base_label].set_index('Model')
    sam  = cost_df[cost_df['Optimizer'] == sam_label].set_index('Model')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    x = np.arange(len(models))
    w = 0.35

    # Sharpness
    ax1.barh(x - w / 2,
             [base.loc[m, 'Sharpness'] for m in models],
             w, label=base_label, color='#5B9BD5')
    ax1.barh(x + w / 2,
             [sam.loc[m, 'Sharpness'] for m in models],
             w, label=sam_label, color='#ED7D31')
    ax1.set_yticks(x)
    ax1.set_yticklabels(models, fontsize=9)
    ax1.set_xlabel('Sharpness (lower = flatter minimum)')
    ax1.set_title('Loss Landscape Sharpness', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

    # Avg sensitivity
    ax2.barh(x - w / 2,
             [base.loc[m, 'Avg Sensitivity'] for m in models],
             w, label=base_label, color='#5B9BD5')
    ax2.barh(x + w / 2,
             [sam.loc[m, 'Avg Sensitivity'] for m in models],
             w, label=sam_label, color='#ED7D31')
    ax2.set_yticks(x)
    ax2.set_yticklabels(models, fontsize=9)
    ax2.set_xlabel('Avg Sensitivity (lower = more robust)')
    ax2.set_title('Avg Loss Sensitivity to Perturbation', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    fig.suptitle('Loss Landscape Analysis — SAM vs Baseline',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f"{out_dir}/sharpness_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {fname}")


# ---------------------------------------------------------------------------
# Computational cost comparison
# ---------------------------------------------------------------------------

def plot_computational_cost(cost_df, models, base_label, sam_label, out_dir):
    """Grouped bar charts for timing, memory, inference speed."""
    base = cost_df[cost_df['Optimizer'] == base_label].set_index('Model')
    sam  = cost_df[cost_df['Optimizer'] == sam_label].set_index('Model')

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    x = np.arange(len(models))
    w = 0.35

    cols = [
        ('Avg Epoch Time (s)',  'Avg Epoch Time (seconds)',   'Per-Epoch Training Time'),
        ('GPU Mem Peak (MB)',   'Peak GPU Memory (MB)',       'GPU Memory Usage'),
        ('Inference (ms/img)',  'Inference Latency (ms/img)', 'Inference Speed'),
    ]

    for ax, (col, xlabel, title) in zip(axes, cols):
        ax.barh(x - w / 2, [base.loc[m, col] for m in models],
                w, label=base_label, color='#5B9BD5')
        ax.barh(x + w / 2, [sam.loc[m, col] for m in models],
                w, label=sam_label, color='#ED7D31')
        ax.set_yticks(x)
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle(f'Computational Cost — {base_label} vs {sam_label}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f"{out_dir}/computational_cost.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {fname}")


# ---------------------------------------------------------------------------
# SAM overhead ratio
# ---------------------------------------------------------------------------

def plot_time_overhead_ratio(cost_df, models, base_label, sam_label, out_dir):
    """Bar chart showing SAM overhead ratio (SAM time / baseline time)."""
    base = cost_df[cost_df['Optimizer'] == base_label].set_index('Model')
    sam  = cost_df[cost_df['Optimizer'] == sam_label].set_index('Model')

    ratios = [sam.loc[m, 'Avg Epoch Time (s)'] /
              base.loc[m, 'Avg Epoch Time (s)']
              for m in models]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(models, ratios, color='#A855F7', edgecolor='white')
    ax.axvline(x=1.0, color='black', ls='-', lw=0.5)
    ax.axvline(x=2.0, color='red',   ls='--', alpha=0.5, label='2× overhead')
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{r:.2f}×', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel(f'Time Ratio  ({sam_label} / {base_label})')
    ax.set_title('SAM Training Overhead per Model',
                 fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fname = f"{out_dir}/sam_overhead_ratio.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {fname}")


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names,
                          model_name, opt_label, best_epoch, out_dir):
    """Save a confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix as cm_func
    cm = cm_func(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} ({opt_label}) — Best Epoch {best_epoch}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    safe = model_name.replace(' ', '_')
    safe_opt = opt_label.replace('+', '_')
    fname = f"{out_dir}/{safe}_{safe_opt}_cm.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fname}")
