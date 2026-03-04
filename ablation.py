#!/usr/bin/env python3
"""
Ablation study — Isolating SAM's contributions.

Three ablation studies on a subset of 6 representative architectures:
    1. SAM ρ sensitivity:       ρ ∈ {0, 0.01, 0.02, 0.05, 0.10, 0.20}
    2. Augmentation strategy:   {None, Basic, Strong} × {Adam, SAM+Adam}
    3. Training strategy:       {Full Fine-Tune, Gradual Unfreeze} × {Adam, SAM+Adam}

Usage:
    python ablation.py --data-path /path/to/dataset --output-dir ablation_results
    python ablation.py --data-path /path/to/dataset --ablation 1       # Run only Ablation 1
    python ablation.py --data-path /path/to/dataset --ablation 1 2 3   # Run all three

Outputs:
    - ablation_results/ablation1_rho.csv, ablation1_rho_sensitivity.png
    - ablation_results/ablation2_augmentation.csv, ablation2_augmentation.png
    - ablation_results/ablation3_training_strategy.csv, ablation3_training_strategy.png
    - ablation_results/ablation_combined_summary.png
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
from sam import SAM
from utils.model_zoo import (
    MODEL_ZOO, get_family, get_head_params,
    get_discriminative_param_groups,
)
from utils.training import (
    train_one_epoch_sam, train_one_epoch_standard, validate, count_parameters,
)


# ===================================================================
# Argument parsing
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation study — Isolating SAM's contributions.")
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to ImageFolder dataset.')
    parser.add_argument('--output-dir', type=str, default='ablation_results',
                        help='Output directory (default: ablation_results).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--ablation', type=int, nargs='+', default=[1, 2, 3],
                        help='Which ablation(s) to run: 1, 2, 3, or all.')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Subset of models. Default: 6 representative models.')
    return parser.parse_args()


# ===================================================================
# Configuration
# ===================================================================

IMG_SIZE = 224
BATCH_SIZE = 32
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 25
NUM_EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS

HEAD_LR = 1e-3
LATE_LR = 1e-4
MID_LR = 1e-5
EARLY_LR = 1e-6
WEIGHT_DECAY = 1e-4

# 6 representative architectures (one per family)
ABLATION_MODELS = [
    'VGG16', 'ResNet-50', 'ResNet-101',
    'MobileNetV3 Large', 'DenseNet-121', 'EfficientNet B3',
]


# ===================================================================
# Augmentation pipelines
# ===================================================================

def _aug_none():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _aug_basic():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _aug_strong():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def _val_transform():
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ===================================================================
# Data loading
# ===================================================================

def make_loaders(data_path, train_transform, seed=42):
    """Create train/val DataLoaders with a fixed 70/30 split."""
    ds = datasets.ImageFolder(data_path, transform=None)
    num_classes = len(ds.classes)

    train_size = int(0.7 * len(ds))
    val_size = len(ds) - train_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=gen)
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = _val_transform()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader, num_classes


# ===================================================================
# Core experiment runner (flexible for ablation variants)
# ===================================================================

def run_single(model_name, use_sam, rho, train_loader, val_loader,
               num_classes, use_gradual_unfreeze=True, tag=""):
    """
    Train one model with the given configuration.

    Args:
        model_name:           Key in MODEL_ZOO.
        use_sam:              True → SAM+Adam, False → Adam.
        rho:                  SAM perturbation radius.
        train_loader:         DataLoader with desired augmentation.
        val_loader:           DataLoader for validation.
        num_classes:          Number of output classes.
        use_gradual_unfreeze: True → warmup + discriminative LRs.
        tag:                  Label string for this ablation run.

    Returns:
        dict with accuracy, precision, recall, f1, val_loss
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt_label = f"SAM(ρ={rho})+Adam" if use_sam else "Adam"

    print(f"\n  → {model_name} | {opt_label} | {tag}")

    model = MODEL_ZOO[model_name](num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    train_fn = train_one_epoch_sam if use_sam else train_one_epoch_standard

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    global_epoch = 0
    t_start = time.time()

    if use_gradual_unfreeze:
        # --- PHASE 1: Warmup — freeze backbone, train head only ---
        for p in model.parameters():
            p.requires_grad = False
        for p in get_head_params(model, model_name):
            p.requires_grad = True
        head_params = [p for p in model.parameters() if p.requires_grad]

        if use_sam:
            opt = SAM(head_params, base_optimizer=torch.optim.Adam,
                      lr=HEAD_LR, weight_decay=WEIGHT_DECAY, rho=rho)
            base = opt.base_optimizer
        else:
            opt = torch.optim.Adam(head_params, lr=HEAD_LR,
                                   weight_decay=WEIGHT_DECAY)
            base = opt
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            base, T_max=WARMUP_EPOCHS, eta_min=1e-5)

        for ep in range(1, WARMUP_EPOCHS + 1):
            global_epoch += 1
            train_fn(model, train_loader, criterion, opt, device)
            sched.step()
            val_acc, _, _, _ = validate(model, val_loader, device, criterion)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = global_epoch
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}

        # --- PHASE 2: Full fine-tune — discriminative LRs ---
        for p in model.parameters():
            p.requires_grad = True
        pg = get_discriminative_param_groups(
            model, model_name, HEAD_LR, LATE_LR, MID_LR, EARLY_LR)

        if use_sam:
            opt = SAM(pg, base_optimizer=torch.optim.Adam,
                      weight_decay=WEIGHT_DECAY, rho=rho)
            base = opt.base_optimizer
        else:
            for g in pg:
                g.pop('name', None)
            opt = torch.optim.Adam(pg, weight_decay=WEIGHT_DECAY)
            base = opt
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            base, T_max=FINETUNE_EPOCHS, eta_min=1e-7)

        for ep in range(1, FINETUNE_EPOCHS + 1):
            global_epoch += 1
            train_fn(model, train_loader, criterion, opt, device)
            sched.step()
            val_acc, _, _, _ = validate(model, val_loader, device, criterion)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = global_epoch
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
    else:
        # --- NO gradual unfreezing — train all layers from epoch 1 ---
        if use_sam:
            opt = SAM(model.parameters(), base_optimizer=torch.optim.Adam,
                      lr=HEAD_LR, weight_decay=WEIGHT_DECAY, rho=rho)
            base = opt.base_optimizer
        else:
            opt = torch.optim.Adam(model.parameters(), lr=HEAD_LR,
                                   weight_decay=WEIGHT_DECAY)
            base = opt
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            base, T_max=NUM_EPOCHS, eta_min=1e-7)

        for ep in range(1, NUM_EPOCHS + 1):
            global_epoch += 1
            train_fn(model, train_loader, criterion, opt, device)
            sched.step()
            val_acc, _, _, _ = validate(model, val_loader, device, criterion)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = global_epoch
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}

    # Final evaluation
    elapsed = time.time() - t_start
    model.load_state_dict(best_state)
    val_acc, y_pred, y_true, val_loss = validate(
        model, val_loader, device, criterion)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')

    print(f"    Acc={val_acc:.4f}  F1={f1:.4f}  Loss={val_loss:.4f}  "
          f"BestEp={best_epoch}  Time={elapsed:.0f}s")

    return {
        'Model': model_name,
        'Accuracy': round(val_acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
        'Val Loss': round(val_loss, 4),
        'Best Epoch': best_epoch,
        'Train Time (s)': round(elapsed, 1),
    }


# ===================================================================
# Ablation 1: SAM ρ sensitivity
# ===================================================================

def ablation_rho(data_path, models, out_dir, seed):
    """Vary ρ ∈ {0, 0.01, 0.02, 0.05, 0.10, 0.20}."""
    print("\n" + "#" * 70)
    print("  ABLATION 1: SAM ρ (perturbation radius) sensitivity")
    print("#" * 70)

    rho_values = [0, 0.01, 0.02, 0.05, 0.10, 0.20]
    train_loader, val_loader, nc = make_loaders(data_path, _aug_basic(), seed)

    results = []
    for model_name in models:
        for rho in rho_values:
            use_sam = (rho > 0)
            tag = f"ρ={rho}" if use_sam else "Adam (ρ=0)"
            res = run_single(model_name, use_sam=use_sam, rho=rho,
                             train_loader=train_loader,
                             val_loader=val_loader, num_classes=nc,
                             use_gradual_unfreeze=True, tag=tag)
            res['ρ'] = rho
            results.append(res)
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f'{out_dir}/ablation1_rho.csv', index=False)

    # --- Plot: line chart per model ---
    n = len(models)
    nrows = (n + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 5 * nrows))
    axes = np.array(axes).flat
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        mdf = df[df['Model'] == model_name]
        ax.plot(mdf['ρ'], mdf['Accuracy'], 'o-', color='#ED7D31',
                linewidth=2, markersize=8, label='Accuracy')
        ax.plot(mdf['ρ'], mdf['F1 Score'], 's--', color='#5B9BD5',
                linewidth=2, markersize=7, label='F1 Score')
        adam_acc = mdf[mdf['ρ'] == 0]['Accuracy'].values[0]
        ax.axhline(y=adam_acc, color='gray', linestyle=':', alpha=0.5,
                   label=f'Adam baseline ({adam_acc:.4f})')
        ax.set_title(model_name, fontweight='bold', fontsize=11)
        ax.set_xlabel('ρ (perturbation radius)')
        ax.set_ylabel('Score')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rho_values)
    # Hide unused axes
    for i in range(n, len(list(axes))):
        fig.delaxes(list(np.array(fig.axes))[i])
    fig.suptitle('Ablation 1 — SAM ρ Sensitivity\n'
                 '(Higher ρ = more aggressive flat-minima search)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/ablation1_rho_sensitivity.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- Heatmap ---
    heat = df.pivot_table(index='Model', columns='ρ', values='Accuracy')
    plt.figure(figsize=(10, 5))
    sns.heatmap(heat, annot=True, fmt='.4f', cmap='YlOrRd', linewidths=0.5)
    plt.title('Ablation 1 — Accuracy Heatmap (Model × ρ)', fontweight='bold')
    plt.xlabel('ρ (perturbation radius)')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/ablation1_rho_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Ablation 1 saved to {out_dir}/ablation1_rho*")
    return df


# ===================================================================
# Ablation 2: Augmentation strategy
# ===================================================================

def ablation_augmentation(data_path, models, out_dir, seed):
    """Vary {None, Basic, Strong} × {Adam, SAM+Adam} at ρ = 0.05."""
    print("\n" + "#" * 70)
    print("  ABLATION 2: Augmentation strategy × Optimizer")
    print("#" * 70)

    aug_configs = [
        ('None',   _aug_none()),
        ('Basic',  _aug_basic()),
        ('Strong', _aug_strong()),
    ]
    rho = 0.05

    results = []
    for aug_name, aug_tf in aug_configs:
        tl, vl, nc = make_loaders(data_path, aug_tf, seed)
        for model_name in models:
            for use_sam in [False, True]:
                opt_name = "SAM+Adam" if use_sam else "Adam"
                tag = f"Aug={aug_name}, {opt_name}"
                res = run_single(model_name, use_sam=use_sam, rho=rho,
                                 train_loader=tl, val_loader=vl,
                                 num_classes=nc,
                                 use_gradual_unfreeze=True, tag=tag)
                res['Augmentation'] = aug_name
                res['Optimizer'] = opt_name
                results.append(res)
                torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f'{out_dir}/ablation2_augmentation.csv', index=False)

    # --- Plot ---
    n = len(models)
    nrows = (n + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(20, 6 * nrows))
    axes = np.array(axes).flat
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        mdf = df[df['Model'] == model_name]
        w = 0.3
        for i, (aug_name, _) in enumerate(aug_configs):
            adam_acc = mdf[(mdf['Augmentation'] == aug_name) &
                          (mdf['Optimizer'] == 'Adam')]['Accuracy'].values[0]
            sam_acc = mdf[(mdf['Augmentation'] == aug_name) &
                         (mdf['Optimizer'] == 'SAM+Adam')]['Accuracy'].values[0]
            ax.bar(i - w / 2, adam_acc, w,
                   label='Adam' if i == 0 else '', color='#5B9BD5')
            ax.bar(i + w / 2, sam_acc, w,
                   label='SAM+Adam' if i == 0 else '', color='#ED7D31')
            delta = sam_acc - adam_acc
            ax.annotate(f'{delta:+.3f}',
                        xy=(i, max(adam_acc, sam_acc) + 0.005),
                        ha='center', fontsize=8, fontweight='bold',
                        color='green' if delta > 0 else 'red')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['None', 'Basic', 'Strong'])
        ax.set_ylabel('Accuracy')
        ax.set_title(model_name, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.7, 1.0)
    for i in range(n, len(list(axes))):
        fig.delaxes(list(np.array(fig.axes))[i])
    fig.suptitle('Ablation 2 — Augmentation Strategy × Optimizer\n'
                 '(numbers show SAM gain Δ)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/ablation2_augmentation.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Ablation 2 saved to {out_dir}/ablation2_augmentation*")
    return df


# ===================================================================
# Ablation 3: Training strategy (gradual unfreezing)
# ===================================================================

def ablation_training_strategy(data_path, models, out_dir, seed):
    """Vary {Full Fine-Tune, Gradual Unfreeze} × {Adam, SAM+Adam}."""
    print("\n" + "#" * 70)
    print("  ABLATION 3: Gradual unfreezing ablation")
    print("#" * 70)

    rho = 0.05
    tl, vl, nc = make_loaders(data_path, _aug_basic(), seed)

    results = []
    for use_unfreeze in [False, True]:
        strat = "Gradual Unfreeze" if use_unfreeze else "Full Fine-Tune"
        for model_name in models:
            for use_sam in [False, True]:
                opt_name = "SAM+Adam" if use_sam else "Adam"
                tag = f"{strat}, {opt_name}"
                res = run_single(model_name, use_sam=use_sam, rho=rho,
                                 train_loader=tl, val_loader=vl,
                                 num_classes=nc,
                                 use_gradual_unfreeze=use_unfreeze, tag=tag)
                res['Strategy'] = strat
                res['Optimizer'] = opt_name
                results.append(res)
                torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f'{out_dir}/ablation3_training_strategy.csv', index=False)

    # --- Plot ---
    strategies = ['Full Fine-Tune', 'Gradual Unfreeze']
    n = len(models)
    nrows = (n + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(20, 6 * nrows))
    axes = np.array(axes).flat
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        mdf = df[df['Model'] == model_name]
        w = 0.3
        for i, strat in enumerate(strategies):
            adam_acc = mdf[(mdf['Strategy'] == strat) &
                          (mdf['Optimizer'] == 'Adam')]['Accuracy'].values[0]
            sam_acc = mdf[(mdf['Strategy'] == strat) &
                         (mdf['Optimizer'] == 'SAM+Adam')]['Accuracy'].values[0]
            ax.bar(i - w / 2, adam_acc, w,
                   label='Adam' if i == 0 else '', color='#5B9BD5')
            ax.bar(i + w / 2, sam_acc, w,
                   label='SAM+Adam' if i == 0 else '', color='#ED7D31')
            delta = sam_acc - adam_acc
            ax.annotate(f'{delta:+.3f}',
                        xy=(i, max(adam_acc, sam_acc) + 0.005),
                        ha='center', fontsize=9, fontweight='bold',
                        color='green' if delta > 0 else 'red')
        ax.set_xticks(range(2))
        ax.set_xticklabels(strategies, fontsize=9)
        ax.set_ylabel('Accuracy')
        ax.set_title(model_name, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.7, 1.0)
    for i in range(n, len(list(axes))):
        fig.delaxes(list(np.array(fig.axes))[i])
    fig.suptitle('Ablation 3 — Training Strategy × Optimizer\n'
                 '(Full Fine-Tune = all layers from start; '
                 'Gradual Unfreeze = warmup head then discriminative LRs)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/ablation3_training_strategy.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Ablation 3 saved to {out_dir}/ablation3_training_strategy*")
    return df


# ===================================================================
# Combined summary
# ===================================================================

def combined_summary(df_rho, df_aug, df_strat, models, out_dir):
    """Generate a combined summary figure across all three ablations."""
    print("\n" + "#" * 70)
    print("  COMBINED ABLATION SUMMARY")
    print("#" * 70)

    # Best ρ per model
    print("\n  Best ρ per model:")
    for m in models:
        sub = df_rho[df_rho['Model'] == m]
        best = sub.loc[sub['Accuracy'].idxmax()]
        print(f"    {m:20s}  ρ={best['ρ']:.2f}  Acc={best['Accuracy']:.4f}")

    # SAM gain by augmentation
    print("\n  Average SAM gain (Δ Accuracy) by augmentation:")
    for aug in ['None', 'Basic', 'Strong']:
        adam_a = df_aug[(df_aug['Augmentation'] == aug) &
                        (df_aug['Optimizer'] == 'Adam')]['Accuracy']
        sam_a = df_aug[(df_aug['Augmentation'] == aug) &
                       (df_aug['Optimizer'] == 'SAM+Adam')]['Accuracy']
        print(f"    Aug={aug:6s}  Avg Δ = "
              f"{(sam_a.values - adam_a.values).mean():+.4f}")

    # SAM gain by training strategy
    print("\n  Average SAM gain (Δ Accuracy) by training strategy:")
    for strat in ['Full Fine-Tune', 'Gradual Unfreeze']:
        adam_a = df_strat[(df_strat['Strategy'] == strat) &
                          (df_strat['Optimizer'] == 'Adam')]['Accuracy']
        sam_a = df_strat[(df_strat['Strategy'] == strat) &
                         (df_strat['Optimizer'] == 'SAM+Adam')]['Accuracy']
        print(f"    {strat:20s}  Avg Δ = "
              f"{(sam_a.values - adam_a.values).mean():+.4f}")

    # --- Master figure ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # (a) ρ sensitivity
    ax = axes[0]
    rhos = sorted(df_rho['ρ'].unique())
    avg_accs = [df_rho[df_rho['ρ'] == r]['Accuracy'].mean() for r in rhos]
    ax.plot(rhos, avg_accs, 'o-', color='#ED7D31', lw=2.5, markersize=10)
    ax.axhline(y=avg_accs[0], color='gray', ls=':', alpha=0.5,
               label=f'Adam baseline ({avg_accs[0]:.4f})')
    for r, a in zip(rhos, avg_accs):
        ax.annotate(f'{a:.4f}', xy=(r, a + 0.003), ha='center', fontsize=9)
    ax.set(xlabel='ρ (perturbation radius)', ylabel='Avg Accuracy',
           title='(a) ρ Sensitivity\n(avg across models)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(rhos)

    # (b) Augmentation
    ax = axes[1]
    augs = ['None', 'Basic', 'Strong']
    adam_m = [df_aug[(df_aug['Augmentation'] == a) &
                     (df_aug['Optimizer'] == 'Adam')]['Accuracy'].mean()
              for a in augs]
    sam_m = [df_aug[(df_aug['Augmentation'] == a) &
                    (df_aug['Optimizer'] == 'SAM+Adam')]['Accuracy'].mean()
             for a in augs]
    x = np.arange(3)
    ax.bar(x - 0.15, adam_m, 0.3, label='Adam', color='#5B9BD5')
    ax.bar(x + 0.15, sam_m, 0.3, label='SAM+Adam', color='#ED7D31')
    for i in range(3):
        d = sam_m[i] - adam_m[i]
        ax.annotate(f'Δ={d:+.4f}',
                    xy=(i, max(adam_m[i], sam_m[i]) + 0.005),
                    ha='center', fontsize=9, fontweight='bold',
                    color='green' if d > 0 else 'red')
    ax.set_xticks(x)
    ax.set_xticklabels(augs)
    ax.set(ylabel='Avg Accuracy',
           title='(b) Augmentation Strategy\n(avg across models)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # (c) Training strategy
    ax = axes[2]
    strats = ['Full Fine-Tune', 'Gradual Unfreeze']
    adam_s = [df_strat[(df_strat['Strategy'] == s) &
                       (df_strat['Optimizer'] == 'Adam')]['Accuracy'].mean()
              for s in strats]
    sam_s = [df_strat[(df_strat['Strategy'] == s) &
                      (df_strat['Optimizer'] == 'SAM+Adam')]['Accuracy'].mean()
             for s in strats]
    x = np.arange(2)
    ax.bar(x - 0.15, adam_s, 0.3, label='Adam', color='#5B9BD5')
    ax.bar(x + 0.15, sam_s, 0.3, label='SAM+Adam', color='#ED7D31')
    for i in range(2):
        d = sam_s[i] - adam_s[i]
        ax.annotate(f'Δ={d:+.4f}',
                    xy=(i, max(adam_s[i], sam_s[i]) + 0.005),
                    ha='center', fontsize=9, fontweight='bold',
                    color='green' if d > 0 else 'red')
    ax.set_xticks(x)
    ax.set_xticklabels(['Full\nFine-Tune', 'Gradual\nUnfreeze'], fontsize=10)
    ax.set(ylabel='Avg Accuracy',
           title='(c) Training Strategy\n(avg across models)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Ablation Study — Isolating SAM Contributions',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/ablation_combined_summary.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Combined summary saved to {out_dir}/ablation_combined_summary.png")


# ===================================================================
# Main
# ===================================================================

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)
    models = args.models if args.models else ABLATION_MODELS

    print("=" * 70)
    print("  ABLATION STUDY — Isolating SAM's Contribution")
    print(f"  Models: {models}")
    print(f"  Dataset: {args.data_path}")
    print(f"  Epochs: {NUM_EPOCHS} (warmup={WARMUP_EPOCHS}, "
          f"finetune={FINETUNE_EPOCHS})")
    print("=" * 70)

    df_rho = df_aug = df_strat = None

    if 1 in args.ablation:
        df_rho = ablation_rho(args.data_path, models,
                              args.output_dir, args.seed)
    if 2 in args.ablation:
        df_aug = ablation_augmentation(args.data_path, models,
                                       args.output_dir, args.seed)
    if 3 in args.ablation:
        df_strat = ablation_training_strategy(args.data_path, models,
                                              args.output_dir, args.seed)

    if df_rho is not None and df_aug is not None and df_strat is not None:
        combined_summary(df_rho, df_aug, df_strat, models, args.output_dir)

    total = 0
    if 1 in args.ablation:
        total += len(models) * 6
    if 2 in args.ablation:
        total += len(models) * 3 * 2
    if 3 in args.ablation:
        total += len(models) * 2 * 2

    print(f"\n{'=' * 70}")
    print(f"  ABLATION STUDY COMPLETE")
    print(f"  Total runs: {total}")
    print(f"  All results in: {args.output_dir}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
