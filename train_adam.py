#!/usr/bin/env python3
"""
Train 17 CNN architectures with Adam vs SAM+Adam on breast ultrasound data.

This script implements Experiments 1 (BUSI) and 3 (BUS-UCLM) from the paper:
    "Sharpness-Aware Minimization for Breast Ultrasound Image Classification"

Training protocol:
    Phase 1 (epochs  1–5):   Backbone frozen, train classification head only.
    Phase 2 (epochs 6–30):   All layers unfrozen with discriminative LRs.

Usage:
    python train_adam.py --data-path /path/to/dataset --output-dir results_adam
    python train_adam.py --data-path /path/to/BUSI --output-dir results_exp1
    python train_adam.py --data-path /path/to/BUS-UCLM --output-dir results_exp3

Outputs:
    - Performance CSV:  <output-dir>/performance_table.csv
    - Cost CSV:         <output-dir>/computational_cost_table.csv
    - Per-model:        learning curves, confusion matrices, overlay plots
    - Summary:          comparison bar charts, improvement heatmap, sharpness
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
)

# Local imports
from sam import SAM
from utils.model_zoo import (
    MODEL_ZOO, get_head_params, get_discriminative_param_groups,
)
from utils.training import (
    train_one_epoch_sam, train_one_epoch_standard,
    validate, count_parameters, get_gpu_memory_mb, benchmark_inference,
)
from utils.sharpness import compute_sharpness
from utils.plotting import (
    plot_learning_curves, plot_overlay_curves,
    plot_performance_comparison, plot_improvement_heatmap,
    plot_sharpness_comparison, plot_computational_cost,
    plot_time_overhead_ratio, plot_confusion_matrix,
)


# ===================================================================
# Argument parsing
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 17 CNNs with Adam vs SAM+Adam on breast US data.")
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to ImageFolder dataset.')
    parser.add_argument('--output-dir', type=str, default='results_adam',
                        help='Directory for outputs (default: results_adam).')
    parser.add_argument('--config', type=str,
                        default='configs/adam_config.yaml',
                        help='YAML config file (default: configs/adam_config.yaml).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Subset of model names to run. '
                             'Default: all 17 architectures.')
    return parser.parse_args()


# ===================================================================
# Data loading
# ===================================================================

def create_data_loaders(data_path, img_size, batch_size, augmentation,
                        seed=42):
    """Create train/val DataLoaders with a 70/30 split."""
    # Augmentation pipelines
    if augmentation == "strong":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:  # basic
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_ds = datasets.ImageFolder(data_path, transform=None)
    num_classes = len(full_ds.classes)
    class_names = full_ds.classes

    train_size = int(0.7 * len(full_ds))
    val_size = len(full_ds) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                    generator=generator)
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    return train_loader, val_loader, num_classes, class_names


# ===================================================================
# Single experiment runner
# ===================================================================

def run_experiment(model_name, use_sam, cfg, train_loader, val_loader,
                   num_classes, class_names, device, out_dir):
    """
    Train one model with two-phase progressive unfreezing.

    Returns:
        result_dict:  metrics (accuracy, precision, recall, f1)
        history:      per-epoch train/val accuracy and loss
        cost_dict:    computational cost data
    """
    opt_label = "SAM+Adam" if use_sam else "Adam"
    warmup_epochs = cfg['WARMUP_EPOCHS']
    finetune_epochs = cfg['FINETUNE_EPOCHS']
    head_lr = cfg['HEAD_LR']
    late_lr = cfg['LATE_LR']
    mid_lr = cfg['MID_LR']
    early_lr = cfg['EARLY_LR']
    weight_decay = cfg['WEIGHT_DECAY']
    sam_rho = cfg['SAM_RHO']
    label_smoothing = cfg.get('LABEL_SMOOTHING', 0.0)
    grad_clip = cfg.get('GRAD_CLIP', None)

    print(f"\n{'=' * 70}")
    print(f"  {model_name}  |  Optimizer: {opt_label}  "
          f"|  Warmup {warmup_epochs} + Finetune {finetune_epochs} epochs")
    print(f"{'=' * 70}")

    # Reset GPU memory counter
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = MODEL_ZOO[model_name](num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing) if label_smoothing > 0 \
        else nn.CrossEntropyLoss()
    train_fn = train_one_epoch_sam if use_sam else train_one_epoch_standard
    total_params, _ = count_parameters(model)

    # History tracking
    history = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': [],
        'epoch_time': [],
    }
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    global_epoch = 0
    total_train_time = 0.0

    # ---------------------------------------------------------------
    # PHASE 1 — Warmup: freeze backbone, train only head
    # ---------------------------------------------------------------
    print(f"\n--- Phase 1: Warmup (head only, backbone frozen) ---")

    for param in model.parameters():
        param.requires_grad = False
    for param in get_head_params(model, model_name):
        param.requires_grad = True

    _, trainable_p1 = count_parameters(model)
    print(f"  Trainable params: {trainable_p1:,} / {total_params:,} "
          f"({100 * trainable_p1 / total_params:.1f}%)")

    head_params = [p for p in model.parameters() if p.requires_grad]

    if use_sam:
        optimizer = SAM(head_params, base_optimizer=torch.optim.Adam,
                        lr=head_lr, weight_decay=weight_decay, rho=sam_rho)
        base_opt = optimizer.base_optimizer
    else:
        optimizer = torch.optim.Adam(head_params, lr=head_lr,
                                     weight_decay=weight_decay)
        base_opt = optimizer

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_opt, T_max=warmup_epochs, eta_min=1e-5)

    for epoch in range(1, warmup_epochs + 1):
        global_epoch += 1
        t0 = time.time()
        train_acc, train_loss = train_fn(
            model, train_loader, criterion, optimizer, device,
            grad_clip=grad_clip)
        epoch_time = time.time() - t0
        total_train_time += epoch_time
        scheduler.step()

        val_acc, y_pred, y_true, val_loss = validate(
            model, val_loader, device, criterion)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted')

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_time'].append(epoch_time)

        print(f"  [Warmup]   Epoch {global_epoch:02d} | "
              f"TrainAcc {train_acc:.4f} TrainLoss {train_loss:.4f} | "
              f"ValAcc {val_acc:.4f} ValLoss {val_loss:.4f} | "
              f"F1 {f1:.4f} | {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = global_epoch
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

    # ---------------------------------------------------------------
    # PHASE 2 — Full fine-tune: unfreeze all, discriminative LRs
    # ---------------------------------------------------------------
    print(f"\n--- Phase 2: Full fine-tune (discriminative LRs) ---")

    for param in model.parameters():
        param.requires_grad = True
    _, trainable_p2 = count_parameters(model)
    print(f"  Trainable params: {trainable_p2:,} / {total_params:,} (100%)")

    param_groups = get_discriminative_param_groups(
        model, model_name, head_lr, late_lr, mid_lr, early_lr)
    print("  Layer-group LRs:")
    for pg in param_groups:
        print(f"    {pg['name']:20s} → {pg['lr']:.1e}")

    if use_sam:
        optimizer = SAM(param_groups, base_optimizer=torch.optim.Adam,
                        weight_decay=weight_decay, rho=sam_rho)
        base_opt = optimizer.base_optimizer
    else:
        for pg in param_groups:
            pg.pop('name', None)
        optimizer = torch.optim.Adam(param_groups,
                                     weight_decay=weight_decay)
        base_opt = optimizer

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_opt, T_max=finetune_epochs, eta_min=1e-7)

    for epoch in range(1, finetune_epochs + 1):
        global_epoch += 1
        t0 = time.time()
        train_acc, train_loss = train_fn(
            model, train_loader, criterion, optimizer, device,
            grad_clip=grad_clip)
        epoch_time = time.time() - t0
        total_train_time += epoch_time
        scheduler.step()

        val_acc, y_pred, y_true, val_loss = validate(
            model, val_loader, device, criterion)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted')

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_time'].append(epoch_time)

        print(f"  [Finetune] Epoch {global_epoch:02d} | "
              f"TrainAcc {train_acc:.4f} TrainLoss {train_loss:.4f} | "
              f"ValAcc {val_acc:.4f} ValLoss {val_loss:.4f} | "
              f"F1 {f1:.4f} | {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = global_epoch
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

    # ---------------------------------------------------------------
    # Final evaluation with best checkpoint
    # ---------------------------------------------------------------
    print(f"\n  ★ Best epoch: {best_epoch}  (Val Acc: {best_val_acc:.4f})")
    model.load_state_dict(best_state)
    val_acc, y_pred, y_true, val_loss = validate(
        model, val_loader, device, criterion)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')

    print(f"\n  Final Report ({model_name} — {opt_label}):")
    print(classification_report(
        y_true, y_pred, target_names=class_names, digits=4))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names,
                          model_name, opt_label, best_epoch, out_dir)

    # Learning curves
    plot_learning_curves(history, model_name, opt_label, best_epoch,
                         warmup_epochs, out_dir)

    # Computational cost
    gpu_mem_mb = get_gpu_memory_mb()
    avg_epoch_time = np.mean(history['epoch_time'])
    inference_ms = benchmark_inference(
        model, device, n_iters=cfg['INFERENCE_BENCHMARK_ITERS'])

    # Sharpness analysis
    print(f"\n  Computing sharpness metric "
          f"(ρ={cfg['SHARPNESS_PERTURBATION_RADIUS']}, "
          f"{cfg['SHARPNESS_NUM_SAMPLES']} directions)...")
    sharpness_info = compute_sharpness(
        model, val_loader, criterion, device,
        rho=cfg['SHARPNESS_PERTURBATION_RADIUS'],
        n_directions=cfg['SHARPNESS_NUM_SAMPLES'])
    print(f"    Base loss:       {sharpness_info['base_loss']:.6f}")
    print(f"    Max pert. loss:  {sharpness_info['max_perturbed_loss']:.6f}")
    print(f"    Sharpness:       {sharpness_info['sharpness']:.6f}")
    print(f"    Avg sensitivity: {sharpness_info['avg_sensitivity']:.6f}")

    cost_dict = {
        'Model': model_name,
        'Optimizer': opt_label,
        'Total Params (M)': round(total_params / 1e6, 2),
        'Avg Epoch Time (s)': round(avg_epoch_time, 2),
        'Total Train Time (s)': round(total_train_time, 1),
        'Total Train Time (min)': round(total_train_time / 60, 1),
        'GPU Mem Peak (MB)': round(gpu_mem_mb, 1),
        'Inference (ms/img)': round(inference_ms, 2),
        'Sharpness': round(sharpness_info['sharpness'], 6),
        'Avg Sensitivity': round(sharpness_info['avg_sensitivity'], 6),
        'Final Val Loss': round(sharpness_info['base_loss'], 6),
    }

    result_dict = {
        'Model': model_name,
        'Optimizer': opt_label,
        'Accuracy': round(val_acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
    }

    return result_dict, history, cost_dict


# ===================================================================
# Main
# ===================================================================

def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, num_classes, class_names = create_data_loaders(
        args.data_path,
        cfg['IMG_SIZE'], cfg['BATCH_SIZE'],
        cfg.get('TRAIN_AUGMENTATION', 'basic'),
        seed=args.seed,
    )
    print(f"Dataset: {args.data_path}")
    print(f"Classes: {class_names} ({num_classes})")

    # Models to run
    models_to_run = args.models if args.models else list(MODEL_ZOO.keys())

    all_results = []
    all_costs = []
    all_histories = {}

    for model_name in models_to_run:
        # --- Adam baseline ---
        res_adam, hist_adam, cost_adam = run_experiment(
            model_name, use_sam=False, cfg=cfg,
            train_loader=train_loader, val_loader=val_loader,
            num_classes=num_classes, class_names=class_names,
            device=device, out_dir=args.output_dir)
        all_results.append(res_adam)
        all_costs.append(cost_adam)
        all_histories[(model_name, 'Adam')] = hist_adam

        # --- SAM+Adam ---
        res_sam, hist_sam, cost_sam = run_experiment(
            model_name, use_sam=True, cfg=cfg,
            train_loader=train_loader, val_loader=val_loader,
            num_classes=num_classes, class_names=class_names,
            device=device, out_dir=args.output_dir)
        all_results.append(res_sam)
        all_costs.append(cost_sam)
        all_histories[(model_name, 'SAM+Adam')] = hist_sam

        # Overlay
        plot_overlay_curves(hist_adam, hist_sam, model_name,
                            'Adam', 'SAM+Adam',
                            cfg['WARMUP_EPOCHS'], args.output_dir)
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Summary tables
    # ---------------------------------------------------------------
    df = pd.DataFrame(all_results)
    cost_df = pd.DataFrame(all_costs)

    adam_df = df[df['Optimizer'] == 'Adam'].set_index('Model')
    sam_df = df[df['Optimizer'] == 'SAM+Adam'].set_index('Model')

    summary = pd.DataFrame({
        ('Adam', 'Accuracy'):     adam_df['Accuracy'],
        ('Adam', 'Precision'):    adam_df['Precision'],
        ('Adam', 'Recall'):       adam_df['Recall'],
        ('Adam', 'F1 Score'):     adam_df['F1 Score'],
        ('SAM+Adam', 'Accuracy'):  sam_df['Accuracy'],
        ('SAM+Adam', 'Precision'): sam_df['Precision'],
        ('SAM+Adam', 'Recall'):    sam_df['Recall'],
        ('SAM+Adam', 'F1 Score'):  sam_df['F1 Score'],
    })
    summary.columns = pd.MultiIndex.from_tuples(summary.columns)

    print(f"\n{'=' * 100}")
    print("  TABLE — Performance: Adam vs SAM+Adam")
    print(f"{'=' * 100}")
    print(summary.to_string())
    summary.to_csv(f'{args.output_dir}/performance_table.csv')

    print(f"\n{'=' * 100}")
    print("  TABLE — Computational Cost")
    print(f"{'=' * 100}")
    print(cost_df.to_string(index=False))
    cost_df.to_csv(f'{args.output_dir}/computational_cost_table.csv',
                    index=False)

    # ---------------------------------------------------------------
    # Summary plots
    # ---------------------------------------------------------------
    plot_performance_comparison(adam_df, sam_df, models_to_run,
                                'Adam', 'SAM+Adam', args.output_dir)
    plot_improvement_heatmap(adam_df, sam_df, 'Adam', 'SAM+Adam',
                             args.output_dir)
    plot_sharpness_comparison(cost_df, models_to_run,
                               'Adam', 'SAM+Adam', args.output_dir)
    plot_computational_cost(cost_df, models_to_run,
                            'Adam', 'SAM+Adam', args.output_dir)
    plot_time_overhead_ratio(cost_df, models_to_run,
                              'Adam', 'SAM+Adam', args.output_dir)

    # Win/loss summary
    wins = (sam_df['Accuracy'] > adam_df['Accuracy']).sum()
    ties = (sam_df['Accuracy'] == adam_df['Accuracy']).sum()
    total = len(models_to_run)
    print(f"\n  SAM+Adam wins: {wins}/{total}  |  "
          f"Ties: {ties}/{total}  |  "
          f"Adam wins: {total - wins - ties}/{total}")

    print(f"\n✓ All results saved to '{args.output_dir}/'")


if __name__ == '__main__':
    main()
