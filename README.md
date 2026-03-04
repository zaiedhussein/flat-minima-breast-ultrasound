# Favoring Flat Minima Improves Generalization of Transfer-Learned Models for Breast Ultrasound Tumor Classification

This repository contains the complete source code for reproducing the experiments presented in:

> **"Favoring Flat Minima Improves Generalization of Transfer-Learned Models for Breast Ultrasound Tumor Classification"**
>
> 

-
### Key Features

- **17 ImageNet-pretrained architectures** spanning 5 families:
  - VGG/AlexNet: VGG16, VGG19, AlexNet
  - ResNets: ResNet-18, -50, -101, -152
  - MobileNets: MobileNetV2, MobileNetV3 Small/Large
  - DenseNets: DenseNet-121, -169, -201
  - EfficientNets: B0, B1, B2, B3

- **Two-phase progressive unfreezing** training protocol:
  - Phase 1 (epochs 1–5): Backbone frozen, train classification head only
  - Phase 2 (epochs 6–30): All layers unfrozen with discriminative learning rates

- **Four experiments** across two datasets and two optimiser families:

  | | BUSI | BUS-UCLM |
  |---|---|---|
  | Adam vs SAM+Adam | Experiment 1 | Experiment 3 |
  | SGD vs SAM+SGD | Experiment 2 | Experiment 4 |

- **Comprehensive analysis** including:
  - Loss-landscape sharpness measurement
  - Computational cost profiling (training time, GPU memory, inference latency)
  - Three-way ablation study (ρ sensitivity, augmentation, training strategy)

---

## Repository Structure

```
├── README.md                   # This file
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── sam.py                      # SAM optimizer implementation
├── train_adam.py               # Adam vs SAM+Adam experiments
├── train_sgd.py                # SGD vs SAM+SGD experiments
├── ablation.py                 # Ablation studies (ρ, augmentation, unfreezing)
├── configs/
│   ├── adam_config.yaml        # Hyperparameters for Adam experiments
│   └── sgd_config.yaml        # Hyperparameters for SGD experiments
└── utils/
    ├── __init__.py
    ├── model_zoo.py            # 17-architecture model zoo
    ├── training.py             # Training/validation loops, benchmarking
    ├── sharpness.py            # Loss-landscape sharpness analysis
    └── plotting.py             # All visualisation utilities
```

---

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0 (with CUDA support recommended)
- NVIDIA GPU with ≥ 8 GB VRAM (for larger models like VGG19, ResNet-152)

### Setup

```bash
git clone https://github.com/YOUR-USERNAME/sam-breast-ultrasound.git
cd sam-breast-ultrasound
pip install -r requirements.txt
```

---

## Datasets

### 1. BUSI (Breast Ultrasound Images Dataset)

- **Source**: [Al-Dhabyani et al. (2020)](https://doi.org/10.1016/j.dib.2019.104863)
- **Size**: 780 images from 600 patients (25–75 years, mean 52)
- **Classes**: Normal (133), Benign (487), Malignant (210)
- **Download**: [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

### 2. BUS-UCLM

- **Source**: [Gómez-Flores et al. (2024)](https://doi.org/10.1016/j.compbiomed.2024.107999)
- **Classes**: Benign, Malignant (binary classification)

Both datasets should be organised in ImageFolder format:
```
dataset_root/
├── class_1/
│   ├── image001.png
│   ├── image002.png
│   └── ...
├── class_2/
│   └── ...
└── class_3/   (BUSI only)
    └── ...
```

---

## Usage

### Experiment 1: Adam vs SAM+Adam on BUSI

```bash
python train_adam.py \
    --data-path /path/to/BUSI \
    --output-dir results_exp1 \
    --config configs/adam_config.yaml \
    --seed 42
```

### Experiment 2: SGD vs SAM+SGD on BUSI

```bash
python train_sgd.py \
    --data-path /path/to/BUSI \
    --output-dir results_exp2 \
    --config configs/sgd_config.yaml \
    --seed 42
```

### Experiment 3: Adam vs SAM+Adam on BUS-UCLM

```bash
python train_adam.py \
    --data-path /path/to/BUS-UCLM \
    --output-dir results_exp3 \
    --config configs/adam_config.yaml \
    --seed 42
```

### Experiment 4: SGD vs SAM+SGD on BUS-UCLM

```bash
python train_sgd.py \
    --data-path /path/to/BUS-UCLM \
    --output-dir results_exp4 \
    --config configs/sgd_config.yaml \
    --seed 42
```

### Run a Subset of Models

```bash
python train_adam.py \
    --data-path /path/to/BUSI \
    --models VGG16 ResNet-50 "EfficientNet B0"
```

### Ablation Studies

```bash
# All three ablations
python ablation.py --data-path /path/to/dataset --output-dir ablation_results

# Only ρ sensitivity (Ablation 1)
python ablation.py --data-path /path/to/dataset --ablation 1

# ρ sensitivity + augmentation (Ablations 1 and 2)
python ablation.py --data-path /path/to/dataset --ablation 1 2
```

---

## Outputs

Each experiment produces:

| File | Description |
|------|-------------|
| `performance_table.csv` | Per-model accuracy, precision, recall, F1 |
| `computational_cost_table.csv` | Training time, GPU memory, inference speed, sharpness |
| `performance_comparison.png` | Grouped bar chart (4 metrics × 17 models) |
| `improvement_heatmap.png` | SAM improvement Δ per model and metric |
| `sharpness_comparison.png` | Sharpness and sensitivity: baseline vs SAM |
| `computational_cost.png` | Timing, memory, inference grouped bars |
| `sam_overhead_ratio.png` | SAM/baseline time ratio per model |
| `<Model>_<Opt>_curves.png` | Per-model learning curves (accuracy + loss) |
| `<Model>_<Opt>_cm.png` | Per-model confusion matrix |
| `<Model>_overlay.png` | Baseline vs SAM training trajectory overlay |

---

## Hyperparameters

### Adam Experiments (Experiments 1 & 3)

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Head LR | 1e-3 |
| Late backbone LR | 1e-4 |
| Mid backbone LR | 1e-5 |
| Early backbone LR | 1e-6 |
| Weight decay | 1e-4 |
| SAM ρ | 0.05 |
| Label smoothing | 0.0 |
| Gradient clipping | None |
| Scheduler | Cosine annealing |

### SGD Experiments (Experiments 2 & 4)

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Head LR | 1e-2 (10× Adam) |
| Late backbone LR | 1e-3 |
| Mid backbone LR | 1e-4 |
| Early backbone LR | 1e-5 |
| Weight decay | 1e-4 |
| SAM ρ | 0.10 |
| Momentum | 0.9 (Nesterov) |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| Augmentation | Strong |

---

## Citation

If you use this code, please cite:


## Acknowledgements

- SAM PyTorch implementation adapted from [davda54/sam](https://github.com/davda54/sam)
- BUSI dataset: [Al-Dhabyani et al. (2020)](https://doi.org/10.1016/j.dib.2019.104863)
- BUS-UCLM dataset: [Gómez-Flores et al. (2024)](https://doi.org/10.1016/j.compbiomed.2024.107999)
- EfficientNet models via [timm](https://github.com/huggingface/pytorch-image-models)
