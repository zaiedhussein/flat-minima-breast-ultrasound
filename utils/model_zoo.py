"""
Model Zoo — 17 ImageNet-pretrained CNN architectures.

Organised into five families:
    1. Classic CNNs   : VGG16, VGG19, AlexNet
    2. ResNets        : ResNet-18, -50, -101, -152
    3. MobileNets     : MobileNetV2, MobileNetV3 Small, MobileNetV3 Large
    4. DenseNets      : DenseNet-121, -169, -201
    5. EfficientNets  : EfficientNet-B0, -B1, -B2, -B3

All models are loaded with ImageNet-pretrained weights. The final
classification head is replaced to match the target number of classes.
"""

import torch.nn as nn
from torchvision import models
from collections import OrderedDict
import timm


# ---------------------------------------------------------------------------
# Head replacement helpers
# ---------------------------------------------------------------------------

def _replace_linear(model, attr, nc):
    """Replace a single nn.Linear attribute with one having `nc` outputs."""
    orig = getattr(model, attr)
    setattr(model, attr, nn.Linear(orig.in_features, nc))
    return model


def _replace_sequential_last(model, attr, nc):
    """Replace the last nn.Linear inside a nn.Sequential attribute."""
    seq = getattr(model, attr)
    seq[-1] = nn.Linear(seq[-1].in_features, nc)
    setattr(model, attr, seq)
    return model


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

MODEL_ZOO = OrderedDict({
    # --- VGG family ---
    'VGG16': lambda nc: _replace_sequential_last(
        models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        'classifier', nc),
    'VGG19': lambda nc: _replace_sequential_last(
        models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1),
        'classifier', nc),
    # --- AlexNet ---
    'AlexNet': lambda nc: _replace_sequential_last(
        models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
        'classifier', nc),
    # --- ResNet family ---
    'ResNet-18': lambda nc: _replace_linear(
        models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        'fc', nc),
    'ResNet-50': lambda nc: _replace_linear(
        models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
        'fc', nc),
    'ResNet-101': lambda nc: _replace_linear(
        models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2),
        'fc', nc),
    'ResNet-152': lambda nc: _replace_linear(
        models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2),
        'fc', nc),
    # --- MobileNet family ---
    'MobileNetV2': lambda nc: _replace_sequential_last(
        models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2),
        'classifier', nc),
    'MobileNetV3 Small': lambda nc: _replace_sequential_last(
        models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        'classifier', nc),
    'MobileNetV3 Large': lambda nc: _replace_sequential_last(
        models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
        'classifier', nc),
    # --- DenseNet family ---
    'DenseNet-121': lambda nc: _replace_linear(
        models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
        'classifier', nc),
    'DenseNet-169': lambda nc: _replace_linear(
        models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1),
        'classifier', nc),
    'DenseNet-201': lambda nc: _replace_linear(
        models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1),
        'classifier', nc),
    # --- EfficientNet family (via timm) ---
    'EfficientNet B0': lambda nc: timm.create_model(
        'efficientnet_b0', pretrained=True, num_classes=nc),
    'EfficientNet B1': lambda nc: timm.create_model(
        'efficientnet_b1', pretrained=True, num_classes=nc),
    'EfficientNet B2': lambda nc: timm.create_model(
        'efficientnet_b2', pretrained=True, num_classes=nc),
    'EfficientNet B3': lambda nc: timm.create_model(
        'efficientnet_b3', pretrained=True, num_classes=nc),
})


# ---------------------------------------------------------------------------
# Architecture family detection
# ---------------------------------------------------------------------------

def get_family(model_name):
    """Return the architecture family string for a given model name."""
    name = model_name.lower()
    if 'vgg' in name:          return 'vgg'
    if 'alexnet' in name:      return 'alexnet'
    if 'resnet' in name:       return 'resnet'
    if 'mobilenet' in name:    return 'mobilenet'
    if 'densenet' in name:     return 'densenet'
    if 'efficientnet' in name: return 'efficientnet'
    raise ValueError(f"Unknown architecture family for: {model_name}")


# ---------------------------------------------------------------------------
# Head parameter access (per family)
# ---------------------------------------------------------------------------

def get_head_params(model, model_name):
    """Return an iterator over the classification-head parameters."""
    family = get_family(model_name)
    if family in ('vgg', 'alexnet', 'mobilenet', 'efficientnet'):
        return model.classifier.parameters()
    elif family == 'resnet':
        return model.fc.parameters()
    elif family == 'densenet':
        return model.classifier.parameters()
    raise ValueError(f"Unsupported family: {family}")


# ---------------------------------------------------------------------------
# Discriminative learning-rate parameter groups (per family)
# ---------------------------------------------------------------------------

def get_discriminative_param_groups(model, model_name,
                                    head_lr, late_lr, mid_lr, early_lr):
    """
    Partition model parameters into four groups with descending LRs.

    Returns a list of dicts suitable for ``torch.optim.Optimizer``.
    """
    family = get_family(model_name)

    # --- VGG / AlexNet ---
    if family in ('vgg', 'alexnet'):
        feats = list(model.features.children())
        t = len(feats) // 3
        ep = [p for m in feats[:t]     for p in m.parameters()]
        mp = [p for m in feats[t:2*t]  for p in m.parameters()]
        lp = [p for m in feats[2*t:]   for p in m.parameters()]
        return [
            {'params': ep, 'lr': early_lr, 'name': 'early_features'},
            {'params': mp, 'lr': mid_lr,   'name': 'mid_features'},
            {'params': lp, 'lr': late_lr,  'name': 'late_features'},
            {'params': model.classifier.parameters(),
             'lr': head_lr, 'name': 'head'},
        ]

    # --- ResNet ---
    if family == 'resnet':
        return [
            {'params': (list(model.conv1.parameters())
                        + list(model.bn1.parameters())),
             'lr': early_lr, 'name': 'stem'},
            {'params': model.layer1.parameters(),
             'lr': early_lr, 'name': 'layer1'},
            {'params': model.layer2.parameters(),
             'lr': mid_lr,   'name': 'layer2'},
            {'params': model.layer3.parameters(),
             'lr': mid_lr,   'name': 'layer3'},
            {'params': model.layer4.parameters(),
             'lr': late_lr,  'name': 'layer4'},
            {'params': model.fc.parameters(),
             'lr': head_lr,  'name': 'head'},
        ]

    # --- MobileNet V2 / V3 ---
    if family == 'mobilenet':
        feats = list(model.features.children())
        t = len(feats) // 3
        ep = [p for m in feats[:t]     for p in m.parameters()]
        mp = [p for m in feats[t:2*t]  for p in m.parameters()]
        lp = [p for m in feats[2*t:]   for p in m.parameters()]
        return [
            {'params': ep, 'lr': early_lr, 'name': 'early_features'},
            {'params': mp, 'lr': mid_lr,   'name': 'mid_features'},
            {'params': lp, 'lr': late_lr,  'name': 'late_features'},
            {'params': model.classifier.parameters(),
             'lr': head_lr, 'name': 'head'},
        ]

    # --- DenseNet ---
    if family == 'densenet':
        children = list(model.features.named_children())
        stem, ep, mp, lp = [], [], [], []
        for name, mod in children:
            if name in ('conv0', 'norm0', 'relu0', 'pool0'):
                stem += list(mod.parameters())
            elif name in ('denseblock1', 'transition1'):
                ep += list(mod.parameters())
            elif name in ('denseblock2', 'transition2'):
                mp += list(mod.parameters())
            else:
                lp += list(mod.parameters())
        return [
            {'params': stem + ep, 'lr': early_lr, 'name': 'stem+early'},
            {'params': mp,        'lr': mid_lr,   'name': 'mid_dense'},
            {'params': lp,        'lr': late_lr,  'name': 'late_dense'},
            {'params': model.classifier.parameters(),
             'lr': head_lr, 'name': 'head'},
        ]

    # --- EfficientNet (timm) ---
    if family == 'efficientnet':
        blocks = list(model.blocks.children())
        n = len(blocks)
        c1, c2 = n // 3, 2 * n // 3
        ep = (list(model.conv_stem.parameters())
              + list(model.bn1.parameters()))
        ep += [p for b in blocks[:c1]   for p in b.parameters()]
        mp  = [p for b in blocks[c1:c2] for p in b.parameters()]
        lp  = [p for b in blocks[c2:]   for p in b.parameters()]
        lp += (list(model.conv_head.parameters())
               + list(model.bn2.parameters()))
        return [
            {'params': ep, 'lr': early_lr, 'name': 'early_blocks'},
            {'params': mp, 'lr': mid_lr,   'name': 'mid_blocks'},
            {'params': lp, 'lr': late_lr,  'name': 'late_blocks'},
            {'params': model.classifier.parameters(),
             'lr': head_lr, 'name': 'head'},
        ]

    raise ValueError(f"Unsupported family: {family}")
