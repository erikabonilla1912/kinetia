"""Arquitectura del contrato: U-Net con encoder MobileNetV3 Large preentrenado en ImageNet."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch

from dataset import NUM_CLASSES

ENCODER_NAME = "timm-mobilenetv3_large_100"


def build_model(pretrained: bool = True) -> torch.nn.Module:
    """La salida son logits: el softmax se aplica en la API (contrato, seccion 2.1)."""
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
