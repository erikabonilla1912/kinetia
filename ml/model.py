"""Arquitectura del contrato: U-Net con encoder MobileNetV3 Large, entrenada DESDE CERO.

Sin transfer learning (decision 14 de docs/PROGRESO.md): todos los pesos arrancan aleatorios.
A proposito, este modulo no ofrece ninguna forma de cargar pesos preentrenados.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch

from dataset import INIT, NUM_CLASSES  # noqa: F401  (INIT se reexporta para train.py y export.py)

ENCODER_NAME = "timm-mobilenetv3_large_100"


def build_model() -> torch.nn.Module:
    """Pesos aleatorios. La salida son logits: el softmax se aplica en la API (contrato, seccion 2.1)."""
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
