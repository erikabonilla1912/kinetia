"""Entrenamiento de la segmentacion de KINETIA, reanudable por epoca, DESDE CERO.

Sin transfer learning: todos los pesos arrancan aleatorios (ml/model.py) y el checkpoint
lleva la marca `init`. Un checkpoint sin esa marca, como el del modelo anterior con pesos
de ImageNet, no se puede reanudar.

Reanudacion: en cada epoca se guardan modelo, optimizador, scheduler, scaler, estado de RNG,
mejor mIoU, contador de early stopping e historial. Al arrancar, si existe `last.ckpt` se
continua sola desde la epoca siguiente; una epoca cortada a medias se repite desde su inicio.

Este script NO puede leer la particion de prueba: solo pide filas de `TRAINING_SPLITS` y el
dataset rechaza cualquier otra. Hay un test automatizado que lo comprueba.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Optional

# Albumentations consulta pypi.org al importarse para avisar de versiones nuevas. El
# entrenamiento no necesita red: se desactiva, y una prueba comprueba que no hay conexiones.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A  # noqa: E402
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (
    CLASSES,
    DEFAULT_MANIFEST,
    IGNORE_INDEX,
    NUM_CLASSES,
    SOURCE_BOWFIRE,
    TRAINING_SPLITS,
    ManifestMismatchError,
    SegDataset,
    atomic_write,
    atomic_write_json,
    limit_rows,
    manifest_sha256,
    read_manifest,
    rows_for_split,
)
from model import ENCODER_NAME, INIT, build_model
from prepare_data import compute_class_weights
from seg_metrics import metrics_from_confusion

LAST_CKPT = "last.ckpt"
PREV_CKPT = "last.prev.ckpt"
BEST_MODEL = "kinetia_seg.pt"
HISTORY = "history.json"
CLASS_WEIGHTS = "class_weights.json"
# BoWFire tiene pocas imagenes frente a COCO: se ven mas veces por epoca.
BOWFIRE_SAMPLE_FACTOR = 8.0


def build_train_transform() -> A.Compose:
    """Aumentos. La salida sigue siendo 320x320; los pixeles rellenados quedan en IGNORE_INDEX.

    El tono se toca poco para que el fuego no deje de parecer fuego.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.25),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                cval=0,
                cval_mask=IGNORE_INDEX,
                mode=cv2.BORDER_CONSTANT,
                p=0.7,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=20, val_shift_limit=15, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ]
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _worker_init(worker_id: int) -> None:
    # La semilla base sale del RNG principal, que se restaura al reanudar.
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    if info is not None and info.dataset.transform is not None:
        info.dataset.transform.set_random_seed(seed)  # Albumentations tiene su propio RNG


def get_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_torch_atomic(obj: object, path: Path) -> None:
    atomic_write(path, lambda tmp: torch.save(obj, tmp))


def load_checkpoint(out_dir: Path, map_location: str = "cpu") -> Optional[dict]:
    """Ultimo checkpoint legible. Si el ultimo esta dañado, prueba con el anterior."""
    for name in (LAST_CKPT, PREV_CKPT):
        path = out_dir / name
        if not path.exists():
            continue
        try:
            # weights_only=False: el checkpoint es nuestro y trae estado de RNG de numpy.
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
            print(f"Checkpoint encontrado: {path} (epocas completas: {ckpt['epoch']})", flush=True)
            return ckpt
        except Exception as exc:
            print(f"AVISO: no se pudo leer {path} ({exc}); pruebo con el anterior.", flush=True)
    return None


def save_checkpoint(ckpt: dict, out_dir: Path) -> None:
    """Guarda atomicamente. Antes conserva el ultimo bueno como respaldo."""
    last = out_dir / LAST_CKPT
    if last.exists():
        # Renombrar, no copiar: el archivo anterior nunca queda a medio escribir.
        last.replace(out_dir / PREV_CKPT)
    save_torch_atomic(ckpt, last)


class InitMismatchError(RuntimeError):
    """El checkpoint no viene de un entrenamiento desde cero."""


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Una sola tasa para todo el modelo: no hay pesos previos que proteger."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


class CombinedLoss(torch.nn.Module):
    """Entropia cruzada ponderada + Dice multiclase, ambas ignorando IGNORE_INDEX."""

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True, ignore_index=IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target) + self.dice(logits, target)


def _confusion(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    index = NUM_CLASSES * target[valid] + pred[valid]
    return torch.bincount(index, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def make_loaders(args, train_rows: list[dict], val_rows: list[dict]) -> tuple[DataLoader, DataLoader]:
    train_ds = SegDataset(args.data_root, train_rows, TRAINING_SPLITS, build_train_transform())
    val_ds = SegDataset(args.data_root, val_rows, TRAINING_SPLITS)
    weights = [BOWFIRE_SAMPLE_FACTOR if r["source"] == SOURCE_BOWFIRE else 1.0 for r in train_rows]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True)
    # persistent_workers=False: los trabajadores se reinician cada epoca con una semilla que
    # sale del RNG principal, asi la reanudacion reproduce los mismos aumentos.
    common = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init,
        persistent_workers=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=len(train_rows) > args.batch_size, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_batches = 0.0, 0
    matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.set_grad_enabled(is_train):
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            if is_train:
                if not torch.isfinite(loss):
                    raise FloatingPointError("La perdida no es finita; revisa los datos o baja el learning rate")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item()
            total_batches += 1
            matrix += _confusion(logits.detach().argmax(dim=1), masks)
    result = metrics_from_confusion(matrix.cpu().numpy())
    result["loss"] = total_loss / max(total_batches, 1)
    return result


def write_best_model(model, out_dir: Path, epoch: int, miou: float, manifest_hash: str) -> None:
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload = {
        "state_dict": state,
        "encoder_name": ENCODER_NAME,
        "init": INIT,
        "classes": CLASSES,
        "input_size": [320, 320],
        "epoch": epoch,
        "val_miou": miou,
        "dataset_manifest_sha256": manifest_hash,
    }
    save_torch_atomic(payload, out_dir / BEST_MODEL)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=Path("data/runs/default"), help="En Colab, una carpeta de Drive")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--stop-after-epochs", type=int, default=None, help="Solo para pruebas: para limpiamente tras guardar esa epoca")
    return parser.parse_args(argv)


def train(args: argparse.Namespace) -> dict:
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = use_amp

    manifest = read_manifest(args.manifest)
    manifest_hash = manifest_sha256(args.manifest)
    train_rows = limit_rows(rows_for_split(manifest, "train", TRAINING_SPLITS), args.limit_train)
    val_rows = limit_rows(rows_for_split(manifest, "val", TRAINING_SPLITS), args.limit_val)
    print(f"Particiones: train={len(train_rows)} val={len(val_rows)}. La de prueba no se abre aqui.", flush=True)

    seed_everything(args.seed)
    model = build_model().to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    weights_path = args.out_dir / CLASS_WEIGHTS
    if weights_path.exists():
        weights_info = json.loads(weights_path.read_text(encoding="utf-8"))
    else:
        weights_info = compute_class_weights(args.data_root, train_rows)
        atomic_write_json(weights_path, weights_info)
    print("Pesos por clase (de los pixeles de train):", dict(zip(CLASSES, np.round(weights_info["weights"], 3))), flush=True)
    criterion = CombinedLoss(torch.tensor(weights_info["weights"], dtype=torch.float32, device=device))

    state = {"epoch": 0, "best_miou": -1.0, "best_epoch": 0, "bad_epochs": 0, "history": [], "finished": False}
    ckpt = load_checkpoint(args.out_dir)
    if ckpt is not None:
        if ckpt.get("init") != INIT:
            raise InitMismatchError(
                f"El checkpoint de {args.out_dir} no viene de un entrenamiento desde cero (es el modelo anterior, "
                "con pesos de ImageNet). Usa otra carpeta: este no se puede reanudar."
            )
        if ckpt["dataset_manifest_sha256"] != manifest_hash:
            raise ManifestMismatchError("El checkpoint se entreno con otro manifiesto; no se puede reanudar con este.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = {k: copy.deepcopy(ckpt[k]) for k in state}
        set_rng_state(ckpt["rng"])
        if not state["finished"]:
            print(f"REANUDANDO: continuo en la epoca {state['epoch'] + 1} de {args.epochs}; mejor mIoU de val: {state['best_miou']:.4f}; "
                  f"epocas sin mejora: {state['bad_epochs']}", flush=True)
    else:
        print("Sin checkpoint previo: empiezo desde la epoca 1.", flush=True)

    if state["finished"]:
        print("El entrenamiento ya habia terminado. No hay nada que hacer.", flush=True)
        return state

    train_loader, val_loader = make_loaders(args, train_rows, val_rows)

    while state["epoch"] < args.epochs:
        epoch = state["epoch"] + 1
        start = time.time()
        # Albumentations tiene su propio RNG: se siembra desde el principal, que se restaura al reanudar.
        train_loader.dataset.transform.set_random_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        train_m = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        train_seconds = time.time() - start
        val_m = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        seconds = time.time() - start

        state["epoch"] = epoch
        improved = val_m["miou"] > state["best_miou"] + args.min_delta
        if improved:
            state.update(best_miou=val_m["miou"], best_epoch=epoch, bad_epochs=0)
            write_best_model(model, args.out_dir, epoch, val_m["miou"], manifest_hash)
        else:
            state["bad_epochs"] += 1
        state["history"].append(
            {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "val_loss": val_m["loss"],
                "train_miou": train_m["miou"],
                "val_miou": val_m["miou"],
                "val_pixel_accuracy": val_m["pixel_accuracy"],
                "val_iou_per_class": {k: v["iou"] for k, v in val_m["per_class"].items()},
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": seconds,
                "train_seconds": train_seconds,
            }
        )
        if state["bad_epochs"] >= args.patience:
            state["finished"] = True
            print(f"Early stopping: {args.patience} epocas sin mejorar el mIoU de validacion.", flush=True)
        if epoch >= args.epochs:
            state["finished"] = True

        ckpt = {
            **copy.deepcopy(state),
            "init": INIT,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": get_rng_state(),
            "dataset_manifest_sha256": manifest_hash,
            "args": {k: str(v) for k, v in vars(args).items()},
        }
        save_checkpoint(ckpt, args.out_dir)
        atomic_write_json(args.out_dir / HISTORY, state["history"])

        recent = [h["seconds"] for h in state["history"][-5:]]
        remaining = (args.epochs - epoch) * float(np.mean(recent))
        print(
            f"Epoca {epoch}/{args.epochs} | train loss {train_m['loss']:.4f} mIoU {train_m['miou']:.4f} | "
            f"val loss {val_m['loss']:.4f} mIoU {val_m['miou']:.4f} | mejor {state['best_miou']:.4f} (epoca {state['best_epoch']}) | "
            f"sin mejora {state['bad_epochs']}/{args.patience} | tiempo medido {seconds:.1f} s | "
            f"restante estimado si no hay early stopping: {remaining / 60:.1f} min | checkpoint guardado",
            flush=True,
        )
        if state["finished"]:
            break
        if args.stop_after_epochs is not None and epoch >= args.stop_after_epochs:
            print(f"Parada solicitada tras la epoca {epoch} (solo para pruebas).", flush=True)
            return state

    print(f"Fin. Mejor mIoU de validacion: {state['best_miou']:.4f} en la epoca {state['best_epoch']}.", flush=True)
    return state


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
