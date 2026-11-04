"""Máscaras con borde suave, relleno de zonas ocultas y mezcla en punto fijo."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

Roi = tuple[int, int, int, int]  # y0, y1, x0, x1


def inner_feather(region: np.ndarray, px: float) -> np.ndarray:
    """Alfa float32 que vale 0 fuera de ``region`` y sube a 1 hacia adentro.

    El borde crece solo hacia el interior: los píxeles de otras clases quedan
    exactamente intactos, sin dejar de tener una transición suave.
    """
    binary = (region > 0).astype(np.uint8)
    if not binary.any():
        return np.zeros(binary.shape, np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    a = np.clip((dist - 1.0) / max(px, 1.0), 0.0, 1.0)
    return a * a * (3.0 - 2.0 * a)


def bounding_roi(active: np.ndarray) -> Optional[Roi]:
    """ROI de los píxeles distintos de cero, o None si no hay ninguno."""
    x, y, w, h = cv2.boundingRect((active > 0).astype(np.uint8))
    if w == 0 or h == 0:
        return None
    return y, y + h, x, x + w


def clip_alpha(alpha: np.ndarray) -> np.ndarray:
    """Alfa float32 de 0 a 1, listo para ``blend``."""
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def blend(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Mezcla por píxel. Con alfa 0 el resultado es ``dst`` exacto, y con 1, ``src``."""
    return cv2.blendLinear(dst, src, 1.0 - alpha, alpha)


def scale_pixels(image: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Multiplica cada píxel uint8 por una ganancia float32 (H, W), con saturación."""
    fixed = cv2.convertScaleAbs(gain, alpha=128.0)
    return cv2.multiply(image, cv2.merge([fixed, fixed, fixed]), scale=1.0 / 128.0)


def add_tinted(image: np.ndarray, amount: np.ndarray, color) -> np.ndarray:
    """Suma ``amount * color`` a la imagen, con saturación. ``amount`` es float32 (H, W)."""
    layers = [cv2.convertScaleAbs(amount, alpha=float(c)) for c in color]
    return cv2.add(image, cv2.merge(layers))


class Region:
    """Píxeles que un efecto debe calcular.

    Si la región llena poco su rectángulo (máscaras dispersas), los mapas se guardan
    como una lista de solo los píxeles con alfa > 0, en filas de ``_STRIP`` columnas:
    remap y las demás operaciones no gastan tiempo en lo que no se va a ver.
    Las columnas se limitan porque remap no admite lados de 32767 o más.
    """

    _STRIP = 1024
    _DENSE_FILL = 0.6

    def __init__(self, alpha: np.ndarray) -> None:
        self.roi = bounding_roi(alpha)
        self.active = self.roi is not None
        if not self.active:
            return
        y0, y1, x0, x1 = self.roi
        self.width = alpha.shape[1]
        local = alpha[y0:y1, x0:x1]
        self.size = (x1 - x0, y1 - y0)
        count = int(np.count_nonzero(local))
        self.flat = None
        self._low_maps: dict = {}
        if count < self._DENSE_FILL * local.size:
            flat = np.flatnonzero(local > 0)
            cols = min(self._STRIP, count)
            rows = int(np.ceil(count / cols))
            # El relleno repite el último píxel: escribirlo dos veces no cambia nada.
            flat = np.concatenate([flat, np.full(rows * cols - count, flat[-1], flat.dtype)])
            self.flat = flat
            self.strip_shape = (rows, cols)
            ys, xs = np.unravel_index(flat, local.shape)
            self.absolute = (ys + y0) * self.width + (xs + x0)
        self.alpha = self.pick(clip_alpha(local))
        self.inverse = 1.0 - self.alpha

    def pick(self, local: np.ndarray) -> np.ndarray:
        """Lleva un mapa (H, W) del tamaño del rectángulo al formato del efecto."""
        if self.flat is None:
            return local
        return local.ravel()[self.flat].reshape(self.strip_shape)

    def from_low(self, low: np.ndarray) -> np.ndarray:
        """Lleva un mapa a baja resolución que cubre el rectángulo al formato del efecto.

        Con lista de píxeles interpola solo esos píxeles, con la misma fórmula que resize.
        """
        if self.flat is None:
            if low.shape[1] == self.size[0] and low.shape[0] == self.size[1]:
                return low
            return cv2.resize(low, self.size, interpolation=cv2.INTER_LINEAR)
        key = low.shape[:2]
        if key not in self._low_maps:
            lh, lw = key
            ys, xs = np.unravel_index(self.flat, (self.size[1], self.size[0]))
            map_x = ((xs + 0.5) * lw / self.size[0] - 0.5).astype(np.float32).reshape(self.strip_shape)
            map_y = ((ys + 0.5) * lh / self.size[1] - 0.5).astype(np.float32).reshape(self.strip_shape)
            self._low_maps[key] = (map_x, map_y)
        map_x, map_y = self._low_maps[key]
        return cv2.remap(low, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def composite(self, canvas: np.ndarray, layer: np.ndarray) -> None:
        """Mezcla ``layer`` sobre ``canvas``, que debe ser contiguo y de tamaño completo."""
        if self.flat is None:
            y0, y1, x0, x1 = self.roi
            canvas[y0:y1, x0:x1] = cv2.blendLinear(canvas[y0:y1, x0:x1], layer, self.inverse, self.alpha)
            return
        pixels = canvas.reshape(-1, 3)
        current = pixels[self.absolute].reshape(self.strip_shape + (3,))
        pixels[self.absolute] = cv2.blendLinear(current, layer, self.inverse, self.alpha).reshape(-1, 3)


def _pyr_down(x: np.ndarray, channels: int) -> np.ndarray:
    down = cv2.pyrDown(x)
    return down.reshape(down.shape[0], down.shape[1], channels)


def push_pull_fill(image: np.ndarray, valid: np.ndarray, coarse: bool = False) -> np.ndarray:
    """Rellena lo no válido por interpolación en pirámide (rápido y suave).

    Admite imágenes uint8 de 3 canales y mapas float32 de 1 canal. En imágenes
    grandes trabaja a media resolución, o a un cuarto con ``coarse`` (video): el relleno
    es suave y los píxeles válidos conservan su valor exacto.
    """
    weight = (valid > 0).astype(np.float32)
    if weight.all() or not weight.any():
        return image.copy()
    h, w = weight.shape
    binary = (valid > 0).astype(np.uint8)
    if max(h, w) >= 640:
        factor = 4 if coarse else 2
        size = (w // factor, h // factor)
        small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        small_valid = cv2.resize(weight, size, interpolation=cv2.INTER_AREA) >= 0.999
        filled = _fill_pyramid(small, small_valid)
        out = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.copyTo(image, binary, out)
        return out
    out = _fill_pyramid(image, weight > 0)
    cv2.copyTo(image, binary, out)
    return out


def _fill_pyramid(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    weight = valid.astype(np.float32)
    if weight.all() or not weight.any():
        return image.copy()
    single = image.ndim == 2
    img = image.astype(np.float32)
    if single:
        img = img[..., None]
    channels = img.shape[2]
    weights = [weight]
    colors = [img * weight[..., None]]
    while min(weights[-1].shape) > 2 and float(weights[-1].min()) < 0.999:
        weights.append(cv2.pyrDown(weights[-1]))
        colors.append(_pyr_down(colors[-1], channels))
    recon = colors[-1] / np.maximum(weights[-1], 1e-4)[..., None]
    for level in range(len(weights) - 2, -1, -1):
        h, w = weights[level].shape
        up = cv2.pyrUp(recon, dstsize=(w, h)).reshape(h, w, channels)
        recon = colors[level] + up * (1.0 - np.clip(weights[level], 0.0, 1.0))[..., None]
    if image.dtype == np.uint8:
        out = np.clip(recon + 0.5, 0, 255).astype(np.uint8)
    else:
        out = recon.astype(image.dtype)
    if single:
        out = out[..., 0]
    return out


def region_plate(image: np.ndarray, region: np.ndarray, erode_px: int = 2, coarse: bool = False) -> np.ndarray:
    """Imagen en la que todo lo que no es ``region`` se rellena con su contenido.

    Así, al desplazar la textura de una región, no entran colores de las vecinas.
    Se descarta también una banda fina del borde, donde los píxeles mezclan clases.
    """
    binary = (region > 0).astype(np.uint8)
    if erode_px > 0:
        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        eroded = cv2.erode(binary, kernel, borderType=cv2.BORDER_REPLICATE)
        if eroded.any():
            binary = eroded
    return push_pull_fill(image, binary, coarse)
