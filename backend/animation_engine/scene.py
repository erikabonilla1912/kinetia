"""Composición de un frame. Es la misma para imagen y para video."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .assets import Assets
from .effects.brush import BrushEffect
from .effects.fire import FireEffect
from .effects.sky import SkyEffect
from .effects.vegetation import VegetationEffect
from .effects.water import WaterEffect
from .grading import Grader
from .masks import blend
from .parallax import ParallaxRig
from .particles import ParticleLayer
from .timing import FrameTime


class Scene:
    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        depth: Optional[np.ndarray],
        assets: Assets,
        particles: ParticleLayer,
        grader: Grader,
        brush: BrushEffect,
        fast: bool,
    ) -> None:
        config = assets.config
        self.image = image
        self.particles = particles
        self.grader = grader
        self.brush = brush
        self.only = config.brush_mode == "only"

        # En modo only no hay parallax: mover la cámara animaría lo no pintado.
        self.rig: Optional[ParallaxRig] = None
        if depth is not None and config.mood.parallax_amplitude > 0 and not self.only:
            self.rig = ParallaxRig(image, mask, depth, assets, fast)

        self.effects = []
        if not self.only:
            self.effects = [
                SkyEffect(image, mask, assets),
                WaterEffect(image, mask, assets),
                VegetationEffect(image, mask, assets),
                FireEffect(image, mask, assets),
            ]
        self.static_graded = grader.apply(image.copy()) if self.only else None

    def render(self, ft: FrameTime) -> np.ndarray:
        source = self.rig.bg_plate if self.rig is not None else self.image
        canvas = source.copy()
        for effect in self.effects:
            effect.apply(canvas, ft)
        if self.brush.active:
            self.brush.apply(canvas, source, ft)
        if self.rig is not None:
            canvas = self.rig.warp_background(canvas, ft)
        self.particles.render(canvas, ft, front=False)
        if self.rig is not None and self.rig.has_subject:
            subject_source = None
            if self.brush.active:
                subject_source = self.rig.fg_plate.copy()
                self.brush.apply(subject_source, self.rig.fg_plate, ft)
            self.rig.composite_subject(canvas, ft, subject_source)
        self.particles.render(canvas, ft, front=True)
        out = self.grader.apply(canvas)
        if self.only:
            if self.brush.active:
                return blend(self.static_graded, out, self.brush.paint_alpha)
            return self.static_graded.copy()
        return out
