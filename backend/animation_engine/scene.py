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
from .particles import ParticleLayer
from .subject import SubjectLayer
from .timing import FrameTime


class Scene:
    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        assets: Assets,
        particles: ParticleLayer,
        grader: Grader,
        brush: BrushEffect,
    ) -> None:
        config = assets.config
        self.image = image
        self.particles = particles
        self.grader = grader
        self.brush = brush
        self.only = config.brush_mode == "only"

        self.subject: Optional[SubjectLayer] = None
        self.effects = []
        if not self.only:
            self.subject = SubjectLayer(mask, max(assets.scale, 0.25))
            self.effects = [
                SkyEffect(image, mask, assets),
                WaterEffect(image, mask, assets),
                VegetationEffect(image, mask, assets),
                FireEffect(image, mask, assets),
            ]
        self.static_graded = grader.apply(image.copy()) if self.only else None

    def render(self, ft: FrameTime) -> np.ndarray:
        canvas = self.image.copy()
        for effect in self.effects:
            effect.apply(canvas, ft)
        if self.brush.active:
            self.brush.apply(canvas, self.image, ft)
        self.particles.render(canvas, ft, front=False)
        if self.subject is not None and self.subject.active:
            source = self.image
            if self.brush.active:
                source = self.image.copy()
                self.brush.apply(source, self.image, ft)
            self.subject.composite(canvas, source)
        self.particles.render(canvas, ft, front=True)
        out = self.grader.apply(canvas)
        if self.only:
            if self.brush.active:
                return blend(self.static_graded, out, self.brush.paint_alpha)
            return self.static_graded.copy()
        return out
