"""Presets de mood. Los valores finales se reportan al orquestador (contrato, sección 5)."""
from __future__ import annotations

from .types import MoodName, MoodParams

_PRESETS: dict[str, MoodParams] = {
    "calma": MoodParams(
        name="calma", speed=0.6, saturation=0.95, warmth=0.05, contrast=1.00,
        particle_kind="polen", particle_density=30.0,
        parallax_amplitude=0.010, vignette=0.10,
    ),
    "epico": MoodParams(
        name="epico", speed=1.4, saturation=1.15, warmth=-0.10, contrast=1.20,
        particle_kind="brasas", particle_density=75.0,
        parallax_amplitude=0.022, vignette=0.35,
    ),
    "magico": MoodParams(
        name="magico", speed=1.0, saturation=1.25, warmth=-0.05, contrast=1.05,
        particle_kind="chispas", particle_density=110.0,
        parallax_amplitude=0.015, vignette=0.20,
    ),
    "nostalgico": MoodParams(
        name="nostalgico", speed=0.8, saturation=0.70, warmth=0.35, contrast=0.95,
        particle_kind="polvo", particle_density=50.0,
        parallax_amplitude=0.012, vignette=0.40,
    ),
}


def get_mood(name: MoodName) -> MoodParams:
    """Devuelve el preset del mood. Un nombre desconocido lanza ValueError."""
    try:
        return _PRESETS[name]
    except KeyError:
        raise ValueError(f"Mood desconocido: {name!r}. Válidos: {sorted(_PRESETS)}") from None
