"""Immutable model registration; never contains a live Blender reference."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelAsset:
    model_path: Path
    object_name: str
    category_name: str
    category_id: int | None
    active_in_simulation: bool = False
    collision_shape: str = "CONVEX_HULL"
    scale: float | tuple[float, float, float] = 1.0
    preprocess_model: bool = True
    shading: str = "FLAT"
    material_preprocessing: str = "replace"
    uv_mapping: str = "smart"
