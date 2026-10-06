"""Validated, Blender-independent distributions for procedural materials."""

import random
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import TypeAlias

from telekinesis.illusion.types.distribution import Uniform

Scalar: TypeAlias = float | tuple[float, float]
Color: TypeAlias = (
    tuple[float, ...] | tuple[tuple[float, ...], tuple[float, ...]]
)
Parameter: TypeAlias = Scalar | Color | Uniform
_BLENDER_FLOAT_MAX = 3.4028234663852886e38

# Public parameter -> Blender 4.2 socket, component count, valid bounds.
PARAMETERS = MappingProxyType(
    {
        "base_color": ("Base Color", 4, 0.0, 1.0),
        "metallic": ("Metallic", 1, 0.0, 1.0),
        "roughness": ("Roughness", 1, 0.0, 1.0),
        "ior": ("IOR", 1, 1.0, 1000.0),
        "specular_ior_level": ("Specular IOR Level", 1, 0.0, 1.0),
        "transmission": ("Transmission Weight", 1, 0.0, 1.0),
        "alpha": ("Alpha", 1, 0.0, 1.0),
        "coat_weight": ("Coat Weight", 1, 0.0, 1.0),
        "coat_roughness": ("Coat Roughness", 1, 0.0, 1.0),
        "subsurface_weight": ("Subsurface Weight", 1, 0.0, 1.0),
        "subsurface_radius": ("Subsurface Radius", 3, 0.0, _BLENDER_FLOAT_MAX),
        "emission_color": ("Emission Color", 4, 0.0, 1.0),
        "emission_strength": ("Emission Strength", 1, 0.0, _BLENDER_FLOAT_MAX),
    }
)


def _bounds(name: str, value: Parameter) -> tuple[tuple, tuple]:
    """Normalize fixed values, tuple ranges and the existing Uniform type."""
    import math

    if name not in PARAMETERS:
        raise ValueError(f"Unknown Principled parameter {name!r}.")
    _, size, minimum, maximum = PARAMETERS[name]
    if isinstance(value, Uniform):
        low, high = value.min.tolist(), value.max.tolist()
    elif size == 1:
        if isinstance(value, Real):
            low = high = value
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            low, high = value
        else:
            raise ValueError(f"{name}: expected a number or (min, max).")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        low, high = value
    else:
        low = high = value

    def vector(item):
        if size == 1:
            item = (item,)
        if not isinstance(item, (tuple, list)):
            raise TypeError(f"{name}: expected {size} components.")
        if size == 4 and len(item) == 3:
            item = (*item, 1.0)
        if len(item) != size or any(
            isinstance(x, bool)
            or not isinstance(x, Real)
            or not math.isfinite(x)
            or not minimum <= x <= maximum
            for x in item
        ):
            raise ValueError(
                f"{name}: expected {size} finite components in "
                f"[{minimum}, {maximum}]."
            )
        return tuple(float(x) for x in item)

    low, high = vector(low), vector(high)
    if any(a > b for a, b in zip(low, high)):
        raise ValueError(f"{name}: minimum must be <= maximum.")
    return low, high


@dataclass(frozen=True)
class MaterialPreset:
    """An immutable collection of fixed values and uniform ranges."""

    name: str
    parameters: Mapping[str, Parameter]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A material preset needs a non-empty name.")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("Preset parameters must be a mapping.")
        normalized = {}
        for name, value in self.parameters.items():
            low, high = _bounds(name, value)
            normalized[name] = (
                (low[0], high[0]) if len(low) == 1 else (low, high)
            )
        object.__setattr__(self, "parameters", MappingProxyType(normalized))

    def with_overrides(self, **parameters: Parameter) -> "MaterialPreset":
        return MaterialPreset(self.name, {**self.parameters, **parameters})

    def sample(self, rng=None) -> dict[str, float | tuple[float, ...]]:
        """Sample in canonical order; omitted RNG uses random.seed()."""
        rng = random if rng is None else rng
        result = {}
        for name in PARAMETERS:
            if name not in self.parameters:
                continue
            low, high = _bounds(name, self.parameters[name])
            values = tuple(
                a if a == b else rng.uniform(a, b) for a, b in zip(low, high)
            )
            result[name] = values[0] if len(values) == 1 else values
        return result


MATERIAL_PRESETS = MappingProxyType(
    {
        "metal": MaterialPreset(
            "metal",
            {
                "base_color": ((0.3, 0.3, 0.3), (0.8, 0.8, 0.8)),
                "metallic": 1.0,
                "roughness": (0.15, 0.45),
            },
        ),
        "plastic": MaterialPreset(
            "plastic",
            {
                "base_color": ((0.05, 0.05, 0.05), (0.8, 0.8, 0.8)),
                "metallic": 0.0,
                "roughness": (0.25, 0.65),
                "ior": 1.46,
            },
        ),
        "rubber": MaterialPreset(
            "rubber",
            {
                "base_color": ((0.01, 0.01, 0.01), (0.08, 0.08, 0.08)),
                "metallic": 0.0,
                "roughness": (0.65, 0.95),
                "specular_ior_level": (0.2, 0.4),
            },
        ),
    }
)


@dataclass(frozen=True, init=False)
class PrincipledMaterial:
    """A procedural material recipe, e.g. ``roughness=(0.2, 0.6)``.

    Colors accept RGB/RGBA or a pair of lower/upper colors in linear space.
    Presets are names from MATERIAL_PRESETS or user-defined MaterialPresets.
    """

    preset: MaterialPreset

    def __init__(
        self,
        preset: str | MaterialPreset | None = None,
        **parameters: Parameter,
    ):
        if isinstance(preset, str):
            if preset not in MATERIAL_PRESETS:
                raise ValueError(
                    f"Unknown material preset {preset!r}; "
                    f"choose from {tuple(MATERIAL_PRESETS)}."
                )
            preset = MATERIAL_PRESETS[preset]
        if preset is None:
            preset = MaterialPreset("principled", {})
        if not isinstance(preset, MaterialPreset):
            raise TypeError("preset must be a name or MaterialPreset.")
        object.__setattr__(self, "preset", preset.with_overrides(**parameters))

    def sample(self, rng=None) -> dict[str, float | tuple[float, ...]]:
        return self.preset.sample(rng)
