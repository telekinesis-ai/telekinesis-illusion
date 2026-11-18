"""Validated, Blender-independent distributions for procedural materials."""

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

from telekinesis.illusion.types.distribution import Uniform

Scalar: TypeAlias = float | tuple[float, float]
Color: TypeAlias = (
    tuple[float, ...] | tuple[tuple[float, ...], tuple[float, ...]]
)
Parameter: TypeAlias = Scalar | Color | Uniform
_BLENDER_FLOAT_MAX = 3.4028234663852886e38
COLOR_SAMPLING = ("channel", "segment")

# Public parameter -> Blender 4.5 socket, component count, valid bounds.
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
        # Velvet/fibre retro-reflection (foam, cloth) and brushed-metal anisotropy.
        "sheen_weight": ("Sheen Weight", 1, 0.0, 1.0),
        "sheen_roughness": ("Sheen Roughness", 1, 0.0, 1.0),
        "anisotropic": ("Anisotropic", 1, 0.0, 1.0),
        "anisotropic_rotation": ("Anisotropic Rotation", 1, 0.0, 1.0),
    }
)


def _bounds(name: str, value: Parameter) -> tuple[tuple, tuple]:
    """Normalize fixed values, tuple ranges and the existing Uniform type."""
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

    def sample(
        self, rng=None, *, color_sampling: str = "channel"
    ) -> dict[str, float | tuple[float, ...]]:
        """Sample in canonical order; omitted RNG uses random.seed().

        ``color_sampling`` says how a vector range (a colour) is drawn.
        ``"channel"`` draws every component on its own, so a dark-grey to
        light-grey range also yields tinted colours. ``"segment"`` draws one
        position along the line between the two end colours, so the sample
        keeps their hue and only its brightness varies.
        """
        if color_sampling not in COLOR_SAMPLING:
            raise ValueError(
                f"color_sampling must be one of {COLOR_SAMPLING}, "
                f"got {color_sampling!r}."
            )
        rng = random if rng is None else rng
        result = {}
        for name in PARAMETERS:
            if name not in self.parameters:
                continue
            low, high = _bounds(name, self.parameters[name])
            if len(low) > 1 and color_sampling == "segment":
                t = rng.uniform(0.0, 1.0)
                values = tuple(a + t * (b - a) for a, b in zip(low, high))
            else:
                values = tuple(
                    a if a == b else rng.uniform(a, b)
                    for a, b in zip(low, high)
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


@dataclass(frozen=True)
class SurfaceImperfections:
    """Optional mask-based roughness and bump detail on procedural recipes.

    Relative directories are resolved against the context's asset directory.
    Numeric settings accept a fixed value or a uniform (minimum, maximum).
    """

    enabled: bool = True
    directory: str | Path = "surface_imperfections"
    roughness_strength: Scalar = (0.1, 0.3)
    bump_strength: Scalar = (0.05, 0.2)
    bump_distance: Scalar = 0.0001
    scale: Scalar = (1.0, 3.0)

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("surface_imperfections.enabled must be a bool.")
        if not isinstance(self.directory, (str, Path)) or not str(
            self.directory
        ):
            raise ValueError("surface_imperfections.directory needs a path.")
        object.__setattr__(self, "directory", Path(self.directory))
        for name in (
            "roughness_strength",
            "bump_strength",
            "bump_distance",
            "scale",
        ):
            value = getattr(self, name)
            bounds = (value, value) if isinstance(value, Real) else value
            maximum = 1.0 if name.endswith("strength") else _BLENDER_FLOAT_MAX
            if (
                not isinstance(bounds, (tuple, list))
                or len(bounds) != 2
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, Real)
                    or not math.isfinite(x)
                    or not 0 <= x <= maximum
                    for x in bounds
                )
                or bounds[0] > bounds[1]
                or (name == "scale" and bounds[0] == 0)
            ):
                raise ValueError(
                    f"Invalid surface_imperfections.{name}: {value!r}."
                )
            object.__setattr__(self, name, tuple(float(x) for x in bounds))

    def sample(self, rng) -> dict[str, float]:
        return {
            name: low if low == high else rng.uniform(low, high)
            for name in (
                "roughness_strength",
                "bump_strength",
                "bump_distance",
                "scale",
            )
            for low, high in [getattr(self, name)]
        }


@dataclass(frozen=True, init=False)
class PrincipledMaterial:
    """A procedural material recipe, e.g. ``roughness=(0.2, 0.6)``.

    Colors accept RGB/RGBA or a pair of lower/upper colors in linear space.
    Presets are names from MATERIAL_PRESETS or user-defined MaterialPresets.
    ``wear`` (a MaterialWear or its settings) layers dust, edge wear and
    scratches over the sampled colour and roughness; None (default) adds none.
    """

    preset: MaterialPreset
    surface_imperfections: SurfaceImperfections
    color_sampling: str
    wear: "MaterialWear | None"

    def __init__(
        self,
        preset: str | MaterialPreset | None = None,
        *,
        surface_imperfections: bool | SurfaceImperfections | Mapping = False,
        color_sampling: str = "channel",
        wear: "MaterialWear | Mapping | None" = None,
        **parameters: Parameter,
    ):
        if color_sampling not in COLOR_SAMPLING:
            raise ValueError(
                f"color_sampling must be one of {COLOR_SAMPLING}, "
                f"got {color_sampling!r}."
            )
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
        if isinstance(surface_imperfections, bool):
            surface_imperfections = SurfaceImperfections(
                enabled=surface_imperfections
            )
        elif isinstance(surface_imperfections, Mapping):
            surface_imperfections = SurfaceImperfections(
                **surface_imperfections
            )
        if not isinstance(surface_imperfections, SurfaceImperfections):
            raise TypeError(
                "surface_imperfections must be a bool, mapping or SurfaceImperfections."
            )
        object.__setattr__(self, "preset", preset.with_overrides(**parameters))
        object.__setattr__(self, "surface_imperfections", surface_imperfections)
        object.__setattr__(self, "color_sampling", color_sampling)
        object.__setattr__(self, "wear", _wear(wear))

    def sample(self, rng=None) -> dict[str, float | tuple[float, ...]]:
        return self.preset.sample(rng, color_sampling=self.color_sampling)


def _range(name: str, value, size: int, minimum: float) -> tuple[tuple, tuple]:
    """A fixed value or a [low, high] pair, each a number (size 1) or a size-vector."""
    def vec(v):
        if size == 1 and isinstance(v, Real) and not isinstance(v, bool):
            return (float(v),)
        if size > 1 and isinstance(v, (tuple, list)) and len(v) == size and all(
            isinstance(x, Real) and not isinstance(x, bool) for x in v
        ):
            return tuple(float(x) for x in v)
        return None

    fixed = vec(value)
    if fixed is not None:
        low = high = fixed
    elif isinstance(value, (tuple, list)) and len(value) == 2 and vec(value[0]) and vec(value[1]):
        low, high = vec(value[0]), vec(value[1])
    else:
        shape = "a number" if size == 1 else f"{size} numbers"
        raise ValueError(f"{name}: expected {shape} or a [low, high] pair of them, got {value!r}.")
    if any(not math.isfinite(x) or x < minimum for x in low + high):
        raise ValueError(f"{name}: values must be finite and >= {minimum}, got {value!r}.")
    if any(a > b for a, b in zip(low, high)):
        raise ValueError(f"{name}: low must not exceed high, got {value!r}.")
    return low, high


_WEAR_FIELDS = MappingProxyType(
    {
        # name: (default, size, minimum, maximum)
        "dust": (0.0, 1, 0.0, 1.0),
        "dust_color": ((0.30, 0.29, 0.27), 3, 0.0, 1.0),
        "dust_roughness": (0.9, 1, 0.0, 1.0),
        "dust_cavity": (1.0, 1, 0.0, 100.0),
        "dust_up": (0.3, 1, 0.0, 100.0),
        "edge_wear": (0.0, 1, 0.0, 1.0),
        "edge_color": ((0.6, 0.6, 0.6), 3, 0.0, 1.0),
        "edge_roughness": (0.3, 1, 0.0, 1.0),
        "edge_width": (0.002, 1, 1e-5, 1.0),
        "scratches": (0.0, 1, 0.0, 1.0),
        "scratch_scale": (0.004, 1, 1e-5, 10.0),
        "grime_scale": (0.03, 1, 1e-5, 10.0),
        "ao_distance": (0.03, 1, 1e-5, 10.0),
    }
)


@dataclass(frozen=True, init=False)
class MaterialWear:
    """Procedural dust, edge wear and scratches layered over a material.

    Every setting is fixed or a ``(low, high)`` range drawn per material;
    colours are linear RGB (a range is drawn per channel). Lengths are in
    object-space metres.

    * ``dust`` (0-1): strength of a darker, rougher ``dust_color`` /
      ``dust_roughness`` layer. Its mask is a noise of feature size
      ``grime_scale`` times ``dust_cavity`` x cavity (one minus ambient
      occlusion within ``ao_distance``: corners and creases) plus ``dust_up``
      x the up-facing part of the normal (top faces).
    * ``edge_wear`` (0-1): strength of an ``edge_color`` / ``edge_roughness``
      layer on convex edges, found by comparing a Bevel normal of radius
      ``edge_width`` with the surface normal (works on low-poly meshes,
      needs Cycles), kept off concave creases by the ambient occlusion and
      broken up by noise.
    * ``scratches`` (0-1): fine streaks of added roughness, ``scratch_scale``
      wide and 30 times as long, in a random direction.

    Example:
        MaterialWear(dust=(0.2, 0.4), edge_wear=0.3)
    """

    dust: tuple[float, float]
    dust_color: tuple[tuple[float, ...], tuple[float, ...]]
    dust_roughness: tuple[float, float]
    dust_cavity: tuple[float, float]
    dust_up: tuple[float, float]
    edge_wear: tuple[float, float]
    edge_color: tuple[tuple[float, ...], tuple[float, ...]]
    edge_roughness: tuple[float, float]
    edge_width: tuple[float, float]
    scratches: tuple[float, float]
    scratch_scale: tuple[float, float]
    grime_scale: tuple[float, float]
    ao_distance: tuple[float, float]

    def __init__(self, **settings):
        unknown = sorted(set(settings) - set(_WEAR_FIELDS))
        if unknown:
            raise ValueError(
                f"wear: unknown setting(s) {unknown}; choose from {sorted(_WEAR_FIELDS)}."
            )
        for name, (default, size, minimum, maximum) in _WEAR_FIELDS.items():
            low, high = _range(f"wear.{name}", settings.get(name, default), size, minimum)
            if any(x > maximum for x in low + high):
                raise ValueError(
                    f"wear.{name}: values must be <= {maximum}, got {settings[name]!r}."
                )
            object.__setattr__(self, name, (low[0], high[0]) if size == 1 else (low, high))

    def sample(self, rng) -> dict:
        """Draw every setting (colours per channel)."""
        result = {}
        for name, (_, size, _, _) in _WEAR_FIELDS.items():
            low, high = getattr(self, name)
            if size == 1:
                low, high = (low,), (high,)
            values = tuple(a if a == b else rng.uniform(a, b) for a, b in zip(low, high))
            result[name] = values[0] if size == 1 else values
        return result


def _wear(value) -> "MaterialWear | None":
    """None, a MaterialWear or a mapping of its settings."""
    if value is None or isinstance(value, MaterialWear):
        return value
    if isinstance(value, Mapping):
        return MaterialWear(**value)
    raise TypeError(f"wear must be a mapping of wear settings, got {value!r}.")


@dataclass(frozen=True, init=False)
class TexturedMaterial:
    """A registered PBR texture set, re-tiled and tinted per sample.

    ``texture`` is a material group (as for a plain PBR choice). ``uv_scale``
    multiplies the texture coordinates of every map, so 2 tiles the set twice
    per UV unit and makes its grain half as large. ``tint`` multiplies the
    colour map by an RGB factor in linear space, e.g. to bring a scanned wood
    to the colour of a measured table. Both take a fixed value or a
    ``(low, high)`` range sampled per material; ``color_sampling`` draws a tint
    range as for PrincipledMaterial. ``displacement=False`` disconnects the
    set's displacement map, which otherwise moves real geometry.
    ``roughness_scale`` multiplies the roughness (map or value), e.g. to make
    a scanned plastic glossier. ``parameters`` sets Principled inputs as for
    PrincipledMaterial (fixed or ranges), e.g. ``coat_weight`` or
    ``specular_ior_level``, which the PBR loader fixes at 0.333 for every set;
    an input with a map connected loses the map.
    ``uv_rotation`` (degrees) rotates and ``uv_offset`` shifts the texture
    coordinates per sample, so a tiled set neither repeats identically on
    every object nor keeps one grain direction: ``uv_offset=True`` draws a
    shift in [0, 1) per axis, a ``(u, v)`` pair is fixed and a pair of pairs a
    range. ``wear`` (a MaterialWear or its settings) layers dust, edge wear
    and scratches over the colour and roughness. All three default to off.
    """

    texture: str
    uv_scale: tuple[float, float]
    tint: tuple[tuple[float, ...], tuple[float, ...]] | None
    color_sampling: str
    displacement: bool
    roughness_scale: tuple[float, float]
    parameters: "MaterialPreset"
    uv_rotation: tuple[float, float] | None
    uv_offset: tuple[tuple[float, ...], tuple[float, ...]] | None
    wear: "MaterialWear | None"

    def __init__(
        self,
        texture: str,
        *,
        uv_scale: Scalar = 1.0,
        tint: Color | None = None,
        color_sampling: str = "channel",
        displacement: bool = True,
        roughness_scale: Scalar = 1.0,
        parameters: Mapping[str, Parameter] | None = None,
        uv_rotation: Scalar | None = None,
        uv_offset: bool | tuple | None = None,
        wear: "MaterialWear | Mapping | None" = None,
    ):
        if not isinstance(texture, str) or not texture.strip():
            raise ValueError("texture must be a non-empty material group name.")
        if color_sampling not in COLOR_SAMPLING:
            raise ValueError(
                f"color_sampling must be one of {COLOR_SAMPLING}, "
                f"got {color_sampling!r}."
            )
        if not isinstance(displacement, bool):
            raise TypeError("displacement must be a boolean.")
        low, high = _range("uv_scale", uv_scale, 1, 0.0)
        if low[0] <= 0.0:
            raise ValueError(f"uv_scale: must be > 0, got {uv_scale!r}.")
        object.__setattr__(self, "texture", texture)
        object.__setattr__(self, "uv_scale", (low[0], high[0]))
        object.__setattr__(self, "tint", None if tint is None else _range("tint", tint, 3, 0.0))
        object.__setattr__(self, "color_sampling", color_sampling)
        object.__setattr__(self, "displacement", displacement)
        rlow, rhigh = _range("roughness_scale", roughness_scale, 1, 0.0)
        object.__setattr__(self, "roughness_scale", (rlow[0], rhigh[0]))
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping.")
        object.__setattr__(self, "parameters", MaterialPreset("textured", dict(parameters or {})))
        if uv_rotation is not None:
            alow, ahigh = _range("uv_rotation", uv_rotation, 1, -math.inf)
            uv_rotation = (alow[0], ahigh[0])
        object.__setattr__(self, "uv_rotation", uv_rotation)
        if uv_offset is False:
            uv_offset = None
        elif uv_offset is True:
            uv_offset = ((0.0, 0.0), (1.0, 1.0))
        elif uv_offset is not None:
            uv_offset = _range("uv_offset", uv_offset, 2, -math.inf)
        object.__setattr__(self, "uv_offset", uv_offset)
        object.__setattr__(self, "wear", _wear(wear))

    def sample(self, rng=None) -> dict:
        """Draw ``uv_scale``, ``tint`` (None when no tint is set), ``roughness_scale`` and ``parameters``.

        ``uv_rotation`` (radians), ``uv_offset`` and ``wear`` are drawn after
        them and only when set (None otherwise), so a choice without them
        draws exactly what it drew before they existed.
        """
        rng = random if rng is None else rng
        low, high = self.uv_scale
        scale = low if low == high else rng.uniform(low, high)
        rlow, rhigh = self.roughness_scale
        roughness_scale = rlow if rlow == rhigh else rng.uniform(rlow, rhigh)
        tint = None
        if self.tint is not None:
            lo, hi = self.tint
            if self.color_sampling == "segment":
                t = rng.uniform(0.0, 1.0)
                tint = tuple(a + t * (b - a) for a, b in zip(lo, hi))
            else:
                tint = tuple(a if a == b else rng.uniform(a, b) for a, b in zip(lo, hi))
        parameters = self.parameters.sample(rng, color_sampling=self.color_sampling)
        rotation = offset = wear = None
        if self.uv_rotation is not None:
            alow, ahigh = self.uv_rotation
            rotation = math.radians(alow if alow == ahigh else rng.uniform(alow, ahigh))
        if self.uv_offset is not None:
            lo, hi = self.uv_offset
            offset = tuple(a if a == b else rng.uniform(a, b) for a, b in zip(lo, hi))
        if self.wear is not None:
            wear = self.wear.sample(rng)
        return {
            "uv_scale": scale,
            "tint": tint,
            "roughness_scale": roughness_scale,
            "parameters": parameters,
            "uv_rotation": rotation,
            "uv_offset": offset,
            "wear": wear,
        }
