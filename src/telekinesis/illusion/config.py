"""Typed configuration dataclasses for assets, scenes, randomizers and output.

Every config is an immutable dataclass that validates its values on
construction and round-trips through plain dictionaries with
:meth:`BaseConfig.to_dict` and :meth:`BaseConfig.from_dict`. Each section of a
YAML configuration file is therefore validated when it is loaded, and invalid
values raise ``ValueError`` or ``TypeError`` with the key path of the offending
value, for example ``params.radius_min: must be a number >= 0.0, got -1.``.
:meth:`BaseConfig.replace` returns a validated copy with changed fields.

Distances are in meters. Angles are in radians unless a field states degrees.
This module does not import bpy, so configs can be built and validated
without it.

Attributes:
    LIGHT_TYPES: Supported light types.
    AREA_SHAPES: Supported shapes of AREA lights.
    LIGHT_PROPERTIES: Numeric light properties by public name, mapped to the
        Blender attribute, the supporting light types, and the valid bounds.
    UV_MAPPING_METHODS: UV projections applied during model preprocessing.
    MODEL_ROLES: Roles a model can have in a bin-picking scene. A model
        declares its role as one of its groups.
    COLLISION_SHAPES: Collision shapes for physics and collision checks.
    SHADING_MODES: Shading modes applied during model preprocessing.
    MATERIAL_PREPROCESSING_MODES: How imported material slots are handled
        during model preprocessing.
    MATERIAL_MODES: Material assignment modes of the material randomizer.
    POSE_SAMPLING_STRATEGIES: Strategies for placing parts in a container.
    RENDER_DEVICES: Render devices. GPU devices fall back to the CPU when they
        are unavailable.
    IMAGE_FORMATS: File formats of written color images.
    DATASET_FORMATS: Formats a merged dataset can be exported to.
    POSE_SAMPLER_CONFIGS: Camera and light pose sampler names mapped to the
        config class of their parameters.
"""

import abc
import dataclasses
import math
import re
from collections.abc import Collection, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Self, TypeVar

import numpy as np

from telekinesis.illusion.randomizer import instance_counts
from telekinesis.illusion.types import distribution, material

LIGHT_TYPES = ("POINT", "SUN", "SPOT", "AREA")
AREA_SHAPES = ("SQUARE", "RECTANGLE", "DISK", "ELLIPSE")
_FLOAT_MAX = float(np.finfo(np.float32).max)
LIGHT_PROPERTIES = {
    "power": ("energy", LIGHT_TYPES, 0.0, _FLOAT_MAX),
    "radius": ("shadow_soft_size", ("POINT", "SPOT"), 0.0, _FLOAT_MAX),
    "size": ("size", ("AREA",), 0.0, _FLOAT_MAX),
    "size_y": ("size_y", ("AREA",), 0.0, _FLOAT_MAX),
    "angle": ("angle", ("SUN",), 0.0, math.pi),
    "spot_size": ("spot_size", ("SPOT",), math.pi / 180, math.pi),
    "spot_blend": ("spot_blend", ("SPOT",), 0.0, 1.0),
}
UV_MAPPING_METHODS = ("smart", "cube", "cylinder", "sphere")
MODEL_ROLES = frozenset({"part", "container", "distractor"})
COLLISION_SHAPES = frozenset(
    {
        "BOX",
        "SPHERE",
        "CAPSULE",
        "CYLINDER",
        "CONE",
        "CONVEX_HULL",
        "MESH",
        "COMPOUND",
    }
)
SHADING_MODES = frozenset({"FLAT", "SMOOTH", "AUTO_SMOOTH"})
MATERIAL_PREPROCESSING_MODES = frozenset({"replace", "preserve"})
MATERIAL_MODES = frozenset({"active", "model", "slots"})
POSE_SAMPLING_STRATEGIES = frozenset({"random", "grid"})
RENDER_DEVICES = frozenset({"CPU", "CUDA", "OPTIX", "HIP", "METAL"})
IMAGE_FORMATS = frozenset({"JPEG", "PNG"})
DATASET_FORMATS = frozenset({"coco", "yolo"})

# Matches messages that already start with a key path, such as "radius_min: "
# or "materials[0].preset: ", so nested paths join with a dot.
_KEY_PATH_PATTERN = re.compile(r"[\w.\[\]]+: ")

ConfigType = TypeVar("ConfigType", bound="BaseConfig")


def _to_plain(value: Any) -> Any:
    """Recursively convert configs and containers into plain YAML values.

    Args:
        value: Value to convert.

    Returns:
        The value with configs and mappings as dictionaries, sequences as
        lists, paths as POSIX strings, and uniform ranges as ``min``/``max``
        dictionaries.
    """
    if isinstance(value, BaseConfig):
        return value.to_dict()
    if isinstance(value, distribution.Uniform):
        return {"min": value.min.tolist(), "max": value.max.tolist()}
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def build_config(
    key_path: str, config_type: type[ConfigType], value: Any
) -> ConfigType:
    """Build a config from a mapping, prefixing errors with its key path.

    Args:
        key_path: Location of the value in the configuration, such as
            ``"randomizers.camera_pose"``. It prefixes every error message.
        config_type: Config class to build.
        value: Mapping of config values, or an existing ``config_type``
            instance, which is returned unchanged.

    Returns:
        The validated config.

    Raises:
        TypeError: If the value is neither a mapping nor a ``config_type``
            instance, or if a nested value has the wrong type.
        ValueError: If the mapping has unknown or missing keys or an invalid
            value.
    """
    if isinstance(value, config_type):
        return value
    try:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"expected a mapping for {config_type.__name__}, got "
                f"{type(value).__name__}."
            )
        return config_type.from_dict(value)
    except (TypeError, ValueError) as error:
        separator = "." if _KEY_PATH_PATTERN.match(str(error)) else ": "
        raise type(error)(f"{key_path}{separator}{error}") from error


def _check_int(name: str, value: Any, minimum: int) -> None:
    """Check that a value is an integer of at least ``minimum``.

    Args:
        name: Field name, used in the error message.
        value: Value to check. Booleans are rejected.
        minimum: Smallest allowed value.

    Raises:
        ValueError: If the value is not an integer or is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name}: must be an integer >= {minimum}, got {value!r}."
        )


def _check_number(
    name: str,
    value: Any,
    minimum: float = -math.inf,
    maximum: float = math.inf,
    *,
    exclusive_minimum: bool = False,
) -> None:
    """Check that a value is a finite number within bounds.

    Args:
        name: Field name, used in the error message.
        value: Value to check. Booleans are rejected.
        minimum: Smallest allowed value.
        maximum: Largest allowed value.
        exclusive_minimum: Whether ``minimum`` itself is excluded.

    Raises:
        ValueError: If the value is not a finite number within the bounds.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value > maximum
        or value < minimum
        or (exclusive_minimum and value == minimum)
    ):
        lower = "> " if exclusive_minimum else ">= "
        bounds = [f"{lower}{minimum}"] if minimum > -math.inf else []
        if maximum < math.inf:
            bounds.append(f"<= {maximum}")
        requirement = " and ".join(bounds) or "finite"
        raise ValueError(
            f"{name}: must be a number {requirement}, got {value!r}."
        )


def _check_choice(name: str, value: Any, choices: Collection) -> None:
    """Check that a value is one of the allowed choices.

    Args:
        name: Field name, used in the error message.
        value: Value to check.
        choices: Allowed values.

    Raises:
        ValueError: If the value is not in ``choices``.
    """
    if value not in choices:
        raise ValueError(
            f"{name}: must be one of {sorted(choices)}, got {value!r}."
        )


def _check_ordered(
    name: str,
    lower: Any,
    upper: Any,
    minimum: float = -math.inf,
    maximum: float = math.inf,
) -> None:
    """Check a lower and an upper bound that are stored in separate fields.

    Args:
        name: Common field name prefix, such as ``"radius"`` for
            ``radius_min`` and ``radius_max``.
        lower: Lower bound.
        upper: Upper bound.
        minimum: Smallest allowed value of both bounds.
        maximum: Largest allowed value of both bounds.

    Raises:
        ValueError: If a bound is not a finite number within ``minimum`` and
            ``maximum``, or if ``lower`` is greater than ``upper``.
    """
    _check_number(f"{name}_min", lower, minimum, maximum)
    _check_number(f"{name}_max", upper, minimum, maximum)
    if lower > upper:
        raise ValueError(
            f"{name}_min: must not exceed {name}_max, got {lower} > {upper}."
        )


def _as_interval(
    name: str,
    value: Any,
    minimum: float = -math.inf,
    maximum: float = math.inf,
) -> tuple[float, float]:
    """Validate a ``[lower, upper]`` pair and return it as a float tuple.

    Args:
        name: Field name, used in the error message.
        value: Two numbers with ``lower <= upper``.
        minimum: Smallest allowed value of both bounds.
        maximum: Largest allowed value of both bounds.

    Returns:
        The pair as a tuple of floats.

    Raises:
        ValueError: If the value is not a pair of finite numbers within the
            bounds, or if the lower bound exceeds the upper bound.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name}: must be a [min, max] pair, got {value!r}.")
    if len(value) != 2:
        raise ValueError(f"{name}: must be a [min, max] pair, got {value!r}.")
    for bound in value:
        _check_number(name, bound, minimum, maximum)
    if value[0] > value[1]:
        raise ValueError(
            f"{name}: minimum must not exceed maximum, got {list(value)}."
        )
    return float(value[0]), float(value[1])


def _as_names(name: str, value: Any) -> tuple[str, ...]:
    """Validate a sequence of non-empty strings and return it as a tuple.

    Args:
        name: Field name, used in the error message.
        value: Sequence of non-empty strings. A single string is rejected.

    Returns:
        The strings as a tuple.

    Raises:
        ValueError: If the value is not a sequence of non-empty strings.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name}: must be a list of names, got {value!r}.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(
            f"{name}: must contain only non-empty names, got {list(value)}."
        )
    return tuple(value)


def _as_target(name: str, value: Any) -> tuple[str, ...] | tuple[float, ...]:
    """Validate a sampler target given as model-name prefixes or a point.

    Args:
        name: Field name, used in the error message.
        value: Non-empty list of model-name prefixes, or three coordinates in
            meters.

    Returns:
        The prefixes as a tuple of strings, or the point as a tuple of three
        floats.

    Raises:
        ValueError: If the value is neither a list of names nor a point.
    """
    if (
        not isinstance(value, (str, bytes))
        and isinstance(value, Sequence)
        and value
        and all(isinstance(item, str) for item in value)
    ):
        return _as_names(name, value)
    try:
        return tuple(as_vector3(value, name).tolist())
    except ValueError as error:
        raise ValueError(
            f"{name}: must be a list of model-name prefixes or three "
            f"coordinates, got {value!r}."
        ) from error


def as_vector3(
    value: Any,
    name: str,
    minimum: float = -_FLOAT_MAX,
    maximum: float = _FLOAT_MAX,
) -> np.ndarray:
    """Validate three finite numbers within bounds.

    Args:
        value: Three numbers, such as a location in meters, Euler angles in
            radians, or an RGB color.
        name: Name of the value, used in the error message.
        minimum: Smallest allowed component.
        maximum: Largest allowed component.

    Returns:
        The values as a float array of shape ``(3,)``.

    Raises:
        ValueError: If the value is not three finite numbers within the
            bounds.
    """
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name}: needs three finite values in [{minimum}, {maximum}], "
            f"got {value!r}."
        ) from error
    if (
        result.shape != (3,)
        or not np.isfinite(result).all()
        or np.any(result < minimum)
        or np.any(result > maximum)
    ):
        raise ValueError(
            f"{name}: needs three finite values in [{minimum}, {maximum}], "
            f"got {value!r}."
        )
    return result


def validate_light_properties(
    light_type: str, properties: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate fixed light property values for a light type.

    Args:
        light_type: One of :data:`LIGHT_TYPES`.
        properties: Property values by public name: ``color`` (linear RGB in
            [0, 1]), ``shape`` (one of :data:`AREA_SHAPES`, AREA lights only),
            or a key of :data:`LIGHT_PROPERTIES`.

    Returns:
        The validated values keyed by their Blender light attribute.

    Raises:
        ValueError: If the light type is unknown, or if a property is
            unknown, unsupported by the light type, or out of bounds.
    """
    _check_choice("light_type", light_type, LIGHT_TYPES)
    values = {}
    for name, value in properties.items():
        if name == "color":
            values[name] = as_vector3(value, name, 0.0, 1.0)
        elif name == "shape":
            if light_type != "AREA" or value not in AREA_SHAPES:
                raise ValueError(
                    f"shape: requires an AREA light and one of {AREA_SHAPES}, "
                    f"got {value!r} for a {light_type} light."
                )
            values[name] = value
        else:
            if name not in LIGHT_PROPERTIES:
                raise ValueError(
                    f"{name}: unknown light property; use color, shape or one "
                    f"of {sorted(LIGHT_PROPERTIES)}."
                )
            attribute, types, minimum, maximum = LIGHT_PROPERTIES[name]
            if light_type not in types:
                raise ValueError(
                    f"{name}: not supported by {light_type} lights."
                )
            _check_number(name, value, minimum, maximum)
            values[attribute] = value
    return values


def find_color_texture(path: Path) -> Path | None:
    """Return the color texture of a PBR texture directory.

    Args:
        path: Texture directory. Its color texture is named
            ``<directory-name>_Color`` with a ``.png``, ``.jpg`` or ``.jpeg``
            extension.

    Returns:
        The first existing color texture, or ``None`` if there is none.
    """
    return next(
        (
            candidate
            for suffix in (".png", ".jpg", ".jpeg")
            if (candidate := path / f"{path.name}_Color{suffix}").is_file()
        ),
        None,
    )


@dataclasses.dataclass(frozen=True)
class BaseConfig:
    """Immutable configuration with dictionary conversion and validation.

    Subclasses validate their values on construction, so every instance is
    valid. Sequences are stored as tuples.
    """

    def to_dict(self, keep_none: bool = False) -> dict[str, Any]:
        """Convert the config into a nested dictionary of plain values.

        Nested configs and mappings become dictionaries, sequences become
        lists, paths become POSIX strings, and uniform ranges become
        ``{"min": ..., "max": ...}`` dictionaries, so the result can be
        written with ``yaml.safe_dump``. A field whose value is ``None`` is
        omitted when its default is also ``None``.

        Args:
            keep_none: Whether to keep fields whose value and default are both
                ``None``.

        Returns:
            The config as a dictionary that :meth:`from_dict` accepts.
        """
        result = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is None and field.default is None and not keep_none:
                continue
            result[field.name] = _to_plain(value)
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """Build the config from a dictionary, such as a YAML section.

        Args:
            values: Field values by name. Nested configs may be given as
                dictionaries or as config instances.

        Returns:
            The validated config.

        Raises:
            TypeError: If ``values`` is not a mapping, or if a value has the
                wrong type.
            ValueError: If ``values`` contains unknown keys, lacks a required
                key, or contains an invalid value.
        """
        if not isinstance(values, Mapping):
            raise TypeError(
                f"expected a mapping for {cls.__name__}, got "
                f"{type(values).__name__}."
            )
        fields = {field.name: field for field in dataclasses.fields(cls)}
        unknown = sorted(str(key) for key in values if key not in fields)
        if unknown:
            raise ValueError(
                f"{', '.join(unknown)}: unknown key(s) for {cls.__name__}. "
                f"Valid keys are {sorted(fields)}."
            )
        missing = [
            name
            for name, field in fields.items()
            if field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
            and name not in values
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)}: required key(s) missing for "
                f"{cls.__name__}."
            )
        return cls(**values)

    def replace(self, **changes: Any) -> Self:
        """Return a copy of the config with the given fields replaced.

        Args:
            **changes: New field values by name.

        Returns:
            A new, validated config.

        Raises:
            TypeError: If a field name is unknown or a value has the wrong
                type.
            ValueError: If a value is invalid.
        """
        return dataclasses.replace(self, **changes)


class AssetConfig(BaseConfig, abc.ABC):
    """Local asset description that can be discovered in a directory tree.

    Example:
        from telekinesis.illusion import config

        models = config.ModelConfig.discover("assets/models")
    """

    @classmethod
    def discover(
        cls, root: str | Path, *, group_prefix: str = ""
    ) -> list[Self]:
        """Discover assets in sorted depth-first order beneath a directory.

        Groups come from each asset's parent relative to root, prefixed with
        group_prefix. Recognized directories are not traversed further.
        Material and background assets directly in root need a group_prefix.
        Directory symlinks are not traversed, preventing cycles.

        Args:
            root: Directory to search.
            group_prefix: Slash-separated group prepended to every derived
                group.

        Returns:
            The discovered assets.

        Raises:
            FileNotFoundError: The root is missing or no assets are found.
            NotADirectoryError: The root is not a directory.
            TypeError: group_prefix is not a string.
            ValueError: A group is invalid or a required group is missing.
        """
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        if not root.is_dir():
            raise NotADirectoryError(root)
        if not isinstance(group_prefix, str):
            raise TypeError("group_prefix must be a string.")
        prefix = cls._normalize_groups(group_prefix) if group_prefix else ()
        pending = [root]
        assets = []
        while pending:
            path = pending.pop()
            if cls._recognizes_path(path):
                parents = (
                    () if path == root else path.parent.relative_to(root).parts
                )
                group = "/".join((*prefix[-1:], *parents))
                groups = cls._normalize_groups(group) if group else ()
                assets.append(cls._from_discovered_path(path, groups))
            elif path.is_dir() and not path.is_symlink():
                pending.extend(sorted(path.iterdir(), reverse=True))
        if not assets:
            raise FileNotFoundError(
                f"No {cls.__name__} assets found under '{root}'."
            )
        return assets

    @classmethod
    def _validate_path(cls, path: str | Path) -> Path:
        """Resolve one local path and require this config's asset type.

        Args:
            path: Path to resolve.

        Returns:
            The absolute path.

        Raises:
            FileNotFoundError: The path does not exist.
            ValueError: The path is not a supported asset.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if not cls._recognizes_path(path):
            raise ValueError(f"'{path}' is not a valid {cls.__name__} path.")
        return path

    @staticmethod
    def _normalize_groups(groups: str | Sequence[str]) -> tuple[str, ...]:
        """Validate slash-separated groups and include each ancestor once.

        Args:
            groups: One group or a sequence of groups.

        Returns:
            The groups and their ancestors in order, without duplicates.

        Raises:
            TypeError: The groups are not strings.
            ValueError: A group is empty or not a forward-slash path.
        """
        if isinstance(groups, str):
            groups = (groups,)
        if not isinstance(groups, Sequence):
            raise TypeError(
                "groups: must be a string or a sequence of strings."
            )
        normalized = {}
        for group in groups:
            if not isinstance(group, str):
                raise TypeError("groups: each group must be a string.")
            parts = group.split("/")
            if "\\" in group or any(
                not part.strip() or part in (".", "..") for part in parts
            ):
                raise ValueError(
                    f"groups: invalid group {group!r}; use forward-slash paths."
                )
            for index in range(1, len(parts) + 1):
                normalized["/".join(parts[:index])] = None
        return tuple(normalized)

    @classmethod
    @abc.abstractmethod
    def _recognizes_path(cls, path: Path) -> bool:
        """Return whether a path is a supported asset of this type.

        Called during validation and discovery for every visited path.

        Args:
            path: Absolute path to check.

        Returns:
            Whether the path is one asset of this type.
        """

    @classmethod
    @abc.abstractmethod
    def _from_discovered_path(cls, path: Path, groups: tuple[str, ...]) -> Self:
        """Create the config of one asset found by :meth:`discover`.

        Called after :meth:`_recognizes_path` accepted the path.

        Args:
            path: Absolute path of the asset.
            groups: Groups derived from the asset's parent directories.

        Returns:
            The asset config with default settings.
        """


@dataclasses.dataclass(frozen=True)
class ModelConfig(AssetConfig):
    """Single-mesh model registered with ``Environment.add``.

    Construction does not load the model file. ``Environment.add`` checks that
    the file exists and returns a copy with an absolute path and resolved
    annotation category; the model is imported when the environment is
    prepared. One model may produce multiple linked instances.

    Supported formats are .obj, .ply, .dae (lowercase extensions), and .stl,
    .fbx, .glb, .gltf, .usd, .usda, .usdc (case-insensitive).

    Attributes:
        model_path: Model file path, absolute or relative to the current
            working directory.
        model_name: Non-empty, unique model name within the environment.
        category_name: Annotation category. If both category fields are
            omitted, model_name is used. ``"distractor"`` disables category
            assignment.
        category_id: Explicit non-negative annotation ID. An existing ID
            supplies its category name when category_name is omitted; a new ID
            uses model_name. Omitted IDs reuse the named category or are
            allocated by the environment. Conflicting name/ID pairs are
            rejected during registration.
        active_in_simulation: Whether active instances participate in physics.
        collision_shape: One of :data:`COLLISION_SHAPES`.
        scale: Positive finite scale factor, or three factors for the X, Y and
            Z axes. Stored as a float or a tuple of three floats.
        preprocess_model: Whether to apply material/UV preprocessing and
            shading on import.
        shading: One of :data:`SHADING_MODES`, applied when preprocessing is
            enabled.
        material_preprocessing: ``"replace"`` uses placeholder materials;
            ``"preserve"`` keeps imported material slots. Applied when
            preprocessing is enabled.
        uv_mapping: One of :data:`UV_MAPPING_METHODS`, applied when
            preprocessing is enabled.
        groups: Selection groups, independent of annotation categories. A
            string or sequence is stored as a tuple including slash-separated
            ancestors.

    Example:
        model = ModelConfig(
            "assets/models/mechanical_parts/gearwheel_1.glb", "gear"
        )
    """

    model_path: str | Path
    model_name: str
    category_name: str | None = None
    category_id: int | None = None
    active_in_simulation: bool = False
    collision_shape: str = "CONVEX_HULL"
    scale: float | tuple[float, float, float] = 1.0
    preprocess_model: bool = True
    shading: str = "FLAT"
    material_preprocessing: str = "replace"
    uv_mapping: str = "smart"
    groups: str | Sequence[str] = ()

    def __post_init__(self) -> None:
        """Validate the model settings and normalize scale and groups.

        Raises:
            TypeError: If groups are not strings.
            ValueError: If a setting is invalid.
        """
        if not isinstance(self.model_path, (str, Path)) or not str(
            self.model_path
        ):
            raise ValueError(
                f"model_path: must be a non-empty path, got "
                f"{self.model_path!r}."
            )
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError(
                f"model_name: must be a non-empty string, got "
                f"{self.model_name!r}."
            )
        if self.category_name is not None and (
            not isinstance(self.category_name, str)
            or not self.category_name.strip()
        ):
            raise ValueError(
                f"category_name: must be a non-empty string or null, got "
                f"{self.category_name!r}."
            )
        if self.category_id is not None:
            _check_int("category_id", self.category_id, 0)
        _check_choice("collision_shape", self.collision_shape, COLLISION_SHAPES)
        _check_choice("shading", self.shading, SHADING_MODES)
        _check_choice(
            "material_preprocessing",
            self.material_preprocessing,
            MATERIAL_PREPROCESSING_MODES,
        )
        _check_choice("uv_mapping", self.uv_mapping, UV_MAPPING_METHODS)
        try:
            scale = np.asarray(self.scale, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"scale: must be a positive number or three positive "
                f"numbers, got {self.scale!r}."
            ) from error
        if (
            isinstance(self.scale, bool)
            or scale.shape not in ((), (3,))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)
        ):
            raise ValueError(
                f"scale: must be a positive number or three positive numbers, "
                f"got {self.scale!r}."
            )
        object.__setattr__(
            self,
            "scale",
            float(scale) if scale.ndim == 0 else tuple(map(float, scale)),
        )
        object.__setattr__(self, "groups", self._normalize_groups(self.groups))

    @property
    def path(self) -> str | Path:
        """Return the model file path."""
        return self.model_path

    @property
    def name(self) -> str:
        """Return the model registration name."""
        return self.model_name

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        model_name: str | None = None,
        category_name: str | None = None,
        category_id: int | None = None,
        active_in_simulation: bool = False,
        collision_shape: str = "CONVEX_HULL",
        scale: float | Sequence[float] = 1.0,
        preprocess_model: bool = True,
        shading: str = "FLAT",
        material_preprocessing: str = "replace",
        uv_mapping: str = "smart",
        *,
        groups: str | Sequence[str] = (),
    ) -> Self:
        """Create the config of one existing local model file.

        The path is resolved to an absolute path; model_name defaults to the
        file stem. The remaining arguments have the meanings of the
        corresponding attributes.

        Raises:
            FileNotFoundError: The file does not exist.
            ValueError: The format is unsupported or a setting is invalid.
        """
        path = cls._validate_path(path)
        return cls(
            path,
            path.stem if model_name is None else model_name,
            category_name,
            category_id,
            active_in_simulation,
            collision_shape,
            scale,
            preprocess_model,
            shading,
            material_preprocessing,
            uv_mapping,
            groups,
        )

    @classmethod
    def _recognizes_path(cls, path: Path) -> bool:
        """Recognize the extensions and case rules of the model loader."""
        return path.is_file() and (
            path.suffix in (".obj", ".ply", ".dae")
            or path.suffix.lower()
            in (".stl", ".fbx", ".glb", ".gltf", ".usd", ".usda", ".usdc")
        )

    @classmethod
    def _from_discovered_path(cls, path: Path, groups: tuple[str, ...]) -> Self:
        """Create a model with default import settings and derived groups."""
        return cls(path, path.stem, groups=groups)


@dataclasses.dataclass(frozen=True, init=False)
class MaterialConfig(AssetConfig):
    """PBR texture directory with a unique name and selection groups.

    The directory must contain ``<directory-name>_Color.png``, ``.jpg`` or
    ``.jpeg``; other maps are optional. Construction checks the file names
    without loading textures into Blender. Use :meth:`discover` to derive
    groups from relative parent directories, with group_prefix when starting
    inside a group or at an asset directory.

    Attributes:
        path: Absolute path of the texture directory.
        name: Non-empty registration name.
        groups: At least one selection group. A string or sequence is stored
            as a tuple including slash-separated ancestors, so
            ``"metal/steel"`` gives ``("metal", "metal/steel")``.

    Example:
        material = MaterialConfig.from_path(
            "assets/materials/metal/Metal019_1K-PNG", groups="metal"
        )
    """

    path: Path
    name: str
    groups: tuple[str, ...]

    def __init__(
        self, path: str | Path, name: str, groups: str | Sequence[str]
    ) -> None:
        """Validate one texture directory, its name, and its groups.

        Raises:
            FileNotFoundError: The directory does not exist.
            TypeError: A group is not a string.
            ValueError: The directory has no color texture, the name is empty,
                or no valid group is given.
        """
        path = self._validate_path(path)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"name: must be a non-empty string, got {name!r}.")
        groups = self._normalize_groups(groups)
        if not groups:
            raise ValueError(
                "groups: MaterialConfig requires at least one group; supply "
                "group_prefix when discovering root assets."
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "groups", groups)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        groups: str | Sequence[str],
        name: str | None = None,
    ) -> Self:
        """Create the config of one texture directory.

        The name defaults to the directory name. Raises the errors of the
        constructor.
        """
        path = cls._validate_path(path)
        return cls(path, path.name if name is None else name, groups)

    @classmethod
    def _recognizes_path(cls, path: Path) -> bool:
        """Recognize directories with a color texture before descending."""
        return path.is_dir() and find_color_texture(path) is not None

    @classmethod
    def _from_discovered_path(cls, path: Path, groups: tuple[str, ...]) -> Self:
        """Create a material named after its directory, with derived groups."""
        return cls(path, path.name, groups)


@dataclasses.dataclass(frozen=True, init=False)
class BackgroundConfig(AssetConfig):
    """HDRI file (.hdr or .exr) with a unique name and selection groups.

    Construction checks the extension and that the file exists without
    loading it. Use :meth:`discover` to derive groups from relative parent
    directories, with group_prefix when starting inside a group.

    Attributes:
        path: Absolute path of the HDRI file.
        name: Non-empty registration name.
        groups: At least one selection group. A string or sequence is stored
            as a tuple including slash-separated ancestors, so
            ``"indoor/industrial"`` gives ``("indoor", "indoor/industrial")``.

    Example:
        background = BackgroundConfig.from_path(
            "assets/hdris/indoor/industrial/boiler_room_1k.exr",
            groups="indoor/industrial",
        )
    """

    path: Path
    name: str
    groups: tuple[str, ...]

    def __init__(
        self, path: str | Path, name: str, groups: str | Sequence[str]
    ) -> None:
        """Validate one HDRI file, its name, and its groups.

        Raises:
            FileNotFoundError: The file does not exist.
            TypeError: A group is not a string.
            ValueError: The file is not an HDRI, the name is empty, or no
                valid group is given.
        """
        path = self._validate_path(path)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"name: must be a non-empty string, got {name!r}.")
        groups = self._normalize_groups(groups)
        if not groups:
            raise ValueError(
                "groups: BackgroundConfig requires at least one group; supply "
                "group_prefix when discovering root assets."
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "groups", groups)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        groups: str | Sequence[str],
        name: str | None = None,
    ) -> Self:
        """Create the config of one HDRI file.

        The name defaults to the file stem. Raises the errors of the
        constructor.
        """
        path = cls._validate_path(path)
        return cls(path, path.stem if name is None else name, groups)

    @classmethod
    def _recognizes_path(cls, path: Path) -> bool:
        """Recognize HDR and EXR files during validation and discovery."""
        return path.is_file() and path.suffix.lower() in (".hdr", ".exr")

    @classmethod
    def _from_discovered_path(cls, path: Path, groups: tuple[str, ...]) -> Self:
        """Create an HDRI named after its file stem, with derived groups."""
        return cls(path, path.stem, groups)


@dataclasses.dataclass(frozen=True)
class CameraConfig(BaseConfig):
    """Scene camera registered with ``Environment.add``.

    Fields set to ``None`` keep the current Blender camera setting.

    Attributes:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        focal_length: Focal length in millimeters, at least 1. Mutually
            exclusive with field_of_view.
        field_of_view: Field of view in radians, in (0, pi). Mutually
            exclusive with focal_length.
        clip_start: Near clipping distance in meters.
        clip_end: Far clipping distance in meters, greater than clip_start.
        pixel_aspect_x: Pixel aspect ratio along the x-axis.
        pixel_aspect_y: Pixel aspect ratio along the y-axis.
        shift_x: Horizontal lens shift, relative to the larger image side.
        shift_y: Vertical lens shift, relative to the larger image side.
        location: Initial XYZ location in meters. ``None`` uses a default
            pose.
        rotation: Initial XYZ Euler rotation in degrees. ``None`` uses a
            default pose.

    Example:
        camera_config = CameraConfig(image_width=640, image_height=480)
    """

    image_width: int | None = 720
    image_height: int | None = 720
    focal_length: float | None = None
    field_of_view: float | None = None
    clip_start: float | None = 0.1
    clip_end: float | None = 1000.0
    pixel_aspect_x: float | None = 1.0
    pixel_aspect_y: float | None = 1.0
    shift_x: float | None = 0.0
    shift_y: float | None = 0.0
    location: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        """Validate the camera settings.

        Raises:
            ValueError: If a setting is invalid, or if both focal_length and
                field_of_view are given.
        """
        for name in ("image_width", "image_height"):
            if getattr(self, name) is not None:
                _check_int(name, getattr(self, name), 1)
        if self.focal_length is not None and self.field_of_view is not None:
            raise ValueError(
                "field_of_view: give either focal_length or field_of_view, "
                "not both."
            )
        if self.focal_length is not None:
            _check_number("focal_length", self.focal_length, 1.0)
        if self.field_of_view is not None:
            _check_number(
                "field_of_view",
                self.field_of_view,
                0.0,
                math.pi,
                exclusive_minimum=True,
            )
        if self.clip_start is not None:
            _check_number(
                "clip_start", self.clip_start, 0.0, exclusive_minimum=True
            )
        if self.clip_end is not None:
            _check_number(
                "clip_end",
                self.clip_end,
                self.clip_start or 0.0,
                exclusive_minimum=True,
            )
        for name in ("pixel_aspect_x", "pixel_aspect_y"):
            if getattr(self, name) is not None:
                _check_number(
                    name, getattr(self, name), 0.0, exclusive_minimum=True
                )
        for name in ("shift_x", "shift_y"):
            if getattr(self, name) is not None:
                _check_number(name, getattr(self, name))
        for name in ("location", "rotation"):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self,
                    name,
                    tuple(as_vector3(getattr(self, name), name).tolist()),
                )


@dataclasses.dataclass(frozen=True)
class LightConfig(BaseConfig):
    """Scene light registered with ``Environment.add``.

    Supported properties and units:

    - ``color``: linear RGB in [0, 1]; ``power``: watts (SUN: watts per
      square meter).
    - POINT and SPOT: ``radius`` in meters.
    - AREA: ``shape`` (one of :data:`AREA_SHAPES`), ``size`` and ``size_y`` in
      meters. ``size_y`` only affects RECTANGLE and ELLIPSE.
    - SUN: angular diameter ``angle`` in radians.
    - SPOT: cone angle ``spot_size`` in radians and ``spot_blend`` in [0, 1].

    Attributes:
        light_name: Unique, non-empty name of the light.
        light_type: One of :data:`LIGHT_TYPES`.
        location: XYZ position in meters.
        rotation: XYZ Euler angles in radians. The light emits along its local
            -Z axis.
        properties: Fixed property values by name. Omitted properties keep
            the Blender light defaults.

    Example:
        light_config = LightConfig("key", "POINT", properties={"power": 40.0})
    """

    light_name: str
    light_type: str = "POINT"
    location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    properties: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the light settings.

        Raises:
            ValueError: If the name is empty, or if the type, a property, the
                location or the rotation is invalid.
        """
        if not isinstance(self.light_name, str) or not self.light_name.strip():
            raise ValueError(
                f"light_name: must be a non-empty string, got "
                f"{self.light_name!r}."
            )
        if not isinstance(self.properties, Mapping):
            raise TypeError("properties: must be a mapping.")
        validate_light_properties(self.light_type, self.properties)
        object.__setattr__(self, "properties", dict(self.properties))
        for name in ("location", "rotation"):
            object.__setattr__(
                self,
                name,
                tuple(as_vector3(getattr(self, name), name).tolist()),
            )


@dataclasses.dataclass(frozen=True)
class RenderConfig(BaseConfig):
    """Render settings of the Blender engine.

    Attributes:
        device: One of :data:`RENDER_DEVICES`.
        samples: Maximum number of samples per pixel. ``None`` keeps the
            Blender default.
        noise_threshold: Adaptive sampling noise threshold. ``None`` keeps
            the Blender default.
        max_retries: Number of times a failed render is retried.
        verbose: Whether to print the Blender render output.
        depth: Whether to render a depth pass.
        normals: Whether to render a surface normal pass.

    Example:
        render_config = RenderConfig(device="CPU", samples=16)
    """

    device: str = "CUDA"
    samples: int | None = None
    noise_threshold: float | None = None
    max_retries: int = 2
    verbose: bool = False
    depth: bool = False
    normals: bool = False

    def __post_init__(self) -> None:
        """Validate the render settings.

        Raises:
            ValueError: If a setting is invalid.
        """
        _check_choice("device", self.device, RENDER_DEVICES)
        if self.samples is not None:
            _check_int("samples", self.samples, 1)
        if self.noise_threshold is not None:
            _check_number(
                "noise_threshold",
                self.noise_threshold,
                0.0,
                exclusive_minimum=True,
            )
        _check_int("max_retries", self.max_retries, 0)


@dataclasses.dataclass(frozen=True)
class PhysicsConfig(BaseConfig):
    """Rigid-body simulation that settles active models before rendering.

    The simulation runs for a duration sampled from min_simulation_time_range
    and stops early, at a duration sampled from max_simulation_time_range at
    the latest, once all models have stopped moving.

    Attributes:
        active: Whether to simulate physics for every scene.
        min_simulation_time_range: Range in seconds of the minimum simulated
            duration.
        max_simulation_time_range: Range in seconds of the maximum simulated
            duration.
        check_model_interval: Interval in seconds between checks whether
            models have stopped.
        model_stopped_location_threshold: Largest location change in meters
            between checks for a model to count as stopped.
        model_stopped_rotation_threshold: Largest rotation change in radians
            between checks for a model to count as stopped.
        substeps_per_frame: Simulation substeps per frame.
        solver_iters: Constraint solver iterations per substep.
        verbose: Whether to log simulation progress.
        use_volume_com: Whether to compute centers of mass from the volume
            instead of the surface.

    Example:
        physics_config = PhysicsConfig(active=True)
    """

    active: bool = False
    min_simulation_time_range: tuple[float, float] = (0.5, 1.0)
    max_simulation_time_range: tuple[float, float] = (2.0, 5.0)
    check_model_interval: float = 2.0
    model_stopped_location_threshold: float = 0.01
    model_stopped_rotation_threshold: float = 0.1
    substeps_per_frame: int = 10
    solver_iters: int = 10
    verbose: bool = False
    use_volume_com: bool = False

    def __post_init__(self) -> None:
        """Validate the simulation settings.

        Raises:
            ValueError: If a setting is invalid.
        """
        for name in ("min_simulation_time_range", "max_simulation_time_range"):
            object.__setattr__(
                self, name, _as_interval(name, getattr(self, name), 0.0)
            )
        _check_number(
            "check_model_interval",
            self.check_model_interval,
            0.0,
            exclusive_minimum=True,
        )
        _check_number(
            "model_stopped_location_threshold",
            self.model_stopped_location_threshold,
            0.0,
        )
        _check_number(
            "model_stopped_rotation_threshold",
            self.model_stopped_rotation_threshold,
            0.0,
        )
        _check_int("substeps_per_frame", self.substeps_per_frame, 1)
        _check_int("solver_iters", self.solver_iters, 1)


@dataclasses.dataclass(frozen=True)
class InstanceRandomizerConfig(BaseConfig):
    """Configuration for ``ModelInstanceRandomizer``.

    Each scene activates a sampled number of instances per model. Counts are
    sampled within the per-model bounds so that their sum lies within the
    total bounds.

    Attributes:
        counts: Inclusive ``[min, max]`` number of visible instances, keyed by
            registered model name.
        min_num_total_models: Smallest total number of visible instances.
            ``None`` uses the sum of the per-model minimums.
        max_num_total_models: Largest total number of visible instances.
            ``None`` uses the sum of the per-model maximums.

    Example:
        instance_config = InstanceRandomizerConfig(counts={"gear": (1, 3)})
    """

    counts: dict[str, tuple[int, int]]
    min_num_total_models: int | None = None
    max_num_total_models: int | None = None

    def __post_init__(self) -> None:
        """Validate the per-model and total bounds.

        Raises:
            TypeError: If counts is not a mapping.
            ValueError: If a bound is invalid or the total bounds conflict
                with the per-model bounds.
        """
        if not isinstance(self.counts, Mapping):
            raise TypeError(
                "counts: must map model names to [min, max] bounds."
            )
        counts = {}
        for name, bounds in self.counts.items():
            if isinstance(bounds, (str, bytes)) or not isinstance(
                bounds, Sequence
            ):
                raise ValueError(
                    f"counts.{name}: must be a [min, max] pair, got {bounds!r}."
                )
            counts[name] = tuple(bounds)
        try:
            instance_counts.validate_counts(
                counts, self.min_num_total_models, self.max_num_total_models
            )
        except ValueError as error:
            raise ValueError(f"counts: {error}") from error
        object.__setattr__(self, "counts", counts)


@dataclasses.dataclass(frozen=True)
class ModelPoseRandomizerConfig(BaseConfig):
    """Configuration for ``ModelPoseRandomizer``.

    Attributes:
        max_tries: Attempts to place each model without collision. A model
            that cannot be placed is hidden for the current scene.
        sample_on_surface: Name, or name substring, of the model whose surface
            the poses are sampled on. ``None`` samples without a fixed
            surface.

    Example:
        pose_config = ModelPoseRandomizerConfig(max_tries=20)
    """

    max_tries: int = 10
    sample_on_surface: str | None = None

    def __post_init__(self) -> None:
        """Validate the placement settings.

        Raises:
            ValueError: If a setting is invalid.
        """
        _check_int("max_tries", self.max_tries, 1)
        if self.sample_on_surface is not None and (
            not isinstance(self.sample_on_surface, str)
            or not self.sample_on_surface.strip()
        ):
            raise ValueError(
                f"sample_on_surface: must be a non-empty string or null, got "
                f"{self.sample_on_surface!r}."
            )


@dataclasses.dataclass(frozen=True)
class PrincipledMaterialConfig(BaseConfig):
    """Procedural Principled BSDF material recipe.

    Attributes:
        preset: Name of a material preset such as ``"metal"``. ``None`` starts
            from the plain Principled BSDF defaults.
        parameters: Principled BSDF inputs that override the preset. Each
            value is fixed or a ``[min, max]`` range sampled per material.
            Colors are RGB or RGBA in linear space.
        surface_imperfections: Whether to add surface imperfections, or a
            mapping of imperfection settings.

    Example:
        material_config = PrincipledMaterialConfig(
            "metal", parameters={"roughness": (0.2, 0.5)}
        )
    """

    preset: str | None = None
    parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    surface_imperfections: bool | dict[str, Any] = False

    def __post_init__(self) -> None:
        """Validate the recipe by building the material it describes.

        Raises:
            TypeError: If parameters or surface_imperfections has the wrong
                type.
            ValueError: If the preset, a parameter, or an imperfection setting
                is invalid.
        """
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters: must be a mapping.")
        if not isinstance(self.surface_imperfections, (bool, Mapping)):
            raise TypeError(
                "surface_imperfections: must be a boolean or a mapping."
            )
        object.__setattr__(self, "parameters", dict(self.parameters))
        if isinstance(self.surface_imperfections, Mapping):
            object.__setattr__(
                self, "surface_imperfections", dict(self.surface_imperfections)
            )
        self.to_principled_material()

    def to_principled_material(self) -> material.PrincipledMaterial:
        """Return the procedural material described by this recipe.

        Returns:
            A new material recipe that samples the configured parameters.
        """
        return material.PrincipledMaterial(
            self.preset,
            surface_imperfections=self.surface_imperfections,
            **self.parameters,
        )


@dataclasses.dataclass(frozen=True)
class MaterialRandomizerConfig(BaseConfig):
    """Configuration for ``MaterialRandomizer``.

    Target models are selected by exact registered name and by model group;
    the selection is their union. Without either, all registered models are
    targeted. Materials come either from registered PBR materials, selected
    by groups and names, or from explicit material choices.

    Attributes:
        target_models: Registered model names whose instances receive
            materials.
        target_groups: Model groups whose models receive materials.
        groups: Material groups to sample registered PBR materials from.
        names: Names of registered PBR materials to sample from.
        materials: Explicit material choices: PBR material group names or
            procedural Principled recipes. Cannot be combined with groups or
            names.
        material_slots: ``"all"`` samples every material slot independently;
            a mapping assigns separate choices to slot names or zero-based
            slot indices. Requires mode ``"slots"`` or no mode.
        mode: One of :data:`MATERIAL_MODES`. ``"active"`` replaces the active
            slot, ``"model"`` uses one sample for all slots of a model, and
            ``"slots"`` samples per slot. ``None`` derives the mode from the
            other settings.
        seed: Seed of a private random number generator. ``None`` uses the
            global generator.

    Example:
        material_config = MaterialRandomizerConfig(
            target_groups=("part",), groups=("metal",)
        )
    """

    target_models: tuple[str, ...] | None = None
    target_groups: tuple[str, ...] | None = None
    groups: tuple[str, ...] | None = None
    names: tuple[str, ...] | None = None
    materials: tuple[str | PrincipledMaterialConfig, ...] | None = None
    material_slots: (
        str | dict[str | int, tuple[str | PrincipledMaterialConfig, ...]] | None
    ) = None
    mode: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the selections and coerce material recipes into configs.

        Raises:
            TypeError: If a material choice has the wrong type.
            ValueError: If a setting is invalid or settings conflict.
        """
        for name in ("target_models", "target_groups", "groups", "names"):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self, name, _as_names(name, getattr(self, name))
                )
        if self.materials is not None:
            object.__setattr__(
                self,
                "materials",
                self._as_material_choices("materials", self.materials),
            )
        if isinstance(self.material_slots, Mapping):
            if not self.material_slots:
                raise ValueError("material_slots: must not be empty.")
            object.__setattr__(
                self,
                "material_slots",
                {
                    slot: self._as_material_choices(
                        f"material_slots.{slot}", choices
                    )
                    for slot, choices in self.material_slots.items()
                },
            )
        elif self.material_slots not in (None, "all"):
            raise ValueError(
                f"material_slots: must be 'all' or a mapping of slots to "
                f"material choices, got {self.material_slots!r}."
            )
        if self.mode is not None:
            _check_choice("mode", self.mode, MATERIAL_MODES)
        if self.seed is not None:
            _check_int("seed", self.seed, 0)
        selects_registered = self.groups is not None or self.names is not None
        if selects_registered and self.materials is not None:
            raise ValueError(
                "materials: cannot be combined with groups or names."
            )
        if self.material_slots is not None and self.mode not in (None, "slots"):
            raise ValueError("material_slots: requires mode 'slots'.")
        if isinstance(self.material_slots, Mapping) and (
            self.materials is not None or selects_registered
        ):
            raise ValueError(
                "material_slots: put the material choices inside the slot "
                "mapping instead of materials, groups or names."
            )

    @staticmethod
    def _as_material_choices(
        name: str, choices: Any
    ) -> tuple[str | PrincipledMaterialConfig, ...]:
        """Validate material choices and coerce recipes into configs.

        Args:
            name: Field name, used in error messages.
            choices: Non-empty sequence of PBR group names and Principled
                recipe mappings or configs.

        Returns:
            The choices as a tuple.

        Raises:
            TypeError: If a choice is neither a name nor a recipe.
            ValueError: If the choices are empty or a recipe is invalid.
        """
        if (
            isinstance(choices, (str, bytes))
            or not isinstance(choices, Sequence)
            or not choices
        ):
            raise ValueError(
                f"{name}: must be a non-empty list of material choices."
            )
        result = []
        for index, choice in enumerate(choices):
            if isinstance(choice, str) and choice.strip():
                result.append(choice)
            elif isinstance(choice, (Mapping, PrincipledMaterialConfig)):
                result.append(
                    build_config(
                        f"{name}[{index}]", PrincipledMaterialConfig, choice
                    )
                )
            else:
                raise TypeError(
                    f"{name}[{index}]: must be a material group name or a "
                    f"Principled recipe, got {choice!r}."
                )
        return tuple(result)


@dataclasses.dataclass(frozen=True)
class BackgroundRandomizerConfig(BaseConfig):
    """Configuration for ``BackgroundRandomizer``.

    The eligible HDRIs are the union of the registered backgrounds in groups
    and the registered backgrounds named in names. Without either, every
    registered background is eligible.

    Attributes:
        groups: Background groups, such as ``"indoor/industrial"``.
        names: Names of individual registered backgrounds.

    Example:
        background_config = BackgroundRandomizerConfig(
            groups=("indoor/industrial",)
        )
    """

    groups: tuple[str, ...] | None = None
    names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the selection.

        Raises:
            ValueError: If groups or names are not lists of non-empty names.
        """
        for name in ("groups", "names"):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self, name, _as_names(name, getattr(self, name))
                )


@dataclasses.dataclass(frozen=True)
class LightRandomizerConfig(BaseConfig):
    """Configuration for ``LightRandomizer``.

    Each property is sampled independently for every target light. A property
    value is fixed, a ``{"min": ..., "max": ...}`` range (also for RGB
    colors) sampled uniformly, or for ``shape`` a list of choices. See
    :class:`LightConfig` for the properties and their units.

    Attributes:
        properties: Light property values, ranges or choices by name.
        seed: Seed of a private random number generator. ``None`` uses the
            global generator.

    Example:
        light_randomizer_config = LightRandomizerConfig(
            properties={"power": {"min": 20.0, "max": 60.0}}
        )
    """

    properties: dict[str, Any]
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the properties and coerce ranges into uniform ranges.

        Each value is checked against the light types that support its
        property. Use :meth:`check_light_type` for a concrete light type.

        Raises:
            TypeError: If properties is not a mapping.
            ValueError: If a property is unknown, empty or out of bounds.
        """
        if not isinstance(self.properties, Mapping) or not self.properties:
            raise TypeError(
                "properties: must be a non-empty mapping of light properties."
            )
        properties = {}
        for name, value in self.properties.items():
            if isinstance(value, Mapping):
                if set(value) != {"min", "max"}:
                    raise ValueError(
                        f"properties.{name}: a range needs exactly the keys "
                        f"'min' and 'max', got {sorted(value)}."
                    )
                try:
                    value = distribution.Uniform(value["min"], value["max"])
                except ValueError as error:
                    raise ValueError(f"properties.{name}: {error}") from error
            elif name == "shape" and not isinstance(value, str):
                if not isinstance(value, Sequence) or not value:
                    raise ValueError(
                        f"properties.shape: must be a shape or a non-empty "
                        f"list of shapes from {AREA_SHAPES}."
                    )
                value = tuple(value)
            properties[name] = value
        object.__setattr__(self, "properties", properties)
        for name in self.properties:
            # Check against a light type that supports the property; the
            # concrete type is checked by check_light_type().
            if name == "shape":
                light_type = "AREA"
            elif name in LIGHT_PROPERTIES:
                light_type = LIGHT_PROPERTIES[name][1][0]
            elif name == "color":
                light_type = "POINT"
            else:
                raise ValueError(
                    f"properties.{name}: unknown light property; use color, "
                    f"shape or one of {sorted(LIGHT_PROPERTIES)}."
                )
            self._check_property(name, light_type)
        if self.seed is not None:
            _check_int("seed", self.seed, 0)

    def check_light_type(self, light_type: str) -> None:
        """Check that every value and range bound is valid for a light type.

        Args:
            light_type: One of :data:`LIGHT_TYPES`.

        Raises:
            ValueError: If the light type does not support a property or a
                value is out of bounds.
        """
        for name in self.properties:
            self._check_property(name, light_type)

    def _check_property(self, name: str, light_type: str) -> None:
        """Check every value, range bound or choice of one property.

        Args:
            name: Property name.
            light_type: Light type the values must be valid for.

        Raises:
            ValueError: If a value is invalid for the light type.
        """
        value = self.properties[name]
        if isinstance(value, distribution.Uniform):
            choices = (value.min.tolist(), value.max.tolist())
        elif name == "shape" and isinstance(value, tuple):
            choices = value
        else:
            choices = (value,)
        for choice in choices:
            try:
                validate_light_properties(light_type, {name: choice})
            except ValueError as error:
                raise ValueError(f"properties.{error}") from error


def _sampler_params(
    sampler: str, params: Any
) -> "ShellSamplerConfig | VolumeSamplerConfig":
    """Validate a pose sampler name and coerce its parameters into a config.

    Args:
        sampler: Key of :data:`POSE_SAMPLER_CONFIGS`.
        params: Sampler parameters as a mapping, as the sampler's config, or
            ``None`` for the sampler defaults.

    Returns:
        The sampler parameters as the sampler's config.

    Raises:
        TypeError: If params is not a mapping or the sampler's config.
        ValueError: If the sampler is unknown or a parameter is invalid.
    """
    _check_choice("sampler", sampler, POSE_SAMPLER_CONFIGS)
    params_type = POSE_SAMPLER_CONFIGS[sampler]
    if params is None:
        return params_type()
    if isinstance(params, BaseConfig) and not isinstance(params, params_type):
        raise TypeError(
            f"params: {sampler} requires {params_type.__name__}, got "
            f"{type(params).__name__}."
        )
    return build_config("params", params_type, params)


@dataclasses.dataclass(frozen=True)
class ShellSamplerConfig(BaseConfig):
    """Parameters of ``shell_sampler``, which samples poses on a shell.

    Locations are sampled on a spherical shell around the center and oriented
    to look at it, with a random rotation about the viewing axis.

    Attributes:
        center: Point the shell is centered on: model-name prefixes whose
            visible models give the center, or three coordinates in meters.
            ``None`` uses all visible models.
        radius_min: Smallest distance from the center in meters.
        radius_max: Largest distance from the center in meters.
        elevation_min: Smallest elevation in degrees, in [-90, 90].
        elevation_max: Largest elevation in degrees, in [-90, 90].
        azimuth_min: Smallest azimuth in degrees.
        azimuth_max: Largest azimuth in degrees.
        inplane_rot_min: Smallest rotation about the viewing axis in degrees.
        inplane_rot_max: Largest rotation about the viewing axis in degrees.

    Example:
        shell_config = ShellSamplerConfig(radius_min=1.0, radius_max=1.5)
    """

    center: tuple[str, ...] | tuple[float, float, float] | None = None
    radius_min: float = 0.4
    radius_max: float = 0.6
    elevation_min: float = -90.0
    elevation_max: float = 90.0
    azimuth_min: float = -180.0
    azimuth_max: float = 180.0
    inplane_rot_min: float = -60.0
    inplane_rot_max: float = 60.0

    def __post_init__(self) -> None:
        """Validate the shell parameters.

        Raises:
            ValueError: If a parameter is invalid or a range is reversed.
        """
        if self.center is not None:
            object.__setattr__(
                self, "center", _as_target("center", self.center)
            )
        _check_ordered("radius", self.radius_min, self.radius_max, 0.0)
        _check_ordered(
            "elevation", self.elevation_min, self.elevation_max, -90.0, 90.0
        )
        _check_ordered("azimuth", self.azimuth_min, self.azimuth_max)
        _check_ordered(
            "inplane_rot", self.inplane_rot_min, self.inplane_rot_max
        )


@dataclasses.dataclass(frozen=True)
class VolumeSamplerConfig(BaseConfig):
    """Parameters of ``volume_sampler``, which samples poses in a volume.

    Locations are sampled in an axis-aligned box and oriented to look at a
    point of interest, with a random rotation about the viewing axis. Unset
    values are derived from the visible models.

    Attributes:
        point_of_interst: Point the camera looks at: model-name prefixes whose
            visible models give the point, or three coordinates in meters.
            ``None`` uses all visible models.
        volume_size: Length, width and height of the box in meters.
        distance_range: ``[min, max]`` distance in meters of the box above
            the point of interest.
        volume_center: Center of the box in meters.
        inplane_rot_min: Smallest rotation about the viewing axis in degrees.
        inplane_rot_max: Largest rotation about the viewing axis in degrees.

    Example:
        volume_config = VolumeSamplerConfig(distance_range=(0.5, 1.0))
    """

    point_of_interst: tuple[str, ...] | tuple[float, float, float] | None = None
    volume_size: tuple[float, float, float] | None = None
    distance_range: tuple[float, float] | None = None
    volume_center: tuple[float, float, float] | None = None
    inplane_rot_min: float = -60.0
    inplane_rot_max: float = 60.0

    def __post_init__(self) -> None:
        """Validate the volume parameters.

        Raises:
            ValueError: If a parameter is invalid or a range is reversed.
        """
        if self.point_of_interst is not None:
            object.__setattr__(
                self,
                "point_of_interst",
                _as_target("point_of_interst", self.point_of_interst),
            )
        if self.volume_size is not None:
            size = as_vector3(self.volume_size, "volume_size")
            if np.any(size <= 0):
                raise ValueError(
                    f"volume_size: must contain only positive values, got "
                    f"{list(self.volume_size)}."
                )
            object.__setattr__(self, "volume_size", tuple(size.tolist()))
        if self.distance_range is not None:
            object.__setattr__(
                self,
                "distance_range",
                _as_interval("distance_range", self.distance_range, 0.0),
            )
        if self.volume_center is not None:
            object.__setattr__(
                self,
                "volume_center",
                tuple(as_vector3(self.volume_center, "volume_center").tolist()),
            )
        _check_ordered(
            "inplane_rot", self.inplane_rot_min, self.inplane_rot_max
        )


POSE_SAMPLER_CONFIGS = {
    "shell_sampler": ShellSamplerConfig,
    "volume_sampler": VolumeSamplerConfig,
}


@dataclasses.dataclass(frozen=True)
class CameraPoseRandomizerConfig(BaseConfig):
    """Configuration for ``CameraPoseRandomizer``.

    Attributes:
        sampler: Pose sampler, a key of :data:`POSE_SAMPLER_CONFIGS`.
        number_of_views: Number of camera views rendered per scene.
        params: Parameters of the sampler, given as a mapping or as the
            sampler's config. ``None`` uses the sampler defaults.

    Example:
        camera_pose_config = CameraPoseRandomizerConfig(
            sampler="shell_sampler",
            number_of_views=2,
            params={"radius_min": 0.5, "radius_max": 0.7},
        )
    """

    sampler: str = "volume_sampler"
    number_of_views: int = 1
    params: ShellSamplerConfig | VolumeSamplerConfig | None = None

    def __post_init__(self) -> None:
        """Validate the settings and coerce params into the sampler's config.

        Raises:
            TypeError: If params has the wrong type.
            ValueError: If a setting or parameter is invalid.
        """
        object.__setattr__(
            self, "params", _sampler_params(self.sampler, self.params)
        )
        _check_int("number_of_views", self.number_of_views, 1)


@dataclasses.dataclass(frozen=True)
class LightPoseRandomizerConfig(BaseConfig):
    """Configuration for ``LightPoseRandomizer``.

    Light poses are sampled with a pose sampler and rejected until the
    emitter keeps min_distance from every visible mesh, measured from the
    emitter's enclosing sphere to each mesh's axis-aligned bounding box. SUN
    lights have no finite position and are always accepted.

    Attributes:
        sampler: Pose sampler, a key of :data:`POSE_SAMPLER_CONFIGS`.
        min_distance: Clearance in meters between emitters and meshes.
        max_tries: Attempts to place each light before giving up.
        params: Parameters of the sampler, given as a mapping or as the
            sampler's config. ``None`` uses the sampler defaults.

    Example:
        light_pose_config = LightPoseRandomizerConfig(
            min_distance=0.3, params={"elevation_min": 35.0}
        )
    """

    sampler: str = "shell_sampler"
    min_distance: float = 0.25
    max_tries: int = 100
    params: ShellSamplerConfig | VolumeSamplerConfig | None = None

    def __post_init__(self) -> None:
        """Validate the settings and coerce params into the sampler's config.

        Raises:
            TypeError: If params has the wrong type.
            ValueError: If a setting or parameter is invalid.
        """
        object.__setattr__(
            self, "params", _sampler_params(self.sampler, self.params)
        )
        _check_number("min_distance", self.min_distance, 0.0)
        _check_int("max_tries", self.max_tries, 1)


@dataclasses.dataclass(frozen=True)
class GridSamplingConfig(BaseConfig):
    """Grid of part start positions on a container's top face.

    Attributes:
        rows: Number of grid rows.
        cols: Number of grid columns.
        layers: Number of stacked grid layers.
        layer_spacing: Vertical distance between layers in meters.
        xy_jitter: Random offset of each position, as a fraction of the cell
            size, in [0, 0.5].
        z_rotation_range: ``[min, max]`` rotation about the vertical axis in
            degrees.
        shuffle: Whether to visit the cells in random order.

    Example:
        grid_config = GridSamplingConfig(rows=3, cols=2)
    """

    rows: int = 5
    cols: int = 5
    layers: int = 1
    layer_spacing: float = 0.03
    xy_jitter: float = 0.0
    z_rotation_range: tuple[float, float] = (0.0, 0.0)
    shuffle: bool = True

    def __post_init__(self) -> None:
        """Validate the grid settings.

        Raises:
            ValueError: If a setting is invalid.
        """
        for name in ("rows", "cols", "layers"):
            _check_int(name, getattr(self, name), 1)
        _check_number("layer_spacing", self.layer_spacing, 0.0)
        _check_number("xy_jitter", self.xy_jitter, 0.0, 0.5)
        object.__setattr__(
            self,
            "z_rotation_range",
            _as_interval("z_rotation_range", self.z_rotation_range),
        )

    @property
    def capacity(self) -> int:
        """Number of grid cells, rows times columns times layers."""
        return self.rows * self.cols * self.layers


@dataclasses.dataclass(frozen=True)
class PoseSamplingConfig(BaseConfig):
    """Placement of parts above a container in a bin-picking scene.

    Attributes:
        strategy: One of :data:`POSE_SAMPLING_STRATEGIES`. ``"random"``
            samples positions above the container's top face with random
            rotations; ``"grid"`` uses the cells of grid.
        min_height: Smallest height in meters above the top face.
        max_height: Largest height in meters above the top face.
        face_sample_range: ``[min, max]`` fraction of the top face's extent
            that positions are sampled in, within [0, 1].
        grid: Grid settings, used by the ``"grid"`` strategy.

    Example:
        pose_sampling_config = PoseSamplingConfig(
            strategy="grid", grid={"rows": 3, "cols": 2}
        )
    """

    strategy: str = "random"
    min_height: float = 0.0
    max_height: float = 0.0
    face_sample_range: tuple[float, float] = (0.25, 0.75)
    grid: GridSamplingConfig = dataclasses.field(
        default_factory=GridSamplingConfig
    )

    def __post_init__(self) -> None:
        """Validate the placement settings and coerce grid into a config.

        Raises:
            TypeError: If grid has the wrong type.
            ValueError: If a setting is invalid.
        """
        _check_choice("strategy", self.strategy, POSE_SAMPLING_STRATEGIES)
        _check_ordered("height", self.min_height, self.max_height)
        object.__setattr__(
            self,
            "face_sample_range",
            _as_interval("face_sample_range", self.face_sample_range, 0.0, 1.0),
        )
        object.__setattr__(
            self, "grid", build_config("grid", GridSamplingConfig, self.grid)
        )


@dataclasses.dataclass(frozen=True)
class CocoLicenseConfig(BaseConfig):
    """License entry of written COCO annotation files.

    Attributes:
        id: Unique license ID.
        name: License name.
        url: License URL.

    Example:
        license_config = CocoLicenseConfig(1, "CC BY 4.0")
    """

    id: int
    name: str
    url: str = ""

    def __post_init__(self) -> None:
        """Validate the license values.

        Raises:
            ValueError: If a value is invalid.
        """
        _check_int("id", self.id, 0)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"name: must be a non-empty string, got {self.name!r}."
            )
        if not isinstance(self.url, str):
            raise ValueError(f"url: must be a string, got {self.url!r}.")


@dataclasses.dataclass(frozen=True)
class CocoConfig(BaseConfig):
    """Info and licenses of written COCO annotation files.

    Attributes:
        description: Dataset description.
        url: Dataset or publisher URL.
        version: Dataset version.
        year: Year of creation.
        contributor: Dataset contributor.
        licenses: Licenses listed in the annotations. Empty uses the default
            license.

    Example:
        coco_config = CocoConfig(description="Gears in bins", year=2026)
    """

    description: str = ""
    url: str = ""
    version: str = "1.0.0"
    year: int | None = None
    contributor: str = ""
    licenses: tuple[CocoLicenseConfig, ...] = ()

    def __post_init__(self) -> None:
        """Validate the info values and coerce the licenses.

        Raises:
            TypeError: If a license has the wrong type.
            ValueError: If a value is invalid.
        """
        for name in ("description", "url", "version", "contributor"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(
                    f"{name}: must be a string, got {getattr(self, name)!r}."
                )
        if self.year is not None:
            _check_int("year", self.year, 0)
        if isinstance(self.licenses, (str, bytes)) or not isinstance(
            self.licenses, Sequence
        ):
            raise ValueError("licenses: must be a list of licenses.")
        object.__setattr__(
            self,
            "licenses",
            tuple(
                build_config(f"licenses[{index}]", CocoLicenseConfig, entry)
                for index, entry in enumerate(self.licenses)
            ),
        )


@dataclasses.dataclass(frozen=True)
class ShardConfig(BaseConfig):
    """Split of a generation run into separately written shards.

    Attributes:
        size: Number of images per shard.
        name_template: Shard directory name. ``{date}`` is replaced by the
            start time and ``{uuid}`` by a random identifier.

    Example:
        shard_config = ShardConfig(size=50)
    """

    size: int
    name_template: str = "shard_{date}_{uuid}"

    def __post_init__(self) -> None:
        """Validate the shard settings.

        Raises:
            ValueError: If the size is invalid or the template uses fields
                other than ``{date}`` and ``{uuid}``.
        """
        _check_int("size", self.size, 1)
        try:
            if not self.name_template.format(date="", uuid="").strip():
                raise ValueError
        except (AttributeError, IndexError, KeyError, ValueError) as error:
            raise ValueError(
                f"name_template: must be a non-empty name that only uses "
                f"{{date}} and {{uuid}}, got {self.name_template!r}."
            ) from error


@dataclasses.dataclass(frozen=True)
class WriterConfig(BaseConfig):
    """Written image files, output size limit, and dataset export.

    Attributes:
        image_format: One of :data:`IMAGE_FORMATS`.
        jpg_quality: JPEG quality from 0 to 100.
        max_size_gb: Output size in gigabytes after which generation stops
            at the end of the current shard.
        dataset_format: Format the merged dataset is exported to, one of
            :data:`DATASET_FORMATS`. ``None`` only merges the shards.
        split_ratios: Train, validation and test fractions of the export,
            summing to 1.
        split_seed: Seed of the export split.
        stratify: Whether the export split keeps the category distribution
            in every subset.
        coco: Info and licenses of the written COCO annotations.

    Example:
        writer_config = WriterConfig(image_format="PNG", dataset_format="yolo")
    """

    image_format: str = "JPEG"
    jpg_quality: int = 90
    max_size_gb: float = 10.0
    dataset_format: str | None = None
    split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1)
    split_seed: int = 42
    stratify: bool = True
    coco: CocoConfig = dataclasses.field(default_factory=CocoConfig)

    def __post_init__(self) -> None:
        """Validate the writer settings and coerce the COCO section.

        Raises:
            TypeError: If the COCO section has the wrong type.
            ValueError: If a setting is invalid.
        """
        _check_choice("image_format", self.image_format, IMAGE_FORMATS)
        _check_int("jpg_quality", self.jpg_quality, 0)
        if self.jpg_quality > 100:
            raise ValueError(
                f"jpg_quality: must be at most 100, got {self.jpg_quality}."
            )
        _check_number(
            "max_size_gb", self.max_size_gb, 0.0, exclusive_minimum=True
        )
        if self.dataset_format is not None:
            _check_choice(
                "dataset_format", self.dataset_format, DATASET_FORMATS
            )
        if (
            isinstance(self.split_ratios, (str, bytes))
            or not isinstance(self.split_ratios, Sequence)
            or len(self.split_ratios) != 3
        ):
            raise ValueError(
                f"split_ratios: must be [train, val, test] fractions, got "
                f"{self.split_ratios!r}."
            )
        for ratio in self.split_ratios:
            _check_number("split_ratios", ratio, 0.0, 1.0)
        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ValueError(
                f"split_ratios: must sum to 1, got {list(self.split_ratios)}."
            )
        object.__setattr__(
            self, "split_ratios", tuple(map(float, self.split_ratios))
        )
        _check_int("split_seed", self.split_seed, 0)
        object.__setattr__(
            self, "coco", build_config("coco", CocoConfig, self.coco)
        )


@dataclasses.dataclass(frozen=True)
class DatasetConfig(BaseConfig):
    """Name, size, sharding, asset source and output of a generated dataset.

    Attributes:
        name: Dataset name, used as the output subdirectory.
        num_images: Number of images to generate.
        shard: Split of the generation run into shards.
        asset_directory: Directory of the models, materials and HDRIs. A
            relative path is resolved against the directory of the
            configuration file. ``None`` uses the default asset lookup.
        output_directory: Directory the dataset directory is created in. A
            relative path is resolved against the current working directory.
        writer: Written image files, COCO metadata, output size limit, and
            dataset export.

    Example:
        dataset_config = DatasetConfig(
            "gears", num_images=100, shard=ShardConfig(size=50)
        )
    """

    name: str
    num_images: int
    shard: ShardConfig
    asset_directory: str | Path | None = None
    output_directory: str | Path = "output"
    writer: WriterConfig = dataclasses.field(default_factory=WriterConfig)

    def __post_init__(self) -> None:
        """Validate the dataset settings and coerce nested sections.

        Raises:
            TypeError: If a nested section has the wrong type.
            ValueError: If a setting is invalid.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"name: must be a non-empty string, got {self.name!r}."
            )
        _check_int("num_images", self.num_images, 1)
        object.__setattr__(
            self, "shard", build_config("shard", ShardConfig, self.shard)
        )
        for name in ("asset_directory", "output_directory"):
            value = getattr(self, name)
            if value is None and name == "asset_directory":
                continue
            if not isinstance(value, (str, Path)) or not str(value):
                raise ValueError(
                    f"{name}: must be a non-empty path, got {value!r}."
                )
            object.__setattr__(self, name, Path(value))
        object.__setattr__(
            self, "writer", build_config("writer", WriterConfig, self.writer)
        )
