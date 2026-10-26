"""Model specifications for deferred environment registration and loading."""

from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from numpy.typing import ArrayLike

from telekinesis.illusion.types.asset import Asset


@dataclass(frozen=True, slots=True)
class ModelAsset(Asset):
    """Frozen single-mesh model specification registered with Environment.add.

    Construction does not load or validate the model file. Environment.add
    returns a normalized copy; the model is imported when the environment is
    prepared. One asset may produce multiple linked Model instances.

    Attributes:
        model_path: Model file path, relative to the current working directory
            or absolute. Registration resolves it to an absolute Path.
        model_name: Non-empty, unique model name within the environment.
        category_name: Annotation category. If both category fields are omitted,
            model_name is used. "distractor" disables category assignment.
        category_id: Explicit annotation ID. An existing ID supplies its category
            name when category_name is omitted; a new ID uses model_name.
            Omitted IDs reuse the named category or are allocated by the environment.
            Conflicting name/ID pairs are rejected during registration.
        active_in_simulation: Whether active instances participate in physics.
        collision_shape: Blender collision shape, such as "CONVEX_HULL" or "MESH".
        scale: Positive finite scale factor or three factors for the X, Y and Z
            axes. Registration stores a float or a tuple of three floats.
        preprocess_model: Apply material/UV preprocessing and shading on import.
        shading: "FLAT", "SMOOTH" or "AUTO_SMOOTH" when preprocessing is enabled.
        material_preprocessing: "replace" uses dummy materials; "preserve" keeps
            imported material slots when preprocessing is enabled.
        uv_mapping: "smart", "cube", "cylinder" or "sphere" UV projection used
            when preprocessing is enabled.
        groups: Optional selection groups, independent of annotation categories.
            A string or sequence is stored as a tuple including slash-separated
            ancestors. Existing positional fields retain their order.

    from_path() checks the file format and defaults model_name to its stem.
    discover() recursively creates descriptors with default import settings,
    names from file stems, and groups from relative parent directories.
    Supported formats match the loader: .obj, .ply, .dae (lowercase extensions),
    and .stl, .fbx, .glb, .gltf, .usd, .usda, .usdc (case-insensitive).

    Example:
        model = ModelAsset("assets/models/mechanical_parts/gearwheel_1.glb",
                           model_name="gear")
    """

    model_path: str | Path
    model_name: str
    category_name: str | None = None
    category_id: int | None = None
    active_in_simulation: bool = False
    collision_shape: str = "CONVEX_HULL"
    scale: ArrayLike = 1.0
    preprocess_model: bool = True
    shading: str = "FLAT"
    material_preprocessing: str = "replace"
    uv_mapping: str = "smart"
    groups: str | Sequence[str] = ()

    def __post_init__(self) -> None:
        """Store selection groups as an immutable normalized tuple."""
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
        scale: ArrayLike = 1.0,
        preprocess_model: bool = True,
        shading: str = "FLAT",
        material_preprocessing: str = "replace",
        uv_mapping: str = "smart",
        *,
        groups: str | Sequence[str] = (),
    ) -> Self:
        """Create one supported local model descriptor without importing it.

        The path is resolved to an absolute Path; model_name defaults to its
        stem. Remaining arguments have the constructor's meanings. Missing
        files raise FileNotFoundError; unsupported formats raise ValueError.
        Category and import-setting validation remains with the environment.
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
        """Recognize precisely the extensions and case rules of load_obj."""
        return path.is_file() and (
            path.suffix in (".obj", ".ply", ".dae")
            or path.suffix.lower()
            in (".stl", ".fbx", ".glb", ".gltf", ".usd", ".usda", ".usdc")
        )

    @classmethod
    def _from_discovered_path(cls, path: Path, groups: tuple[str, ...]) -> Self:
        """Construct a model with default import settings and derived groups."""
        return cls(path, path.stem, groups=groups)
