"""Model specifications for deferred scene registration and loading."""

from dataclasses import dataclass
from pathlib import Path

from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """Frozen specification for a single-mesh model registered with Scene.add.

    Construction does not load or validate the model. Scene.add returns a
    normalized copy; the model is imported when the scene is prepared.

    Attributes:
        model_path: Model file path, relative to the current working directory
            or absolute. Registration resolves it to an absolute Path.
        object_name: Non-empty, unique model name within the scene.
        category_name: Annotation category. If both category fields are omitted,
            object_name is used. "distractor" disables category assignment.
        category_id: Explicit annotation ID. An existing ID supplies its category
            name when category_name is omitted; a new ID uses object_name.
            Omitted IDs reuse the named category or are allocated by the scene.
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

    Example:
        model = ModelAsset("assets/models/mechanical_parts/gearwheel_1.glb",
                           object_name="gear")
    """

    model_path: str | Path
    object_name: str
    category_name: str | None = None
    category_id: int | None = None
    active_in_simulation: bool = False
    collision_shape: str = "CONVEX_HULL"
    scale: ArrayLike = 1.0
    preprocess_model: bool = True
    shading: str = "FLAT"
    material_preprocessing: str = "replace"
    uv_mapping: str = "smart"
