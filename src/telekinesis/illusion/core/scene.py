"""
Owns model definitions, linked instance pools, and current scene activity.
"""

import random
from dataclasses import asdict
from types import MappingProxyType
from telekinesis.illusion.types.model_asset import ModelAsset
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger
from tqdm import tqdm

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported. See blender_env.py for why.
isolate_user_extensions()

# pylint: disable=wrong-import-order
import bpy
from telekinesis.illusion.blender import initialization as initialization
from telekinesis.illusion.blender.labels import LabelIdMapping
from telekinesis.illusion.blender.types.material import Material
from telekinesis.illusion.blender.initialization import clean_temp_dir
from telekinesis.illusion.blender.setup import is_using_external_bpy_module

from telekinesis.illusion.blender.loaders.material_loader import (
    load_ccmaterials,
)
from telekinesis.illusion.types.object import Object
from telekinesis.illusion.types.background import Background
from telekinesis.illusion.types.camera import Camera, CameraConfig
from telekinesis.illusion.types.light import Light
from telekinesis.illusion.utils.assets import (
    pick_default_hdri,
    resolve_asset_dir,
)


# pylint: disable=too-many-instance-attributes
class Scene:
    """
    Class for managing the context for synthetic data generation.
    """

    def __init__(
        self,
        auto_config: bool = True,
        camera_config: CameraConfig = None,
        asset_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize internal states and BlenderProc.

        Args:
            asset_dir: str, Path or None
                Directory holding the models, hdris and materials. If 'None',
                it is resolved by
                telekinesis.illusion.utils.assets.resolve_asset_dir().
        """
        # Intialize BlenderProc
        initialization.init()

        # Resolve the asset directory once and share it with everything in the
        # context that reads assets
        self._asset_dir = resolve_asset_dir(asset_dir)

        # Initialize internal data structures
        self._assets: dict[str, ModelAsset] = {}
        self._prepared = False
        self._closed = False
        self._objects: Dict[str, Object] = {}
        self._lights: dict[str, Light] = {}
        self._object_groups: Dict[str, List[Object]] = {}
        self._categories = LabelIdMapping()
        self._next_category_id = 1
        self._visible_object_names = []

        # Initialize background
        self._background = Background(assets_root=self._asset_dir)

        # Initialize randomizers
        self._pose_randomizer = None
        self._background_randomizer = None

        # Initialize materials
        self._materials = None
        self.material_manager = MaterialManager(asset_dir=self._asset_dir)

        self._camera = Camera(camera_config)

    def get_asset_dir(self) -> Path:
        """
        Get the asset directory the context reads models, hdris and materials
        from.
        """
        return self._asset_dir

    def clean_up(self) -> None:
        """Release scene and temporary resources owned by the context."""
        if self._closed:
            return
        initialization.clean_up()
        self._objects.clear()
        self._object_groups.clear()
        self._visible_object_names.clear()
        self._closed = True
        if is_using_external_bpy_module():
            clean_temp_dir()

    def get_camera(self) -> Camera:
        """
        Get the camera used for rendering.
        """
        return self._camera

    def add_light(
        self, light_name: str, light_type: str = "POINT", **kwargs
    ) -> Light:
        """Create a named light; kwargs are Light pose and property values."""
        if light_name in self._lights:
            raise ValueError(f"Light {light_name!r} already exists.")
        light = Light(light_name, light_type, **kwargs)
        self._lights[light_name] = light
        return light

    def get_lights(self) -> dict[str, Light]:
        """Return lights by their exact registered names, separate from meshes."""
        return self._lights

    def get_light(self, light_name: str) -> Light:
        """Return a registered light; raise ValueError for an unknown name."""
        if light_name not in self._lights:
            raise ValueError(f"No light with the name {light_name!r}.")
        return self._lights[light_name]

    def get_categories(self) -> LabelIdMapping:
        """
        Get the categories in the context.
        """
        return self._categories

    def get_objects(self) -> Dict:
        """
        Get all objects from the context.
        """
        return self._objects

    def get_objects_by_name(
        self,
        name: str,
    ) -> List[Object] | Object:
        """
        Get a specific object from the context by its name.
        If multiple instances of this object are present, a list of all the
        objects is returned.
        """
        matches = [value for key, value in self._objects.items() if name in key]

        if not matches:
            raise ValueError(f"No object with the name {name}")
        if len(matches) == 1:
            return matches[0]  # return single value
        return matches

    def get_object_group(
        self,
        object_name: str,
    ) -> List[Object]:
        """
        Get every instance of a model, keyed by the 'object_name' it was added
        under (i.e. the spec's model name, not the '<name>_INSTANCE_<n>' keys
        used by get_objects()).

        Args:
            object_name: str
                The name the model was registered with in add_model().

        Returns:
            The list of Object instances, or an empty list if the name is
            not registered.
        """
        return self._object_groups.get(object_name, [])

    def get_background(self) -> Background:
        """
        Return the background of the scene.
        """
        return self._background

    def get_materials(self) -> List:
        """
        Return all the materials available in the context.
        """
        return self._materials

    def set_materials(self, material_list: List) -> None:
        """
        Set the available materials in the context.
        """
        self._materials = material_list

    def get_visible_object_names(self) -> List[str]:
        """
        Return the name of the hidden objects in the scene.
        """
        return list(self._visible_object_names)

    def set_visible_object_names(self, visible_object_names: List[str]) -> None:
        """Compatibility setter; keep visibility and physics consistent."""
        names = set(visible_object_names)
        unknown = names - self._objects.keys()
        if unknown:
            raise ValueError(f"Unknown scene objects: {unknown}")
        for name in tuple(self._visible_object_names):
            if name not in names:
                self.deactivate(self._objects[name])
        for name in dict.fromkeys(visible_object_names):
            self.activate(self._objects[name])

    def auto_config(
        self,
    ) -> None:
        """
        Configure the scene for default data generation by picking a default
        background HDRI.
        """
        self._background.set_background_texture(
            str(
                pick_default_hdri(
                    self._asset_dir / "hdris",
                    preferred_category="indoor/industrial",
                )
            ),
        )

    def _get_next_available_category_id(self) -> int:
        """
        Finds the next available category ID by skipping over IDs that are already in use.
        This method modifies self._next_category_id to point to the returned ID.

        Returns:
            The next category ID as int.
        """
        while self._categories.has_id(self._next_category_id):
            self._next_category_id += 1
        return self._next_category_id

    def _resolve_category_id_and_name(
        self,
        object_name: str,
        category_name: str | None,
        category_id: int | None = None,
    ) -> tuple[str, int]:
        """
        Return (final_category_name, final_category_id) following the 4 rules described below:

        1. If no ID and category_name is provided:
            - Use the mandatory object_name as the category_name
            - Check if that category_name already has an ID assigned
            - If no ID is assigned, get the next_id
            - If ID exists, use the existing ID

        2. If category_name is provided:
            - If the name is 'distractor', return 'distractor', None
            - Check if a category already exists with that name
            - Use that category ID and category name

        3. If category_id is provided:
            - Check if the category ID exists
            - Use that ID and respective category name

        4. If both category_id and category_name are provided:
            - Check if that exact combination already exists
            - If it exists, use that combination
            - If the exact combination doesn't exist, but either the ID or the name is already used,
                throw an error saying that combination is already used and only pass either id or name
            - If the combination doesn't exist and neither the ID nor the name are assigned,
                create a new mapping and assign it

        Args:
            object_name: str
                The name of the object to set category id and name.
            category_name: str or None
                The category name for the object.
            category_id: int or None
                The category id for the object.

        Returns:
            A tuple consisting of the category_name and category_id

        Raises:
            ValueError:
                - If a combination of category name and category id is provided
                that contains an already registered category name or category id
                but doesn't match the already registered combination.
        """
        cats = self._categories

        def bump_next(start: int) -> None:
            """Set _next_category_id to the next free id >= start."""
            self._next_category_id = max(self._next_category_id, start)
            while cats.has_id(self._next_category_id):
                self._next_category_id += 1

        def alloc_id() -> int:
            """Get next available id and advance _next_category_id."""
            cid = self._get_next_available_category_id()
            bump_next(cid + 1)
            return cid

        def add_mapping(name: str, cid: int) -> tuple[str, int]:
            cats.add(name, cid)
            bump_next(cid + 1)
            return name, cid

        # Normalize fallback name (rule 1 + part of rule 3)
        if category_name is None and category_id is None:
            category_name = object_name
            logger.info(
                f"No category name provided. Using object name '{object_name}' as category name."
            )

        # Case A: name provided (id optional)
        if category_name is not None:
            if category_name == "distractor":
                logger.info("Skipping category name update for distractor...")
                return "distractor", None
            name_exists = cats.has_label(category_name)
            id_exists = category_id is not None and cats.has_id(category_id)

            # If name exists, it wins unless the user also provided a conflicting id
            if name_exists:
                existing_id = cats.id_from_label(category_name)
                if category_id is not None and category_id != existing_id:
                    raise ValueError(
                        f"Category combination conflict: category name '{category_name}' is already mapped "
                        f"to ID {existing_id}, but you provided category_id={category_id}. "
                        f"Please provide only category_name or category_id, not both."
                    )
                logger.info(
                    f"Category name '{category_name}' already exists with ID {existing_id}. Reusing existing mapping."
                )
                return category_name, existing_id

            # Name doesn't exist
            if category_id is None:
                cid = alloc_id()
                logger.info(
                    f"Created new category '{category_name}' with auto-assigned ID {cid}."
                )
                return add_mapping(category_name, cid)

            # Both provided and name is new:
            if id_exists:
                existing_name = cats.label_from_id(category_id)
                raise ValueError(
                    f"Category combination conflict: category ID {category_id} is already mapped to "
                    f"'{existing_name}', but you provided category_name='{category_name}'. "
                    f"Please provide only category_id or category_name, not both category_id and category_name."
                )

            logger.info(
                f"Created new category '{category_name}' with ID {category_id}."
            )
            return add_mapping(category_name, category_id)

        # Case B: only id provided
        assert category_id is not None  # (since name is None here)

        if cats.has_id(category_id):
            name = cats.label_from_id(category_id)
            logger.info(
                f"Category ID {category_id} already exists with name '{name}'. Reusing existing mapping."
            )
            return name, category_id

        # ID doesn't exist: create mapping using object_name
        name = object_name
        logger.info(
            f"Category ID {category_id} does not exist. Using object name '{object_name}' as category name "
            f"and creating new mapping."
        )

        return add_mapping(name, category_id)

    def add_model(
        self,
        model_path: str | Path,
        object_name: str,
        category_name: str | None = None,
        category_id: int | None = None,
        active_in_simulation: bool = False,
        collision_shape: str = "CONVEX_HULL",
        scale: float | np.ndarray = 1.0,
        preprocess_model: bool = True,
        shading: str = "FLAT",
        material_preprocessing: str = "replace",
        uv_mapping: str = "smart",
    ) -> ModelAsset:
        """Register a single-mesh model without importing or duplicating it.

        Instance ranges belong to ObjectInstanceRandomizer.counts. Register all
        assets before preparation. A new Scene replaces the current bpy scene.
        """
        if self._prepared or self._closed:
            raise RuntimeError("Register models before preparing the scene.")
        if (
            not object_name
            or object_name in self._assets
            or object_name in self._object_groups
        ):
            raise ValueError(
                f"Model name {object_name!r} is empty or already registered."
            )
        path = Path(model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        values = np.asarray(scale, dtype=float)
        if (
            values.shape not in ((), (3,))
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(
                "scale must be a positive scalar or three positive values."
            )
        scale = float(values) if values.ndim == 0 else tuple(map(float, values))
        category_name, category_id = self._resolve_category_id_and_name(
            object_name, category_name, category_id
        )
        asset = ModelAsset(
            path,
            object_name,
            category_name,
            category_id,
            active_in_simulation,
            collision_shape,
            scale,
            preprocess_model,
            shading,
            material_preprocessing,
            uv_mapping,
        )
        self._assets[object_name] = asset
        return asset

    def get_assets(self):
        """Definitions by exact model name; the scene owns this mapping."""
        return MappingProxyType(self._assets)

    def prepare(self, required_instances: dict[str, int]) -> None:
        """Import each requested asset once, then create linked duplicates.

        Repeated preparation reuses the pool. Increasing capacity requires a
        new Scene; there is no hidden rebuilding during dataset generation.
        """
        if self._closed:
            raise RuntimeError(
                "This scene was cleaned up; construct a new Scene."
            )
        for name, count in required_instances.items():
            if name not in self._assets:
                raise ValueError(f"Unknown registered model: {name!r}")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError(
                    f"Capacity for {name!r} must be a nonnegative integer."
                )
        if self._prepared:
            for name, count in required_instances.items():
                if count > len(self._object_groups.get(name, ())):
                    raise ValueError(
                        f"{name!r} now needs {count} instances; build a new Scene."
                    )
            return
        try:
            for name, count in required_instances.items():
                if count == 0:
                    continue
                kwargs = asdict(self._assets[name])
                kwargs["model_path"] = str(kwargs["model_path"])
                if isinstance(kwargs["scale"], tuple):
                    kwargs["scale"] = np.asarray(kwargs["scale"])
                obj = Object(**kwargs)
                self._object_groups[name] = [obj]
                self._objects[obj.get_name()] = obj
                self.deactivate(obj)
                for _ in range(count - 1):
                    duplicate = obj.create_linked_duplicate()
                    self._object_groups[name].append(duplicate)
                    self._objects[duplicate.get_name()] = duplicate
                    self.deactivate(duplicate)
        except Exception:
            # A failed import can leave partial Blender data; don't reuse it.
            self.clean_up()
            raise
        self._prepared = True

    def activate(self, obj: Object) -> None:
        """Make a pooled object participate in rendering and physics."""
        name = obj.get_name()
        if self._objects.get(name) is not obj:
            raise ValueError("Object does not belong to this scene.")
        if name not in self._visible_object_names:
            obj.hide(False)
            obj.enable_rigid_body()
            self._visible_object_names.append(name)

    def deactivate(self, obj: Object) -> None:
        """Hide an object and remove its rigid body and annotation eligibility."""
        obj.hide(True)
        obj.disable_rigid_body()
        name = obj.get_name()
        if name in self._visible_object_names:
            self._visible_object_names.remove(name)

    def begin_sample(self) -> None:
        """Reset object activity before the next sample."""
        for name in tuple(self._visible_object_names):
            self.deactivate(self._objects[name])


class MaterialManager:
    """
    Class for managing the materials in the synthetic data generation context.
    Currently, it implements a simple dictionary for managing different material
    types.
    """

    def __init__(
        self,
        asset_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            asset_dir: str, Path or None
                Directory holding the materials, under a 'materials'
                sub-directory. If 'None', it is resolved by
                telekinesis.illusion.utils.assets.resolve_asset_dir().
        """
        self._materials = {}
        self._asset_dir = resolve_asset_dir(asset_dir)

    def get_asset_dir(self) -> Path:
        """Return the root used for PBR and surface-imperfection assets."""
        return self._asset_dir

    def update_materials(
        self,
        types: Sequence[str] | None = None,
    ) -> None:
        """
        Updates the available materials in the context.
        """
        # Check if materials are already loaded
        # If not, load them
        assets_path = self._asset_dir

        if types:
            # Check which materials are not loaded yet
            missing = list(
                dict.fromkeys(t for t in types if t not in self._materials)
            )
            for material in missing:
                # Load missing material
                materials_directory = assets_path / "materials" / material
                self._materials[material] = load_ccmaterials(
                    str(materials_directory),
                    preload=False,
                    use_all_materials=True,
                )
        else:
            # Load all the available materials
            materials_directory = assets_path / "materials"
            for iter_dir in sorted(materials_directory.iterdir()):
                if not iter_dir.is_dir():
                    continue
                material_type = iter_dir.name
                if material_type in self._materials:
                    continue
                self._materials[material_type] = load_ccmaterials(
                    str(iter_dir), preload=False, use_all_materials=True
                )
        available = types or list(self._materials)
        if not available:
            raise ValueError("No PBR material types are available.")
        for name in available:
            if not self._materials.get(name):
                raise ValueError(
                    f"No PBR materials available for type {name!r}."
                )

    def get_random_material(
        self,
        types: Sequence[str] | None = None,
        rng=None,
    ) -> Material:
        """
        Sample a random material of the types specified in types.

        Args:
            types: List[str] or None:
                A list containing the material types for sampling a random
                material. If 'None' is provided, the material is sampled from
                all the available material types.
        """
        rng = random if rng is None else rng
        available = types or list(self._materials)
        if not available:
            raise ValueError("No PBR material types are available.")
        for name in available:
            if not self._materials.get(name):
                raise ValueError(
                    f"No PBR materials available for type {name!r}."
                )
        random_type = rng.choice(available)
        return rng.choice(self._materials[random_type])
