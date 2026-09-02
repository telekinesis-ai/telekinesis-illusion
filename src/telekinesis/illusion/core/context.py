"""
Defines the Context class that is the main orchestrator of the synthetic data
generation
"""

import random
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
import blenderproc as bproc  # type: ignore
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping  # type: ignore

from telekinesis.illusion.loader.material_loader import load_ccmaterials
from telekinesis.illusion.types.object import Object
from telekinesis.illusion.types.background import Background
from telekinesis.illusion.types.camera import Camera, CameraConfig
from telekinesis.illusion.utils.assets import (
    pick_default_hdri,
    resolve_asset_dir,
)


# pylint: disable=too-many-instance-attributes
class Context:
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
        bproc.init()

        # Resolve the asset directory once and share it with everything in the
        # context that reads assets
        self._asset_dir = resolve_asset_dir(asset_dir)

        # Initialize internal data structures
        self._objects: Dict[str, Object] = {}
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

    def get_camera(self) -> Camera:
        """
        Get the camera used for rendering.
        """
        return self._camera

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
        return self._visible_object_names

    def set_visible_object_names(
        self, visible_object_names: List[str] | List
    ) -> None:
        """
        Sets the name of the hidden objects.
        """
        self._visible_object_names = visible_object_names

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
        model_path: str,
        object_name: str,
        category_name: str | None = None,
        category_id: int | None = None,
        min_number_instances: int = 1,
        max_number_instances: int = 1,
        active_in_simulation: bool = False,
        collision_shape: str = "CONVEX_HULL",
        scale: float | np.ndarray = 1.0,
        preprocess_model: bool = True,
    ) -> None:
        """
        Add a 3D model to the scene and assign it to a COCO category.

        Object names must be unique within the scene; providing an existing
        object_name raises a ValueError. Category assignment follows a strict
        one-to-one mapping between category_id and category_name
        (COCO-compliant) — see `_resolve_category_id_and_name` for the exact
        rules used to resolve category_id and category_name.

        Args:
            model_path: str
                Path to the 3D model file on disk.
            object_name: str
                Unique name for the object within the scene.
            category_name: str or None
                Name of the category for annotation.
            category_id: int or None
                Optional explicit category ID.
            min_number_instances: int
                Minimum number of instances of this model visible in the
                generated scenes.
            max_number_instances: int
                Maximum number of instances of this model visible in the
                generated scenes.
            scale: float or None
                Scale factor when adding the model.
            preprocess_model: bool
                Whether to preprocess the model upon loading.

        Returns:
            None

        Raises:
            ValueError:
                - If object_name is not unique.
                - If both category_id and category_name are provided but
                conflict with existing mappings.
        """
        logger.info(f"Adding model '{object_name}' to the scene...")

        # Input validation: Ensure that object name prefix doesn't exist as
        # object names need to be unique
        if any(object_name in name for name in self._objects):
            raise ValueError(
                "Redundant model import: Instead of loading the same model "
                "multiple times consider increasing the number of instances "
                "when importing the model."
            )

        # Resolve category_id and category_name
        final_category_name, final_category_id = (
            self._resolve_category_id_and_name(
                object_name=object_name,
                category_name=category_name,
                category_id=category_id,
            )
        )

        # Create the object with the determined category_id and category_name
        object = Object(
            model_path=model_path,
            object_name=object_name,
            category_name=final_category_name,
            category_id=final_category_id,
            active_in_simulation=active_in_simulation,
            collision_shape=collision_shape,
            scale=scale,
            preprocess_model=preprocess_model,
            min_number_instances=min_number_instances,
            max_number_instances=max_number_instances,
        )

        # Add object to the objects dictionary
        self._objects[object.get_name()] = object
        self._object_groups[object_name] = [object]

        # Create multiple instances if needed
        if max_number_instances is not None and max_number_instances > 1:
            # Modify the name of the original object
            for _ in range(max_number_instances - 1):
                # Create linked duplicates
                linked_duplicate = object.create_linked_duplicate()
                # Add object to the objects dictionary
                self._objects[linked_duplicate.get_name()] = linked_duplicate
                self._object_groups[object_name].append(linked_duplicate)

    def randomize_instance_visibility(
        self,
    ) -> None:
        """
        Simple uniform randomization of the instance visibility.
        """
        for base_name, instances in self._object_groups.items():
            min_inst = instances[0].min_number_instances
            max_inst = instances[0].max_number_instances

            # Sample how many should be visible
            num_visible = random.randint(min_inst, max_inst)
            num_to_hide = max_inst - num_visible

            if num_to_hide > 0:
                # Randomly choose which instances to hide
                to_hide = random.sample(instances, num_to_hide)
                for obj in to_hide:
                    obj.hide(True)
                    self._visible_object_names.append(obj.get_name())
                    # Disable the rigid body property of the object to not
                    # participate in the simulation and collision checking
                    obj.disable_rigid_body()


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

    def update_materials(
        self,
        types: List[str] | None = None,
    ) -> None:
        """
        Updates the available materials in the context.
        """
        # Check if materials are already loaded
        # If not, load them
        assets_path = self._asset_dir

        if types:
            # Check which materials are not loaded yet
            missing = [t for t in types if t not in self._materials]
            for material in missing:
                # Load missing material
                materials_directory = assets_path / "materials" / material
                self._materials[material] = load_ccmaterials(
                    str(materials_directory), preload=False
                )
        else:
            # Load all the available materials
            materials_directory = assets_path / "materials"
            for iter_dir in materials_directory.iterdir():
                material_type = iter_dir.name
                self._materials[material_type] = load_ccmaterials(
                    str(iter_dir), preload=False
                )

    def get_random_material(
        self,
        types: List[str] | None = None,
    ) -> None:
        """
        Sample a random material of the types specified in types.

        Args:
            types: List[str] or None:
                A list containing the material types for sampling a random
                material. If 'None' is provided, the material is sampled from
                all the available material types.
        """
        if types:
            random_type = random.choice(types)
        else:
            random_type = random.choice(list(self._materials.keys()))
        return random.choice(self._materials[random_type])
