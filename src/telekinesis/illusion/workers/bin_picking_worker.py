"""
Worker class for generating shards for the bin picking use case.
"""

import math
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional
from uuid import uuid4
import yaml

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported. See blender_env.py for why.
isolate_user_extensions()

import bpy  # Required before we can import mathutils
import numpy as np
from mathutils import Matrix
from loguru import logger

from telekinesis.illusion.core.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from telekinesis.illusion.core.context import Context
from telekinesis.illusion.types.camera import CameraConfig
from telekinesis.illusion.types.object import Object
from telekinesis.illusion.sampler.camera_pose_sampler import (
    CAMERA_POSE_SAMPLERS,
)
from telekinesis.illusion.sampler.upper_region_sampler import (
    upper_region_sampler,
)
from telekinesis.illusion.sampler.grid_region_sampler import GridRegionSampler
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.randomizer.randomizer_node import (
    ObjectPoseRandomizer,
    ObjectInstanceRandomizer,
    BackgroundRandomizer,
    MaterialRandomizer,
    CameraPoseRandomizer,
    NodeConfig,
    STAGE_COMPOSITION,
    STAGE_POSE,
    STAGE_APPEARANCE,
    STAGE_CAMERA,
)
from telekinesis.illusion.writer.writer import CocoWriter
from telekinesis.illusion.utils.assets import (
    resolve_asset_dir,
    resolve_asset_path,
)


DEFAULT_CAMERA_POSE_RANDOMIZER_CFG = {
    "sampler": "volume_sampler",
    "number_of_views": 2,
    "params": {
        "distance_range": (1.3, 0.5),
        "inplane_rot_min": -30.0,
        "inplane_rot_max": 30.0,
    },
}

DEFAULT_POSE_SAMPLING_CFG = {
    "strategy": "random",
    "sample_on_surface": None,
    "params": {
        "min_height": 0.0,
        "max_height": 0.0,
        "face_sample_range": [0.25, 0.75],
    },
}

DEFAULT_GRID_CFG = {
    "rows": 5,
    "cols": 5,
    "layers": 1,
    "layer_spacing": 0.03,
    "shuffle": True,
    "xy_jitter": 0.0,
    "z_rotation_range": [0.0, 0.0],
}

DEFAULT_PHYSICS_SIMULATOR_PARAMS = {
    "active": False,
    "min_simulation_time_range": [0.5, 1.0],
    "max_simulation_time_range": [2.0, 5.0],
    "check_object_interval": 0.5,
    "object_stopped_location_threshold": 0.01,
    "object_stopped_rotation_threshold": 1.0,
    "substeps_per_frame": 10,
    "solver_iters": 10,
    "verbose": False,
    "use_volume_com": False,
    "clean_up_scene": False,
}

# Material type tags per supercategory, matched against the sub-directories of
# <asset_dir>/materials. Overridable via the spec's `material_randomizer`
# section; an empty list for a role skips that MaterialRandomizer entirely.
DEFAULT_MATERIAL_RANDOMIZER_CFG = {
    "part": ["metal"],
    "container": ["plastic"],
    "distractor": ["metal"],
}

# HDRI categories, matched against the sub-directories of <asset_dir>/hdris.
# Overridable via the spec's `background_randomizer` section.
DEFAULT_BACKGROUND_RANDOMIZER_CFG = {
    "categories": ["indoor/industrial", "indoor/misc"],
}

# Only 'container' has a default here; 'part' and 'distractor' fall back to the
# top-level min/max_number_visible_models / _distractors keys, which every
# existing spec already sets. Overridable via `instance_randomizer`.
DEFAULT_INSTANCE_RANDOMIZER_CFG = {
    "container": {"min": 1, "max": 1},
}


class BinPickingWorker:
    """
    A wroker class for generating shard for the bin picking use case.
    """

    # The stages randomize_geometry() re-runs. Defined here rather than at the
    # call site so that a future stage (say a lighting randomizer that belongs
    # with the camera) can be folded in without touching the callers.
    GEOMETRY_STAGES = frozenset({STAGE_POSE, STAGE_CAMERA})

    def __init__(
        self,
        spec_file_path: Path,
    ) -> None:
        """
        Initial internal states.

        Args:
            spec_file_path: Path
                The path to the specifications YAML-file.

        """
        self._spec_file_path = Path(spec_file_path)
        self._specs = self._load_specs(self._spec_file_path)

        # A relative metadata.asset_directory is anchored on the spec file, so
        # a spec stays valid no matter which directory it is run from.
        self._asset_dir = resolve_asset_dir(
            self._specs["metadata"].get("asset_directory"),
            base_dir=self._spec_file_path.parent,
        )

        self._num_image = self._specs["metadata"]["num_images"]
        self._num_shards = math.ceil(
            self._num_image / self._specs["shard"]["size"]
        )

        max_size_gb = self._specs.get("output", {}).get("max_size_gb", 10)
        self._max_output_size_bytes = max_size_gb * (1024**3)

        self._grid_sampler = None
        self._parse_tunable_cfg()

        # Prepare configs from spec
        self._camera_config = CameraConfig(**self._specs.get("camera", {}))

        # Create the context
        self._context = Context(
            camera_config=self._camera_config,
            asset_dir=self._asset_dir,
        )

        self._info = self._specs["metadata"]["info"]
        self._licenses = self._specs["metadata"]["licenses"]

        # Add models to context
        self._model_supercatgory_map = {}
        self._id_supercategory_map = {}
        self._add_models_to_context()

        # Add dsitractors to context
        self._distractor_names = []
        self._add_distractors_to_context()

        # Create the randomizer
        self._randomizer = Randomizer()
        self._add_randomizers()

        # Create the writer
        self._output_base = None
        self._writer = CocoWriter(
            create_output_dir_on_init=False,
            info=self._info,
            licenses=self._licenses,
            supercategory_map=self._id_supercategory_map,
        )

        # Create the data generator with context, randomizer and writer
        self._data_generator = SyntheticDataGenerator(
            context=self._context,
            randomizer=self._randomizer,
            writer=self._writer,
        )

        # Create shard merger and arguments for splitter
        self._merger = None
        self._dataset_format = self._specs.get("output", {}).get(
            "dataset_format", None
        )
        self._train_val_tes_ratio = self._specs.get("output", {}).get(
            "train_val_tes_ratio", (0.7, 0.2, 0.1)
        )
        self._splitter_seed = self._specs.get("output", {}).get("seed", 42)
        self._splitter_stratify = self._specs.get("output", {}).get(
            "stratify", True
        )

    def _parse_tunable_cfg(self) -> None:
        """
        Derive the tunable randomizer configs from self._specs.

        Split out of __init__ so apply_spec_updates() can re-derive them from
        an edited spec without rebuilding the worker.
        """
        pose_sampling_cfg = self._specs.get(
            "pose_sampling", DEFAULT_POSE_SAMPLING_CFG
        )
        self._sample_on_surface = pose_sampling_cfg.get(
            "sample_on_surface", DEFAULT_POSE_SAMPLING_CFG["sample_on_surface"]
        )
        self._pose_sampling_strategy = pose_sampling_cfg.get(
            "strategy", DEFAULT_POSE_SAMPLING_CFG["strategy"]
        )
        ps_params = pose_sampling_cfg.get("params", {})
        self._pose_sampling_params = {
            **DEFAULT_POSE_SAMPLING_CFG["params"],
            **{k: v for k, v in ps_params.items() if k != "grid"},
        }
        self._grid_cfg = {
            **DEFAULT_GRID_CFG,
            **ps_params.get("grid", {}),
        }
        # Physics simulation params
        self._physics_simulator_params = self._specs.get(
            "physics_simulator", DEFAULT_PHYSICS_SIMULATOR_PARAMS
        )

        # Randomizer params that used to be hard-coded in _add_randomizers().
        # Merged per key so a spec that overrides only one role still picks up
        # the defaults for the others.
        self._material_cfg = {
            **DEFAULT_MATERIAL_RANDOMIZER_CFG,
            **self._specs.get("material_randomizer", {}),
        }
        self._background_cfg = {
            **DEFAULT_BACKGROUND_RANDOMIZER_CFG,
            **self._specs.get("background_randomizer", {}),
        }
        self._instance_cfg = {
            **DEFAULT_INSTANCE_RANDOMIZER_CFG,
            **self._specs.get("instance_randomizer", {}),
        }

    def get_output_base_directory(self) -> Path:
        """
        Get the output base directory.

        Raises:
            RuntimeError: When the base directory is None

        Returns:
            Path: The base output directory
        """
        if self._output_base:
            return self._output_base
        else:
            raise RuntimeError("self._output_base is None.")

    def get_context(self) -> Context:
        """
        Get the Context built from the spec (models loaded, camera configured).
        """
        return self._context

    def get_randomizer(self) -> Randomizer:
        """
        Get the Randomizer built from the spec.
        """
        return self._randomizer

    def randomize_geometry(self, context: Context) -> None:
        """
        Re-sample object poses and camera poses only.

        Which instances are visible, their materials and the background are all
        left exactly as they are, so the caller can iterate on the layout and
        the framing without the scene changing underneath them. Interactive
        tools (the Blender spec editor's "Preview Scene") use this; a real
        generation run always calls randomize() unfiltered.

        Note that the visible objects are the ones that get re-posed, so the
        caller is responsible for having composed a scene first - on an empty
        scene this is a no-op.

        Args:
            context: Context
                A shared context with assets.
        """
        self._randomizer.randomize(context, stages=self.GEOMETRY_STAGES)

    def _load_specs(self, spec_file_path: Path) -> Dict:
        """
        Helper function to load and print the provided specs.

        Args:
            spec_file_path: Path
                The path to the specifications YAML-file.

        Returns:
            The specifications as a dict.
        """
        with open(spec_file_path, "r", encoding="utf-8") as f:
            specs = yaml.load(f, Loader=yaml.SafeLoader)
        self._print_specs(specs)
        return specs

    def _get_output_size_bytes(self) -> int:
        """Return the total size in bytes of all files under the output directory."""
        if self._output_base is None or not self._output_base.exists():
            return 0
        return sum(
            f.stat().st_size
            for f in self._output_base.rglob("*")
            if f.is_file()
        )

    def _print_specs(self, specs: Dict) -> None:
        """
        Helper function to print the provided specs.

        Args:
            specs: Dict
                A dictionary containing the specifications.
        """
        md = specs["metadata"]
        width = max(len(k) for k in md.keys())
        for k, v in md.items():
            logger.info(f"{k:<{width}} : {v}")

    def _add_models_to_context(self) -> None:
        """
        Add the listed models from the specs to the context.
        """
        model_supercategory_map = defaultdict(list)
        id_supercategory_map = defaultdict(list)
        for model in self._specs["models"]:
            model_path = str(resolve_asset_path(model["path"], self._asset_dir))
            self._context.add_model(
                model_path,
                object_name=model["name"],
                category_name=model["category_name"],
                category_id=model["id"],
                min_number_instances=model["instances"]["min"],
                max_number_instances=model["instances"]["max"],
                active_in_simulation=model["simulation"]["active"],
                collision_shape=model["simulation"]["collision_shape"],
                scale=model["scale"],
                preprocess_model=model["preprocess_model"],
            )
            model_supercategory_map[model["supercategory"]].append(
                model["name"]
            )
            id_supercategory_map[model["id"]] = model["supercategory"]
        # Convert from defaultdict back to normal dict
        self._model_supercatgory_map = dict(model_supercategory_map)
        self._id_supercategory_map = dict(id_supercategory_map)

    def _add_distractors_to_context(self) -> None:
        """
        Add the listed distractors from the specs to the context.
        """
        for distractor in self._specs.get("distractors", []):
            distractor_path = str(
                resolve_asset_path(distractor["path"], self._asset_dir)
            )
            self._context.add_model(
                distractor_path,
                object_name=distractor["name"],
                category_name=distractor["category_name"],
                category_id=distractor["id"],
                min_number_instances=distractor["instances"]["min"],
                max_number_instances=distractor["instances"]["max"],
                active_in_simulation=distractor["simulation"]["active"],
                collision_shape=distractor["simulation"]["collision_shape"],
                scale=distractor["scale"],
                preprocess_model=distractor["preprocess_model"],
            )
            self._distractor_names.append(distractor["name"])
            self._id_supercategory_map[distractor["id"]] = distractor[
                "supercategory"
            ]

    def _instance_bounds(
        self,
        role: str,
        legacy_min_key: Optional[str] = None,
        legacy_max_key: Optional[str] = None,
    ) -> tuple:
        """
        Resolve the instance-count bounds for a supercategory.

        Prefers the spec's `instance_randomizer.<role>` section and falls back
        to the top-level min/max_number_visible_* keys, which every spec
        predating that section already sets.

        Args:
            role: str
                The supercategory to look up ('part', 'container', ...).
            legacy_min_key: Optional[str]
                Top-level spec key holding the lower bound, if any.
            legacy_max_key: Optional[str]
                Top-level spec key holding the upper bound, if any.

        Returns:
            The (min, max) instance counts as a tuple.
        """
        cfg = self._instance_cfg.get(role)
        if cfg:
            return cfg["min"], cfg["max"]
        return self._specs[legacy_min_key], self._specs[legacy_max_key]

    def apply_spec_updates(self, specs: Dict) -> List[str]:
        """
        Re-apply the tunable parts of a spec to the already-built Context and
        Randomizer, without re-importing any models.

        Built for interactive tools (the Blender spec editor) where rebuilding
        the whole worker on every slider drag is far too slow. Everything the
        randomizers read per scene is updated in place; anything that is baked
        into the imported geometry is reported back instead of being silently
        ignored.

        Args:
            specs: Dict
                The edited spec, same schema as the file passed to __init__.

        Returns:
            Human-readable descriptions of changes that could NOT be applied
            and need a full reload. Empty when everything took effect.
        """
        self._specs = specs
        self._parse_tunable_cfg()

        needs_reload = self._apply_model_scales()

        # Instance counts - plain attributes, re-read on every randomize().
        for role, node_name, legacy in (
            (
                "part",
                "instance_randomizer_objects",
                ("min_number_visible_models", "max_number_visible_models"),
            ),
            ("container", "instance_randomizer_containers", None),
            (
                "distractor",
                "instance_randomizer_distractors",
                (
                    "min_number_visible_distractors",
                    "max_number_visible_distractors",
                ),
            ),
        ):
            node = self._randomizer.get_randomizer_node(node_name)
            if node is None:
                continue
            minimum, maximum = self._instance_bounds(role, *(legacy or ()))
            node.min_num_total_objects = minimum
            node.max_num_total_objects = maximum

        # Pose sampling. The 'random' strategy's parameters are read live by
        # the closure, so only the grid geometry and a strategy switch need
        # explicit work here.
        pose_node = self._randomizer.get_randomizer_node(
            "pose_randomizer_models"
        )
        if pose_node is not None:
            is_grid = self._pose_sampling_strategy == "grid"
            if is_grid and self._grid_sampler is not None:
                self._grid_sampler.update_grid_params(
                    rows=self._grid_cfg["rows"],
                    cols=self._grid_cfg["cols"],
                    layers=self._grid_cfg["layers"],
                    layer_spacing=self._grid_cfg["layer_spacing"],
                    face_sample_range=self._pose_sampling_params.get(
                        "face_sample_range", (0.05, 0.95)
                    ),
                    min_height=self._pose_sampling_params.get(
                        "min_height", 0.0
                    ),
                    max_height=self._pose_sampling_params.get(
                        "max_height", 0.0
                    ),
                    shuffle=self._grid_cfg["shuffle"],
                    xy_jitter=self._grid_cfg["xy_jitter"],
                )
            elif is_grid:
                # Switched random -> grid: no sampler exists yet.
                pose_node.pose_sampling_function = self._make_sample_pose_grid()
            else:
                # Switched grid -> random (or still random): the random
                # closure reads its params live, so just make sure it's the
                # one installed.
                pose_node.pose_sampling_function = self._make_sample_pose()

        # Materials: the type list is baked into the context's loaded
        # materials, so a change needs a fresh randomizer.
        for role, node_name in (
            ("part", "material_randomizer_parts"),
            ("container", "material_randomizer_crate"),
            ("distractor", "material_randomizer_distractors"),
        ):
            node = self._randomizer.get_randomizer_node(node_name)
            types = self._material_cfg.get(role)
            if node is None or not types:
                continue
            if list(types) != list(node.types or []):
                self._randomizer.replace_randomizer(
                    node_name,
                    MaterialRandomizer(
                        target_objects=node.target_objects,
                        types=list(types),
                        context=self._context,
                    ),
                )

        # Background: the hdri catalog is built at construction.
        background_node = self._randomizer.get_randomizer_node(
            "background_randomizer"
        )
        categories = self._background_cfg.get("categories")
        if background_node is not None and categories:
            if list(categories) != list(background_node.categories or []):
                self._randomizer.replace_randomizer(
                    "background_randomizer",
                    BackgroundRandomizer(
                        hdris_root=self._asset_dir / "hdris",
                        categories=list(categories),
                    ),
                )

        # Camera: sampler kwargs are captured at construction.
        camera_node = self._randomizer.get_randomizer_node(
            "camera_pose_randomizer"
        )
        if camera_node is not None:
            cpr_cfg = self._specs.get(
                "camera_pose_randomizer", DEFAULT_CAMERA_POSE_RANDOMIZER_CFG
            )
            sampler_name = cpr_cfg.get(
                "sampler", DEFAULT_CAMERA_POSE_RANDOMIZER_CFG["sampler"]
            )
            if sampler_name not in CAMERA_POSE_SAMPLERS:
                raise ValueError(
                    f"Unknown camera pose sampler '{sampler_name}'. "
                    f"Available: {sorted(CAMERA_POSE_SAMPLERS)}"
                )
            self._randomizer.replace_randomizer(
                "camera_pose_randomizer",
                CameraPoseRandomizer(
                    pose_sampling_function=CAMERA_POSE_SAMPLERS[sampler_name],
                    number_of_views=cpr_cfg.get(
                        "number_of_views",
                        DEFAULT_CAMERA_POSE_RANDOMIZER_CFG["number_of_views"],
                    ),
                    **cpr_cfg.get("params", {}),
                ),
            )

        return needs_reload

    def _apply_model_scales(self) -> List[str]:
        """
        Re-scale already-imported models whose spec scale changed.

        Object.__init__ bakes the scale into the mesh vertices and resets the
        object scale to 1, so this applies the *relative* factor to the same
        mesh data. Instances are linked duplicates sharing one mesh datablock,
        which must be transformed exactly once - so this iterates the object
        group and transforms each distinct datablock a single time.

        Returns:
            Descriptions of models that could not be re-scaled.
        """
        problems: List[str] = []

        for entry in self._specs.get("models", []) + self._specs.get(
            "distractors", []
        ):
            name = entry["name"]
            new_scale = entry.get("scale", 1.0)
            group = self._context.get_object_group(name)
            if not group:
                problems.append(
                    f"'{name}' is not loaded - reload assets to add it"
                )
                continue

            old_scale = group[0].get_scale()
            if new_scale == old_scale:
                continue
            if not isinstance(new_scale, (int, float)) or not isinstance(
                old_scale, (int, float)
            ):
                problems.append(
                    f"'{name}' uses a non-uniform scale - reload assets to apply it"
                )
                continue
            if old_scale == 0.0:
                problems.append(
                    f"'{name}' was loaded at scale 0 - reload assets"
                )
                continue

            factor = float(new_scale) / float(old_scale)
            scale_matrix = Matrix.Diagonal((factor, factor, factor)).to_4x4()
            transformed = set()
            for obj in group:
                mesh = obj.get_object().blender_obj.data
                # Linked duplicates share this datablock - transform once, but
                # update every Object's bookkeeping so the next call computes
                # its factor from a fresh baseline.
                if mesh.name not in transformed:
                    mesh.transform(scale_matrix)
                    transformed.add(mesh.name)
                obj.set_scale(new_scale)
            logger.info(
                f"Re-scaled '{name}' by {factor} to {new_scale} "
                f"({len(transformed)} mesh datablock(s))."
            )

        return problems

    def _make_sample_pose(self) -> Callable[[Object], None]:
        """
        Build the 'random' pose-sampling function.

        Reads self._sample_on_surface / self._pose_sampling_params on every
        call rather than capturing them, so apply_spec_updates() can retune
        the sampling box without rebuilding the randomizer.

        Returns:
            A callable taking the Object to place.
        """

        def sample_pose(obj: Object):
            """
            Randomly samples and applies a 6-DoF pose to an object.

            The object's location is sampled uniformly within an axis-aligned box
            centered around the origin.
            """
            if self._sample_on_surface:
                target = [self._sample_on_surface]
            else:
                # Get visible container in the context
                visible = self._context.get_visible_object_names()
                containers = self._model_supercatgory_map["container"]
                target = next(
                    (
                        obj
                        for obj in visible
                        if any(name in obj for name in containers)
                    ),
                    None,
                )
            sample_location = upper_region_sampler(
                self._context,
                object_names_to_sample_on=target,  # type: ignore[arg-type]
                **self._pose_sampling_params,
            )
            obj.set_location(sample_location)
            obj.set_rotation(
                np.random.uniform((-180, -180, -180), (180, 180, 180))
            )

        return sample_pose

    def _make_sample_pose_grid(self) -> Callable[[Object], None]:
        """
        Build the 'grid' pose-sampling function, creating self._grid_sampler.

        The z-rotation range is read from self._grid_cfg per call instead of
        being captured, so apply_spec_updates() can retune it in place - the
        grid geometry itself is retuned through
        GridRegionSampler.update_grid_params().

        Returns:
            A callable taking the Object to place.
        """
        self._grid_sampler = GridRegionSampler(
            context=self._context,
            rows=self._grid_cfg["rows"],
            cols=self._grid_cfg["cols"],
            layers=self._grid_cfg["layers"],
            layer_spacing=self._grid_cfg["layer_spacing"],
            face_sample_range=tuple(
                self._pose_sampling_params.get(
                    "face_sample_range", (0.05, 0.95)
                )
            ),
            min_height=self._pose_sampling_params.get("min_height", 0.0),
            max_height=self._pose_sampling_params.get("max_height", 0.0),
            shuffle=self._grid_cfg["shuffle"],
            xy_jitter=self._grid_cfg["xy_jitter"],
        )

        def sample_pose_grid(obj: Object):
            """
            Place an object at the next grid cell on the bin's top face
            with optional z-axis rotation randomization.
            """
            location = self._grid_sampler.next_location(
                self._model_supercatgory_map["container"]
            )
            obj.set_location(location)
            z_rot_min_deg, z_rot_max_deg = self._grid_cfg["z_rotation_range"]
            z_rot_min, z_rot_max = (
                np.deg2rad(z_rot_min_deg),
                np.deg2rad(z_rot_max_deg),
            )
            z_rot = (
                np.random.uniform(z_rot_min, z_rot_max)
                if z_rot_min != z_rot_max
                else 0.0
            )
            obj.set_rotation(np.array([0.0, 0.0, z_rot]))

        return sample_pose_grid

    def _add_randomizers(
        self,
    ) -> None:
        """
        Define and add the randomizers.
        """
        # Add object instance ranodmizer
        min_parts, max_parts = self._instance_bounds(
            "part", "min_number_visible_models", "max_number_visible_models"
        )
        object_instance_randomizer = ObjectInstanceRandomizer(
            target_objects=self._model_supercatgory_map["part"],
            min_num_total_objects=min_parts,
            max_num_total_objects=max_parts,
        )

        self._randomizer.add_randomizer(
            randomizer_node=object_instance_randomizer,
            node_name="instance_randomizer_objects",
            node_config=NodeConfig(stage=STAGE_COMPOSITION),
        )

        # Add container instance ranodmizer
        min_containers, max_containers = self._instance_bounds("container")
        container_instance_randomizer = ObjectInstanceRandomizer(
            target_objects=self._model_supercatgory_map["container"],
            min_num_total_objects=min_containers,
            max_num_total_objects=max_containers,
        )

        self._randomizer.add_randomizer(
            randomizer_node=container_instance_randomizer,
            node_name="instance_randomizer_containers",
            node_config=NodeConfig(stage=STAGE_COMPOSITION),
        )

        # Add model pose randomizer
        # The target objects are all the models activly participating in the
        # simulation
        sample_pose = self._make_sample_pose()

        if self._pose_sampling_strategy == "grid":
            sample_pose_grid = self._make_sample_pose_grid()

            def resolve_container_surface(context: Context) -> str:
                visible = context.get_visible_object_names()
                containers = self._model_supercatgory_map["container"]
                match = next(
                    (
                        name
                        for name in visible
                        if any(c in name for c in containers)
                    ),
                    None,
                )
                if match is None:
                    raise RuntimeError(
                        "No visible container found for on-surface "
                        "grid sampling."
                    )
                return match

            object_pose_randomizer = ObjectPoseRandomizer(
                pose_sampling_function=sample_pose_grid,
                target_objects=self._model_supercatgory_map["part"],
                surface_name_resolver=resolve_container_surface,
            )
        elif self._pose_sampling_strategy == "random":
            object_pose_randomizer = ObjectPoseRandomizer(
                pose_sampling_function=sample_pose,
                target_objects=self._model_supercatgory_map["part"],
                sample_on_surface=self._sample_on_surface,
            )
        else:
            raise ValueError(
                f"Unknown pose_sampling.strategy "
                f"'{self._pose_sampling_strategy}'. Expected 'random' or 'grid'."
            )

        self._randomizer.add_randomizer(
            randomizer_node=object_pose_randomizer,
            node_name="pose_randomizer_models",
            node_config=NodeConfig(stage=STAGE_POSE),
        )

        if self._distractor_names:
            min_distractors, max_distractors = self._instance_bounds(
                "distractor",
                "min_number_visible_distractors",
                "max_number_visible_distractors",
            )
            distractor_instance_randomizer = ObjectInstanceRandomizer(
                target_objects=self._distractor_names,
                min_num_total_objects=min_distractors,
                max_num_total_objects=max_distractors,
            )
            self._randomizer.add_randomizer(
                randomizer_node=distractor_instance_randomizer,
                node_name="instance_randomizer_distractors",
                node_config=NodeConfig(stage=STAGE_COMPOSITION),
            )

            distractor_pose_randomizer = ObjectPoseRandomizer(
                pose_sampling_function=sample_pose,
                target_objects=self._distractor_names,
            )
            self._randomizer.add_randomizer(
                randomizer_node=distractor_pose_randomizer,
                node_name="pose_randomizer_distractors",
                node_config=NodeConfig(stage=STAGE_POSE),
            )

        # Add matrial randomizer
        # An empty type list means "do not randomize the material for this
        # role" - MaterialRandomizer would otherwise load no materials and
        # fail later in get_random_material().
        def add_material_randomizer(
            role: str, target_objects, node_name: str
        ) -> None:
            types = self._material_cfg.get(role)
            if not types:
                logger.info(
                    f"No material types configured for '{role}' - "
                    f"skipping {node_name}."
                )
                return
            self._randomizer.add_randomizer(
                randomizer_node=MaterialRandomizer(
                    target_objects=target_objects,
                    types=types,
                    context=self._context,
                ),
                node_name=node_name,
                node_config=NodeConfig(stage=STAGE_APPEARANCE),
            )

        add_material_randomizer(
            "part",
            self._model_supercatgory_map["part"],
            "material_randomizer_parts",
        )
        add_material_randomizer(
            "container",
            self._model_supercatgory_map["container"],
            "material_randomizer_crate",
        )
        if self._distractor_names:
            add_material_randomizer(
                "distractor",
                self._distractor_names,
                "material_randomizer_distractors",
            )

        # Add background randomizer
        background_categories = self._background_cfg.get("categories")
        if background_categories:
            background_randomizer = BackgroundRandomizer(
                hdris_root=self._asset_dir / "hdris",
                categories=background_categories,
            )

            self._randomizer.add_randomizer(
                randomizer_node=background_randomizer,
                node_name="background_randomizer",
                node_config=NodeConfig(stage=STAGE_APPEARANCE),
            )
        else:
            logger.info(
                "No background categories configured - "
                "skipping background_randomizer."
            )

        # Add camera pose randomizer
        cpr_cfg = self._specs.get(
            "camera_pose_randomizer", DEFAULT_CAMERA_POSE_RANDOMIZER_CFG
        )
        sampler_name = cpr_cfg.get(
            "sampler", DEFAULT_CAMERA_POSE_RANDOMIZER_CFG["sampler"]
        )
        if sampler_name not in CAMERA_POSE_SAMPLERS:
            raise ValueError(
                f"Unknown camera pose sampler '{sampler_name}'. "
                f"Available: {sorted(CAMERA_POSE_SAMPLERS)}"
            )
        camera_pose_randomizer = CameraPoseRandomizer(
            pose_sampling_function=CAMERA_POSE_SAMPLERS[sampler_name],
            number_of_views=cpr_cfg.get(
                "number_of_views",
                DEFAULT_CAMERA_POSE_RANDOMIZER_CFG["number_of_views"],
            ),
            **cpr_cfg.get("params", {}),
        )

        self._randomizer.add_randomizer(
            randomizer_node=camera_pose_randomizer,
            node_name="camera_pose_randomizer",
            node_config=NodeConfig(stage=STAGE_CAMERA),
        )

    def generate(self) -> None:
        """
        Generate shards based on the provided specs.

        Each shard is written to its own output directory whose name includes a
        UUID so that multiple workers can run simultaneously without collisions.
        """
        shard_size = self._specs["shard"]["size"]
        remaining_images = self._num_image
        output_cfg = self._specs.get("output", {})
        name_template = output_cfg.get(
            "shard_name_template", "shard_{date}_{uuid}"
        )

        dataset_name = self._specs["metadata"]["dataset_name"]
        base_output_directory = self._specs["metadata"]["base_output_directory"]
        if base_output_directory:
            self._output_base = Path(base_output_directory) / dataset_name
        else:
            self._output_base = Path.cwd() / "output" / dataset_name

        for i in range(self._num_shards):
            current_shard_size = min(shard_size, remaining_images)
            shard_name = name_template.format(
                date=datetime.now().strftime("%Y%m%d_%H%M%S"),
                uuid=uuid4().hex[:8],
            )
            shard_dir = self._output_base / shard_name
            self._writer.set_output_dir(shard_dir)

            logger.info(
                f"[shard {i + 1}/{self._num_shards}] "
                f"Generating {current_shard_size} scenes -> {shard_dir.name}"
            )

            self._data_generator.generate(
                num_images=current_shard_size,
                simulate_physics=self._physics_simulator_params["active"],
                min_simulation_time_range=tuple(
                    self._physics_simulator_params["min_simulation_time_range"]
                ),
                max_simulation_time_range=tuple(
                    self._physics_simulator_params["max_simulation_time_range"]
                ),
                check_object_interval=self._physics_simulator_params[
                    "check_object_interval"
                ],
                object_stopped_location_threshold=self._physics_simulator_params[
                    "object_stopped_location_threshold"
                ],
                object_stopped_rotation_threshold=self._physics_simulator_params[
                    "object_stopped_rotation_threshold"
                ],
                substeps_per_frame=self._physics_simulator_params[
                    "substeps_per_frame"
                ],
                solver_iters=self._physics_simulator_params["solver_iters"],
                verbose=self._physics_simulator_params["verbose"],
                use_volume_com=self._physics_simulator_params["use_volume_com"],
                clean_up_scene=self._physics_simulator_params["clean_up_scene"],
            )
            remaining_images -= current_shard_size

            output_size = self._get_output_size_bytes()
            logger.info(
                f"[shard {i + 1}/{self._num_shards}] "
                f"Current output size: {output_size / (1024**3):.2f} GB "
                f"(limit: {self._max_output_size_bytes / (1024**3):.1f} GB)"
            )
            if output_size >= self._max_output_size_bytes:
                logger.warning(
                    f"Output size limit reached "
                    f"({output_size / (1024**3):.2f} GB >= "
                    f"{self._max_output_size_bytes / (1024**3):.1f} GB). "
                    f"Stopping after shard {i + 1}/{self._num_shards}."
                )
                break

        self._data_generator.clean_up()

    def merge_shards(self, preview: bool = True) -> None:
        """
        Merge the generated shards.

        Args:
            preview (bool):
                Whether the preview the merged dataset.
        """
        # Deferred: pycocotools (and, via CocoShardMerger.preview(), tkinter)
        # aren't available in every Python environment BinPickingWorker gets
        # constructed in (e.g. Blender's bundled Python) - only needed once
        # shard merging/splitting is actually run.
        from telekinesis.illusion.dataset import CocoShardMerger
        from telekinesis.illusion.dataset.converter import DatasetConverter

        self._merger = CocoShardMerger(str(self._output_base))
        self._merger.merge()

        if self._dataset_format:
            output_dir = Path(self._specs["metadata"]["dataset_name"]) / Path(
                self._dataset_format + "_format"
            )
            output_dir = self._output_base.parent / output_dir

            if not output_dir.exists() or self._confirm_overwrite(output_dir):
                splitter = DatasetConverter(
                    merged_dataset_dir=self._output_base,
                    output_dir=output_dir,
                    dataset_format=self._dataset_format,
                    ratios=tuple(self._train_val_tes_ratio),
                    stratify=self._splitter_stratify,
                    seed=self._splitter_seed,
                )
                splitter.split()

        if preview:
            self._merger.preview()

    def _confirm_overwrite(self, output_dir: Path) -> bool:
        """
        Prompt the user about an existing dataset-format output directory.

        Args:
            output_dir (Path):
                The existing directory that the format conversion would write to.

        Returns:
            bool: True to proceed with the conversion (the directory is removed
                first for a clean write); False to skip the conversion.
        """
        while True:
            answer = (
                input(
                    f"Output directory '{output_dir}' already exists. Overwrite? [y/N]: "
                )
                .strip()
                .lower()
            )
            if answer in ("y", "yes"):
                logger.warning(
                    f"Overwriting existing output directory: {output_dir}"
                )
                shutil.rmtree(output_dir)
                return True
            if answer in ("", "n", "no"):
                logger.info(
                    "Aborting dataset-format conversion; merged dataset left in place."
                )
                return False
            logger.warning(
                f"Unrecognized response '{answer}'. Please answer 'y' or 'n'."
            )
