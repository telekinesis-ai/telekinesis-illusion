"""
Worker class for generating shards for the bin picking use case.
"""

import math
import shutil
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4
import yaml

from telekinesis.illusion import config
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
from telekinesis.illusion.core.environment import Environment
from telekinesis.illusion.core.engine import BlenderEngine
from telekinesis.illusion.types.model import Model
from telekinesis.illusion.sampler.upper_region_sampler import (
    upper_region_sampler,
)
from telekinesis.illusion.sampler.grid_region_sampler import GridRegionSampler
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.randomizer.randomizer_node import (
    ModelPoseRandomizer,
    ModelInstanceRandomizer,
    BackgroundRandomizer,
    MaterialRandomizer,
    CameraPoseRandomizer,
    LightRandomizer,
    LightPoseRandomizer,
    NodeConfig,
    RandomizerNode,
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


class BinPickingWorker:
    """Generate bin-picking dataset shards from a YAML configuration file.

    The configuration file is the single description of the dataset. Each
    top-level section is validated as the config of the same purpose in
    telekinesis.illusion.config: dataset (DatasetConfig, with its shard and
    writer sections), renderer (RenderConfig), camera (CameraConfig),
    physics (PhysicsConfig), models (ModelConfig per entry), lights
    (LightConfig per entry), pose_sampling (PoseSamplingConfig) and
    randomizers. The randomizers section holds instance (per model role),
    model_pose, material (per rule name), background, light and light_pose
    (per light name), and camera_pose. Every model has exactly one role
    group: part, container or distractor.

    Example:
        worker = BinPickingWorker("configs/bin_picking_example.yaml")
        worker.generate()
        worker.merge_shards(preview=False)
    """

    # Geometry previews move models, lights and cameras, retaining properties.
    GEOMETRY_STAGES = frozenset({STAGE_POSE, STAGE_CAMERA})

    def __init__(
        self,
        config_file_path: str | Path,
    ) -> None:
        """
        Validate the configuration and build the environment and randomizers.

        Args:
            config_file_path: str or Path
                The path to the YAML configuration file.

        Raises:
            TypeError: A configuration value has the wrong type.
            ValueError: A configuration value is invalid. The message names
                the file and the key path of the value.
            FileNotFoundError: The configuration file, the asset directory or
                a model file does not exist.
        """
        self._config_file_path = Path(config_file_path)
        with open(self._config_file_path, encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)
        try:
            self._load_configs(raw_config)
        except (TypeError, ValueError) as error:
            raise type(error)(
                f"Invalid configuration file '{self._config_file_path}': "
                f"{error}"
            ) from error
        logger.info(
            f"Dataset '{self._dataset_config.name}': "
            f"{self._dataset_config.num_images} images in shards of "
            f"{self._dataset_config.shard.size}."
        )

        # A relative dataset.asset_directory is anchored on the config file, so
        # a config stays valid no matter which directory it is run from.
        self._asset_dir = resolve_asset_dir(
            self._dataset_config.asset_directory,
            base_dir=self._config_file_path.parent,
        )

        # TODO: Reuse blender.camera.settings.set_intrinsics_from_K_matrix
        # for measured camera calibration and add_depth_of_field for the
        # target lens's aperture and focus distance.
        self._environment = Environment(asset_dir=self._asset_dir)
        self._environment.add(self._camera_config)

        materials_directory = self._asset_dir / "materials"
        if materials_directory.is_dir():
            # Transparent sets (those with an opacity map) are not used for
            # opaque parts, so they are not registered.
            self._environment.add(
                material_config
                for material_config in config.MaterialConfig.discover(
                    materials_directory
                )
                if not Path(
                    str(
                        config.find_color_texture(material_config.path)
                    ).replace("Color", "Opacity")
                ).exists()
            )
        hdris_directory = self._asset_dir / "hdris"
        if hdris_directory.is_dir():
            self._environment.add(
                config.BackgroundConfig.discover(hdris_directory)
            )

        for model_config in self._model_configs:
            self._environment.add(
                model_config.replace(
                    model_path=resolve_asset_path(
                        model_config.model_path, self._asset_dir
                    )
                )
            )
        self._environment.add(self._light_configs)

        # Roles of the registered models; they stay fixed for this worker.
        self._model_names_by_role = {
            role: [
                model_config.model_name
                for model_config in self._model_configs
                if role in model_config.groups
            ]
            for role in config.MODEL_ROLES
        }
        # category_id may be allocated by the environment when the config
        # omits it. Record the resolved ID that rendering will actually emit.
        assets = self._environment.get_assets()
        supercategory_map = {
            assets[name].category_id: role
            for role, names in self._model_names_by_role.items()
            for name in names
        }

        self._randomizer = Randomizer()
        for node_name, node, stage in self._create_randomizer_nodes():
            self._randomizer.add_randomizer(
                randomizer_node=node,
                node_name=node_name,
                node_config=NodeConfig(stage=stage),
            )

        # TODO: Reuse blender.camera.lens_distortion.set_lens_distortion
        # before rendering and apply_lens_distortion on captured RGB and
        # instance maps before COCO writing. Use the same mapping for both,
        # nearest-neighbor interpolation for IDs, and derive masks/boxes
        # from the remapped instance maps.
        self._output_base = None
        coco_info = self._dataset_config.writer.coco.to_dict()
        coco_licenses = coco_info.pop("licenses")
        self._writer = CocoWriter(
            create_output_dir_on_init=False,
            info=coco_info,
            licenses=coco_licenses,
            supercategory_map=supercategory_map,
            color_file_format=self._dataset_config.writer.image_format,
            jpg_quality=self._dataset_config.writer.jpg_quality,
        )

        # TODO: For RGB-D datasets, enable RenderConfig.depth and reuse
        # blender.postprocessing.add_kinect_azure_noise/add_gaussian_shifts
        # only when they match the target sensor. Preserve clean depth and
        # write noisy sensor depth separately; CocoWriter writes neither.
        self._data_generator = SyntheticDataGenerator(
            environment=self._environment,
            randomizer=self._randomizer,
            writer=self._writer,
            engine=BlenderEngine(self._render_config, self._physics_config),
        )
        self._merger = None

    def _load_configs(self, raw_config: Any) -> None:
        """
        Validate every configuration section and store it as its config.

        Sections are validated individually by their config classes, and
        relations between sections are checked afterwards. All configs are
        stored only after everything is valid, so an invalid configuration
        leaves the previous configs in place.

        Args:
            raw_config: Any
                The configuration as loaded from YAML.

        Raises:
            TypeError: A section or value has the wrong type.
            ValueError: A section is unknown or missing, a value is invalid,
                or sections contradict each other. The message starts with
                the key path of the value.
        """
        sections = {
            "dataset",
            "renderer",
            "camera",
            "physics",
            "models",
            "lights",
            "pose_sampling",
            "randomizers",
        }
        if not isinstance(raw_config, Mapping):
            raise TypeError("The configuration must be a mapping of sections.")
        unknown = sorted(str(key) for key in raw_config if key not in sections)
        if unknown:
            raise ValueError(
                f"{', '.join(unknown)}: unknown section(s). Valid sections "
                f"are {sorted(sections)}."
            )
        missing = [
            section
            for section in ("dataset", "models", "randomizers")
            if section not in raw_config
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)}: required section(s) missing."
            )

        dataset_config = config.build_config(
            "dataset", config.DatasetConfig, raw_config["dataset"]
        )
        render_config = config.build_config(
            "renderer", config.RenderConfig, raw_config.get("renderer", {})
        )
        camera_config = config.build_config(
            "camera", config.CameraConfig, raw_config.get("camera", {})
        )
        physics_config = config.build_config(
            "physics", config.PhysicsConfig, raw_config.get("physics", {})
        )
        pose_sampling_config = config.build_config(
            "pose_sampling",
            config.PoseSamplingConfig,
            raw_config.get("pose_sampling", {}),
        )
        if (
            not isinstance(raw_config["models"], list)
            or not raw_config["models"]
        ):
            raise ValueError("models: must be a non-empty list of models.")
        model_configs = tuple(
            config.build_config(f"models[{index}]", config.ModelConfig, entry)
            for index, entry in enumerate(raw_config["models"])
        )
        if not isinstance(raw_config.get("lights", []), list):
            raise ValueError("lights: must be a list of lights.")
        light_configs = tuple(
            config.build_config(f"lights[{index}]", config.LightConfig, entry)
            for index, entry in enumerate(raw_config.get("lights", []))
        )

        randomizers = raw_config["randomizers"]
        randomizer_sections = {
            "instance",
            "model_pose",
            "material",
            "background",
            "light",
            "light_pose",
            "camera_pose",
        }
        if not isinstance(randomizers, Mapping):
            raise TypeError(
                "randomizers: must be a mapping of randomizer sections."
            )
        unknown = sorted(
            str(key) for key in randomizers if key not in randomizer_sections
        )
        if unknown:
            raise ValueError(
                f"randomizers.{', '.join(unknown)}: unknown section(s). Valid "
                f"sections are {sorted(randomizer_sections)}."
            )
        if "instance" not in randomizers:
            raise ValueError("randomizers.instance: required section missing.")

        def build_named_configs(section, config_type):
            entries = randomizers.get(section, {})
            if not isinstance(entries, Mapping):
                raise TypeError(
                    f"randomizers.{section}: must be a mapping of names to "
                    "settings."
                )
            return {
                name: config.build_config(
                    f"randomizers.{section}.{name}", config_type, entry
                )
                for name, entry in entries.items()
            }

        instance_configs = build_named_configs(
            "instance", config.InstanceRandomizerConfig
        )
        material_configs = build_named_configs(
            "material", config.MaterialRandomizerConfig
        )
        light_randomizer_configs = build_named_configs(
            "light", config.LightRandomizerConfig
        )
        light_pose_configs = build_named_configs(
            "light_pose", config.LightPoseRandomizerConfig
        )
        model_pose_config = config.build_config(
            "randomizers.model_pose",
            config.ModelPoseRandomizerConfig,
            randomizers.get("model_pose", {}),
        )
        background_config = (
            config.build_config(
                "randomizers.background",
                config.BackgroundRandomizerConfig,
                randomizers["background"],
            )
            if "background" in randomizers
            else None
        )
        camera_pose_config = config.build_config(
            "randomizers.camera_pose",
            config.CameraPoseRandomizerConfig,
            randomizers.get("camera_pose", {}),
        )

        names_by_role = {role: [] for role in config.MODEL_ROLES}
        model_names = set()
        for index, model_config in enumerate(model_configs):
            if model_config.model_name in model_names:
                raise ValueError(
                    f"models[{index}].model_name: duplicate model name "
                    f"{model_config.model_name!r}."
                )
            model_names.add(model_config.model_name)
            roles = [
                group
                for group in model_config.groups
                if group in config.MODEL_ROLES
            ]
            if len(roles) != 1:
                raise ValueError(
                    f"models[{index}].groups: must contain exactly one role "
                    f"of {sorted(config.MODEL_ROLES)}, got "
                    f"{list(model_config.groups)}."
                )
            names_by_role[roles[0]].append(model_config.model_name)
        for role in ("part", "container"):
            if not names_by_role[role]:
                raise ValueError(
                    f"models: at least one model needs the role '{role}'."
                )

        roles = {role for role, names in names_by_role.items() if names}
        if set(instance_configs) != roles:
            raise ValueError(
                f"randomizers.instance: must configure exactly the model "
                f"roles {sorted(roles)}, got {sorted(instance_configs)}."
            )
        for role, instance_config in instance_configs.items():
            for name in instance_config.counts:
                if name not in names_by_role[role]:
                    raise ValueError(
                        f"randomizers.instance.{role}.counts.{name}: not a "
                        f"model with the role '{role}'."
                    )

        model_groups = {
            group
            for model_config in model_configs
            for group in model_config.groups
        }
        for name, material_config in material_configs.items():
            unknown = sorted(
                set(material_config.target_models or ()) - model_names
            )
            if unknown:
                raise ValueError(
                    f"randomizers.material.{name}.target_models: unknown "
                    f"models {unknown}."
                )
            unknown = sorted(
                set(material_config.target_groups or ()) - model_groups
            )
            if unknown:
                raise ValueError(
                    f"randomizers.material.{name}.target_groups: no model has "
                    f"the groups {unknown}."
                )

        light_types = {
            light_config.light_name: light_config.light_type
            for light_config in light_configs
        }
        for section, configs in (
            ("light", light_randomizer_configs),
            ("light_pose", light_pose_configs),
        ):
            unknown = sorted(set(configs) - set(light_types))
            if unknown:
                raise ValueError(
                    f"randomizers.{section}: unknown lights {unknown}; declare "
                    "them under 'lights'."
                )
        for name, light_randomizer_config in light_randomizer_configs.items():
            try:
                light_randomizer_config.check_light_type(light_types[name])
            except ValueError as error:
                raise ValueError(f"randomizers.light.{name}.{error}") from error

        if pose_sampling_config.strategy == "grid":
            if model_pose_config.sample_on_surface is not None:
                raise ValueError(
                    "randomizers.model_pose.sample_on_surface: must be null "
                    "for the 'grid' pose_sampling strategy, which samples on "
                    "the visible container."
                )
            # The grid hands out cell centers in order and starts over once it
            # runs out, so more parts than cells means parts share a center and
            # only the jitter keeps them apart.
            part_config = instance_configs["part"]
            visible_parts = sum(
                maximum for _, maximum in part_config.counts.values()
            )
            if part_config.max_num_total_models is not None:
                visible_parts = min(
                    visible_parts, part_config.max_num_total_models
                )
            grid = pose_sampling_config.grid
            if visible_parts > grid.capacity:
                logger.warning(
                    f"Grid capacity is {grid.capacity} cells "
                    f"({grid.rows}x{grid.cols}x{grid.layers}) but up to "
                    f"{visible_parts} parts can be visible - cells get reused, "
                    "so parts will share positions. Add rows/cols/layers or "
                    "lower the part instance count."
                )

        self._dataset_config = dataset_config
        self._render_config = render_config
        self._camera_config = camera_config
        self._physics_config = physics_config
        self._pose_sampling_config = pose_sampling_config
        self._model_configs = model_configs
        self._light_configs = light_configs
        self._instance_configs = instance_configs
        self._model_pose_config = model_pose_config
        self._material_configs = material_configs
        self._background_config = background_config
        self._light_randomizer_configs = light_randomizer_configs
        self._light_pose_configs = light_pose_configs
        self._camera_pose_config = camera_pose_config

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

    def get_environment(self) -> Environment:
        """
        Get the Environment built from the config, with its models and camera.
        """
        return self._environment

    def get_randomizer(self) -> Randomizer:
        """
        Get the Randomizer built from the config.
        """
        return self._randomizer

    def randomize_geometry(self, environment: Environment) -> None:
        """
        Re-sample models, light and camera poses.

        Visible instances, materials, background and sampled light properties
        are retained while the caller iterates on layout and framing. Interactive
        tools use this; a real generation run always calls randomize()
        unfiltered.

        Visible models are the ones that get re-posed, so the caller is
        responsible for composing a scene before sampling poses.

        Args:
            environment: Environment
                The environment whose current scene receives the new poses.
        """
        self._randomizer.randomize(environment, stages=self.GEOMETRY_STAGES)

    def _get_output_size_bytes(self) -> int:
        """Return the total size in bytes of all files under the output directory."""
        if self._output_base is None or not self._output_base.exists():
            return 0
        return sum(
            f.stat().st_size
            for f in self._output_base.rglob("*")
            if f.is_file()
        )

    def prepare(self) -> None:
        """Prepare linked pools before an interactive preview or generation."""
        self._data_generator.prepare()

    def apply_config_updates(self, raw_config: Mapping[str, Any]) -> list[str]:
        """
        Re-apply an edited configuration to the already-built Environment
        and Randomizer, without re-importing any models.

        Built for interactive tools where rebuilding the whole worker on every
        edit is far too slow. Model scales, physics and every randomizer are
        updated in place. Changes that are baked into the imported scene are
        reported back instead of being silently ignored: models or lights that
        changed beyond model scales, and instance counts that exceed the
        prepared capacity, leave the randomizers unchanged. Camera, renderer
        and image format changes are reported and do not block the other
        updates.

        Args:
            raw_config: Mapping
                The edited configuration, in the same format as the file
                passed to __init__.

        Returns:
            Human-readable descriptions of changes that could NOT be applied
            and need a full reload. Empty when everything took effect.

        Raises:
            TypeError: A configuration value has the wrong type.
            ValueError: A configuration value is invalid. The previous
                configuration then stays in effect.
        """
        previous_models = self._model_configs
        previous_lights = self._light_configs
        previous_camera = self._camera_config
        previous_render = self._render_config
        previous_writer = self._dataset_config.writer
        self._load_configs(raw_config)

        needs_reload = self._apply_model_scales()
        # Scales are applied in place, every other model setting needs a
        # fresh import.
        blocking = [
            model_config.replace(scale=1.0)
            for model_config in self._model_configs
        ] != [
            model_config.replace(scale=1.0) for model_config in previous_models
        ]
        if blocking:
            needs_reload.append(
                "Models were added, removed, reordered or changed beyond "
                "their scale - reload assets to apply it"
            )
        if self._light_configs != previous_lights:
            blocking = True
            needs_reload.append(
                "Lights were added, removed or changed - reload assets to "
                "apply it"
            )
        if self._camera_config != previous_camera:
            needs_reload.append(
                "The camera changed - reload assets to apply it"
            )
        if self._render_config != previous_render:
            needs_reload.append(
                "The renderer settings changed - reload assets to apply them"
            )
        if (
            self._dataset_config.writer.image_format,
            self._dataset_config.writer.jpg_quality,
        ) != (previous_writer.image_format, previous_writer.jpg_quality):
            needs_reload.append(
                "The image format or JPEG quality changed - reload assets to "
                "apply it"
            )

        self._data_generator.engine.physics_config = self._physics_config
        if blocking:
            return needs_reload

        nodes = self._create_randomizer_nodes()
        changed_node_names = {name for name, _, _ in nodes} ^ set(
            self._randomizer.get_node_stages()
        )
        if changed_node_names:
            needs_reload.append(
                f"Randomizers {sorted(changed_node_names)} were added or "
                "removed - reload assets to apply it"
            )
            return needs_reload
        # Prepared instance pools cannot grow.
        if self._environment.model_manager.is_prepared:
            exceeded = [
                name
                for _, node, _ in nodes
                for name, count in node.required_instances().items()
                if count > len(self._environment.get_model_group(name))
            ]
            if exceeded:
                needs_reload.extend(
                    f"'{name}' capacity increased; reload assets to apply it"
                    for name in exceeded
                )
                return needs_reload
        for node_name, node, _ in nodes:
            self._randomizer.replace_randomizer(node_name, node)
        return needs_reload

    def _apply_model_scales(self) -> list[str]:
        """
        Re-scale already-imported models whose config scale changed.

        Model.__init__ bakes the scale into the mesh vertices and resets the
        models scale to 1, so this applies the *relative* factor to the same
        mesh data. Instances are linked duplicates sharing one mesh datablock,
        which must be transformed exactly once - so this iterates the model
        group and transforms each distinct datablock a single time.

        Returns:
            Descriptions of models that could not be re-scaled.
        """
        problems: list[str] = []

        for model_config in self._model_configs:
            name = model_config.model_name
            new_scale = model_config.scale
            group = self._environment.get_model_group(name)
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
            for model in group:
                mesh = model.get_mesh_object().blender_obj.data
                # Linked duplicates share this datablock - transform once, but
                # update every Model's bookkeeping so the next call computes
                # its factor from a fresh baseline.
                if mesh.name not in transformed:
                    mesh.transform(scale_matrix)
                    transformed.add(mesh.name)
                model.set_scale(new_scale)
            logger.info(
                f"Re-scaled '{name}' by {factor} to {new_scale} "
                f"({len(transformed)} mesh datablock(s))."
            )

        return problems

    def _make_sample_pose(self) -> Callable[[Model], None]:
        """
        Build the 'random' pose-sampling function.

        Reads the pose sampling and model pose configs on every call rather
        than capturing them, so the function follows configuration updates.

        Returns:
            A callable taking the Model to place.
        """

        def sample_pose(model: Model):
            """
            Randomly sample and apply a 6-DoF pose to a model above the
            visible container, or above the configured surface.
            """
            target = self._model_pose_config.sample_on_surface
            if target is None:
                visible = self._environment.get_visible_model_names()
                containers = self._model_names_by_role["container"]
                target = next(
                    (
                        model_name
                        for model_name in visible
                        if any(name in model_name for name in containers)
                    ),
                    None,
                )
            sample_location = upper_region_sampler(
                self._environment,
                model_names_to_sample_on=target,  # type: ignore[arg-type]
                face_sample_range=self._pose_sampling_config.face_sample_range,
                min_height=self._pose_sampling_config.min_height,
                max_height=self._pose_sampling_config.max_height,
            )
            model.set_location(sample_location)
            model.set_rotation(
                np.random.uniform((-180, -180, -180), (180, 180, 180))
            )

        return sample_pose

    def _make_sample_pose_grid(self) -> Callable[[Model], None]:
        """
        Build the 'grid' pose-sampling function with its own grid sampler.

        The grid geometry is taken from the current pose sampling config; the
        z-rotation range is read on every call.

        Returns:
            A callable taking the Model to place.
        """
        pose_sampling_config = self._pose_sampling_config
        grid_sampler = GridRegionSampler(
            environment=self._environment,
            rows=pose_sampling_config.grid.rows,
            cols=pose_sampling_config.grid.cols,
            layers=pose_sampling_config.grid.layers,
            layer_spacing=pose_sampling_config.grid.layer_spacing,
            face_sample_range=pose_sampling_config.face_sample_range,
            min_height=pose_sampling_config.min_height,
            max_height=pose_sampling_config.max_height,
            shuffle=pose_sampling_config.grid.shuffle,
            xy_jitter=pose_sampling_config.grid.xy_jitter,
        )

        def sample_pose_grid(model: Model):
            """
            Place a model at the next grid cell on the bin's top face
            with optional z-axis rotation randomization.
            """
            location = grid_sampler.next_location(
                self._model_names_by_role["container"]
            )
            model.set_location(location)
            z_rot_min, z_rot_max = np.deg2rad(
                self._pose_sampling_config.grid.z_rotation_range
            )
            z_rot = (
                np.random.uniform(z_rot_min, z_rot_max)
                if z_rot_min != z_rot_max
                else 0.0
            )
            model.set_rotation(np.array([0.0, 0.0, z_rot]))

        return sample_pose_grid

    def _create_randomizer_nodes(
        self,
    ) -> list[tuple[str, RandomizerNode, str]]:
        """
        Create the randomizer nodes described by the current configs.

        The order is the execution order: instance and pose randomizers,
        materials, background, light properties and poses, and the camera.

        Returns:
            A list of (node name, node, stage) tuples.
        """
        nodes = [
            (
                "instance_randomizer_models",
                ModelInstanceRandomizer(self._instance_configs["part"]),
                STAGE_COMPOSITION,
            ),
            (
                "instance_randomizer_containers",
                ModelInstanceRandomizer(self._instance_configs["container"]),
                STAGE_COMPOSITION,
            ),
        ]

        sample_pose = self._make_sample_pose()
        if self._pose_sampling_config.strategy == "grid":

            def resolve_container_surface(environment: Environment) -> str:
                visible = environment.get_visible_model_names()
                containers = self._model_names_by_role["container"]
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

            model_pose_randomizer = ModelPoseRandomizer(
                self._model_pose_config,
                self._make_sample_pose_grid(),
                self._model_names_by_role["part"],
                surface_name_resolver=resolve_container_surface,
            )
        else:
            model_pose_randomizer = ModelPoseRandomizer(
                self._model_pose_config,
                sample_pose,
                self._model_names_by_role["part"],
            )
        nodes.append(
            ("pose_randomizer_models", model_pose_randomizer, STAGE_POSE)
        )

        if "distractor" in self._instance_configs:
            nodes.append(
                (
                    "instance_randomizer_distractors",
                    ModelInstanceRandomizer(
                        self._instance_configs["distractor"]
                    ),
                    STAGE_COMPOSITION,
                )
            )
            nodes.append(
                (
                    "pose_randomizer_distractors",
                    ModelPoseRandomizer(
                        self._model_pose_config.replace(sample_on_surface=None),
                        sample_pose,
                        self._model_names_by_role["distractor"],
                    ),
                    STAGE_POSE,
                )
            )

        # TODO: Use the existing material rules with PrincipledMaterial and
        # SurfaceImperfections for plausible roughness, scratches and wear;
        # see examples/preview_bin_surface_imperfections.py. Reuse the local
        # Material.infuse_texture/infuse_material helpers for further layers,
        # duplicating shared materials before per-instance shader edits.
        for name, material_config in self._material_configs.items():
            nodes.append(
                (
                    f"material_randomizer_{name}",
                    MaterialRandomizer(material_config, self._environment),
                    STAGE_APPEARANCE,
                )
            )

        # TODO: Extend HDRI variation using Background's existing mapping
        # rotation and background strength nodes, alongside image selection,
        # to vary reflection direction and illumination without new loaders.
        if self._background_config is not None:
            nodes.append(
                (
                    "background_randomizer",
                    BackgroundRandomizer(self._background_config),
                    STAGE_APPEARANCE,
                )
            )

        for light_config in self._light_configs:
            name = light_config.light_name
            if name in self._light_randomizer_configs:
                nodes.append(
                    (
                        f"light_properties_{name}",
                        LightRandomizer(
                            self._light_randomizer_configs[name], [name]
                        ),
                        STAGE_APPEARANCE,
                    )
                )
            if name in self._light_pose_configs:
                nodes.append(
                    (
                        f"light_pose_{name}",
                        LightPoseRandomizer(
                            self._light_pose_configs[name], [name]
                        ),
                        STAGE_POSE,
                    )
                )

        # TODO: Run camera validation in the generator after physics settles
        # and before rendering. Refresh evaluated geometry and reuse
        # blender.types.mesh_object.create_bvh_tree_multi_objects for the
        # visible parts, bin and distractors. Exclude inactive pool instances
        # and invisible physics helpers from all visibility checks.
        # TODO: Recompute model-relative look-at targets from settled poses
        # before evaluating each camera view.
        # TODO: Reuse blender.camera.validation.perform_obstacle_in_view_check
        # for minimum viewing distance and visible_objects to check whether
        # target parts are visible from each candidate camera pose.
        # TODO: Make acceptance follow the intended visibility, cropping and
        # occlusion distribution; do not require every part to be fully
        # visible. Ray-grid checks are approximate: use rendered instance
        # masks when exact visible pixel areas are required.
        # TODO: Resample only failed camera views with a bounded retry budget
        # and report exhaustion. Preserve the settled arrangement and accepted
        # views; randomized views of one arrangement remain separate images.
        nodes.append(
            (
                "camera_pose_randomizer",
                CameraPoseRandomizer(self._camera_pose_config),
                STAGE_CAMERA,
            )
        )
        return nodes

    def generate(self) -> None:
        """
        Generate shards based on the provided config.

        Each shard is written to its own output directory whose name includes a
        UUID so that multiple workers can run simultaneously without collisions.
        Generation stops early once the output exceeds
        dataset.writer.max_size_gb.

        The configuration file is saved unchanged to
        <output_directory>/<name>/config.yaml, headed by a comment with the
        installed telekinesis-illusion version and the generation time. An
        existing copy from an earlier run is overwritten.
        """
        shard_size = self._dataset_config.shard.size
        remaining_images = self._dataset_config.num_images
        num_shards = math.ceil(remaining_images / shard_size)
        max_output_size_bytes = (
            self._dataset_config.writer.max_size_gb * (1024**3)
        )
        self._output_base = (
            Path(self._dataset_config.output_directory)
            / self._dataset_config.name
        )
        # The header is a YAML comment, so the copy can be used to rerun the
        # generation as-is.
        self._output_base.mkdir(parents=True, exist_ok=True)
        (self._output_base / "config.yaml").write_text(
            f"# Generated with telekinesis-illusion "
            f"{metadata.version('telekinesis-illusion')} on "
            f"{datetime.now().isoformat(timespec='seconds')}\n"
            + self._config_file_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        try:
            for i in range(num_shards):
                current_shard_size = min(shard_size, remaining_images)
                shard_name = self._dataset_config.shard.name_template.format(
                    date=datetime.now().strftime("%Y%m%d_%H%M%S"),
                    uuid=uuid4().hex[:8],
                )
                shard_dir = self._output_base / shard_name
                self._writer.set_output_dir(shard_dir)

                logger.info(
                    f"[shard {i + 1}/{num_shards}] "
                    f"Generating {current_shard_size} environments -> {shard_dir.name}"
                )

                self._data_generator.generate(
                    num_images=current_shard_size,
                    clean_up_environment=False,
                )
                remaining_images -= current_shard_size

                output_size = self._get_output_size_bytes()
                logger.info(
                    f"[shard {i + 1}/{num_shards}] "
                    f"Current output size: {output_size / (1024**3):.2f} GB "
                    f"(limit: {max_output_size_bytes / (1024**3):.1f} GB)"
                )
                if output_size >= max_output_size_bytes:
                    logger.warning(
                        f"Output size limit reached "
                        f"({output_size / (1024**3):.2f} GB >= "
                        f"{max_output_size_bytes / (1024**3):.1f} GB). "
                        f"Stopping after shard {i + 1}/{num_shards}."
                    )
                    break

        finally:
            self._data_generator.clean_up()

    def merge_shards(self, preview: bool = True) -> None:
        """
        Merge the generated shards and export them in
        dataset.writer.dataset_format.

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

        dataset_format = self._dataset_config.writer.dataset_format
        if dataset_format:
            output_dir = (
                self._output_base.parent
                / self._dataset_config.name
                / f"{dataset_format}_format"
            )

            if not output_dir.exists() or self._confirm_overwrite(output_dir):
                splitter = DatasetConverter(
                    merged_dataset_dir=self._output_base,
                    output_dir=output_dir,
                    dataset_format=dataset_format,
                    ratios=self._dataset_config.writer.split_ratios,
                    stratify=self._dataset_config.writer.stratify,
                    seed=self._dataset_config.writer.split_seed,
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
