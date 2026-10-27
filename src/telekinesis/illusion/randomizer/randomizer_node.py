"""
Defines the RandomizerNode class that is the base class for all randomizers
nodes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple
import random
import numpy as np
from loguru import logger
from collections.abc import Sequence

from telekinesis.illusion import config
from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported. See blender_env.py for why.
isolate_user_extensions()

import bpy
import mathutils
from mathutils import Euler

from telekinesis.illusion.blender.types.mesh_object import (
    MeshObject,
    get_all_mesh_objects,
)
from telekinesis.illusion.blender.collision import CollisionUtility

from telekinesis.illusion.types.distribution import Uniform
from telekinesis.illusion.core.environment import Environment
from telekinesis.illusion.types.model import Model
from telekinesis.illusion.types.light import Light
from telekinesis.illusion.randomizer import instance_counts
from telekinesis.illusion.randomizer.materials import (
    AllSlots,
    MaterialPool,
    PBRMaterial,
    _names,
    assign_material,
    clean_generated_materials,
    slot_pools,
)
from telekinesis.illusion.sampler.camera_pose_sampler import (
    CAMERA_POSE_SAMPLERS,
)

# Stages group the randomizer nodes by what part of the scene they change, so
# a caller can re-run only some of them (see Randomizer.randomize's 'stages'
# argument). Interactive tools use this to re-sample the geometry without
# rebuilding the whole environment.
STAGE_COMPOSITION = "composition"  # which and how many instances are visible
STAGE_POSE = "pose"  # where the visible models are placed
STAGE_APPEARANCE = "appearance"  # materials and background/HDRI
STAGE_CAMERA = "camera"  # camera pose sampling

# Nodes left at this stage are never filtered out - a node that doesn't opt
# into a stage keeps running in every pass.
STAGE_DEFAULT = "default"


@dataclass(slots=True)
class NodeConfig:
    """
    A data class for providing additional node configs for the Randomizer
    """

    enabled: bool = True
    stage: str = STAGE_DEFAULT
    priority: int = 0
    seed_key: str | None = None
    apply_prob: float = 1.0
    max_tries: int = 1
    on_failure: str = "raise"  # "raise" | "skip_node" | "skip_sample"
    profile: bool = False


class RandomizerNode(ABC):
    """
    Declares the method for executing the randomization.
    """

    def prepare(self, environment: Environment) -> None:
        """Prepare resources in the environment before sampling scenes."""

    def required_instances(self) -> dict[str, int]:
        """Finite upper bounds for additional instances; modifiers return {}."""
        return {}

    @abstractmethod
    def randomize(self, environment: Environment) -> None:
        """Apply this node's rules to the environment's current scene.

        Called in dependency order when this node is enabled and selected for
        the current randomization pass. Mutate the realization in place and
        return None.
        """
        pass


class ModelPoseRandomizer(RandomizerNode):
    """Randomize the poses of models.

    Example:
        def sample_pose(model):
            model.set_location(np.zeros(3))

        randomizer = ModelPoseRandomizer(
            config.ModelPoseRandomizerConfig(), sample_pose, ["part"]
        )
    """

    def __init__(
        self,
        randomizer_config: config.ModelPoseRandomizerConfig,
        pose_sampling_function: Callable[[Model], None],
        target_models: Sequence[str] = (),
        surface_name_resolver: Callable[["Environment"], str] | None = None,
    ) -> None:
        """
        Initialize the internal state. The function 'pose_sampling_function'
        should take a Model as its argument and implement the sampling logic
        for the models named by 'target_models'.

        Example for a valid sampling function:

        .. code-block:: python

            def sample_pose(model: Model):
                model.set_location(
                    np.random.uniform((-0.1, -0.1, 0.1), (0.1, 0.1, 0.1))
                )
                model.set_rotation(
                    np.random.uniform((-180, -180, -180), (180, 180, 180))
                )

        Args:
            randomizer_config: config.ModelPoseRandomizerConfig
                Placement attempts per model and the optional surface the
                poses are sampled on.
            pose_sampling_function: Callable[[Model], None]
                A sampling function that takes a Model as its argument and
                sets its location and orientation according to some sampling
                logic. This function is used to randomly place the models in
                the current scene and returns None.
            target_models: Sequence[str]
                Prefixes of the instance names of the target models. Empty
                targets all visible models.
            surface_name_resolver: Callable or None
                If provided and the config has no sample_on_surface, this
                callable is invoked with the Environment at randomize-time to
                dynamically resolve the surface name.
        """
        self.randomizer_config = randomizer_config
        self.pose_sampling_function = pose_sampling_function
        self.target_models = list(target_models)
        self._surface_name_resolver = surface_name_resolver

    def randomize(self, environment: Environment) -> None:
        """
        Randomize the pose of the target models by sampling their pose with
        'self.pose_sampling_function' and positioning them wihtout collision
        in the current scene.

        Args:
            environment: Environment
                The environment owning the target models for this scene.
        """
        if self.target_models:
            prefixes = tuple(self.target_models)

            def is_target(name: str) -> bool:
                return name.startswith(prefixes)

            models = environment.get_models()
            visible_target_models = [
                models[name]
                for name in models
                if is_target(name) and not models[name].is_hidden()
            ]
        else:
            # Randomize all visible models in the current scene.
            visible_target_models = [
                model
                for model in environment.get_models().values()
                if not model.is_hidden()
            ]

        if not visible_target_models:
            return

        surface_name = self.randomizer_config.sample_on_surface
        if surface_name is None and self._surface_name_resolver is not None:
            surface_name = self._surface_name_resolver(environment)

        if surface_name:
            surface_model = environment.get_models_by_name(surface_name)
            self.sample_poses_on_surface(
                environment=environment,
                models_to_sample=visible_target_models,
                sample_pose_func=self.pose_sampling_function,
                surface_model=surface_model,
            )
        else:
            self.sample_poses(
                environment,
                visible_target_models,
                models_to_check_collisions=visible_target_models,
                sample_pose_func=self.pose_sampling_function,
                max_tries=self.randomizer_config.max_tries,
            )

    def sample_poses(
        self,
        environment: Environment,
        models_to_sample: List[Model],
        sample_pose_func: Callable[[Model], None],
        models_to_check_collisions: List[Model] = None,
        max_tries: int = 1000,
        mode_on_failure: str = "last_pose",
    ) -> Dict[Model, Tuple[int, bool]]:
        """
        Sample positions and rotations of selected models inside the sampling
        volume while performing mesh and bounding box collision checks.

        Args:
            environment: Environment
                The environment from which to retrieve models for this scene.
            models_to_sample: List[Model]
                A list of models whose poses are sampled based on the
                given function.
            sample_pose_func: Callable[[Model], None]
                The function to use for sampling the pose of a given model.
            models_to_check_collisions: List[Model]
                A list of models that should not be considered when
                checking for collisions.
            max_tries: int
                Amount of tries before giving up on a model and moving to the
                next one.
            mode_on_failure: str
                Define final state of models that could not be placed without
                collisions within max_tries attempts. Options: 'last_pose',
                'initial_pose'.

        Returns:
            A dict with the models to sample as keys and a Tuple with the
            number of executed attempts to place the model as first element,
            and a bool whether it has been successfully placed without
            collisions.

        Raises:
            ValueError:
                - If provided failure mode is not allowed
                - If the value of max_tries is not greater than zero
                - If the list of models_to_sample is empty
        """
        # Check if mode on failure is allowed
        allowed_modes_on_failure = ["last_pose", "initial_pose"]
        if mode_on_failure not in allowed_modes_on_failure:
            raise ValueError(
                f"{mode_on_failure} is not an allowed mode_on_failure."
            )

        # After this many tries we give up on the current model.
        if models_to_check_collisions is None:
            models_to_check_collisions = list(
                environment.get_models().values()
            )

        # Among sampled models, only check against already placed models.
        current_models_to_check_collisions = list(
            set(models_to_check_collisions) - set(models_to_sample)
        )

        # Collision checks require mesh wrappers.
        if current_models_to_check_collisions:
            current_models_to_check_collisions = [
                model.get_mesh_object()
                for model in current_models_to_check_collisions
            ]

        if max_tries <= 0:
            raise ValueError(
                f"The value of max_tries must be greater than zero: {max_tries}"
            )

        if not models_to_sample:
            raise RuntimeError(
                "The list of models_to_sample can not be empty!"
            )

        # cache to fasten collision detection
        bvh_cache: Dict[str, mathutils.bvhtree.BVHTree] = {}

        sample_results: Dict[Model, Tuple[int, bool]] = {}
        # Track model names that were hidden because of collision
        visible_model_names: List[str] = list(
            environment.get_visible_model_names()
        )
        for model in models_to_sample:
            # Store the model's initial pose in case we need to place it back
            if mode_on_failure == "initial_pose":
                initial_location = model.get_location()
                initial_rotation = model.get_rotation()

            no_collision = True

            amount_of_tries_done = -1

            # Try max_iter amount of times
            for i in range(max_tries):
                # Put the next model at the sampled point in space.
                sample_pose_func(model)

                # Remove the BVH cache because the model moved.
                if model.get_name() in bvh_cache:
                    del bvh_cache[model.get_name()]

                mesh_obj = model.get_mesh_object()

                no_collision = CollisionUtility.check_intersections(
                    mesh_obj, bvh_cache, current_models_to_check_collisions, []
                )

                # If no collision then keep the position
                if no_collision:
                    amount_of_tries_done = i
                    break

            # After placing a model, check collisions with it.
            if no_collision:
                current_models_to_check_collisions.append(mesh_obj)

            if no_collision:
                logger.info(
                    f"It took {amount_of_tries_done + 1} tries to "
                    f"place {model.get_name()}"
                )
            else:
                amount_of_tries_done = max_tries
                logger.warning(
                    f"Could not place {model.get_name()} without "
                    f"collision after {amount_of_tries_done} tries."
                )
                logger.warning("Hiding model for this render.")
                environment.deactivate(model)

                if mode_on_failure == "initial_pose":
                    model.set_location(initial_location)
                    model.set_rotation(initial_rotation)

            sample_results[model] = (amount_of_tries_done, no_collision)

        return sample_results

    def sample_poses_on_surface(
        self,
        environment: Environment,
        models_to_sample: List[Model],
        sample_pose_func: Callable[[Model], None],
        surface_model: Model,
        max_tries: int = 100,
        min_distance: float = 0.01,
        max_distance: float = 0.6,
        up_direction: np.ndarray | None = None,
        check_all_bb_corners_over_surface: bool = True,
    ) -> List[Model]:
        """
        TODO: Check if it makes more sense to use a geo node set up...
        Sample model poses on a surface.

        The models are positioned slightly above the surface due to the
        non-axis aligned nature of used bounding boxes and possible
        non-alignment of the sampling surface (i.e. on the X-Y hyperplane, can
        be somewhat mitigated with precise "up_direction" value), which leads to
        the models hovering slightly above the surface. So it is recommended to
        use the PhysicsPositioning module afterwards for realistically looking
        placements of models on the sampling surface. If placing fails due to
        collisions, the model will be moved back to the intial pose and hidden
        from rendering.

        Args:
            environment: Environment
                The environment from which to retrieve models for this scene.
            models_to_sample: List[Model]
                A list of models that should be sampled above the surface.
            surface_model: Model
                Model to place models_to_sample on.
            sample_pose_func: Callable[[Model], None]
                The function to use for sampling the pose of a given model.
            max_tries: int
                Amount of tries before giving up on a model (deleting it) and
                moving to the next one.
            min_distance: float
                Minimum distance to the closest other model from
                models_to_sample. Center to center.
            max_distance: float
                Maximum distance to the closest other model from
                models_to_sample. Center to center.
            up_direction: np.ndarray or None
                Normal vector of the side of surface the models should be
                placed on.
            check_all_bb_corners_over_surface: bool
                If this is True all bounding box corners have to be above the
                surface, else only the center of the model has to be above the
                surface

        Return:
            The list of placed models' mesh objects.
        """
        if len(set(models_to_sample)) != len(models_to_sample):
            raise ValueError(
                "The given list of models to sample contains duplicates."
                "This leads to complications in the sampling process."
            )

        if up_direction is None:
            up_direction = np.array([0.0, 0.0, 1.0])
        else:
            up_direction /= np.linalg.norm(up_direction)

        # cache to fasten collision detection
        bvh_cache: Dict[str, mathutils.bvhtree.BVHTree] = {}

        placed_models: List[MeshObject] = []

        for model in models_to_sample:
            logger.info(f"Trying to put {model.get_name()}")
            initial_pose = model.get_local2world_mat()
            placed_successfully = False

            for i in range(max_tries):
                sample_pose_func(model)
                # Remove bvh cache, as model has changed
                if model.get_name() in bvh_cache:
                    del bvh_cache[model.get_name()]

                mesh_obj = model.get_mesh_object()
                surface_obj = surface_model.get_mesh_object()

                no_collision = CollisionUtility.check_intersections(
                    mesh_obj, bvh_cache, placed_models, []
                )

                if not no_collision:
                    logger.info("Collision detected, retrying!")
                    continue

                # Remove bvh cache, as model has changed
                if model.get_name() in bvh_cache:
                    del bvh_cache[model.get_name()]

                # Select the current object
                bpy.ops.object.select_all(action="DESELECT")
                mesh_obj.blender_obj.select_set(True)
                bpy.context.view_layer.objects.active = mesh_obj.blender_obj
                # Start ray slightly above models origin
                origin = mesh_obj.blender_obj.location + mathutils.Vector(
                    (0, 0, 0.001)
                )
                direction = mathutils.Vector((0, 0, -1))  # cast downward

                depsgraph = bpy.context.evaluated_depsgraph_get()

                # Cast against the target surface only — by construction this
                # cannot self-hit, regardless of the model's geometry or
                # origin placement.
                surface_blender_obj = surface_obj.blender_obj
                mw = surface_blender_obj.matrix_world
                mw_inv = mw.inverted()

                origin_local = mw_inv @ origin
                direction_local = mw_inv.to_3x3() @ direction

                hit, location_local, normal_local, _ = (
                    surface_blender_obj.ray_cast(origin_local, direction_local)
                )

                if not hit:
                    logger.warning(
                        f'Ray casting of the model "{mesh_obj.get_name()}" '
                        f"into the direction {direction} didn't hit the target "
                        f"surface model {surface_model.get_name()} in the "
                        "environment. "
                        f"Skipping this part."
                    )
                    continue

                location = mw @ location_local
                normal = (mw.to_3x3() @ normal_local).normalized()

                obj_eval = mesh_obj.blender_obj.evaluated_get(depsgraph)
                bbox_world = [
                    obj_eval.matrix_world @ mathutils.Vector(c)
                    for c in obj_eval.bound_box
                ]

                n = normal.normalized()
                o = obj_eval.matrix_world.translation

                # Find the minimum signed distance from origin along the normal
                # among bbox corners
                # (i.e., the "lowest" point in direction -n)
                min_proj = min((v - o).dot(n) for v in bbox_world)

                # If min_proj is negative, that means there's geometry "below"
                # the origin along normal. To put that point onto the surface,
                # shift by -min_proj along n.
                mesh_obj.blender_obj.location = location - min_proj * n

                logger.info(
                    f'Placed model "{mesh_obj.get_name()}" successfully at'
                    f"{mesh_obj.get_location()} after {i + 1} iterations!"
                )

                spacing_is_ok = _OnSurfaceSampler.check_spacing(
                    mesh_obj, placed_models, min_distance, max_distance
                )

                if not spacing_is_ok:
                    logger.info("Bad spacing after drop, retrying!")
                    continue

                no_collision = CollisionUtility.check_intersections(
                    mesh_obj, bvh_cache, placed_models, []
                )

                if not no_collision:
                    logger.info("Collision detected after drop, retrying!")
                    continue

                placed_models.append(mesh_obj)
                placed_successfully = True
                break

            if not placed_successfully:
                logger.info(f"Giving up on {model.get_name()}, hiding...")
                environment.deactivate(model)
                model.set_local2world_mat(initial_pose)

        return placed_models


class _OnSurfaceSampler:
    @staticmethod
    def check_above_surface(
        mesh_object: MeshObject,
        surface: MeshObject,
        up_direction: np.ndarray,
        check_all_bb_corners_over_surface: bool = True,
    ) -> bool:
        """
        Check if all corners of the bounding box are "above" the surface

        Args:
        mesh_object:
            Mesh object for which the check is carried out. Type: blender object.
        surface:
            The surface mesh object.
        up_direction:
            The direction that indicates "above" direction.
        check_all_bb_corners_over_surface:
            If this is True all bounding box corners have to be above the
            surface, else only the center of the mesh object has to be above the
            surface

        Return:
            True if the bounding box is above the surface, False - if not.
        """
        if check_all_bb_corners_over_surface:
            for point in mesh_object.get_bound_box():
                position_is_above_object = surface.position_is_above_object(
                    point + up_direction,
                    -up_direction,
                    check_no_objects_in_between=False,
                )
                if not position_is_above_object:
                    return False
            return True
        center = np.mean(mesh_object.get_bound_box(), axis=0)
        position_is_above_object = surface.position_is_above_object(
            center + up_direction,
            -up_direction,
            check_no_objects_in_between=False,
        )
        return position_is_above_object

    @staticmethod
    def check_spacing(
        mesh_object: MeshObject,
        placed_models: List[MeshObject],
        min_distance: float,
        max_distance: float,
    ) -> bool:
        """
        Check if mesh object is not too close or too far from previous objects.

        Args:
            mesh_object: MeshObject
                Mesh object for which the check is carried out.
            placed_models: List[MeshObject]
                A list of already placed objects that should be used for
                checking spacing.
            min_distance: float
                Minimum distance to the closest other object from
                placed_models. Center to center.
            max_distance: float
                Maximum distance to the closest other object from
                placed_models. Center to center.

            Return:
                True, if the spacing is correct
        """
        closest_distance = None

        for already_placed in placed_models:
            distance = np.linalg.norm(
                already_placed.get_location() - mesh_object.get_location()
            )
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance

        return closest_distance is None or (
            min_distance <= closest_distance <= max_distance
        )

    @staticmethod
    def drop(
        mesh_object: MeshObject, up_direction: np.ndarray, surface_height: float
    ) -> None:
        """
        Moves mesh object "down" until its bounding box touches the bounding box of
        the surface. This uses bounding boxes which are not aligned optimally,
        this will cause objects to be placed slightly to high.

        Args:
            mesh_object: MeshObject
                Object to move. Type: blender object.
            up_direction: np.ndarray
                Vector which points into the opposite drop direction.
            surface_height: float
                Height of the surface above its origin.
        """
        mesh_object_bounds = mesh_object.get_bound_box()
        mesh_object_height = min(up_direction.dot(corner) for corner in mesh_object_bounds)

        mesh_object.set_location(
            mesh_object.get_location() - up_direction * (mesh_object_height - surface_height)
        )


class ModelInstanceRandomizer(RandomizerNode):
    """Sample bounded counts and activate instances from environment pools.

    Multiple nodes requesting the same model have additive capacity and select
    disjoint instances. Pose/material nodes don't reserve additional models.

    Example:
        randomizer = ModelInstanceRandomizer(
            config.InstanceRandomizerConfig(counts={"gear": (1, 3)})
        )
    """

    def __init__(
        self, randomizer_config: config.InstanceRandomizerConfig
    ) -> None:
        """Create the node from validated instance bounds.

        Args:
            randomizer_config: Per-model and total instance bounds, keyed by
                registered model name.
        """
        self.randomizer_config = randomizer_config

    def required_instances(self) -> dict[str, int]:
        cfg = self.randomizer_config
        _, total_max = instance_counts.validate_counts(
            cfg.counts, cfg.min_num_total_models, cfg.max_num_total_models
        )
        minimum_sum = sum(bounds[0] for bounds in cfg.counts.values())
        return {
            name: int(min(bounds[1], total_max - minimum_sum + bounds[0]))
            for name, bounds in cfg.counts.items()
        }

    def randomize(self, environment: Environment) -> None:
        cfg = self.randomizer_config
        selected = instance_counts.sample_counts(
            cfg.counts, cfg.min_num_total_models, cfg.max_num_total_models
        )
        active = set(environment.get_visible_model_names())
        for name, count in selected.items():
            available = [
                model
                for model in environment.get_model_group(name)
                if model.get_name() not in active
            ]
            if count > len(available):
                raise RuntimeError(
                    f"Insufficient prepared instances for {name!r}."
                )
            random.shuffle(available)
            for model in available[:count]:
                environment.activate(model)


class BackgroundRandomizer(RandomizerNode):
    """Choose one of the HDRIs registered in the environment.

    Selects the union of the backgrounds in the configured groups and the
    configured names. Groups are exact normalized groups, including ancestors
    such as "indoor". Without a selection, every registered background is
    eligible. Backgrounds are registered with Environment.add(); this
    randomizer does not discover files.

    Example:
        randomizer = BackgroundRandomizer(
            config.BackgroundRandomizerConfig(groups=("indoor/industrial",))
        )
        randomizer.randomize(environment)
    """

    def __init__(
        self, randomizer_config: config.BackgroundRandomizerConfig
    ) -> None:
        """Create a selection over registered backgrounds.

        Args:
            randomizer_config: Groups and names of the eligible backgrounds.

        Unknown groups or names, or an empty selection, raise ValueError when
        the randomizer is prepared against an environment.
        """
        self.randomizer_config = randomizer_config

    def prepare(self, environment: Environment) -> None:
        """Validate the selection against the environment's backgrounds."""
        environment.background_manager.get_names(
            self.randomizer_config.groups, self.randomizer_config.names
        )

    def randomize(self, environment: Environment) -> None:
        """Select a registered HDRI and apply it to the environment's world."""
        manager = environment.background_manager
        name = random.choice(
            manager.get_names(
                self.randomizer_config.groups, self.randomizer_config.names
            )
        )
        environment.get_background().set_background_texture(
            str(manager.assets_by_name[name].path)
        )


class MaterialRandomizer(RandomizerNode):
    """Randomize the materials of registered models.

    Targets are the instances of the models selected by the config's
    target_models and target_groups. Mode 'active' replaces the active
    material slot, mode 'model' uses one sample across every slot of each
    target, and mode 'slots' samples slots independently. A mapping of
    material slots assigns separate choices to original slot names or
    zero-based indices; named slots must exist on every target. Use separate
    nodes when targets have different slot layouts. A seed gives this node a
    private random number generator.

    Example:
        randomizer = MaterialRandomizer(
            config.MaterialRandomizerConfig(
                target_groups=("part",), groups=("metal",)
            ),
            environment,
        )
    """

    def __init__(
        self,
        randomizer_config: config.MaterialRandomizerConfig,
        environment: Environment | None = None,
    ) -> None:
        """Create the material sources described by the config.

        Args:
            randomizer_config: Target models, material sources, assignment
                mode and seed.
            environment: If given, the node is prepared immediately, which
                validates the targets and loads the selected PBR materials.
        """
        cfg = randomizer_config
        self.randomizer_config = randomizer_config
        self._mode = cfg.mode or (
            "slots"
            if cfg.material_slots is not None
            else "model"
            if cfg.materials is not None
            else "active"
        )
        self._rng = random if cfg.seed is None else random.Random(cfg.seed)
        if self._mode == "slots" and cfg.material_slots != "all":
            self._slot_pools = slot_pools(
                {
                    slot: self._material_choices(choices)
                    for slot, choices in cfg.material_slots.items()
                }
            )
            self._pool = None
        else:
            self._pool = MaterialPool(
                self._material_choices(cfg.materials)
                if cfg.materials is not None
                else [PBRMaterial(cfg.groups, cfg.names)]
            )
            self._slot_pools = (
                [(AllSlots(), self._pool)] if self._mode == "slots" else []
            )
        self._prepared_manager = None
        if environment is not None:
            self.prepare(environment)

    @staticmethod
    def _material_choices(choices) -> list:
        """Turn configured choices into material pool choices.

        Args:
            choices: PBR group names and Principled recipe configs.

        Returns:
            The group names unchanged and each recipe as a PrincipledMaterial.
        """
        return [
            choice
            if isinstance(choice, str)
            else choice.to_principled_material()
            for choice in choices
        ]

    def prepare(self, environment: Environment) -> None:
        """Validate the targets and load the selected materials once."""
        environment.model_manager.get_names(
            self.randomizer_config.target_groups,
            self.randomizer_config.target_models,
        )
        if self._prepared_manager is environment.material_manager:
            return
        pools = [p for _, p in self._slot_pools] or [self._pool]
        for pool in pools:
            pool.prepare(environment.material_manager)
        self._prepared_manager = environment.material_manager

    def randomize(self, environment: Environment) -> None:
        self.prepare(environment)
        targets = [
            model
            for name in environment.model_manager.get_names(
                self.randomizer_config.target_groups,
                self.randomizer_config.target_models,
            )
            for model in environment.get_model_group(name)
        ]
        # Resolve every target before sampling or mutation, so bad selectors
        # cannot leave a partially randomized asset behind.
        plan = [(model, self._assignments(model)) for model in targets]
        try:
            for model, assignments in plan:
                for indices, pool in assignments:
                    material = pool.generate(
                        environment.material_manager, self._rng
                    )
                    assign_material(model, indices, material)
        finally:
            clean_generated_materials()

    def _assignments(self, model):
        if self._mode == "slots":
            model.require_preserved_material_slots()
            assignments = []
            seen = set()
            for selector, pool in self._slot_pools:
                for index in selector.resolve(model):
                    if index in seen:
                        raise ValueError(
                            "Multiple selectors target the same slot."
                        )
                    seen.add(index)
                    assignments.append(((index,), pool))
            if not assignments:
                raise ValueError(f"No material slots on {model.get_name()!r}.")
            return assignments
        mesh = model.get_mesh_object().blender_obj
        indices = (
            (mesh.active_material_index,)
            if self._mode == "active"
            else tuple(range(len(mesh.material_slots)))
        )
        return [(indices or (0,), self._pool)]


def _light_targets(environment: Environment, names) -> list[Light]:
    available = environment.get_lights()
    names = (
        tuple(available) if names is None else _names(names, "target_lights")
    )
    if not names or len(set(names)) != len(names):
        raise ValueError(
            "target_lights must select at least one light, without duplicates."
        )
    return [environment.get_light(name) for name in names]


class LightRandomizer(RandomizerNode):
    """Sample properties independently for each named light.

    None selects all registered lights; explicit names match exactly. Each
    configured property is fixed, a uniform range (including RGB), or for
    ``shape`` a list of area shapes. See config.LightConfig for properties and
    units. Properties unsupported by a target's light type raise before any
    target changes; use separate nodes for different light types.

    Example:
        randomizer = LightRandomizer(
            config.LightRandomizerConfig(
                properties={"power": {"min": 20.0, "max": 60.0}}
            ),
            ["key"],
        )
    """

    def __init__(
        self,
        randomizer_config: config.LightRandomizerConfig,
        target_lights: Sequence[str] | None = None,
    ) -> None:
        """Create the node for a set of lights.

        Args:
            randomizer_config: Property values, ranges or choices, and seed.
            target_lights: Exact names of the target lights. None targets all
                registered lights.
        """
        self.randomizer_config = randomizer_config
        self.target_lights = (
            None
            if target_lights is None
            else _names(target_lights, "target_lights")
        )
        self._rng = (
            random
            if randomizer_config.seed is None
            else random.Random(randomizer_config.seed)
        )

    def randomize(self, environment: Environment) -> None:
        lights = _light_targets(environment, self.target_lights)
        # Validate range endpoints and all choices before sampling or mutation.
        for light in lights:
            self.randomizer_config.check_light_type(light.get_type())
        for light in lights:
            sampled = {}
            for name, value in self.randomizer_config.properties.items():
                if isinstance(value, Uniform):
                    values = [
                        self._rng.uniform(low, high)
                        for low, high in zip(value.min.flat, value.max.flat)
                    ]
                    value = values if value.min.shape else values[0]
                elif name == "shape" and not isinstance(value, str):
                    value = self._rng.choice(value)
                sampled[name] = value
            light.set_properties(**sampled)


class LightPoseRandomizer(RandomizerNode):
    """Sample poses with clearance from every visible mesh in the environment.

    The configured sampler returns a location in meters and XYZ Euler angles
    in radians; shell_sampler aims the light at its center, so use positive
    elevations for overhead lights. Run after model placement and
    LightRandomizer so the bounds and emitter sizes are current. Clearance is
    checked at the time this node runs.

    The clearance is measured from the emitter's enclosing sphere to each
    mesh's world AABB, including containers and unregistered environment
    meshes. SUN has no finite position and is exempt. An exhausted retry
    budget raises RuntimeError and leaves every targeted light's pose
    unchanged.

    Example:
        randomizer = LightPoseRandomizer(
            config.LightPoseRandomizerConfig(min_distance=0.3), ["key"]
        )
    """

    def __init__(
        self,
        randomizer_config: config.LightPoseRandomizerConfig,
        target_lights: Sequence[str] | None = None,
    ) -> None:
        """Create the node for a set of lights.

        Args:
            randomizer_config: Pose sampler, its parameters, clearance and
                retry budget.
            target_lights: Exact names of the target lights. None targets all
                registered lights.
        """
        self.randomizer_config = randomizer_config
        self.target_lights = (
            None
            if target_lights is None
            else _names(target_lights, "target_lights")
        )
        self._pose_sampling_function = CAMERA_POSE_SAMPLERS[
            randomizer_config.sampler
        ]

    def randomize(self, environment: Environment) -> None:
        cfg = self.randomizer_config
        lights = _light_targets(environment, self.target_lights)
        bpy.context.view_layer.update()
        # shortcut: AABBs over-reject; use mesh distances for tight placements.
        bounds = [
            mesh_object.get_bound_box()
            for mesh_object in get_all_mesh_objects()
            if not mesh_object.is_hidden()
        ]
        boxes = [
            (corners.min(axis=0), corners.max(axis=0)) for corners in bounds
        ]
        params = cfg.params.to_dict()
        poses = []
        for light in lights:
            clearance = cfg.min_distance + light.get_emitter_radius()
            for _ in range(cfg.max_tries):
                location, rotation = self._pose_sampling_function(
                    environment=environment, **params
                )
                location = config.as_vector3(
                    location, "sampled light location"
                )
                rotation = config.as_vector3(
                    rotation, "sampled light rotation"
                )
                if light.get_type() == "SUN" or all(
                    np.linalg.norm(location - np.clip(location, lower, upper))
                    > clearance
                    for lower, upper in boxes
                ):
                    poses.append((light, location, rotation))
                    break
            else:
                raise RuntimeError(
                    f"Could not place light {light.get_name()!r} with "
                    f"{cfg.min_distance} m clearance after {cfg.max_tries} attempts."
                )
        for light, location, rotation in poses:
            light.set_location(location)
            light.set_rotation(rotation)


class CameraPoseRandomizer(RandomizerNode):
    """Sample one camera pose per rendered view.

    Example:
        randomizer = CameraPoseRandomizer(
            config.CameraPoseRandomizerConfig(
                sampler="shell_sampler", number_of_views=2
            )
        )
    """

    def __init__(
        self, randomizer_config: config.CameraPoseRandomizerConfig
    ) -> None:
        """
        Initialize the internal states.

        Args:
            randomizer_config: config.CameraPoseRandomizerConfig
                Pose sampler, its parameters, and the number of views per
                scene.
        """
        self.randomizer_config = randomizer_config
        self._pose_sampling_function = CAMERA_POSE_SAMPLERS[
            randomizer_config.sampler
        ]

    def randomize(self, environment: Environment) -> None:
        """
        Sample a camera pose for each configured view with the configured
        sampler and its parameters.

        Args:
            environment: Environment
                Environment containing the camera.
        """
        camera = environment.get_camera()
        params = self.randomizer_config.params.to_dict()
        for frame in range(self.randomizer_config.number_of_views):
            location, rotation_euler = self._pose_sampling_function(
                environment=environment, **params
            )

            rotation_matrix = Euler(rotation_euler).to_matrix()

            camera.set_camera_pose(
                location=location, rotation=rotation_matrix, frame=frame
            )
