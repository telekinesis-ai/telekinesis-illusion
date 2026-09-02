"""
Defines the RandomizerNode class that is the base class for all randomizers
nodes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Any
import random
import numpy as np
from loguru import logger
from pathlib import Path

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported. See blender_env.py for why.
isolate_user_extensions()

import bpy
import mathutils
from mathutils import Matrix, Vector, Euler
import blenderproc as bproc
from blenderproc.python.types.MeshObjectUtility import (
    MeshObject,
    get_all_mesh_objects,
)
from blenderproc.python.utility.CollisionUtility import CollisionUtility
from blenderproc.python.types.EntityUtility import Entity

from telekinesis.illusion.types.distribution import uniform
from telekinesis.illusion.core.context import Context
from telekinesis.illusion.types.camera import Camera
from telekinesis.illusion.types.object import Object
from telekinesis.illusion.utils.assets import resolve_asset_dir

# Stages group the randomizer nodes by what part of the scene they change, so
# a caller can re-run only some of them (see Randomizer.randomize's 'stages'
# argument). Interactive tools use this to re-sample the geometry without
# rebuilding the whole scene.
STAGE_COMPOSITION = "composition"  # which and how many instances are visible
STAGE_POSE = "pose"  # where the visible objects are placed
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

    # NOTE: Maybe check whether it makes more sense to move the context
    # higher in the hierarchy, e.g. to the Randomizer class and provide the
    # context from there.
    @abstractmethod
    def randomize(self, context: Context) -> None:
        pass


class ObjectPoseRandomizer(RandomizerNode):
    """
    Class for randomizing the pose of objects.
    """

    def __init__(
        self,
        pose_sampling_function: Callable[[Object], None],
        target_objects: List[str] = [],
        max_tries: int = 10,
        sample_on_surface: str | None = None,
        surface_name_resolver: Callable[["Context"], str] | None = None,
    ) -> None:
        """
        Initialize the internal state. The function 'pose_sampling_function'
        should take an Object as its argument and implement the sampling logic
        for the objects, whose names are provided in 'target_objects' as a list.

        Example for a valid sampling function:

        .. code-block:: python

            def sample_pose(obj: Object):
                obj.set_location(
                    np.random.uniform((-0.1, -0.1, 0.1), (0.1, 0.1, 0.1))
                )
                obj.set_rotation(
                    np.random.uniform((-180, -180, -180), (180, 180, 180))
                )

        Args:
            target_objects: List[str]
                A list of the target object names.
            pose_sampling_function: Callable[[Object], None]
                A sampling function that takes an Object as its argument and
                sets its location and orientation according to some sampling
                logic. This function is used to randomly place the objects in
                the scene and returns None.
            max_tries: int
                The maximum number of attempts for trying to place the object
                collisionfree with the provided 'pose_sampling_function'.
                If the object can not be placed wihtout collison after
                'max_tries' attempts, it is hidden for the current rendering.
            sample_on_surface: str or None
                If provided, the sampling function samples the poses on this
                surface (static name).
            surface_name_resolver: Callable or None
                If provided and sample_on_surface is None, this callable is
                invoked with the Context at randomize-time to dynamically
                resolve the surface name.
        """
        self.pose_sampling_function = pose_sampling_function
        self.target_objects = target_objects
        self.max_tries = max_tries
        self._sample_on_surface = sample_on_surface
        self._surface_name_resolver = surface_name_resolver

    def randomize(self, context: Context) -> None:
        """
        Randomized the pose of the target objects by sampling their pose with
        'self.pose_sampling_function' and positioning them wihtout collision
        in the scene.

        Args:
            context: Context
                The context where the target objects are.
        """
        if self.target_objects:
            # Get target objects for context
            prefixes = tuple(self.target_objects)

            def is_target(name: str) -> bool:
                return name.startswith(prefixes)

            objects = context.get_objects()
            # Sample on the visible target from self.target_objects
            visible_target_objects = [
                objects[name]
                for name in objects
                if is_target(name) and not objects[name].is_hidden()
            ]
        else:
            # Randomize all visible objects in the scene.
            visible_target_objects = [
                obj
                for obj in context.get_objects().values()
                if not obj.is_hidden()
            ]

        surface_name = self._sample_on_surface
        if surface_name is None and self._surface_name_resolver is not None:
            surface_name = self._surface_name_resolver(context)

        if surface_name:
            surface = context.get_objects_by_name(surface_name)
            self.sample_poses_on_surface(
                context=context,
                objects_to_sample=visible_target_objects,
                sample_pose_func=self.pose_sampling_function,
                surface=surface,
            )
        else:
            self.sample_poses(
                context,
                visible_target_objects,
                objects_to_check_collisions=visible_target_objects,
                sample_pose_func=self.pose_sampling_function,
                max_tries=self.max_tries,
            )

    def sample_poses(
        self,
        context: Context,
        objects_to_sample: List[Object],
        sample_pose_func: Callable[[Object], None],
        objects_to_check_collisions: List[Object] = None,
        max_tries: int = 1000,
        mode_on_failure: str = "last_pose",
    ) -> Dict[Object, Tuple[int, bool]]:
        """
        From: https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/object/ObjectPoseSampler.py
        Samples positions and rotations of selected object inside the sampling
        volume while performing mesh and bounding box collision checks.

        Args:
            context: Context
                The conext for the data generation for retrieving objects.
            objects_to_sample: List[Object]
                A list of mesh objects whose poses are sampled based on the
                given function.
            sample_pose_func: Callable[[Object], None]
                The function to use for sampling the pose of a given object.
            objects_to_check_collisions: List[Object]
                A list of mesh objects who should not be considered when
                checking for collisions.
            max_tries: int
                Amount of tries before giving up on an object and moving to the
                next one.
            mode_on_failure: str
                Define final state of objects that could not be placed without
                collisions within max_tries attempts. Options: 'last_pose',
                'initial_pose'.

        Returns:
            A dict with the objects to sample as keys and a Tuple with the
            number of executed attempts to place the object as first element,
            and a bool whether it has been successfully placed without
            collisions.

        Raises:
            ValueError:
                - If provided failure mode is not allowed
                - If the value of max_tries is not greater than zero
                - If the list of objects_to_sample is empty
        """
        # Check if mode on failure is allowed
        allowed_modes_on_failure = ["last_pose", "initial_pose"]
        if mode_on_failure not in allowed_modes_on_failure:
            raise ValueError(
                f"{mode_on_failure} is not an allowed mode_on_failure."
            )

        # After this many tries we give up on current object and continue with the rest
        if objects_to_check_collisions is None:
            objects_to_check_collisions = list(context.get_objects().values())

        # Among objects_to_sample only check collisions against already placed objects
        cur_objects_to_check_collisions = list(
            set(objects_to_check_collisions) - set(objects_to_sample)
        )

        # Convert to MeshObject in order to use BlenderProc utility functions
        if cur_objects_to_check_collisions:
            cur_objects_to_check_collisions = [
                o.get_object() for o in cur_objects_to_check_collisions
            ]

        if max_tries <= 0:
            raise ValueError(
                f"The value of max_tries must be greater than zero: {max_tries}"
            )

        if not objects_to_sample:
            raise RuntimeError(
                "The list of objects_to_sample can not be empty!"
            )

        # cache to fasten collision detection
        bvh_cache: Dict[str, mathutils.bvhtree.BVHTree] = {}

        sample_results: Dict[Object, Tuple[int, bool]] = {}
        # Track object names that were hidden because of collision
        visible_object_names: List[str] = list(context.get_visible_object_names())
        for obj in objects_to_sample:
            # Store the obejct's initial pose in case we need to place it back
            if mode_on_failure == "initial_pose":
                initial_location = obj.get_location()
                initial_rotation = obj.get_rotation()

            no_collision = True

            amount_of_tries_done = -1

            # Try max_iter amount of times
            for i in range(max_tries):
                # Put the top object in queue at the sampled point in space
                sample_pose_func(obj)

                # Remove bvh cache, as object has changed
                if obj.get_name() in bvh_cache:
                    del bvh_cache[obj.get_name()]

                mesh_obj = obj.get_object()

                no_collision = CollisionUtility.check_intersections(
                    mesh_obj, bvh_cache, cur_objects_to_check_collisions, []
                )

                # If no collision then keep the position
                if no_collision:
                    amount_of_tries_done = i
                    break

            # After placing an object, we will check collisions with it
            cur_objects_to_check_collisions.append(mesh_obj)

            if no_collision:
                logger.info(
                    f"It took {amount_of_tries_done + 1} tries to "
                    f"place {obj.get_name()}"
                )
            else:
                amount_of_tries_done = max_tries
                logger.warning(
                    f"Could not place {obj.get_name()} without "
                    f"collision after {amount_of_tries_done} tries."
                )
                logger.warning("Hiding object for this render.")
                # Hide object, disable rigid body and remove from the visible
                # object names
                mesh_obj.hide()
                obj.disable_rigid_body()
                if obj.get_name() in visible_object_names:
                    visible_object_names.remove(obj.get_name())
                context.set_visible_object_names(visible_object_names)

                if mode_on_failure == "initial_pose":
                    obj.set_location(initial_location)
                    obj.set_rotation(initial_rotation)

            sample_results[obj] = (amount_of_tries_done, no_collision)

        return sample_results

    def sample_poses_on_surface(
        self,
        context: Context,
        objects_to_sample: List[Object],
        sample_pose_func: Callable[[Object], None],
        surface: Object,
        max_tries: int = 100,
        min_distance: float = 0.01,
        max_distance: float = 0.6,
        up_direction: np.ndarray | None = None,
        check_all_bb_corners_over_surface: bool = True,
    ) -> List[Object]:
        """
        TODO: Check if it makes more sense to use a geo node set up...
        From: https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/object/OnSurfaceSampler.py
        Samples objects poses on a surface.

        The objects are positioned slightly above the surface due to the
        non-axis aligned nature of used bounding boxes and possible
        non-alignment of the sampling surface (i.e. on the X-Y hyperplane, can
        be somewhat mitigated with precise "up_direction" value), which leads to
        the objects hovering slightly above the surface. So it is recommended to
        use the PhysicsPositioning module afterwards for realistically looking
        placements of objects on the sampling surface. If placing fails due to
        collisions, the object will be moved back to the intial pose and hidden
        from rendering.

        Args:
            context: Context
                The conext for the data generation for retrieving objects.
            objects_to_sample: List[Object]
                A list of objects that should be sampled above the surface.
            surface: Object
                Object to place objects_to_sample on.
            sample_pose_func: Callable[[Object], None]
                The function to use for sampling the pose of a given object.
            max_tries: int
                Amount of tries before giving up on an object (deleting it) and
                moving to the next one.
            min_distance: float
                Minimum distance to the closest other object from
                objects_to_sample. Center to center.
            max_distance: float
                Maximum distance to the closest other object from
                objects_to_sample. Center to center.
            up_direction: np.ndarray or None
                Normal vector of the side of surface the objects should be
                placed on.
            check_all_bb_corners_over_surface: bool
                If this is True all bounding box corners have to be above the
                surface, else only the center of the object has to be above the
                surface

        Return:
            The list of placed objects.
        """
        if len(set(objects_to_sample)) != len(objects_to_sample):
            raise ValueError(
                "The given list of objects to sample contains duplicates."
                "This leads to complications in the sampling process."
            )

        if up_direction is None:
            up_direction = np.array([0.0, 0.0, 1.0])
        else:
            up_direction /= np.linalg.norm(up_direction)

        # cache to fasten collision detection
        bvh_cache: Dict[str, mathutils.bvhtree.BVHTree] = {}

        placed_objects: List[MeshObject] = []

        for obj in objects_to_sample:
            logger.info(f"Trying to put {obj.get_name()}")
            initial_pose = obj.get_local2world_mat()
            placed_successfully = False

            for i in range(max_tries):
                sample_pose_func(obj)
                # Remove bvh cache, as object has changed
                if obj.get_name() in bvh_cache:
                    del bvh_cache[obj.get_name()]

                mesh_obj = obj.get_object()
                surface_obj = surface.get_object()

                no_collision = CollisionUtility.check_intersections(
                    mesh_obj, bvh_cache, placed_objects, []
                )

                if not no_collision:
                    logger.info("Collision detected, retrying!")
                    continue

                # Remove bvh cache, as object has changed
                if obj.get_name() in bvh_cache:
                    del bvh_cache[obj.get_name()]

                # Select the current object
                bpy.ops.object.select_all(action="DESELECT")
                mesh_obj.blender_obj.select_set(True)
                bpy.context.view_layer.objects.active = mesh_obj.blender_obj
                # Start ray slightly above object origin
                origin = mesh_obj.blender_obj.location + mathutils.Vector(
                    (0, 0, 0.001)
                )
                direction = mathutils.Vector((0, 0, -1))  # cast downward

                depsgraph = bpy.context.evaluated_depsgraph_get()

                # Cast against the target surface only — by construction this
                # cannot self-hit, regardless of the object's geometry or
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
                        f'Ray casting of the object "{mesh_obj.get_name()}" '
                        f"into the direction {direction} didn't hit the target "
                        f"surface object {surface.get_name()} in the scene. "
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
                    f'Placed object "{mesh_obj.get_name()}" successfully at'
                    f"{mesh_obj.get_location()} after {i + 1} iterations!"
                )

                spacing_is_ok = _OnSurfaceSampler.check_spacing(
                    mesh_obj, placed_objects, min_distance, max_distance
                )

                if not spacing_is_ok:
                    logger.info("Bad spacing after drop, retrying!")
                    continue

                no_collision = CollisionUtility.check_intersections(
                    mesh_obj, bvh_cache, placed_objects, []
                )

                if not no_collision:
                    logger.info("Collision detected after drop, retrying!")
                    continue

                placed_objects.append(mesh_obj)
                placed_successfully = True
                break

            if not placed_successfully:
                logger.info(f"Giving up on {obj.get_name()}, hiding...")
                obj.hide(True)
                obj.set_local2world_mat(initial_pose)

        return placed_objects


class _OnSurfaceSampler:
    @staticmethod
    def check_above_surface(
        obj: MeshObject,
        surface: MeshObject,
        up_direction: np.ndarray,
        check_all_bb_corners_over_surface: bool = True,
    ) -> bool:
        """
        Check if all corners of the bounding box are "above" the surface

        Args:
        obj:
            Object for which the check is carried out. Type: blender object.
        surface:
            The surface object.
        up_direction:
            The direction that indicates "above" direction.
        check_all_bb_corners_over_surface:
            If this is True all bounding box corners have to be above the
            surface, else only the center of the object has to be above the
            surface

        Return:
            True if the bounding box is above the surface, False - if not.
        """
        if check_all_bb_corners_over_surface:
            for point in obj.get_bound_box():
                position_is_above_object = surface.position_is_above_object(
                    point + up_direction,
                    -up_direction,
                    check_no_objects_in_between=False,
                )
                if not position_is_above_object:
                    return False
            return True
        center = np.mean(obj.get_bound_box(), axis=0)
        position_is_above_object = surface.position_is_above_object(
            center + up_direction,
            -up_direction,
            check_no_objects_in_between=False,
        )
        return position_is_above_object

    @staticmethod
    def check_spacing(
        obj: MeshObject,
        placed_objects: List[MeshObject],
        min_distance: float,
        max_distance: float,
    ) -> bool:
        """
        Check if object is not too close or too far from previous objects.

        Args:
            obj: MeshObject
                Object for which the check is carried out.
            placed_objects: List[MeshObject]
                A list of already placed objects that should be used for
                checking spacing.
            min_distance: float
                Minimum distance to the closest other object from
                placed_objects. Center to center.
            max_distance: float
                Maximum distance to the closest other object from
                placed_objects. Center to center.

            Return:
                True, if the spacing is correct
        """
        closest_distance = None

        for already_placed in placed_objects:
            distance = np.linalg.norm(
                already_placed.get_location() - obj.get_location()
            )
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance

        return closest_distance is None or (
            min_distance <= closest_distance <= max_distance
        )

    @staticmethod
    def drop(
        obj: MeshObject, up_direction: np.ndarray, surface_height: float
    ) -> None:
        """
        Moves object "down" until its bounding box touches the bounding box of
        the surface. This uses bounding boxes which are not aligned optimally,
        this will cause objects to be placed slightly to high.

        Args:
            obj: MeshObject
                Object to move. Type: blender object.
            up_direction: np.ndarray
                Vector which points into the opposite drop direction.
            surface_height: float
                Height of the surface above its origin.
        """
        obj_bounds = obj.get_bound_box()
        obj_height = min(up_direction.dot(corner) for corner in obj_bounds)

        obj.set_location(
            obj.get_location() - up_direction * (obj_height - surface_height)
        )


class ObjectInstanceRandomizer(RandomizerNode):
    """
    Class for randomizing the number of instances in the scene.
    """

    def __init__(
        self,
        target_objects: List[str] = [],
        min_num_total_objects: int | None = None,
        max_num_total_objects: int | None = None,
    ) -> None:
        """
        Initialize the internal state.

        Args:
            target_objects: List[str]
                A list of the target object names for which we want to randomize
                the number of instances.
            min_num_total_objects (int or None):
                Minimum number of total objects visible in the scene. If None,
                the minimum number of total objects consists of the sum of the
                minimum number of the target_objects instances. Else, the
                randomizer guarantees to make at least the provided minimum
                number of total objects visible in the scene.
            max_num_total_objects: int or None
                Maximum number of total objects visible in the scene. If None,
                the maximum number of total objects consists of the sum of the
                maximum number of the target_objects instances. Else, the
                randomizer guarantees to have maximum the provided number of
                total objects visible in the scene.

        """
        if (
            min_num_total_objects is not None
            and max_num_total_objects is not None
            and min_num_total_objects > max_num_total_objects
        ):
            raise ValueError(
                "min_num_total_objects should be smaller than "
                "max_num_total_objects"
            )

        self.target_objects = target_objects
        self.min_num_total_objects = min_num_total_objects
        self.max_num_total_objects = max_num_total_objects

        # Check if min and max num of total objects is feasable

    def randomize(self, context: Context) -> None:
        """
        Randomized the pose of the target objects by sampling their pose with
        'self.pose_sampling_function' and positioning them wihtout collision
        in the scene.

        Args:
            context: Context
                The context where the target objects are.
        """
        if self.target_objects:
            object_group = context._object_groups
            target_object_group = {
                target_name: list(object_group[target_name])
                for target_name in self.target_objects
            }
            min_per_obj: dict[str, int] = {}
            max_per_obj: dict[str, int] = {}

            # Shuffle each object's instances so we sample uniformly within-object
            for obj_name, instances in target_object_group.items():
                # assuming these are identical across instances of the same object type
                min_per_obj[obj_name] = instances[0].min_number_instances
                max_per_obj[obj_name] = instances[
                    0
                ].max_number_instances  # == len(instances) per your note
                random.shuffle(instances)

            objs = list(target_object_group.keys())

            # Global total selection
            min_total = (
                self.min_num_total_objects
                if self.min_num_total_objects is not None
                else sum(min_per_obj.values())
            )
            max_total = (
                self.max_num_total_objects
                if self.max_num_total_objects is not None
                else sum(max_per_obj.values())
            )
            max_total = min(max_total, sum(max_per_obj.values()))

            if min_total > max_total:
                raise ValueError(
                    f"min_total_objects ({min_total}) > max_total_objects ({max_total})"
                )

            total = random.randint(min_total, max_total)

            # Feasibility checks
            min_sum = 0
            for obj_name in objs:
                mn = min_per_obj.get(obj_name, 0)
                mx = max_per_obj[obj_name]
                if mn > mx:
                    raise ValueError(f"{obj_name}: min {mn} > max {mx}")
                min_sum += mn

            if min_sum > total:
                raise ValueError(
                    f"Sum of minima ({min_sum}) exceeds total ({total})"
                )

            # Allocate minima: Either the min or 0 if not present
            k = {obj_name: min_per_obj.get(obj_name, 0) for obj_name in objs}
            remaining = total - sum(k.values())

            # Distribute remaining slots fairly (balanced by fill ratio)
            while remaining > 0:
                candidates = [o for o in objs if k[o] < max_per_obj[o]]
                if not candidates:
                    break

                # Tie-break randomness, then pick lowest fill ratio
                random.shuffle(candidates)
                chosen = min(candidates, key=lambda o: k[o] / max_per_obj[o])

                k[chosen] += 1
                remaining -= 1

            # Materialize sampled instances (lists already shuffled)
            currently_visible_objects = list(context.get_visible_object_names())
            for obj_name in objs:
                for obj in target_object_group[obj_name][: k[obj_name]]:
                    obj.enable_rigid_body()
                    obj.hide(False)
                    currently_visible_objects.append(obj.get_name())
            context.set_visible_object_names(currently_visible_objects)
        else:
            raise RuntimeError("No target objects to randomize.")


class BackgroundRandomizer(RandomizerNode):
    """
    Class for randomizing the background of the scene.
    """

    def __init__(
        self,
        hdris_root: Path | None = None,
        categories: List[str] | None = None,
    ) -> None:
        """
        Initialize internal states and create the catalog dictionary
        usage.

        Args:
            hdris_root: Path or None
                Path to the root directory where the hdris are located. If
                'None', the 'hdris' sub-directory of the default asset
                directory (see resolve_asset_dir()) is used.
            categories: List[str] or None
                List of hdri categories that should be used for the background
                randomization. Available categories are:
                - indoor/industrial
                - indoor/studio
                - indoor/misc
                - outdoor/day
                - outdorr/evening
                - outdorr/morning
                - outdoor/night

        """
        self._background = None
        self._categories = categories
        self._hdris_root = self._validate_hdris_root(hdris_root)
        self.catalog = self._build_catalog(self._categories)

    @property
    def categories(self) -> List[str] | None:
        """
        The hdri categories this randomizer samples the background from.
        """
        return self._categories

    @staticmethod
    def _validate_hdris_root(
        hdris_root: str | Path = None,
    ) -> Path:
        """
        Validates the hdris root directory.

        Args:
            hdris_root: str, Path or None
                Path to the root directory of the hdri assets. If 'None', the
                'hdris' sub-directory of the default asset directory is used.
        """
        if hdris_root is None:
            # Fall back to the 'hdris' sub-directory of the default asset dir
            default_hdris_root = resolve_asset_dir() / "hdris"
            logger.info(f"Using the default HDRIs root '{default_hdris_root}'.")
            return default_hdris_root

        path = Path(hdris_root)

        if not path.exists():
            raise FileNotFoundError(f"HDRIs root '{path}' does not exist.")

        if not path.is_dir():
            raise NotADirectoryError(f"HDRIs root '{path}' is not a directory.")

        return path

    def _build_catalog(self, categories: List[str] | None = None) -> list[Path]:
        """
        Scans self._hdris_root for .exr files and returns them as a list.

        Args:
            categories: List[str] or None
                List of hdri categories that should be used for the background
                randomization. Available categories are lsited in the docstring
                of the __init__ method.

        Raises:
            FileNotFoundError: if no .exr files are found.
        """
        if categories:
            # Include only the exrs for the provided categories
            exrs = sorted(
                p
                for p in self._hdris_root.rglob("*.exr")
                if p.is_file()
                and any(category in p.as_posix() for category in categories)
            )
        else:
            # Include all the categories
            exrs = sorted(
                p for p in self._hdris_root.rglob("*.exr") if p.is_file()
            )

        if not exrs:
            raise FileNotFoundError(
                f"No .exr files found under HDRI root '{self._hdris_root}'."
            )

        return exrs

    def get_random_hdri(self) -> Path:
        """
        Return a random .exr-path from the catalog.

        Returns:
            A random .exr-path as Path() instance.
        """
        return random.choice(self.catalog)

    def randomize(self, context: Context) -> None:
        """
        Sets a random background texture from the hdris root.
        """
        random_hdri_path = self.get_random_hdri()
        background = context.get_background()
        background.set_background_texture(str(random_hdri_path))


class MaterialRandomizer(RandomizerNode):
    """
    Class for randomizing the material of objets.
    """

    def __init__(
        self,
        target_objects: List[str],
        context: Context,
        types: List[str] | None = None,
    ) -> None:
        self.target_objects = target_objects
        self._types = types
        context.material_manager.update_materials(self._types)

    @property
    def types(self) -> List[str] | None:
        """
        The material type tags this randomizer picks from.
        """
        return self._types

    def randomize(self, context: Context) -> None:
        # Get target objects for context
        prefixes = tuple(self.target_objects)

        def is_target(name: str) -> bool:
            return name.startswith(prefixes)

        objects = context.get_objects()
        target_objects = [
            objects[name].get_object() for name in objects if is_target(name)
        ]

        for objects in target_objects:
            random_material = context.material_manager.get_random_material(
                self._types
            )
            objects.blender_obj.active_material = random_material.blender_obj


class CameraPoseRandomizer(RandomizerNode):
    """
    Class for randomizing the pose of the camera.
    """

    def __init__(
        self,
        pose_sampling_function: Callable[..., Tuple[np.ndarray, np.ndarray]],
        number_of_views: int = 0,
        **kwargs,
    ) -> None:
        """
        Initialize the internal states.

        Args:
            pose_sampling_function: Callable[..., Any]
                A sampling function that samples 6-DoF pose and returns a
                location and XYZ Euler angles as seperate np.ndarray.
            number_of_views: int
                Number of views to render a randomized scene.
            **kwargs:
                Named arguments for the pose sampling function.
        """
        self._number_of_views = number_of_views
        self._kwargs = kwargs
        self._pose_sampling_function = pose_sampling_function

    def randomize(self, context: Context) -> None:
        """
        Randomizes the camera pose(s) for the configured number of views
        according to the provided 'pose_sampling_function' and its arguments
        'kwargs'.

        Args:
            context: Context
                Context in that contains the camera.
        """
        camera = context.get_camera()
        for frame in range(self._number_of_views):
            location, rotation_euler = self._pose_sampling_function(
                context=context, **self._kwargs
            )

            rotation_matrix = Euler(rotation_euler).to_matrix()

            camera.set_camera_pose(
                location=location, rotation=rotation_matrix, frame=frame
            )
