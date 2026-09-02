"""
Defines helper function for sampling pose over the upper region of an object.
"""

from typing import List, Tuple
import numpy as np
import math
import random

from loguru import logger

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported (blenderproc pulls it in). See blender_env.py.
isolate_user_extensions()

import blenderproc as bproc  # type: ignore
from mathutils import Vector  # Only importable once bpy is loaded

from telekinesis.illusion.core.context import Context
from telekinesis.illusion.types.object import Object


def upper_region_sampler(
    context: Context,
    object_names_to_sample_on: str | List[str],
    face_sample_range: np.ndarray | List[float] | None = None,
    min_height: float = 0.0,
    max_height: float = 1.0,
    use_ray_trace_check: bool = False,
    upper_dir: np.ndarray | List[float] | None = None,
    use_upper_dir: bool = True,
) -> np.ndarray:
    """
    Based on:
    https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/sampler/UpperRegionSampler.py

    Uniformly samples 3-dimensional value over the bounding box of the specified
    objects (can be just a plane) in the defined upper direction.
    If "use_upper_dir" is False, samples along the face normal closest to
    "upper_dir". The sampling volume results in a parallelepiped.
    "min_height" and "max_height" define the sampling distance from
    the face.

    .. code-block:: python

    UpperRegionSampler.sample(
        objects_to_sample_on=objs,
        min_height=1.5,
        max_height=1.8
    )

    Args:
        object_names_to_sample_on: str or List[str]
            Object names, on which to sample on.
        face_sample_range: np.ndarray or List[float] or None
            Restricts the area on the face where objects are sampled.
            Specifically describes relative lengths of both face vectors between
            which points are sampled. Default: [0.0, 1.0].

            Example:
            [0.0, 1.0] (default) — samples anywhere on the entire face.

            [0.25, 0.75] — samples only in the center 50% of the face along both
            axes, effectively a smaller rectangle in the middle.

            [0.0, 0.5] — samples only in the first half of the face along both
            axes, confining points to one quarter (corner region) of the face.

        min_height: float
            Minimum distance to the bounding box that a point is sampled on.
        max_height: float
            Maximum distance to the bounding box that a point is sampled on.
        use_ray_trace_check: bool
            Toggles using a ray casting towards the sampled object (if the
            object is directly below the sampled position is the position
            accepted).
        upper_dir: np.ndarray or List[float] or None
            The 'up' direction of the sampling box. Default: [0.0, 0.0, 1.0].
        use_upper_dir: bool
            Toggles using a ray casting towards the sampled object (if the
            object is directly below the sampled position is the position
            accepted).

    Return:
        A sampled value as a np.ndarray.
    """

    if face_sample_range is None:
        face_sample_range = [0.0, 1.0]
    if upper_dir is None:
        upper_dir = [0.0, 0.0, 1.0]

    face_sample_range = np.array(face_sample_range)
    upper_dir = np.array(upper_dir)
    upper_dir /= np.linalg.norm(upper_dir)

    if not isinstance(object_names_to_sample_on, list):
        object_names_to_sample_on = [object_names_to_sample_on]

    # Retrieve the MeshObjects from provided illusion objects
    # Get object from context
    objects_to_sample_on = []
    for object_name in object_names_to_sample_on:
        illusion_object = context.get_objects_by_name(object_name)
        bproc_mesh_object = illusion_object.get_object()
        objects_to_sample_on.append(bproc_mesh_object)

    if max_height < min_height:
        raise RuntimeError(
            f"The minimum height ({min_height}) must be smaller than the "
            f"maximum height ({max_height})!"
        )

    # Determine for each object in objects the region, where to sample on
    regions = [
        select_upper_region(obj, upper_dir) for obj in objects_to_sample_on
    ]

    if regions and len(regions) == len(objects_to_sample_on):
        selected_region_id = random.randint(0, len(regions) - 1)
        selected_region = regions[selected_region_id]
        obj = objects_to_sample_on[selected_region_id]

        if use_ray_trace_check:
            inv_world_matrix = np.linalg.inv(obj.get_local2world_mat())
        while True:
            ret = selected_region.sample_point(face_sample_range)
            dir_val = upper_dir if use_upper_dir else selected_region.normal()
            ret += dir_val * random.uniform(min_height, max_height)
            if use_ray_trace_check:
                # Transform the coords into the reference frame of the object
                c_ret = inv_world_matrix @ np.concatenate((ret, [1]), 0)
                neg_dir_val = -1.0 * dir_val
                c_dir = inv_world_matrix @ np.concatenate((neg_dir_val, [0]), 0)
                # Check if the object was hit
                hit, _, _, _ = obj.ray_cast(c_ret[:3], c_dir[:3])
                if hit:  # If the object was hit return
                    break
            else:
                break
        logger.info(
            f"Sampled location {ret} over the upper region of object of "
            f"{object_names_to_sample_on}."
        )
        return np.array(ret)
    raise RuntimeError(
        "The amount of regions is either zero or does not match the amount of"
        "objects!"
    )


def calc_vec_and_normals(
    face: List[np.ndarray],
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Calculates the two vectors, which lie in the plane of the face and the
    normal of the face.

    Args:
        face: List[np.ndarray]
            Four corner coordinates of a face. Type: [4x[3xfloat]].

    Return:
        Returns a tuple of two vectors in the plane, and the normal.
    """
    vec1 = face[1] - face[0]
    vec2 = face[3] - face[0]
    normal = np.cross(vec1, vec2)
    normal /= np.linalg.norm(normal)
    return (vec1, vec2), normal


def select_upper_region(bproc_mesh_object, upper_dir: np.ndarray) -> "Region2D":
    """
    Pick the face of the object's bounding box whose normal is most aligned with
    'upper_dir' and return it as a Region2D (base point + in-plane vectors +
    normal).

    Args:
        bproc_mesh_object:
            A BlenderProc mesh object exposing 'get_bound_box()' and 'get_name()'.
        upper_dir: np.ndarray
            Unit-length 'up' direction to compare each face normal against.

    Returns:
        A Region2D describing the selected face.

    Raises:
        RuntimeError: If no suitable face is found.
    """
    bb = bproc_mesh_object.get_bound_box()
    faces = [
        [bb[0], bb[1], bb[2], bb[3]],
        [bb[0], bb[4], bb[5], bb[1]],
        [bb[1], bb[5], bb[6], bb[2]],
        [bb[6], bb[7], bb[3], bb[2]],
        [bb[3], bb[7], bb[4], bb[0]],
        [bb[7], bb[6], bb[5], bb[4]],
    ]
    # Select the face, which has the smallest angle to the upper direction
    min_diff_angle = 2 * math.pi
    selected_face = None
    for face in faces:
        # Calc the normal of all faces
        _, normal = calc_vec_and_normals(face)
        diff_angle = math.acos(normal.dot(upper_dir))
        if diff_angle < min_diff_angle:
            min_diff_angle = diff_angle
            selected_face = face
    # Save the selected face values
    if selected_face is None:
        raise RuntimeError(
            f"Couldn't find a face, for this obj: {bproc_mesh_object.get_name()}"
        )
    vectors, normal = calc_vec_and_normals(selected_face)
    base_point = selected_face[0]
    return Region2D(vectors, normal, base_point)


class Region2D:
    """Helper class for UpperRegionSampler: Defines a 2D region in 3D."""

    def __init__(
        self,
        vectors: Tuple[np.ndarray, np.ndarray],
        normal: np.ndarray,
        base_point: np.ndarray,
    ) -> None:
        self._vectors = (
            vectors  # the two vectors which lie in the selected face
        )
        self._normal = normal  # the normal of the selected face
        self._base_point = base_point  # the base point of the selected face

    def sample_point(self, face_sample_range: np.ndarray) -> np.ndarray:
        """
        Samples a point in the 2D Region

        Args:
            face_sample_range: np.ndarray
                Relative lengths of both face vectors between which points are
                sampled.

        Return:
            A sampled point as a np.ndarray.
        """
        ret = self._base_point.copy()
        # Walk over both vectors in the plane and determine a distance in both
        # direction
        for vec in self._vectors:
            ret += vec * random.uniform(
                face_sample_range[0], face_sample_range[1]
            )
        return ret

    def normal(self):
        """
        Returns the normal of the region.
        """
        return self._normal

    def base_point(self) -> np.ndarray:
        """
        Returns the base point (corner) of the region.
        """
        return self._base_point

    def vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the two in-plane vectors spanning the region.
        """
        return self._vectors
