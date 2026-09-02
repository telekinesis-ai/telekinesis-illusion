"""
Defines helper function for sampling camera pose.
"""

from typing import List
import numpy as np

from loguru import logger

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported (blenderproc pulls it in). See blender_env.py.
isolate_user_extensions()

import blenderproc as bproc  # type: ignore
from mathutils import Matrix  # Only importable once bpy is loaded

from telekinesis.illusion.core.context import Context
from telekinesis.illusion.types.object import compute_poi


def _resolve_target_objects(
    context: Context,
    target_names: List[str] | None = None,
    visible_only: bool = True,
) -> List:
    """
    Resolve objects from context by optional name prefixes and visibility.
    """
    objects = context.get_objects()

    if target_names is None:
        resolved = list(objects.values())
    else:
        prefixes = tuple(target_names)
        resolved = [
            obj for name, obj in objects.items() if name.startswith(prefixes)
        ]

    if visible_only:
        visible = [obj for obj in resolved if not obj.is_hidden()]
        if visible:
            return visible

    return resolved


def shell_sampler(
    context: Context = None,
    center: np.ndarray | None | List[str] = None,
    radius_min: float = 0.4,
    radius_max: float = 0.6,
    elevation_min: float = -90.0,
    elevation_max: float = 90.0,
    azimuth_min: float = -180.0,
    azimuth_max: float = 180.0,
    inplane_rot_min: float = -60.0,
    inplane_rot_max: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Samples the camera location on a spherical shell around the center.

    This method determines a point of interest (POI) to look at based on
    `center`:
    - If `center` is None, the POI is computed from visible scene objects.
    - If `center` is a list, the POI is computed from these object prefixes.
    - If `center` is an np.ndarray, it is used directly as the POI.
    """
    if center is None:
        target_objects = _resolve_target_objects(context, visible_only=True)
        if not target_objects:
            raise ValueError("No objects available for shell_sampler center.")
        center = compute_poi(target_objects)
    elif isinstance(center, list):
        target_objects = _resolve_target_objects(
            context=context, target_names=center, visible_only=True
        )
        if not target_objects:
            raise ValueError(
                "No objects resolved for provided `center` name prefixes."
            )
        center = compute_poi(target_objects)
    elif isinstance(center, np.ndarray):
        pass
    else:
        raise TypeError("`center` must be None, list[str], or np.ndarray.")

    location = bproc.sampler.shell(
        center=center,
        radius_min=radius_min,
        radius_max=radius_max,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        azimuth_min=azimuth_min,
        azimuth_max=azimuth_max,
    )

    forward_vec = center - location
    # BlenderProc expects in-plane rotation in radians.
    inplane_rot = np.deg2rad(
        np.random.uniform(inplane_rot_min, inplane_rot_max)
    )
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        forward_vec, inplane_rot=inplane_rot
    )
    rotation_euler = Matrix(rotation_matrix.tolist()).to_euler("XYZ")

    return location, np.array(rotation_euler)


def volume_sampler(
    context: Context = None,
    point_of_interst: np.ndarray | None | List[str] = None,
    volume_size: tuple[float, float, float] | None = None,
    distance_range: tuple[float, float] | None = None,
    volume_center: np.ndarray | None = None,
    inplane_rot_min: float = -60.0,
    inplane_rot_max: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a camera location in a volume and orient it towards a POI.
    """
    if volume_size and not all(dimension > 0 for dimension in volume_size):
        raise ValueError("volume_size must contain only positive values.")

    poi = np.array([0.0, 0.0, 0.0], dtype=float)
    bbox_target_names = None

    if point_of_interst is None:
        target_objects = _resolve_target_objects(context, visible_only=True)
        if not target_objects:
            raise ValueError("No objects available to compute camera POI.")
        poi = compute_poi(target_objects)
        logger.info(
            "point_of_interst is None. Using POI {} from visible objects.", poi
        )
    elif isinstance(point_of_interst, list):
        if all(isinstance(x, str) for x in point_of_interst):
            bbox_target_names = point_of_interst
            target_objects = _resolve_target_objects(
                context=context,
                target_names=point_of_interst,
                visible_only=True,
            )
            if not target_objects:
                raise ValueError(
                    "No objects resolved for provided `point_of_interst` prefixes."
                )
            poi = compute_poi(target_objects)
        elif all(isinstance(x, (float, int)) for x in point_of_interst):
            # list of numeric coordinates
            poi = np.array(point_of_interst, dtype=float)
    elif isinstance(point_of_interst, np.ndarray):
        poi = point_of_interst
    else:
        raise TypeError(
            "`point_of_interst` must be None, list[str], or np.ndarray."
        )

    bbox = None
    if volume_size is None or volume_center is None:
        bbox = compute_axis_aligned_bbox(
            context=context, target_objects=bbox_target_names, visible_only=True
        )

    if volume_size is None:
        length, width, height, _ = bbox
        logger.info(
            "volume_size is None. Using AABB-based size (l={}, w={}, h={}).",
            length,
            width,
            height,
        )
    else:
        length, width, height = volume_size

    if volume_center is None:
        if bbox is None:
            _, _, _, center = compute_axis_aligned_bbox(
                context=context,
                target_objects=bbox_target_names,
                visible_only=True,
            )
        else:
            center = bbox[3]
        volume_center = center
        logger.info(
            "volume_center is None. Using AABB center {}.", volume_center
        )

    local_location = np.random.uniform(
        (-length / 2.0, -width / 2.0, -height / 2.0),
        (length / 2.0, width / 2.0, height / 2.0),
    )
    location = local_location + volume_center

    if distance_range is None:
        focal_length_mm = context.get_camera().focal_length
        sensor_width = context.get_camera().sensor_width
        sensor_height = context.get_camera().sensor_height

        max_distance = min_distance = autocompute_distance(
            focal_length_mm, sensor_width, sensor_height, length, width, height
        )
        logger.info(
            "distance_range is None. Autocomputed min=max={} .", max_distance
        )
    else:
        min_distance, max_distance = distance_range

    distance = np.random.uniform(min_distance, max_distance)
    location += np.array([0.0, 0.0, distance])

    forward_vec = poi - location
    # BlenderProc expects in-plane rotation in radians.
    inplane_rot = np.deg2rad(
        np.random.uniform(inplane_rot_min, inplane_rot_max)
    )
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        forward_vec, inplane_rot=inplane_rot
    )
    rotation_euler = Matrix(rotation_matrix.tolist()).to_euler("XYZ")

    return location, np.array(rotation_euler)


def compute_axis_aligned_bbox(
    context: Context = None,
    target_objects: None | List[str] = None,
    visible_only: bool = True,
) -> tuple[float, float, float, np.ndarray]:
    """
    Compute the world-space axis-aligned bounding box of selected objects.
    """
    resolved = _resolve_target_objects(
        context=context, target_names=target_objects, visible_only=visible_only
    )
    if not resolved:
        raise ValueError(
            "No objects available to compute an axis-aligned bbox."
        )

    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = -mins
    for obj in resolved:
        bb = obj.get_object().get_bound_box()  # (8, 3)
        mins = np.minimum(mins, bb.min(axis=0))
        maxs = np.maximum(maxs, bb.max(axis=0))
    length, width, height = maxs - mins

    x_min, y_min, z_min = mins
    x_max, y_max, z_max = maxs
    volume_center = np.array(
        [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    )

    return length, width, height, volume_center


def autocompute_distance(
    focal_length_mm: float,
    sensor_width: float,
    sensor_height: float,
    bbox_length: float,
    bbox_width: float,
    bbox_height: float,
) -> float:
    """
    Estimate a camera distance that frames the scene bounding box.
    """
    x_c_max = max(bbox_length, bbox_width)
    x_i_max = max(sensor_width, sensor_height)
    distance = focal_length_mm * x_c_max / x_i_max + bbox_height / 2.0
    return distance


CAMERA_POSE_SAMPLERS = {
    "volume_sampler": volume_sampler,
    "shell_sampler": shell_sampler,
}
