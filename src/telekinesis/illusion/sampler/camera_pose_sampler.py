"""
Defines helper function for sampling camera pose.
"""

from typing import List
import numpy as np

from loguru import logger

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported. See blender_env.py for why.
isolate_user_extensions()

from telekinesis.illusion.blender.camera import settings as camera_settings
from telekinesis.illusion.blender.sampling.shell import shell as sample_shell
from mathutils import Matrix  # Only importable once bpy is loaded

from telekinesis.illusion.core.environment import Environment
from telekinesis.illusion.types.model import compute_poi


def _resolve_target_models(
    environment: Environment,
    target_names: List[str] | None = None,
    visible_only: bool = True,
) -> List:
    """
    Resolve models from environment by optional name prefixes and visibility.
    """
    models = environment.get_models()

    if target_names is None:
        resolved = list(models.values())
    else:
        prefixes = tuple(target_names)
        resolved = [
            model
            for name, model in models.items()
            if name.startswith(prefixes)
        ]

    if visible_only:
        visible = [model for model in resolved if not model.is_hidden()]
        if visible:
            return visible

    return resolved


def shell_sampler(
    environment: Environment = None,
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
    - If `center` is None, the POI is computed from visible scene models.
    - If `center` is a list, the POI is computed from these model prefixes.
    - If `center` is an np.ndarray, it is used directly as the POI.
    """
    if center is None:
        target_models = _resolve_target_models(environment, visible_only=True)
        if not target_models:
            raise ValueError("No models available for shell_sampler center.")
        center = compute_poi(target_models)
    elif isinstance(center, list):
        target_models = _resolve_target_models(
            environment=environment, target_names=center, visible_only=True
        )
        if not target_models:
            raise ValueError(
                "No models resolved for provided `center` name prefixes."
            )
        center = compute_poi(target_models)
    elif isinstance(center, np.ndarray):
        pass
    else:
        raise TypeError("`center` must be None, list[str], or np.ndarray.")

    location = sample_shell(
        center=center,
        radius_min=radius_min,
        radius_max=radius_max,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        azimuth_min=azimuth_min,
        azimuth_max=azimuth_max,
    )

    forward_vec = center - location
    # Camera pose sampling expects in-plane rotation in radians.
    inplane_rot = np.deg2rad(
        np.random.uniform(inplane_rot_min, inplane_rot_max)
    )
    rotation_matrix = camera_settings.rotation_from_forward_vec(
        forward_vec, inplane_rot=inplane_rot
    )
    rotation_euler = Matrix(rotation_matrix.tolist()).to_euler("XYZ")

    return location, np.array(rotation_euler)


def volume_sampler(
    environment: Environment = None,
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
        target_models = _resolve_target_models(environment, visible_only=True)
        if not target_models:
            raise ValueError("No models available to compute camera POI.")
        poi = compute_poi(target_models)
        logger.info(
            "point_of_interst is None. Using POI {} from visible models.", poi
        )
    elif isinstance(point_of_interst, list):
        if all(isinstance(x, str) for x in point_of_interst):
            bbox_target_names = point_of_interst
            target_models = _resolve_target_models(
                environment=environment,
                target_names=point_of_interst,
                visible_only=True,
            )
            if not target_models:
                raise ValueError(
                    "No models resolved for provided `point_of_interst` "
                    "prefixes."
                )
            poi = compute_poi(target_models)
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
            environment=environment,
            target_models=bbox_target_names,
            visible_only=True,
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
                environment=environment,
                target_models=bbox_target_names,
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
        focal_length_mm = environment.get_camera().focal_length
        sensor_width = environment.get_camera().sensor_width
        sensor_height = environment.get_camera().sensor_height

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
    # Camera pose sampling expects in-plane rotation in radians.
    inplane_rot = np.deg2rad(
        np.random.uniform(inplane_rot_min, inplane_rot_max)
    )
    rotation_matrix = camera_settings.rotation_from_forward_vec(
        forward_vec, inplane_rot=inplane_rot
    )
    rotation_euler = Matrix(rotation_matrix.tolist()).to_euler("XYZ")

    return location, np.array(rotation_euler)


def compute_axis_aligned_bbox(
    environment: Environment = None,
    target_models: None | List[str] = None,
    visible_only: bool = True,
) -> tuple[float, float, float, np.ndarray]:
    """
    Compute the world-space axis-aligned bounding box of selected models.
    """
    resolved = _resolve_target_models(
        environment=environment,
        target_names=target_models,
        visible_only=visible_only,
    )
    if not resolved:
        raise ValueError(
            "No models available to compute an axis-aligned bbox."
        )

    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = -mins
    for model in resolved:
        bb = model.get_mesh_object().get_bound_box()  # (8, 3)
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
    Estimate a camera distance that frames the environment bounding box.
    """
    x_c_max = max(bbox_length, bbox_width)
    x_i_max = max(sensor_width, sensor_height)
    distance = focal_length_mm * x_c_max / x_i_max + bbox_height / 2.0
    return distance


CAMERA_POSE_SAMPLERS = {
    "volume_sampler": volume_sampler,
    "shell_sampler": shell_sampler,
}
