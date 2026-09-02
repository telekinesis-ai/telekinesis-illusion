"""
Defines the camera class that is reponsible for the camera settings in the
context
"""

from typing import List
from dataclasses import dataclass
import numpy as np

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported. See blender_env.py for why.
isolate_user_extensions()

import bpy
from mathutils import Matrix, Vector, Euler
import blenderproc as bproc  # type: ignore


@dataclass
class CameraConfig:
    # Defaults from BlenderProc
    image_width: int | None = 720
    image_height: int | None = 720
    focal_length: float | None = None
    field_of_view: float | None = None
    clip_start: float | None = 0.1
    clip_end: float | None = 1000
    pixel_aspect_x: float | None = 1
    pixel_aspect_y: float | None = 1
    shift_x: float | None = 0
    shift_y: float | None = 0
    location: np.ndarray | None = None
    rotation: np.ndarray | None = None

    def __post_init__(self) -> None:
        """
        Method for validating the args.

        Raises:
            ValueError:
                - If both focal_length and field_of_view are provided.
                - If focal_length is smaller than 1mm.
        """
        if self.focal_length is not None and self.field_of_view is not None:
            raise ValueError(
                "Provide either 'focal_length' or 'field_of_view', not both."
            )

        if self.focal_length is not None and self.focal_length < 1:
            raise ValueError(
                "The focal length must be greater or equal to 1mm."
            )


class Camera:
    """
    Class for managing the camera in the context.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        """
        Initialize the internal camera states.

        Args:
            image_width: int
                The image width in pixels.
            image_height: int
                The image height in pixels.
            focal_length: float
                Focal length of the lens in millimeters. Must be greater or
                equal 1.
            field_of_view: float
                Field of view (FOV) in radians
            clip_start: float
                Near clipping distance.
            clip_end: float
                Far clipping distance.
            pixel_aspect_x: float
                Pixel aspect ratio along the x-axis.
            pixel_aspect_y: float
                Pixel aspect ratio along the y-axis.
            shift_x: float
                Lens shift in the x direction.
            shift_y: float
                Lens shift in the y direction.

        """
        if config is None:
            config = CameraConfig()

        self.apply_config(config)

    # Use a lightweight proxy-style accessor to avoid updating bpy refs
    def _camera_obj(self):
        return bpy.context.scene.camera  # always fresh

    def _camera_data(self):
        return self._camera_obj().data

    def apply_config(self, config: CameraConfig) -> None:
        """
        Apply the camera configs on the camera class.
        """
        if config.image_width is not None:
            self.image_width = config.image_width
        if config.image_height is not None:
            self.image_height = config.image_height
        if config.focal_length is not None:
            self.focal_length = config.focal_length
        if config.field_of_view is not None:
            self.field_of_view = config.field_of_view
        if config.clip_start is not None:
            self.clip_start = config.clip_start
        if config.clip_end is not None:
            self.clip_end = config.clip_end
        if config.pixel_aspect_x is not None:
            self.pixel_aspect_x = config.pixel_aspect_x
        if config.pixel_aspect_y is not None:
            self.pixel_aspect_y = config.pixel_aspect_y
        if config.shift_x is not None:
            self.shift_x = config.shift_x
        if config.shift_y is not None:
            self.shift_y = config.shift_y
        if config.location is not None:
            self._location = config.location
        else:
            self._location = np.array([0.22912, -0.572652, 0.366659])
        if config.rotation is not None:
            self._rotation = config.rotation
        else:
            self._rotation = np.array([60.0, 0.0, 20.0])

        self._cam2world_matrix = self.set_camera_pose(
            location=self._location, rotation=self._rotation, frame=0
        )

    @property
    def image_width(self) -> int:
        return bpy.context.scene.render.resolution_x

    @image_width.setter
    def image_width(self, value: int) -> None:
        bpy.context.scene.render.resolution_x = value

    @property
    def image_height(self) -> int:
        return bpy.context.scene.render.resolution_y

    @image_height.setter
    def image_height(self, value: int) -> None:
        bpy.context.scene.render.resolution_y = value

    @property
    def focal_length(self) -> float:
        # return self._camera_data.lens
        return self._camera_data().lens

    @focal_length.setter
    def focal_length(self, value: float) -> None:
        if value < 1:
            raise ValueError(
                "The focal length must be greater or equal to 1mm."
            )
        self._camera_data().lens = value

    @property
    def field_of_view(self) -> float:
        return self._camera_data().angle

    @field_of_view.setter
    def field_of_view(self, value: float) -> None:
        self._camera_data().angle = value

    @property
    def clip_start(self) -> float:
        return self._camera_data().clip_start

    @clip_start.setter
    def clip_start(self, value: float) -> None:
        self._camera_data().clip_start = value

    @property
    def clip_end(self) -> float:
        return self._camera_data().clip_end

    @clip_end.setter
    def clip_end(self, value: float) -> None:
        self._camera_data().clip_end = value

    @property
    def pixel_aspect_x(self) -> float:
        return bpy.context.scene.render.pixel_aspect_x

    @pixel_aspect_x.setter
    def pixel_aspect_x(self, value: float) -> None:
        bpy.context.scene.render.pixel_aspect_x = value

    @property
    def pixel_aspect_y(self) -> float:
        return bpy.context.scene.render.pixel_aspect_y

    @pixel_aspect_y.setter
    def pixel_aspect_y(self, value: float) -> None:
        bpy.context.scene.render.pixel_aspect_y = value

    @property
    def shift_x(self) -> float:
        return self._camera_data().shift_x

    @shift_x.setter
    def shift_x(self, value: float) -> None:
        self._camera_data().shift_x = value

    @property
    def shift_y(self) -> float:
        return self._camera_data().shift_y

    @shift_y.setter
    def shift_y(self, value: float) -> None:
        self._camera_data().shift_y = value

    @property
    def sensor_width(self) -> float:
        return self._camera_data().sensor_width

    @property
    def sensor_height(self) -> float:
        return self._camera_data().sensor_height

    def set_camera_pose(
        self,
        location: np.ndarray,
        rotation: np.ndarray | Matrix,
        frame: int = 0,
    ) -> np.ndarray:
        """
        Set the camera pose from a location and rotation np.ndarray.

        Note:
            Blender internally works with keyframes. Illusion uses frame 0 as
            the default frame and utilizes other keyframes only when sampling
            multiple camera poses for rendering the scene.

        Args:
            location: np.ndarray
                The 3D cartesian location (x,y,z) of the camera.
            rotation: np.ndarray or Matrix
                Either XYZ-Euler rotation of the camera in degrees or the
                rotation matrix as Matrix.
            frame: int
                Keyframe that is used for adding a camera pose. Multiple camera
                poses are added as frames. The default camera pose is at frame
                0.

        Returns:
            The camera to world transformation matrix as a np.ndarray.
        """

        def validate_shape(array: np.ndarray) -> bool:
            """
            Helper function to validate whether the provided array describes a
            3D vector.

            Args:
                array: np.ndarray
                    The array that needs to be validated.

            Returns:
                Returns true if the array has either shape (3,) or (3,1).

            Raises:
                ValueError:
                    - If array has not shape (3,) or (3,1).
            """
            valid_shape = (
                array.size == 3
                and array.ndim in (1, 2)
                and (array.ndim == 1 or array.shape[1] == 1)
            )

            if not valid_shape:
                raise ValueError(
                    f"Expected an array of shape (3,) or (3,1), "
                    f"got {array.shape}."
                )

            return valid_shape

        if isinstance(rotation, Matrix):
            rotation_matrix = rotation
        elif isinstance(rotation, np.ndarray):
            if validate_shape(rotation):
                rotation_euler = Euler(np.deg2rad(rotation))
                rotation_matrix = rotation_euler.to_matrix()
        else:
            raise ValueError(
                f"Expected rotation_matrix to be of Type np.ndarray or Matrix, "
                f"got type {type(rotation_matrix)}"
            )

        cam2world_matrix = bproc.math.build_transformation_mat(
            location, rotation_matrix
        )
        bproc.camera.add_camera_pose(cam2world_matrix, frame=frame)
        return cam2world_matrix
