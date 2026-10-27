"""The scene camera, created from its config."""

import dataclasses

from telekinesis.illusion.types import camera


class CameraManager:
    """Create and own the scene camera described by a registered config.

    The scene has one camera. add() creates it from a CameraConfig, whose
    unset fields fall back to the CameraConfig defaults. No camera exists
    until one is added.

    Example:
        manager = CameraManager()
        manager.add(camera.CameraConfig(image_width=128, image_height=128))
        scene_camera = manager.get_camera()
    """

    def __init__(self) -> None:
        """Create the manager without a camera."""
        self._camera: camera.Camera | None = None

    def add(self, config: camera.CameraConfig) -> camera.CameraConfig:
        """Create the camera from its config and apply it to the Blender scene.

        Returns a copy of the config. Raises TypeError for the wrong config
        type and ValueError if a camera was already added.
        """
        if not isinstance(config, camera.CameraConfig):
            raise TypeError("config must be a CameraConfig.")
        if self._camera is not None:
            raise ValueError("A camera was already added.")
        self._camera = camera.Camera(config)
        return dataclasses.replace(config)

    def get_camera(self) -> camera.Camera:
        """Return the camera used for rendering.

        Raises RuntimeError if no camera was added.
        """
        if self._camera is None:
            raise RuntimeError("Add a CameraConfig to the environment first.")
        return self._camera
