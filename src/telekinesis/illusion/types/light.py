"""Named scene lights with validated, type-specific Blender properties."""

import math

import numpy as np

from telekinesis.illusion import config
from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import bpy
from telekinesis.illusion.blender.types.light import Light as BlenderLight


class Light:
    """A named scene light, using meters and XYZ Euler radians.

    The properties accepted by :meth:`set_properties` and their units are
    those of ``config.LightConfig``. Omitted properties retain the light
    defaults. Lights have no annotation category or rigid body. Their local
    -Z axis is the emission direction.

    Example:
        light = Light(config.LightConfig("key", "POINT", properties={"power": 40.0}))
    """

    def __init__(self, light_config: config.LightConfig) -> None:
        """Create the light in the Blender scene.

        Args:
            light_config: config.LightConfig
                Name, type, pose and properties of the light.

        Raises:
            ValueError: A scene object with the light's name already exists.
        """
        if light_config.light_name in bpy.data.objects:
            raise ValueError(
                f"A scene object named {light_config.light_name!r} already "
                "exists."
            )
        self._light = BlenderLight(
            light_type=light_config.light_type, name=light_config.light_name
        )
        self.set_properties(**light_config.properties)
        self.set_location(light_config.location)
        self.set_rotation(light_config.rotation)

    def set_properties(self, **properties) -> None:
        """Validate all supplied properties before changing the light."""
        values = config.validate_light_properties(self.get_type(), properties)
        for attribute, value in values.items():
            setattr(self._light.blender_obj.data, attribute, value)

    def get_light(self) -> BlenderLight:
        """Return the underlying Blender light wrapper."""
        return self._light

    def get_properties(self) -> dict:
        """Return a snapshot using the same names as set_properties()."""
        data = self._light.blender_obj.data
        properties = {"color": tuple(data.color)}
        properties.update(
            (name, getattr(data, attribute))
            for name, (attribute, types, _, _) in config.LIGHT_PROPERTIES.items()
            if data.type in types
        )
        if data.type == "AREA":
            properties["shape"] = data.shape
        return properties

    def get_name(self) -> str:
        return self._light.get_name()

    def get_type(self) -> str:
        return self._light.get_type()

    def get_location(self) -> np.ndarray:
        return self._light.get_location()

    def set_location(self, location) -> None:
        self._light.set_location(config.as_vector3(location, "location"))

    def get_rotation(self) -> np.ndarray:
        return self._light.get_rotation_euler()

    def set_rotation(self, rotation) -> None:
        """Set XYZ Euler angles in radians."""
        self._light.set_rotation_euler(config.as_vector3(rotation, "rotation"))

    def get_emitter_radius(self) -> float:
        """Radius of a sphere enclosing the emitter, for placement clearance."""
        data = self._light.blender_obj.data
        if data.type == "SUN":
            return 0.0
        if data.type != "AREA":
            return data.shadow_soft_size
        if data.shape == "DISK":
            return data.size / 2
        if data.shape == "ELLIPSE":
            return max(data.size, data.size_y) / 2
        height = data.size_y if data.shape == "RECTANGLE" else data.size
        return math.hypot(data.size, height) / 2
