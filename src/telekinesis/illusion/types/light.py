"""Named scene lights with validated, type-specific Blender properties."""

import math
from numbers import Real

import numpy as np

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import bpy
from blenderproc.python.types.LightUtility import Light as BlenderLight

LIGHT_TYPES = ("POINT", "SUN", "SPOT", "AREA")
AREA_SHAPES = ("SQUARE", "RECTANGLE", "DISK", "ELLIPSE")
_FLOAT_MAX = np.finfo(np.float32).max
# Public name -> Blender attribute, supported types, valid bounds.
_PROPERTIES = {
    "power": ("energy", LIGHT_TYPES, 0.0, _FLOAT_MAX),
    "radius": ("shadow_soft_size", ("POINT", "SPOT"), 0.0, _FLOAT_MAX),
    "size": ("size", ("AREA",), 0.0, _FLOAT_MAX),
    "size_y": ("size_y", ("AREA",), 0.0, _FLOAT_MAX),
    "angle": ("angle", ("SUN",), 0.0, math.pi),
    "spot_size": ("spot_size", ("SPOT",), math.pi / 180, math.pi),
    "spot_blend": ("spot_blend", ("SPOT",), 0.0, 1.0),
}


def _vector(value, name, minimum=-_FLOAT_MAX, maximum=_FLOAT_MAX):
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (3,)
        or not np.isfinite(result).all()
        or np.any(result < minimum)
        or np.any(result > maximum)
    ):
        raise ValueError(
            f"{name} needs three finite values in [{minimum}, {maximum}]."
        )
    return result


class Light:
    """Wrap a BlenderProc light, using meters and XYZ Euler radians.

    ``light_type`` is POINT, SUN, SPOT or AREA. Properties accepted by
    construction and :meth:`set_properties` are:

    - color: linear RGB in [0, 1]; power: watts (SUN: watts/meter squared).
    - POINT/SPOT: radius in meters.
    - AREA: shape (SQUARE, RECTANGLE, DISK, ELLIPSE), size and size_y in meters.
      size_y only affects RECTANGLE and ELLIPSE.
    - SUN: angular diameter ``angle`` in radians.
    - SPOT: cone angle ``spot_size`` in radians and ``spot_blend`` in [0, 1].

    Omitted properties retain BlenderProc's defaults. Lights have no annotation
    category or rigid body. Their local -Z axis is the emission direction.
    """

    def __init__(
        self,
        light_name: str,
        light_type: str = "POINT",
        *,
        location=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        **properties,
    ):
        if not isinstance(light_name, str) or not light_name.strip():
            raise ValueError("light_name must be a non-empty string.")
        if light_name in bpy.data.objects:
            raise ValueError(
                f"A scene object named {light_name!r} already exists."
            )
        values = self._validate_properties(light_type, properties)
        location = _vector(location, "location")
        rotation = _vector(rotation, "rotation")
        self._light = BlenderLight(light_type=light_type, name=light_name)
        for attribute, value in values.items():
            setattr(self._light.blender_obj.data, attribute, value)
        self.set_location(location)
        self.set_rotation(rotation)

    @staticmethod
    def _validate_properties(light_type, properties):
        if light_type not in LIGHT_TYPES:
            raise ValueError(f"light_type must be one of {LIGHT_TYPES}.")
        values = {}
        for name, value in properties.items():
            if name == "color":
                values[name] = _vector(value, name, 0.0, 1.0)
            elif name == "shape":
                if light_type != "AREA" or value not in AREA_SHAPES:
                    raise ValueError(
                        f"shape requires AREA and one of {AREA_SHAPES}."
                    )
                values[name] = value
            else:
                if name not in _PROPERTIES:
                    raise ValueError(f"Unknown light property {name!r}.")
                attribute, types, minimum, maximum = _PROPERTIES[name]
                if light_type not in types:
                    raise ValueError(
                        f"{name} is not supported by {light_type} lights."
                    )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(value)
                    or not minimum <= value <= maximum
                ):
                    raise ValueError(
                        f"{name} must be finite and in [{minimum}, {maximum}]."
                    )
                values[attribute] = value
        return values

    def set_properties(self, **properties) -> None:
        """Validate all supplied properties before changing the light."""
        values = self._validate_properties(self.get_type(), properties)
        for attribute, value in values.items():
            setattr(self._light.blender_obj.data, attribute, value)

    def get_light(self) -> BlenderLight:
        """Return the underlying BlenderProc light."""
        return self._light

    def get_properties(self) -> dict:
        """Return a snapshot using the same names as set_properties()."""
        data = self._light.blender_obj.data
        properties = {"color": tuple(data.color)}
        properties.update(
            (name, getattr(data, attribute))
            for name, (attribute, types, _, _) in _PROPERTIES.items()
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
        self._light.set_location(_vector(location, "location"))

    def get_rotation(self) -> np.ndarray:
        return self._light.get_rotation_euler()

    def set_rotation(self, rotation) -> None:
        """Set XYZ Euler angles in radians."""
        self._light.set_rotation_euler(_vector(rotation, "rotation"))

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
