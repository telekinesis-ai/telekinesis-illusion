"""Registered scene lights, created from their configs."""

import dataclasses

from telekinesis.illusion import config
from telekinesis.illusion.types import light


class LightManager:
    """Create and own the scene lights described by registered configs.

    add() creates the Blender light from a LightConfig and stores it under its
    exact name. Lights have no selection groups; randomizers select them by
    name. Treat the dictionary returned by get_lights() as read-only.

    Example:
        manager = LightManager()
        manager.add(
            config.LightConfig("key", "POINT", properties={"power": 40.0})
        )
    """

    def __init__(self) -> None:
        """Create the owned, empty light registry."""
        self._lights: dict[str, light.Light] = {}

    def add(self, light_config: config.LightConfig) -> config.LightConfig:
        """Create one light from its config and register it by name.

        Returns a copy of the config. Raises TypeError for the wrong config
        type and ValueError if the name is already registered or a scene object
        with that name exists. Existing lights are never overwritten.
        """
        if not isinstance(light_config, config.LightConfig):
            raise TypeError("light_config must be a LightConfig.")
        if light_config.light_name in self._lights:
            raise ValueError(
                f"Light {light_config.light_name!r} already exists."
            )
        self._lights[light_config.light_name] = light.Light(light_config)
        return dataclasses.replace(light_config)

    def get_lights(self) -> dict[str, light.Light]:
        """Return lights by their exact registered names."""
        return self._lights

    def get_light(self, light_name: str) -> light.Light:
        """Return a registered light; raise ValueError for an unknown name."""
        if light_name not in self._lights:
            raise ValueError(f"No light with the name {light_name!r}.")
        return self._lights[light_name]
