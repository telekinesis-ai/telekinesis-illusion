"""Compatibility imports. New code should use Scene."""

from telekinesis.illusion.core.scene import Scene, MaterialManager

Context = Scene
__all__ = ["Context", "Scene", "MaterialManager"]
