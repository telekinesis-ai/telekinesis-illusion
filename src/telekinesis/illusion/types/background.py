"""
Defines the Background class that manages the background appearance in the
scenes.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union
from loguru import logger
import random
import weakref

from telekinesis.illusion.utils.assets import (
    pick_default_hdri,
    resolve_asset_dir,
)
from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Unlike the other bpy-touching modules here, nothing in this one pulls in
# blenderproc, so this is the only chance to keep Blender's user extensions off
# sys.path before bpy is imported.
isolate_user_extensions()

import bpy


class Background:
    """
    Class for managing the background in the synthetic scenes.
    """

    # Contains weak refs to all background instances
    # As it only uses weak references, instances can still be removed by GC when
    # all other references are gone. If that happens, the instances' weak ref is
    # also automatically removed from the set.
    __refs__: weakref.WeakSet = weakref.WeakSet()

    # Which HDRI pack the initial background is taken from. Discovered rather
    # than hardcoded to a file name, since asset trees ship different HDRIs.
    DEFAULT_HDRI_CATEGORY = "indoor/industrial"

    def __init__(self, assets_root: str | Path = None) -> None:
        """
        Initialize internal states and creates the shader node setup for HDRI
        usage.

        Args:
            assets_root: str, Path or None
                Directory holding the assets. If 'None', it is resolved by
                telekinesis.illusion.utils.assets.resolve_asset_dir().
        """
        self.assets_path = self._validate_assets_root(assets_root)
        self._create_shader_node_setup()
        # Set default background
        self.set_background_texture(
            str(
                pick_default_hdri(
                    self.assets_path / "hdris",
                    preferred_category=self.DEFAULT_HDRI_CATEGORY,
                )
            )
        )
        # Remember that this instance exists
        Background.__refs__.add(self)

    @staticmethod
    def _validate_assets_root(
        assets_root: str | Path = None,
    ) -> Path:
        """
        Validates the assets root directory.

        Args:
            assets_root: str, Path or None
                Path to the root directory of the assets. If 'None', the
                default asset directory is resolved by resolve_asset_dir().
        """
        return resolve_asset_dir(assets_root)

    def _create_shader_node_setup(
        self,
    ) -> None:

        # Variabel for organizing the nodes in Blender GUI
        self.node_gui_location_x = 0

        self.world = bpy.context.scene.world
        self.nodes = self.world.node_tree.nodes
        self.links = self.world.node_tree.links

        self.nodes.clear()

        # Add a texture node, load image and link it
        self.texture_node = self.nodes.new(type="ShaderNodeTexEnvironment")

        # Create background node
        self.background_node = self.nodes.new(type="ShaderNodeBackground")
        self.background_node.inputs["Strength"].default_value = 1.0
        self.background_node.location.x = self.node_gui_location_x

        # Add a mapping node and a texture coordinate node for rotating HDRI
        # This allows to change sun and shadow directions and reflections
        self.mapping_node = self.nodes.new("ShaderNodeMapping")
        # mapping_node.inputs["Rotation"].default_value = rotation_euler
        self.node_gui_location_x += 300
        self.mapping_node.location.x = self.node_gui_location_x

        self.tex_coords_node = self.nodes.new("ShaderNodeTexCoord")
        self.node_gui_location_x += 300
        self.tex_coords_node.location.x = self.node_gui_location_x

        # Create the output node
        self.world_output_node = self.nodes.new(type="ShaderNodeOutputWorld")
        self.node_gui_location_x += 300
        self.world_output_node.location.x = self.node_gui_location_x

        # Connect the nodes
        self.links.new(
            self.texture_node.outputs["Color"],
            self.background_node.inputs["Color"],
        )
        self.links.new(
            self.tex_coords_node.outputs["Generated"],
            self.mapping_node.inputs["Vector"],
        )
        self.links.new(
            self.mapping_node.outputs["Vector"],
            self.texture_node.inputs["Vector"],
        )
        self.links.new(
            self.background_node.outputs["Background"],
            self.world_output_node.inputs["Surface"],
        )

    def _preload_hdris(self):
        """
        TODO: Preloading a large amount of HDRIs might speed up the SDG at
        cost of memory: The ranodmization might be faster because the hdris are
        not loaded every time, but it might be very costly to keep all the hdris
        in the memory during the generation.

        A potential optimization could be the usage of an Least Recently Used -
        style pattern: A smart storage system that holds a fixed amount of data,
        automatically removing the least recently accessed item when it's full.
        This is based on the principle that recently used data is more likely to
        be needed again soon.
        """
        pass

    def _build_default_catalog(self) -> None:
        pass

    def set_background_texture(self, hdri: str) -> None:
        """
        Sets the background of the environment from a the path of the HDRI-file.

        Args:
            hdri: str
                Path to the .exr-file of the HDRI.

        Raises:
            FileNotFoundError:
                - If provided file does not exist.
        """
        if isinstance(hdri, str) and not Path(hdri).is_file():
            raise FileNotFoundError(
                f"Provided file {Path(hdri)} does not exist."
            )

        texture_image = bpy.data.images.load(hdri, check_existing=True)
        self.texture_node.image = texture_image

    def get_name(self) -> str:
        """
        Returns the name of the bpy reference.
        """
        return self.world.name

    def update_blender_ref(
        self,
    ) -> None:
        """
        Updates ALL the contained blender reference, else they get stall when
        using UndoAfterExecution for the physics simulation.
        """
        # Simply update the world reference
        self.world = bpy.context.scene.world
        self.nodes = self.world.node_tree.nodes
        self.links = self.world.node_tree.links

        self.texture_node = self._find_node("ShaderNodeTexEnvironment")
        self.background_node = self._find_node("ShaderNodeBackground")
        self.mapping_node = self._find_node("ShaderNodeMapping")
        self.tex_coords_node = self._find_node("ShaderNodeTexCoord")
        self.world_output_node = self._find_node("ShaderNodeOutputWorld")

    def _find_node(self, bl_idname: str) -> bpy.types.Node:
        for node in self.nodes:
            if node.bl_idname == bl_idname:
                return node
        raise RuntimeError(
            f"Expected a '{bl_idname}' node in the world shader tree, but none was found."
        )
