"""
Defines the Background class that manages the background appearance in the
scenes.
"""

from pathlib import Path
import weakref

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported. See blender_env.py for why.
isolate_user_extensions()

import bpy


class Background:
    """Apply HDRI textures to the active Blender world's shader nodes.

    Construction creates the world node setup without choosing or loading an
    image. Environment owns asset registration and default HDRI selection.

    Example:
        background = Background()
        background.set_background_texture("studio.exr")
    """

    # Contains weak refs to all background instances
    # As it only uses weak references, instances can still be removed by GC when
    # all other references are gone. If that happens, the instances' weak ref is
    # also automatically removed from the set.
    __refs__: weakref.WeakSet = weakref.WeakSet()

    def __init__(self) -> None:
        """Create the shader node setup for applying HDRI textures."""
        self._create_shader_node_setup()
        Background.__refs__.add(self)

    def _create_shader_node_setup(
        self,
    ) -> None:
        """Build the environment texture, mapping, and world output nodes."""

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

    def set_background_texture(self, hdri: str) -> None:
        """
        Sets the background of the environment from a the path of the HDRI-file.

        Args:
            hdri: str
                Path to the .hdr or .exr file of the HDRI.

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
        """Find one required world shader node after Blender reference updates."""
        for node in self.nodes:
            if node.bl_idname == bl_idname:
                return node
        raise RuntimeError(
            f"Expected a '{bl_idname}' node in the world shader tree, but none was found."
        )
