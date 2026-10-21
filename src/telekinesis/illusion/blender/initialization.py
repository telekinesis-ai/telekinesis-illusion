# Adapted code; see NOTICE.md for attribution and licensing.
"""Initialize and reset the Blender scene and rendering defaults."""

import atexit
import os
import random
import shutil

from numpy import random as np_random
from typing import Optional
import bpy

from telekinesis.illusion.blender.storage import GlobalStorage
from telekinesis.illusion.blender.utility import reset_keyframes
from telekinesis.illusion.blender.setup import (
    SetupUtility,
    is_using_external_bpy_module,
)
from telekinesis.illusion.blender.camera import settings as CameraUtility
from telekinesis.illusion.blender.defaults import DefaultConfig
from telekinesis.illusion.blender.rendering import renderer as RendererUtility


def clean_temp_dir():
    # pylint: disable=import-outside-toplevel
    from telekinesis.illusion.blender.utility import Utility

    # pylint: enable=import-outside-toplevel
    assert is_using_external_bpy_module()
    if os.path.exists(Utility.temp_dir):
        print("Cleaning temporary directory")
        shutil.rmtree(Utility.temp_dir)


def handle_sigterm(_signum, _frame):
    clean_temp_dir()


def init(clean_up_scene: bool = True, temp_dir: Optional[str] = None):
    """Initialize the Blender scene and rendering defaults for Illusion.

    By default, reset scene data and create a world and camera before applying
    rendering and camera defaults. A new scene replaces the process-wide scene;
    multiple live scenes are unsupported. Use clean_up() to reset scene data
    without reapplying these defaults.

    A nonempty ILLUSION_RANDOM_SEED seeds Python and NumPy random generators.
    Data blocks with a truthy illusion_preserve custom property survive cleanup.

    Args:
        clean_up_scene: Whether to reset scene data before initialization.
        temp_dir: Parent directory for a new owned temporary directory. None
            uses the system temporary directory. An existing temporary
            directory is reused until it is removed.

    Raises:
        ValueError: If ILLUSION_RANDOM_SEED is not a valid NumPy integer seed.
    """
    from telekinesis.illusion.blender.utility import Utility

    if not Utility.temp_dir or not os.path.isdir(Utility.temp_dir):
        SetupUtility.setup_utility_paths(
            SetupUtility.determine_temp_dir(temp_dir)
        )
        atexit.register(clean_temp_dir)
    # A new Scene replaces the process-wide bpy scene. Multiple live scenes
    # are deliberately unsupported; wrappers retain Blender's normal lifetime.
    if clean_up_scene:
        GlobalStorage._storage_dict.clear()
        clean_up(clean_up_camera=True)

    # Set language if necessary if not using external bpy (that has only DEFAULT language)
    if (
        not is_using_external_bpy_module()
        and bpy.context.preferences.view.language != "en_US"
    ):
        print("Setting blender language settings to english during this run")
        bpy.context.preferences.view.language = "en_US"

    # Use cycles
    bpy.context.scene.render.engine = "CYCLES"

    # Set default render devices
    RendererUtility.set_render_devices(desired_gpu_device_type="CUDA")

    # Set default parameters
    _Initializer.set_default_parameters()

    random_seed = os.getenv("ILLUSION_RANDOM_SEED")
    if random_seed:
        print(f"Got random seed: {random_seed}")
        try:
            random_seed = int(random_seed)
        except ValueError as e:
            raise e
        random.seed(random_seed)
        np_random.seed(random_seed)

    # Remember init was completed


def clean_up(clean_up_camera: bool = False):
    """Resets the scene to its clean state.

    This method removes all objects, camera poses and cleans up the world background.
    All (renderer) settings and the UI are kept as they are.
    Data blocks with a truthy illusion_preserve custom property are retained.

    :param clean_up_camera: If True, also the camera is set back to its clean state.
    """
    # Switch to right context
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    # Clean up
    _Initializer.remove_all_data(clean_up_camera)
    _Initializer.remove_custom_properties()

    # Make the default 'LayerCollection' as active, so bpy.context.collection points to it.
    # This is the 'Scene Collection' in the outliner.
    bpy.context.view_layer.active_layer_collection = (
        bpy.context.view_layer.layer_collection
    )

    # Create new world
    new_world = bpy.data.worlds.new("World")
    bpy.context.scene.world = new_world
    new_world["category_id"] = 0
    new_world.use_nodes = True

    if clean_up_camera:
        # Create the camera
        cam = bpy.data.cameras.new("Camera")
        cam_ob = bpy.data.objects.new("Camera", cam)
        bpy.context.scene.collection.objects.link(cam_ob)
        bpy.context.scene.camera = cam_ob

    # Make sure keyframes are cleaned up
    reset_keyframes()


class _Initializer:
    """
    Configure Blender defaults and remove scene data.
    """

    @staticmethod
    def set_default_parameters():
        """Loads and sets default parameters defined in DefaultConfig.py"""
        # Set default intrinsics
        CameraUtility.set_intrinsics_from_blender_params(
            DefaultConfig.fov,
            DefaultConfig.resolution_x,
            DefaultConfig.resolution_y,
            DefaultConfig.clip_start,
            DefaultConfig.clip_end,
            DefaultConfig.pixel_aspect_x,
            DefaultConfig.pixel_aspect_y,
            DefaultConfig.shift_x,
            DefaultConfig.shift_y,
            DefaultConfig.lens_unit,
        )
        CameraUtility.set_stereo_parameters(
            DefaultConfig.stereo_convergence_mode,
            DefaultConfig.stereo_convergence_distance,
            DefaultConfig.stereo_interocular_distance,
        )

        # Init renderer
        RendererUtility.render_init()
        RendererUtility.set_world_background(DefaultConfig.world_background)
        RendererUtility.set_max_amount_of_samples(DefaultConfig.samples)
        RendererUtility.set_noise_threshold(
            DefaultConfig.sampling_noise_threshold
        )
        RendererUtility.set_tiling(
            DefaultConfig.use_auto_tile, DefaultConfig.cycles_tile_size
        )

        # Set number of cpu cores used for rendering (1 thread is always used for coordination => 1
        # cpu thread means GPU-only rendering)
        RendererUtility.set_cpu_threads(0)
        RendererUtility.set_denoiser(
            DefaultConfig.denoiser, DefaultConfig.denoiser_configs
        )
        # For now disable the light tree per default, as it seems to increase render time for most of our tests
        RendererUtility.toggle_light_tree(False)

        RendererUtility.set_simplify_subdivision_render(
            DefaultConfig.simplify_subdivision_render,
            DefaultConfig.texture_limit_render,
        )

        RendererUtility.set_light_bounces(
            DefaultConfig.diffuse_bounces,
            DefaultConfig.glossy_bounces,
            DefaultConfig.ao_bounces_render,
            DefaultConfig.max_bounces,
            DefaultConfig.transmission_bounces,
            DefaultConfig.transparency_bounces,
            DefaultConfig.volume_bounces,
        )

        RendererUtility.set_output_format(
            DefaultConfig.file_format,
            DefaultConfig.color_depth,
            DefaultConfig.enable_transparency,
            DefaultConfig.jpg_quality,
            DefaultConfig.view_transform,
            DefaultConfig.look,
            DefaultConfig.exposure,
            DefaultConfig.gamma,
        )

    @staticmethod
    def remove_all_data(remove_camera: bool = True):
        """Remove all data blocks except opened scripts, the default scene and the camera.

        :param remove_camera: If True, also the default camera is removed.
        """
        # Go through all attributes of bpy.data
        for collection in dir(bpy.data):
            data_structure = getattr(bpy.data, collection)
            # Check that it is a data collection
            if (
                isinstance(data_structure, bpy.types.bpy_prop_collection)
                and hasattr(data_structure, "remove")
                and collection not in ["texts"]
            ):
                # Go over all entities in that collection
                for block in data_structure:
                    # Skip the default scene
                    if (
                        isinstance(block, bpy.types.Scene)
                        and block.name == "Scene"
                    ):
                        continue
                    # If desired, skip camera
                    if (
                        not remove_camera
                        and isinstance(
                            block, (bpy.types.Object, bpy.types.Camera)
                        )
                        and block.name == "Camera"
                    ):
                        continue
                    # Skip anything explicitly marked to survive a reset. This
                    # loop otherwise wipes *every* collection in bpy.data,
                    # including bpy.data.node_groups - which destroys add-on
                    # owned datablocks (e.g. an add-on's own configuration
                    # node tree) when running inside a live
                    # Blender session rather than a headless one-shot run.
                    try:
                        if block.get("illusion_preserve"):
                            continue
                    except (AttributeError, TypeError):
                        pass
                    data_structure.remove(block)

    @staticmethod
    def remove_custom_properties():
        """Remove all custom properties registered at global entities like the scene."""
        for key in list(bpy.context.scene.keys()):
            # Do not remove blender's internal property "cycles"
            if key not in ["cycles"]:
                del bpy.context.scene[key]
