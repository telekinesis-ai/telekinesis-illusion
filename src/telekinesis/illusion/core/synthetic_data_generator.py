"""
Defines the SyntheticDataGenerator class that is the main orchestrator of the
synthetic data generation process.
"""

from loguru import logger
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import random

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported (blenderproc pulls it in). See blender_env.py.
isolate_user_extensions()

import blenderproc as bproc
import bpy

from telekinesis.illusion.core.context import Context
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.writer.writer import Writer


class SyntheticDataGenerator:
    """
    Class for managing the synthetic data generation process.
    """

    def __init__(
        self, context: Context, randomizer: Randomizer, writer: Writer
    ) -> None:
        """
        Initialize internal states.
        """
        self._context = context
        self._randomizer = randomizer
        self._writer = writer

    def clean_up(
        self,
    ) -> None:
        """
        Clean up the bproc scene and bpy after finishing the generation.
        """
        logger.info("Cleaning up the scene...")
        bproc.clean_up()

    def generate(
        self,
        num_images: int = 5,
        simulate_physics: bool = False,
        min_simulation_time_range: tuple[float, float] = (0.5, 1.0),
        max_simulation_time_range: tuple[float, float] = (2.0, 5.0),
        check_object_interval: float = 2.0,
        object_stopped_location_threshold: float = 0.01,
        object_stopped_rotation_threshold: float = 0.1,
        substeps_per_frame: int = 10,
        solver_iters: int = 10,
        verbose: bool = False,
        use_volume_com: bool = False,
        save_blender_scene: bool = False,
        clean_up_scene: bool = True,
        render_max_retries: int = 2,
    ) -> None:
        """
        Main method for generating the synthetic data. On every iteration, the
        context is randomized, then the data is rendered and finally the results
        written into the desired format.

        Args:
            num_images: int
                Number of scenes to be rendered.
            simulate_physics: bool
                Whether to run a physics simulation on the objects before
                rendering each scene.
            min_simulation_time_range: tuple[float, float]
                Minimum and maximum min_simulation time for physics simulation.
                The actual value is sampled uniformly from this range. The default
                fps is 24, so 1.0 second simulates 24 frames. Tune this value
                by running the simulation in debug mode, saving the blender file
                and observing the simulation over the frames.
            max_simulation_time_range: tuple[float, float]
                Minimum and maximum max_simulation time for physics simulation.
                The actual value is sampled uniformly from this range.
            check_object_interval: float
                Interval, in seconds of simulated time, at which object poses
                are checked for the stopped-motion thresholds below.
            object_stopped_location_threshold: float
                Maximum location change, in meters, allowed between checks for
                an object to be considered at rest.
            object_stopped_rotation_threshold: float
                Maximum rotation change, in radians, allowed between checks for
                an object to be considered at rest.
            substeps_per_frame: int
                Number of physics substeps computed per simulation frame.
            solver_iters: int
                Number of solver iterations used by the physics simulation.
            verbose: bool
                Whether to enable verbose logging from the physics simulation.
            use_volume_com: bool
                Whether to compute the center of mass of simulated objects from
                their volume instead of their mesh vertices.
            save_blender_scene: bool
                Only for debugging to save the first context as .blend-file.
            clean_up_scene: bool
                Whether to clean up the scene after the generation or not.
                Should be set to False if the generation happens in a loop, like
                in a worker.
            render_max_retries: int
                Number of times to retry rendering a scene after a transient
                render failure before raising the error.
        """
        # Record the start time
        start_time = datetime.now()
        logger.info("Generating synthetic images...")

        bproc.renderer.enable_segmentation_output(
            map_by=["category_id", "instance", "name"],
            default_values={"category_id": 0},
        )
        bproc.renderer.set_render_devices(desired_gpu_device_type="OPTIX")
        # Render the full frame in a single tile (no disk-spooled tile buffers).
        bproc.python.renderer.RendererUtility.set_tiling(
            use_auto_tile=False, tile_size=4096
        )
        # Save us default user preferences
        bpy.ops.wm.save_userpref()

        # Route loguru through tqdm.write() so log lines don't break the bar
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

        progress_bar = tqdm(total=num_images - 1)
        num_total_images = 0
        while num_total_images < (num_images - 1):
            # Hide all hidden objects and disable their rigid body properties
            visible_object_names = self._context.get_visible_object_names()
            objects = self._context.get_objects()
            if visible_object_names:
                for object_name in visible_object_names:
                    objects[object_name].hide(True)
                    objects[object_name].disable_rigid_body()
                self._context.set_visible_object_names([])
                visible_object_names = []

            # Randomize the context
            progress_bar.set_postfix_str("Randomizing...")
            self._randomizer.randomize(self._context)

            # Simulate physics
            if simulate_physics:
                min_simulation_time = random.uniform(
                    min_simulation_time_range[0], min_simulation_time_range[1]
                )
                max_simulation_time = random.uniform(
                    max_simulation_time_range[0], max_simulation_time_range[1]
                )
                bproc.object.simulate_physics_and_fix_final_poses(
                    min_simulation_time,
                    max_simulation_time,
                    check_object_interval,
                    object_stopped_location_threshold,
                    object_stopped_rotation_threshold,
                    substeps_per_frame,
                    solver_iters,
                    verbose,
                    use_volume_com,
                )

            # Render the context
            progress_bar.set_postfix_str("Rendering...")
            last_err = None
            for attempt in range(render_max_retries + 1):
                try:
                    data = bproc.renderer.render()
                    break
                except (
                    Exception
                ) as err:  # tile-write / transient GPU render failures
                    last_err = err
                    logger.warning(
                        f"Render failed on attempt "
                        f"{attempt + 1}/{render_max_retries + 1}: {err}"
                    )
            else:
                logger.error(
                    f"Render failed after {render_max_retries + 1} attempts; aborting."
                )
                raise last_err

            # Write the data
            progress_bar.set_postfix_str("Writing...")
            num_written_images = self._writer.write(
                data=data, categories=self._context.get_categories()
            )

            progress_bar.set_postfix_str("")
            progress_bar.update(num_written_images)
            num_total_images += num_written_images

        time_elapsed = datetime.now() - start_time
        hours, remainder = divmod(time_elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        logger.info(
            f"Time elapsed (hh:mm:ss.ms): "
            f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"
        )

        if clean_up_scene:
            self.clean_up()
