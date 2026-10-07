"""
Defines the SyntheticDataGenerator class that is the main orchestrator of the
synthetic data generation process.
"""

import random
from datetime import datetime

from loguru import logger
from tqdm import tqdm

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported (blenderproc pulls it in). See blender_env.py.
isolate_user_extensions()

import numpy as np

import blenderproc as bproc
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

    @staticmethod
    def _catch_plane_geometry(
        container_bounds: np.ndarray,
        distance_fraction: float,
        minimum_distance: float,
        size_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the location and scale of a plane below a container."""
        bounds = np.asarray(container_bounds, dtype=float)
        if bounds.shape != (8, 3):
            raise ValueError("container_bounds must have shape (8, 3).")
        if distance_fraction < 0:
            raise ValueError("distance_fraction must be non-negative.")
        if minimum_distance <= 0:
            raise ValueError("minimum_distance must be positive.")
        if size_factor <= 0:
            raise ValueError("size_factor must be positive.")

        lower = bounds.min(axis=0)
        upper = bounds.max(axis=0)
        extent = upper - lower
        distance = max(extent[2] * distance_fraction, minimum_distance)
        location = np.array(
            [
                (lower[0] + upper[0]) / 2,
                (lower[1] + upper[1]) / 2,
                lower[2] - distance,
            ]
        )
        # Blender's plane primitive is two units wide, so its XY scale is
        # half the desired footprint. Z scale is irrelevant for a plane.
        scale = np.array(
            [
                max(extent[0] * size_factor / 2, minimum_distance),
                max(extent[1] * size_factor / 2, minimum_distance),
                1.0,
            ]
        )
        return location, scale

    def _create_physics_catch_plane(self, config: dict):
        """Create an invisible passive plane beneath the visible container."""
        container_prefixes = tuple(config.get("container_names", ()))
        visible_names = self._context.get_visible_object_names()
        container_name = next(
            (
                name
                for name in visible_names
                if name.startswith(container_prefixes)
            ),
            None,
        )
        if container_name is None:
            raise RuntimeError(
                "Cannot create a physics catch plane without a visible "
                "container."
            )

        container = self._context.get_objects()[container_name]
        location, scale = self._catch_plane_geometry(
            container.get_bound_box(),
            distance_fraction=float(config.get("distance_fraction", 0.25)),
            minimum_distance=float(config.get("minimum_distance", 0.05)),
            size_factor=float(config.get("size_factor", 6.0)),
        )
        plane = bproc.object.create_primitive(
            "PLANE", location=location, scale=scale
        )
        plane.set_name("ILLUSION_PHYSICS_CATCH_PLANE")
        plane.enable_rigidbody(active=False, collision_shape="BOX")
        # Do not call Entity.hide(): hide_set() can remove the plane from
        # dependency-graph evaluation. hide_render keeps its collision active.
        plane.blender_obj.hide_render = True
        logger.debug(
            f"Created physics catch plane at z={location[2]:.4f} "
            f"below '{container_name}'."
        )
        return plane, float(location[2])

    def _cull_objects_on_catch_plane(
        self, plane_z: float, config: dict
    ) -> list[str]:
        """Hide tracked objects whose bounding boxes touch the catch plane."""
        tolerance = float(config.get("contact_tolerance", 0.05))
        if tolerance < 0:
            raise ValueError("contact_tolerance must be non-negative.")

        tracked_prefixes = tuple(config.get("tracked_object_names", ()))
        visible_names = list(self._context.get_visible_object_names())
        objects = self._context.get_objects()
        escaped = []
        for name in visible_names:
            if not name.startswith(tracked_prefixes):
                continue
            bounds = objects[name].get_bound_box()
            if float(np.min(bounds[:, 2])) <= plane_z + tolerance:
                objects[name].hide(True)
                objects[name].disable_rigid_body()
                escaped.append(name)

        if escaped:
            escaped_set = set(escaped)
            self._context.set_visible_object_names(
                [name for name in visible_names if name not in escaped_set]
            )
            logger.info(
                f"Culled {len(escaped)} object(s) that escaped the bin: "
                f"{escaped}"
            )
        return escaped

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
        render_verbose: bool = False,
        physics_catch_plane: dict | None = None,
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
            render_verbose: bool
                Show Blender's live frame and sample progress while rendering.
            physics_catch_plane: dict or None
                Optional catch-plane configuration. A passive, render-invisible
                plane is placed beneath the visible container before physics;
                tracked objects resting on it are hidden before rendering.
        """
        # Record the start time
        start_time = datetime.now()
        logger.info("Generating synthetic images...")

        segmentation_config = self._writer.get_segmentation_output_config()
        bproc.renderer.enable_segmentation_output(**segmentation_config)
        bproc.renderer.set_render_devices(desired_gpu_device_type="OPTIX")
        # Render the full frame in a single tile (no disk-spooled tile buffers).
        bproc.python.renderer.RendererUtility.set_tiling(
            use_auto_tile=False, tile_size=4096
        )
        # Route loguru through tqdm.write() so log lines don't break the bar
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

        progress_bar = tqdm(total=num_images)
        num_total_images = 0
        while num_total_images < num_images:
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
                catch_plane = None
                catch_plane_z = None
                catch_config = physics_catch_plane or {}
                if catch_config.get("active", False):
                    catch_plane, catch_plane_z = (
                        self._create_physics_catch_plane(catch_config)
                    )
                try:
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
                    if catch_plane_z is not None:
                        self._cull_objects_on_catch_plane(
                            catch_plane_z, catch_config
                        )
                finally:
                    if catch_plane is not None:
                        catch_plane.delete()

            # Render the context
            progress_bar.set_postfix_str("Rendering...")
            last_err = None
            for attempt in range(render_max_retries + 1):
                try:
                    data = bproc.renderer.render(verbose=render_verbose)
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
