"""Simulation and capture using the local bpy algorithms."""

from dataclasses import dataclass, field
import random
import numpy as np
from loguru import logger
from telekinesis.illusion.blender import physics
from telekinesis.illusion.blender.types import mesh_object
from telekinesis.illusion.blender.rendering import renderer
from telekinesis.illusion.types.render_result import RenderResult
import bpy


@dataclass(frozen=True)
class RenderConfig:
    device: str = "CUDA"
    samples: int | None = None
    noise_threshold: float | None = None
    max_retries: int = 2
    verbose: bool = False
    depth: bool = False
    normals: bool = False


@dataclass(frozen=True)
class PhysicsConfig:
    active: bool = False
    min_simulation_time_range: tuple[float, float] = (0.5, 1.0)
    max_simulation_time_range: tuple[float, float] = (2.0, 5.0)
    check_object_interval: float = 2.0
    object_stopped_location_threshold: float = 0.01
    object_stopped_rotation_threshold: float = 0.1
    substeps_per_frame: int = 10
    solver_iters: int = 10
    verbose: bool = False
    use_volume_com: bool = False
    catch_plane: dict = field(default_factory=dict)


class BlenderEngine:
    def __init__(
        self,
        render_config: RenderConfig | None = None,
        physics_config: PhysicsConfig | None = None,
    ):
        self.render_config = render_config or RenderConfig()
        self.physics_config = physics_config or PhysicsConfig()
        self._prepared = False
        self._physics_catch_plane: mesh_object.MeshObject | None = None

    def clean_up(self) -> None:
        """Release engine-owned physics and rendering resources.

        Standalone ``bpy`` can unregister Cycles RNA properties before its
        render engine is destroyed during interpreter shutdown. This causes
        non-fatal ``AttributeError`` and ``ReferenceError`` messages after a
        completed run. Switching to Workbench here destroys the Cycles engine
        while its RNA types are still available. Interactive Blender is left
        unchanged because it manages the renderer lifecycle itself.
        """
        if self._physics_catch_plane is not None:
            plane_object = self._physics_catch_plane.blender_obj
            plane_mesh = plane_object.data
            bpy.data.objects.remove(plane_object, do_unlink=True)
            bpy.data.meshes.remove(plane_mesh)
            self._physics_catch_plane = None

        # Release Cycles before standalone bpy unregisters its RNA properties.
        if (
            not bpy.app.binary_path
            and bpy.context.scene.render.engine == "CYCLES"
        ):
            bpy.context.scene.render.engine = "BLENDER_WORKBENCH"

    def prepare(self, segmentation_config: dict) -> None:
        if self._prepared:
            return
        config = self.render_config
        if config.max_retries < 0:
            raise ValueError("Render retries must be nonnegative.")
        renderer.enable_segmentation_output(**segmentation_config)
        if config.device == "CPU":
            renderer.set_render_devices(use_only_cpu=True)
        else:
            renderer.set_render_devices(desired_gpu_device_type=config.device)
        renderer.set_tiling(use_auto_tile=False, tile_size=4096)
        if config.samples is not None:
            if config.samples <= 0:
                raise ValueError("Render samples must be positive.")
            renderer.set_max_amount_of_samples(config.samples)
        if config.noise_threshold is not None:
            renderer.set_noise_threshold(config.noise_threshold)
        if config.depth:
            renderer.enable_depth_output(activate_antialiasing=False)
        if config.normals:
            renderer.enable_normals_output()
        self._prepared = True

    def simulate(self, scene, config: PhysicsConfig | None = None) -> None:
        config = config or self.physics_config
        if not config.active:
            return
        plane_z = None
        catch = config.catch_plane
        try:
            if catch.get("active", False):
                _, plane_z = self._prepare_physics_catch_plane(scene, catch)
            physics.simulate_physics_and_fix_final_poses(
                random.uniform(*config.min_simulation_time_range),
                random.uniform(*config.max_simulation_time_range),
                config.check_object_interval,
                config.object_stopped_location_threshold,
                config.object_stopped_rotation_threshold,
                config.substeps_per_frame,
                config.solver_iters,
                config.verbose,
                config.use_volume_com,
            )
            if plane_z is not None:
                self._cull_objects_on_catch_plane(scene, plane_z, catch)
        finally:
            if self._physics_catch_plane is not None:
                if self._physics_catch_plane.has_rigidbody_enabled():
                    self._physics_catch_plane.disable_rigidbody()
                self._physics_catch_plane.hide(True)

    def render(self, scene, max_views: int | None = None) -> RenderResult:
        """Capture pass arrays and metadata before undo or scene cleanup."""
        blender_scene = bpy.context.scene
        original_end = blender_scene.frame_end
        original_frame = blender_scene.frame_current
        if max_views is not None:
            blender_scene.frame_end = min(
                original_end, blender_scene.frame_start + max_views
            )
        try:
            captures = []
            for frame in range(
                blender_scene.frame_start, blender_scene.frame_end
            ):
                blender_scene.frame_set(frame)
                camera = blender_scene.camera
                captures.append(
                    {
                        "frame": frame,
                        "camera_matrix_world": [
                            list(row) for row in camera.matrix_world
                        ],
                        "camera_lens_mm": camera.data.lens,
                    }
                )
            if not captures:
                raise ValueError(
                    "Configure at least one camera pose before rendering."
                )
            for attempt in range(self.render_config.max_retries + 1):
                try:
                    data = renderer.render(verbose=self.render_config.verbose)
                    break
                except Exception:
                    if attempt == self.render_config.max_retries:
                        raise
                    logger.exception(f"Retrying render ({attempt + 1})")
            if len(data.get("colors", ())) != len(captures):
                raise RuntimeError(
                    "Rendered views and capture metadata are not aligned."
                )
            categories = dict(scene.get_categories()._id_label_map)
            return RenderResult(data, categories, tuple(captures))
        finally:
            blender_scene.frame_end = original_end
            blender_scene.frame_set(original_frame)

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

    def _prepare_physics_catch_plane(
        self, scene, config: dict
    ) -> tuple[mesh_object.MeshObject, float]:
        """Create or reposition the reusable plane beneath the visible bin."""
        container_prefixes = tuple(config.get("container_names", ()))
        visible_names = scene.get_visible_object_names()
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

        container = scene.get_objects()[container_name]
        location, scale = self._catch_plane_geometry(
            container.get_bound_box(),
            distance_fraction=float(config.get("distance_fraction", 0.25)),
            minimum_distance=float(config.get("minimum_distance", 0.05)),
            size_factor=float(config.get("size_factor", 6.0)),
        )
        if self._physics_catch_plane is None:
            self._physics_catch_plane = mesh_object.create_primitive("PLANE")
            self._physics_catch_plane.set_name("ILLUSION_PHYSICS_CATCH_PLANE")
        plane = self._physics_catch_plane
        plane.set_location(location)
        plane.set_scale(scale)
        # Restore dependency-graph evaluation for physics, but keep the plane
        # invisible to rendering. It is hidden again after simulation.
        plane.hide(False)
        plane.blender_obj.hide_render = True
        plane.enable_rigidbody(active=False, collision_shape="BOX")
        logger.debug(
            f"Prepared physics catch plane at z={location[2]:.4f} "
            f"below '{container_name}'."
        )
        return plane, float(location[2])

    def _cull_objects_on_catch_plane(
        self, scene, plane_z: float, config: dict
    ) -> list[str]:
        """Hide tracked objects whose bounding boxes touch the catch plane."""
        tolerance = float(config.get("contact_tolerance", 0.05))
        if tolerance < 0:
            raise ValueError("contact_tolerance must be non-negative.")

        tracked_prefixes = tuple(config.get("tracked_object_names", ()))
        visible_names = list(scene.get_visible_object_names())
        objects = scene.get_objects()
        escaped = []
        for name in visible_names:
            if not name.startswith(tracked_prefixes):
                continue
            bounds = objects[name].get_bound_box()
            if float(np.min(bounds[:, 2])) <= plane_z + tolerance:
                scene.deactivate(objects[name])
                escaped.append(name)

        if escaped:
            escaped_set = set(escaped)
            scene.set_visible_object_names(
                [name for name in visible_names if name not in escaped_set]
            )
            logger.info(
                f"Culled {len(escaped)} object(s) that escaped the bin: "
                f"{escaped}"
            )
        return escaped
