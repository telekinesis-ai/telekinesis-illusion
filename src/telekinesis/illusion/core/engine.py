"""Simulation and capture using the local bpy algorithms."""

import math
import random
from loguru import logger
from telekinesis.illusion import config
from telekinesis.illusion.blender.types import mesh_object
from telekinesis.illusion.blender import physics
from telekinesis.illusion.blender.rendering import renderer
from telekinesis.illusion.types.render_result import RenderResult
import bpy


class BlenderEngine:
    def __init__(
        self,
        render_config: config.RenderConfig | None = None,
        physics_config: config.PhysicsConfig | None = None,
    ):
        self.render_config = render_config or config.RenderConfig()
        self.physics_config = physics_config or config.PhysicsConfig()
        self._prepared = False

    def clean_up(self) -> None:
        """Release engine-owned rendering resources.

        Standalone ``bpy`` can unregister Cycles RNA properties before its
        render engine is destroyed during interpreter shutdown. This causes
        non-fatal ``AttributeError`` and ``ReferenceError`` messages after a
        completed run. Switching to Workbench here destroys the Cycles engine
        while its RNA types are still available. Interactive Blender is left
        unchanged because it manages the renderer lifecycle itself.
        """
        # Release Cycles before standalone bpy unregisters its RNA properties.
        if (
            not bpy.app.binary_path
            and bpy.context.scene.render.engine == "CYCLES"
        ):
            bpy.context.scene.render.engine = "BLENDER_WORKBENCH"

    def prepare(self, segmentation_config: dict) -> None:
        if self._prepared:
            return
        render_config = self.render_config
        renderer.enable_segmentation_output(**segmentation_config)
        if render_config.device == "CPU":
            renderer.set_render_devices(use_only_cpu=True)
        else:
            renderer.set_render_devices(
                desired_gpu_device_type=render_config.device
            )
        renderer.set_tiling(use_auto_tile=False, tile_size=4096)
        if render_config.samples is not None:
            renderer.set_max_amount_of_samples(render_config.samples)
        if render_config.noise_threshold is not None:
            renderer.set_noise_threshold(render_config.noise_threshold)
        if render_config.depth:
            renderer.enable_depth_output(activate_antialiasing=False)
        if render_config.normals:
            renderer.enable_normals_output()
        renderer.set_output_format(
            view_transform=render_config.view_transform,
            look=render_config.look,
        )
        self._prepared = True

    def randomize_look(self) -> None:
        """Sample per-render exposure, HDRI strength/rotation and bevel."""
        render_config = self.render_config
        if render_config.bevel_radius is not None:
            self._apply_bevel(
                random.uniform(*render_config.bevel_radius),
                render_config.bevel_models,
            )
        if render_config.exposure is not None:
            renderer.set_output_format(
                exposure=random.uniform(*render_config.exposure)
            )
        world = bpy.context.scene.world
        if world is None or not world.use_nodes:
            return
        for node in world.node_tree.nodes:
            if (
                node.type == "BACKGROUND"
                and render_config.background_strength is not None
            ):
                node.inputs["Strength"].default_value = random.uniform(
                    *render_config.background_strength
                )
            elif (
                node.type == "MAPPING"
                and render_config.background_rotation is not None
            ):
                node.inputs["Rotation"].default_value[2] = math.radians(
                    random.uniform(*render_config.background_rotation)
                )

    @staticmethod
    def _apply_bevel(radius: float, model_prefixes=None) -> None:
        """Round shading edges of Principled materials by ``radius``.

        A Bevel node feeds the BSDF normal; when a Bump node already drives
        the normal (surface imperfections), the bevel feeds the bump instead.
        With ``model_prefixes``, only materials on matching objects are
        bevelled (the node is costly: it traces extra rays per sample).
        """
        if model_prefixes is None:
            materials = list(bpy.data.materials)
        else:
            materials = {
                slot.material
                for obj in bpy.context.scene.objects
                if obj.type == "MESH" and obj.name.startswith(tuple(model_prefixes))
                for slot in obj.material_slots
                if slot.material is not None
            }
        for material in materials:
            if not material.use_nodes:
                continue
            tree = material.node_tree
            shaders = [n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"]
            if not shaders:
                continue
            bevel = tree.nodes.get("illusion_bevel")
            if bevel is None:
                bevel = tree.nodes.new("ShaderNodeBevel")
                bevel.name = "illusion_bevel"
                bevel.samples = 4
                for shader in shaders:
                    normal = shader.inputs["Normal"]
                    if normal.is_linked:
                        source = normal.links[0].from_node
                        if source.type == "BUMP" and not source.inputs["Normal"].is_linked:
                            tree.links.new(bevel.outputs["Normal"], source.inputs["Normal"])
                    else:
                        tree.links.new(bevel.outputs["Normal"], normal)
            bevel.inputs["Radius"].default_value = radius

    def simulate(
        self, scene, physics_config: config.PhysicsConfig | None = None
    ) -> None:
        """Simulate rigid-body physics and keep the settled model poses.

        Models that fall below the lowest point of every static model, such
        as parts that miss the container, have nothing left to land on. They
        do not delay the end of the simulation and are deactivated
        afterwards, so they are neither rendered nor annotated.

        Args:
            scene: Environment whose active models are simulated.
            physics_config: Simulation settings. None uses the engine's
                physics_config. Nothing happens when the settings are inactive.
        """
        physics_config = physics_config or self.physics_config
        if not physics_config.active:
            return
        mesh_object.COLLISION_MAX_VERTICES = physics_config.collision_max_vertices
        fallen_names = physics.simulate_physics_and_fix_final_poses(
            random.uniform(*physics_config.min_simulation_time_range),
            random.uniform(*physics_config.max_simulation_time_range),
            physics_config.check_model_interval,
            physics_config.model_stopped_location_threshold,
            physics_config.model_stopped_rotation_threshold,
            physics_config.substeps_per_frame,
            physics_config.solver_iters,
            physics_config.verbose,
            physics_config.use_volume_com,
        )
        models = scene.get_models()
        fallen_names = [name for name in fallen_names if name in models]
        for name in fallen_names:
            scene.deactivate(models[name])
        if fallen_names:
            logger.info(
                f"Deactivated {len(fallen_names)} model(s) that fell below "
                f"all static models: {fallen_names}"
            )

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
            self.randomize_look()
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
