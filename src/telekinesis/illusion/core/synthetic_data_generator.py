"""The generation lifecycle: prepare, randomize, simulate, render, write."""

from dataclasses import replace
from pathlib import Path
import uuid
from loguru import logger
from tqdm import tqdm

from telekinesis.illusion.core.environment import Environment
from telekinesis.illusion.core.engine import BlenderEngine
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.writer.writer import Writer


class SyntheticDataGenerator:
    """Generate scene realizations and capture their camera views.

    One environment supplies reusable assets and the active Blender scene.
    Each sample executes the randomizer tree to realize a scene, optionally
    simulates it, and captures its views before the next realization.

    GPU rendering uses CUDA by default.

    Warning:
        OptiX leaked approximately four Windows Timer/Semaphore handles per
        rendered image in tests with bpy 4.2.17 and NVIDIA driver 591.86.
        Environment cleanup did not release them, so long generation runs can
        accumulate native resources. CUDA did not exhibit this specific leak.
    """

    def __init__(
        self,
        environment: Environment,
        randomizer: Randomizer,
        writer: Writer,
        engine: BlenderEngine | None = None,
    ):
        self.environment = environment
        if self.environment is None or randomizer is None or writer is None:
            raise TypeError("environment, randomizer and writer are required.")
        self.randomizer = randomizer
        self.writer = writer
        self.engine = engine or BlenderEngine()

    def prepare(self) -> None:
        """Prepare environment resources for reuse across scenes and shards."""
        self.environment.prepare(self.randomizer.required_instances())
        self.randomizer.prepare(self.environment)
        self.engine.prepare(self.writer.get_segmentation_output_config())

    def clean_up(self) -> None:
        try:
            self.engine.clean_up()
        finally:
            self.environment.clean_up()

    def generate(
        self,
        num_images: int | None = None,
        *,
        num_samples: int | None = None,
        clean_up_environment: bool = True,
        save_blender_scene: bool = False,
        **legacy_options,
    ) -> int:
        """Write exactly num_images views, or all views of num_samples scenes.

        Each sample is one scene realization of the shared environment under
        the randomizer tree's rules. A scene can produce multiple camera views.
        The legacy physics keyword arguments remain accepted for existing
        workers. New scripts pass PhysicsConfig/RenderConfig to their engine.
        Keep clean_up_environment=False between shards to reuse the environment;
        explicitly clean up at the end.
        Asset registration is closed after preparation. Bounds may change only
        within prepared capacity. No models are imported in the sample loop.
        With save_blender_scene=True, save the final scene as scene_<uuid>.blend
        in the writer's output directory, using a fresh UUID for each save.
        """
        if num_images is not None and num_samples is not None:
            raise ValueError("Choose num_images or num_samples.")
        if num_images is None and num_samples is None:
            num_images = 5
        target = num_images if num_images is not None else num_samples
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target <= 0
        ):
            raise ValueError("Generation count must be a positive integer.")
        physics_overrides = {}
        render_overrides = {}
        aliases = {
            "simulate_physics": "active",
            "physics_catch_plane": "catch_plane",
        }
        for name, value in legacy_options.items():
            if name in ("render_max_retries", "render_verbose"):
                render_overrides[
                    {
                        "render_max_retries": "max_retries",
                        "render_verbose": "verbose",
                    }[name]
                ] = value
            else:
                physics_overrides[aliases.get(name, name)] = value
        physics_config = replace(
            self.engine.physics_config, **physics_overrides
        )
        if render_overrides:
            self.engine.render_config = replace(
                self.engine.render_config, **render_overrides
            )
        images_written = 0
        samples_written = 0
        try:
            self.prepare()
            with tqdm(total=target, desc="Generating") as progress:
                while (
                    images_written
                    if num_images is not None
                    else samples_written
                ) < target:
                    self.environment.begin_sample()
                    self.randomizer.randomize(self.environment)
                    if physics_config.active:
                        self.engine.simulate(self.environment, physics_config)
                        self.randomizer.randomize_light_poses(self.environment)
                    remaining = (
                        target - images_written
                        if num_images is not None
                        else None
                    )
                    result = self.engine.render(self.environment, max_views=remaining)
                    written = self.writer.write(result)
                    if written != len(result) or written <= 0:
                        raise RuntimeError(
                            "Writer did not write the captured batch."
                        )
                    images_written += written
                    samples_written += 1
                    progress.update(written if num_images is not None else 1)
            logger.info(
                f"Generation complete: {images_written} images from {samples_written} samples."
            )
            if save_blender_scene:
                import bpy

                bpy.ops.wm.save_as_mainfile(
                    filepath=str(
                        Path(self.writer.get_output_dir())
                        / f"scene_{uuid.uuid4()}.blend"
                    )
                )
            return images_written
        finally:
            if clean_up_environment:
                self.clean_up()
