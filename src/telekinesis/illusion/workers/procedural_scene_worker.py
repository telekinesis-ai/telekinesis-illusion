"""Generic YAML-driven worker for callback-built procedural scenes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import blenderproc as bproc
import bpy

from telekinesis.illusion.core.context import Context
from telekinesis.illusion.core.synthetic_data_generator import SyntheticDataGenerator
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.randomizer.randomizer_node import (
    BackgroundRandomizer,
    CameraPoseRandomizer,
    MaterialRandomizer,
)
from telekinesis.illusion.sampler.camera_pose_sampler import CAMERA_POSE_SAMPLERS
from telekinesis.illusion.types.camera import CameraConfig
from telekinesis.illusion.writer.writer import CocoWriter


LayoutBuilder = Callable[[Context, int, Mapping[str, Any]], None]
RandomizerBuilder = Callable[
    [Randomizer, Context, Mapping[str, Any]], None
]


class ProceduralSceneWorker:
    """Generate seeded procedural layouts and multi-view COCO annotations."""

    def __init__(
        self,
        spec_path: str | Path,
        layout_builder: LayoutBuilder,
        randomizer_builders: Sequence[RandomizerBuilder] = (),
    ) -> None:
        self.spec_path = Path(spec_path).resolve()
        self.layout_builder = layout_builder
        self.randomizer_builders = tuple(randomizer_builders)
        with self.spec_path.open("r", encoding="utf-8") as stream:
            self.spec = yaml.safe_load(stream) or {}

        metadata = self.spec.get("metadata", {})
        output_root = Path(metadata.get("base_output_directory") or "output")
        if not output_root.is_absolute():
            output_root = (self.spec_path.parent / output_root).resolve()
        self.output_dir = output_root / metadata.get(
            "dataset_name", "procedural_scene"
        )
        writer = self.spec.get("writer", {})
        self._seed_metadata_key = writer.get(
            "seed_metadata_key", "arrangement_seed"
        )
        self._static_image_metadata = dict(writer.get("image_metadata", {}))
        self.writer = CocoWriter(
            output_dir=str(self.output_dir),
            info=metadata.get("info"),
            licenses=metadata.get("licenses"),
            compose_parent_masks=writer.get("compose_parent_masks", False),
            include_camera_metadata=writer.get(
                "include_camera_metadata", False
            ),
        )

    def _camera_config(self) -> CameraConfig:
        config = self.spec.get("camera", {})
        focal_length = config.get("focal_length")
        field_of_view = config.get("field_of_view")
        if focal_length is None and field_of_view is None:
            focal_length = 50.0
        return CameraConfig(
            image_width=config.get("image_width", 1024),
            image_height=config.get("image_height", 1024),
            focal_length=focal_length,
            field_of_view=field_of_view,
            clip_start=config.get("clip_start", 0.001),
            clip_end=config.get("clip_end", 100.0),
            pixel_aspect_x=config.get("pixel_aspect_x", 1.0),
            pixel_aspect_y=config.get("pixel_aspect_y", 1.0),
            shift_x=config.get("shift_x", 0.0),
            shift_y=config.get("shift_y", 0.0),
        )

    @staticmethod
    def _mapping(config: Any, section: str) -> Mapping[str, Any]:
        if not isinstance(config, Mapping):
            raise TypeError(f"The YAML '{section}' section must be a mapping.")
        return config

    def _randomizer(self, context: Context) -> Randomizer:
        randomizer = Randomizer()

        for builder in self.randomizer_builders:
            builder(randomizer, context, self.spec)

        material_config = self._mapping(
            self.spec.get("material_randomizer", {}),
            "material_randomizer",
        )
        if material_config.get("active", False):
            rules = material_config.get("rules", [])
            if not isinstance(rules, list) or not rules:
                raise ValueError(
                    "An active material_randomizer requires a non-empty "
                    "'rules' list."
                )
            for index, rule in enumerate(rules):
                rule = self._mapping(
                    rule, f"material_randomizer.rules[{index}]"
                )
                target_objects = list(rule.get("target_objects", []))
                if not target_objects:
                    raise ValueError(
                        "Each material randomizer rule requires "
                        "'target_objects'."
                    )
                randomizer.add_randomizer(
                    MaterialRandomizer(
                        target_objects=target_objects,
                        types=rule.get("types"),
                        context=context,
                    ),
                    node_name=rule.get("name", f"material_{index}"),
                )

        background_config = self._mapping(
            self.spec.get("background_randomizer", {}),
            "background_randomizer",
        )
        if background_config.get("active", False):
            randomizer.add_randomizer(
                BackgroundRandomizer(
                    hdris_root=context.get_asset_dir() / "hdris",
                    categories=background_config.get("categories"),
                ),
                node_name="background_randomizer",
            )

        config = self._mapping(
            self.spec.get("camera_pose_randomizer", {}),
            "camera_pose_randomizer",
        )
        sampler_name = config.get("sampler", "shell_sampler")
        if sampler_name not in CAMERA_POSE_SAMPLERS:
            raise ValueError(
                f"Unknown camera sampler '{sampler_name}'. Expected one of "
                f"{sorted(CAMERA_POSE_SAMPLERS)}."
            )
        params = dict(config.get("params", {}))
        if "center" in params:
            params["center"] = np.asarray(params["center"], dtype=float)
        randomizer.add_randomizer(
            CameraPoseRandomizer(
                pose_sampling_function=CAMERA_POSE_SAMPLERS[sampler_name],
                number_of_views=config.get("number_of_views", 6),
                **params,
            ),
            node_name="camera_pose_randomizer",
        )
        return randomizer

    def generate(self) -> Path:
        metadata = self.spec.get("metadata", {})
        arrangements = int(metadata.get("num_arrangements", 1))
        seed_start = int(metadata.get("seed_start", 1000))
        camera_seed = int(metadata.get("camera_seed", 4200))
        asset_dir = metadata.get("asset_directory")
        if asset_dir is not None:
            asset_dir = (self.spec_path.parent / asset_dir).resolve()

        for arrangement in range(arrangements):
            seed = seed_start + arrangement
            np.random.seed(camera_seed + arrangement)
            context = Context(
                camera_config=self._camera_config(), asset_dir=asset_dir
            )
            layout_config = self._mapping(
                self.spec.get("layout", {}), "layout"
            )
            self.layout_builder(context, seed, layout_config)
            samples = int(self.spec.get("renderer", {}).get("samples", 32))
            renderer = self.spec.get("renderer", {})
            bproc.renderer.set_max_amount_of_samples(samples)
            bproc.renderer.set_noise_threshold(
                float(renderer.get("noise_threshold", 0.05))
            )
            if renderer.get("denoiser", "OPTIX"):
                bproc.renderer.set_denoiser(
                    renderer.get("denoiser", "OPTIX"), {}
                )
            bpy.context.scene.render.use_persistent_data = renderer.get(
                "persistent_data", True
            )
            generator = SyntheticDataGenerator(
                context=context,
                randomizer=self._randomizer(context),
                writer=self.writer,
            )
            image_metadata = dict(self._static_image_metadata)
            if self._seed_metadata_key:
                image_metadata[self._seed_metadata_key] = seed
            self.writer.set_image_metadata(image_metadata)
            # One randomized arrangement; CameraPoseRandomizer creates all
            # configured views as Blender frames in that render call.
            generator.generate(
                num_images=1,
                clean_up_scene=True,
                render_verbose=renderer.get("verbose", True),
            )

        return self.output_dir

