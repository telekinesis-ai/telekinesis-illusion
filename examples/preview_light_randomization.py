"""Compare nine lighting setups on the same mechanical part.

Run: python examples/preview_light_randomization.py
Writes a labeled comparison PNG, individual renders and sampled settings.
The camera, model, steel material, floor and exposure stay fixed throughout.
All scene operations use telekinesis.illusion's public API.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from telekinesis.illusion.core.context import Context
from telekinesis.illusion.core.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.randomizer.randomizer_node import (
    LightPoseRandomizer,
    LightRandomizer,
    MaterialRandomizer,
)
from telekinesis.illusion.sampler.camera_pose_sampler import shell_sampler
from telekinesis.illusion.types.camera import CameraConfig
from telekinesis.illusion.types.distribution import uniform
from telekinesis.illusion.types.material import PrincipledMaterial
from telekinesis.illusion.utils.assets import resolve_asset_dir
from telekinesis.illusion.writer.writer import CocoWriter


def lighting_variants(distance):
    """Edit these recipes to try other property ranges and area shapes."""
    # Scale finite-light power with distance squared to support other models.
    power = 100 * distance**2
    point = {"color": (1, 1, 1), "power": power, "radius": distance * 0.005}
    area = {"color": (1, 1, 1), "power": power, "size": distance * 0.5}
    return [
        ("00_point_reference", "POINT", point),
        (
            "01_point_power",
            "POINT",
            {**point, "power": uniform(power * 2, power * 3)},
        ),
        (
            "02_point_radius",
            "POINT",
            {**point, "radius": uniform(distance * 0.12, distance * 0.2)},
        ),
        (
            "03_point_color",
            "POINT",
            {**point, "color": uniform((1, 0.25, 0.08), (1, 0.5, 0.2))},
        ),
        (
            "04_sun_angle",
            "SUN",
            {"color": (1, 1, 1), "power": 2, "angle": uniform(0.01, 0.1)},
        ),
        (
            "05_spot_cone",
            "SPOT",
            {
                **point,
                "spot_size": uniform(np.deg2rad(20), np.deg2rad(35)),
                "spot_blend": uniform(0.15, 0.5),
            },
        ),
        (
            "06_area_shape_size",
            "AREA",
            {
                **area,
                "shape": ["RECTANGLE", "ELLIPSE"],
                "size": uniform(distance * 0.2, distance * 0.35),
                "size_y": uniform(distance * 0.4, distance * 0.6),
            },
        ),
        ("07_area_disk", "AREA", {**area, "shape": "DISK"}),
        ("08_area_pose", "AREA", {**area, "shape": "DISK"}),
    ]


def setup_scene(context, model_path, output_dir):
    """Center the part on a matte floor and frame a fixed three-quarter view."""
    context.add_model(str(model_path), "part", shading="AUTO_SMOOTH")
    part = context.get_object_group("part")[0]
    part.hide(False)
    bounds = part.get_bound_box()
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    extent = upper - lower
    if not np.isfinite(extent).all() or np.max(extent) <= 0:
        raise ValueError("The model needs a non-empty, finite bounding box.")
    part.set_location(
        part.get_location()
        - [(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, lower[2]]
    )
    center = np.array([0.0, 0.0, extent[2] / 2])
    span = float(np.max(extent))
    MaterialRandomizer(
        ["part"],
        context,
        materials=[
            PrincipledMaterial(
                "metal",
                base_color=(0.5, 0.55, 0.6),
                metallic=0.8,
                roughness=0.28,
            )
        ],
    ).randomize(context)

    # Supply a simple floor and constant environment through the asset API.
    scene_assets = output_dir / "scene_assets"
    scene_assets.mkdir(exist_ok=True)
    floor_path = scene_assets / "floor.ply"
    floor_path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "-1 -1 0\n1 -1 0\n1 1 0\n-1 1 0\n4 0 1 2 3\n",
        encoding="ascii",
    )
    context.add_model(
        str(floor_path), "floor", category_name="distractor", scale=span * 20
    )
    context.get_object_group("floor")[0].hide(False)
    MaterialRandomizer(
        ["floor"],
        context,
        materials=[
            PrincipledMaterial(
                "plastic", base_color=(0.24, 0.24, 0.24), roughness=0.7
            )
        ],
    ).randomize(context)
    ambient_path = scene_assets / "ambient.png"
    Image.new("RGB", (2, 1), (40, 40, 40)).save(ambient_path)
    context.get_background().set_background_texture(str(ambient_path))

    camera = context.get_camera()
    distance = (
        float(np.linalg.norm(extent))
        / (2 * np.sin(camera.field_of_view / 2))
        * 1.25
    )
    direction = np.array([1, -1.5, 1.3])
    direction /= np.linalg.norm(direction)
    location = center + direction * distance
    # Camera.set_camera_pose accepts XYZ Euler degrees.
    rotation = np.rad2deg(
        [np.arccos(direction[2]), 0, np.arctan2(direction[0], -direction[1])]
    )
    camera.clip_start = distance / 1000
    camera.clip_end = distance * 100
    camera_pose = camera.set_camera_pose(location, rotation)
    return center, span, camera_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        help="Model path; defaults to the bundled gearwheel_2.glb.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/light_preview/gearwheel"),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Pixels per square comparison panel.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Build and randomize the scene and write inventory.json without rendering.",
    )
    args = parser.parse_args()
    if args.resolution < 1:
        parser.error("--resolution must be positive.")
    assets = resolve_asset_dir()
    model_path = (
        args.model or assets / "models/mechanical_parts/gearwheel_2.glb"
    ).resolve()
    if not model_path.is_file():
        parser.error(
            f"Model not found: {model_path}. Supply its path with --model."
        )
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = Context(
        asset_dir=assets,
        camera_config=CameraConfig(
            image_width=args.resolution,
            image_height=args.resolution,
            field_of_view=np.deg2rad(45),
        ),
    )
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        center, span, camera_pose = setup_scene(
            context, model_path, args.output_dir
        )
        distance = max(0.6, span * 4)
        location = center + np.array([-1, -1, 2]) / np.sqrt(6) * distance
        rotation = np.array([np.arccos(2 / np.sqrt(6)), 0, -np.pi / 4])
        if not args.no_render:
            writer = CocoWriter(
                output_dir=str(args.output_dir),
                color_file_format="PNG",
                include_camera_metadata=True,
            )
            generator = SyntheticDataGenerator(context, Randomizer(), writer)

        previews = {}
        previous = None
        for index, (name, light_type, properties) in enumerate(
            lighting_variants(distance)
        ):
            if previous is not None:
                previous.set_properties(power=0)
            light = context.add_light(
                name, light_type, location=location, rotation=rotation
            )
            LightRandomizer(
                [name], seed=args.seed + index, **properties
            ).randomize(context)
            if name == "08_area_pose":
                LightPoseRandomizer(
                    shell_sampler,
                    target_lights=[name],
                    center=center,
                    radius_min=distance,
                    radius_max=distance * 1.1,
                    elevation_min=35,
                    elevation_max=70,
                    azimuth_min=-30,
                    azimuth_max=80,
                    inplane_rot_min=0,
                    inplane_rot_max=0,
                    min_distance=span * 0.5,
                ).randomize(context)

            values = light.get_properties()
            sampled = {key: values[key] for key in properties}
            image_name = None
            if not args.no_render:
                writer.set_image_metadata({"light_variant": name})
                generator.generate(num_images=1, clean_up_scene=False)
                annotations = json.loads(
                    (args.output_dir / "coco_annotations.json").read_text()
                )
                image_name = annotations["images"][-1]["file_name"]
            previews[name] = {
                "type": light_type,
                "properties": sampled,
                "location": light.get_location().tolist(),
                "rotation": light.get_rotation().tolist(),
                "active_lights": [
                    n
                    for n, item in context.get_lights().items()
                    if item.get_properties()["power"] > 0
                ],
                "image": image_name,
            }
            previous = light

        if not args.no_render:
            caption_height = max(32, args.resolution // 10)
            sheet = Image.new(
                "RGB",
                (args.resolution * 3, (args.resolution + caption_height) * 3),
                (24, 27, 32),
            )
            draw = ImageDraw.Draw(sheet)
            font = ImageFont.load_default(size=max(10, args.resolution // 28))
            for index, (name, preview) in enumerate(previews.items()):
                x = (index % 3) * args.resolution
                y = (index // 3) * (args.resolution + caption_height)
                with Image.open(args.output_dir / preview["image"]) as panel:
                    sheet.paste(panel.convert("RGB"), (x, y))
                draw.text(
                    (x + 8, y + args.resolution + 8),
                    name.replace("_", " "),
                    fill="white",
                    font=font,
                )
            sheet.save(args.output_dir / "light_variants.png")

        (args.output_dir / "inventory.json").write_text(
            json.dumps(
                {
                    "model": str(model_path),
                    "seed": args.seed,
                    "rendered": not args.no_render,
                    "resolution": args.resolution,
                    "camera_pose": camera_pose.tolist(),
                    "row_major_order": list(previews),
                    "previews": previews,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        context.clean_up()
    print(f"Light preview complete: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        # Match the material preview: cleanup completed; bypass faulty bpy teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
