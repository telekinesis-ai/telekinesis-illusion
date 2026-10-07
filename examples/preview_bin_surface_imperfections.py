"""Compare clean, scratched and worn plastic bins using Illusion's public API.

Run:
    python examples/preview_bin_surface_imperfections.py --asset-dir E:/telekinesis-illusion/assets

The PNG shows clean plastic, scratches and patchy wear from left to right.
All scene operations use telekinesis.illusion; no direct Blender APIs are used.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from telekinesis.illusion.core.context import Context
from telekinesis.illusion.core.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from telekinesis.illusion.randomizer.materials import MaterialTarget
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.randomizer.randomizer_node import (
    MaterialRandomizer,
    ObjectInstanceRandomizer,
)
from telekinesis.illusion.types.camera import CameraConfig
from telekinesis.illusion.types.material import (
    PrincipledMaterial,
    SurfaceImperfections,
)
from telekinesis.illusion.utils.assets import (
    pick_default_hdri,
    resolve_asset_dir,
    resolve_asset_path,
)
from telekinesis.illusion.writer.writer import CocoWriter


def arrange_and_frame(context, bins):
    """Fit the row from above so both the bin interiors and walls are visible."""
    bounds = bins[0].get_bound_box()
    spacing = float(np.ptp(bounds[:, 0])) * 1.25
    for index, bin_model in enumerate(bins):
        bounds = bin_model.get_bound_box()
        center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
        target = np.array([(index - 1) * spacing, 0.0, 0.0])
        bin_model.set_location(bin_model.get_location() + target - center)

    corners = np.concatenate([model.get_bound_box() for model in bins])
    center = (corners.min(axis=0) + corners.max(axis=0)) / 2
    # Camera Euler X=40 degrees gives a view 50 degrees above the horizon.
    angle = np.deg2rad(40)
    up = np.array([0.0, np.cos(angle), np.sin(angle)])
    backward = np.array([0.0, -np.sin(angle), np.cos(angle)])
    relative = corners - center
    camera = context.get_camera()
    tan_x = np.tan(camera.field_of_view / 2)
    tan_y = tan_x * camera.image_height / camera.image_width
    distance = 1.15 * float(
        np.max(
            relative @ backward
            + np.maximum(
                np.abs(relative[:, 0]) / tan_x,
                np.abs(relative @ up) / tan_y,
            )
        )
    )
    camera.clip_start = distance / 1000
    camera.clip_end = distance * 100
    camera.set_camera_pose(
        center + backward * distance,
        np.array([40.0, 0.0, 0.0]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-dir",
        default=r"E:/telekinesis-illusion/assets",
        type=Path,
        help="Root containing models, hdris and surface_imperfections.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/bins/plastic_bin_3.glb"),
        help="Absolute path or path relative to --asset-dir.",
    )
    parser.add_argument(
        "--imperfections-dir",
        type=Path,
        default=Path("surface_imperfections"),
        help="Folder containing the supplied scratch and surface-imperfection sets.",
    )
    parser.add_argument(
        "--hdri",
        type=Path,
        help="Optional lighting EXR; defaults to the first studio HDRI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/material_preview/plastic_bin_imperfections"),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image height; image width is three times this value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Build and randomize the scene, then write only the inventory.",
    )
    args = parser.parse_args()
    if args.resolution < 1:
        parser.error("--resolution must be positive.")

    assets = resolve_asset_dir(args.asset_dir)
    model_path = resolve_asset_path(args.model, assets)
    imperfections = resolve_asset_path(args.imperfections_dir, assets)
    # Pin the map families for an informative comparison. To sample from all
    # sets instead, use surface_imperfections=True on a PrincipledMaterial.
    scratch_dir = imperfections / "Scratches003_1K-JPG"
    wear_dir = imperfections / "SurfaceImperfections015_1K-JPG"
    for path in (model_path, scratch_dir, wear_dir):
        if not path.exists():
            parser.error(
                f"Asset not found: {path}. Check --asset-dir and --imperfections-dir."
            )
    hdri = (
        resolve_asset_path(args.hdri, assets)
        if args.hdri
        else pick_default_hdri(
            assets / "hdris", preferred_category="indoor/studio"
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = Context(
        asset_dir=assets,
        camera_config=CameraConfig(
            image_width=args.resolution * 3,
            image_height=args.resolution,
            field_of_view=np.deg2rad(55),
        ),
    )
    try:
        context.get_background().set_background_texture(str(hdri))
        context.add_model(
            str(model_path),
            "bin",
            category_name="plastic_bin",
            min_number_instances=3,
            max_number_instances=3,
            shading="AUTO_SMOOTH",
            uv_mapping="smart",
        )
        bins = context.get_object_group("bin")
        arrange_and_frame(context, bins)

        # Deliberately stronger than the feature defaults for a visible demo.
        variants = [
            ("clean", False),
            (
                "scratches",
                SurfaceImperfections(
                    directory=scratch_dir,
                    roughness_strength=0.8,
                    bump_strength=0.3,
                    bump_distance=0.0003,
                    scale=1.0,
                ),
            ),
            (
                "patchy_wear",
                SurfaceImperfections(
                    directory=wear_dir,
                    roughness_strength=0.8,
                    bump_strength=0.3,
                    bump_distance=0.0003,
                    scale=1.0,
                ),
            ),
        ]
        randomizer = Randomizer()
        randomizer.add_randomizer(
            ObjectInstanceRandomizer(
                target_objects=["bin"],
                min_num_total_objects=3,
                max_num_total_objects=3,
            ),
            node_name="visible_bins",
        )
        for bin_model, (name, imperfections) in zip(bins, variants):
            randomizer.add_randomizer(
                MaterialRandomizer(
                    MaterialTarget(objects=(bin_model.get_name(),)),
                    context,
                    materials=[
                        PrincipledMaterial(
                            "plastic",
                            base_color=(0.025, 0.12, 0.35),
                            roughness=0.22,
                            surface_imperfections=imperfections,
                        )
                    ],
                    seed=args.seed,
                ),
                node_name=f"material_{name}",
            )

        if args.no_render:
            randomizer.randomize(context)
        else:
            writer = CocoWriter(
                output_dir=str(args.output_dir), color_file_format="PNG"
            )
            writer.set_image_metadata(
                {
                    "left_to_right": [name for name, _ in variants],
                    "material_seed": args.seed,
                }
            )
            SyntheticDataGenerator(context, randomizer, writer).generate(
                num_images=1,
                simulate_physics=False,
                clean_up_scene=False,
            )
        (args.output_dir / "inventory.json").write_text(
            json.dumps(
                {
                    "model": str(model_path),
                    "hdri": str(hdri),
                    "seed": args.seed,
                    "rendered": not args.no_render,
                    "left_to_right": [name for name, _ in variants],
                    "variants": {
                        name: {
                            "object": model.get_name(),
                            "surface_imperfections": settings is not False,
                            "directory": str(settings.directory)
                            if settings
                            else None,
                        }
                        for model, (name, settings) in zip(bins, variants)
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        context.clean_up()
    print(
        f"Plastic bin imperfection preview complete: {args.output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
