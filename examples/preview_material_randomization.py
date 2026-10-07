"""Preview seven material variants on heavy-duty wheels in one scene.

Run: python examples/preview_material_randomization.py
The default model is in this repository's assets. Override it with --model.
Edit the material_slots mappings below to try different materials and ranges.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import bpy
import numpy as np
from mathutils import Vector

from telekinesis.illusion.core.context import Context
from telekinesis.illusion.randomizer.materials import (
    MaterialTarget,
    PBRMaterial,
)
from telekinesis.illusion.randomizer.randomizer_node import MaterialRandomizer
from telekinesis.illusion.types.material import PrincipledMaterial

DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "assets/models/mechanical_parts/heavy_duty_wheel.glb"
)


def arrange_models(models):
    """Place equally sized model variants in one centered horizontal row."""
    bounds = models[0].get_bound_box()
    extent = bounds.max(axis=0) - bounds.min(axis=0)
    spacing = max(float(extent[0]) * 1.35, 0.01)
    row_center = (len(models) - 1) / 2
    for index, model in enumerate(models):
        bounds = model.get_bound_box()
        current_center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
        target_center = np.array([(index - row_center) * spacing, 0.0, 0.0])
        model.set_location(
            model.get_location() + target_center - current_center
        )


def frame_scene(context, models, resolution):
    """Fit Illusion's camera and studio lights to all model variants."""
    bounds = np.concatenate([model.get_bound_box() for model in models])
    center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    extent = bounds.max(axis=0) - bounds.min(axis=0)
    width = max(float(extent[0]), 0.01)
    height = max(float(extent[2]), 0.01)
    camera = context.get_camera()
    camera.image_width = resolution * len(models)
    camera.image_height = resolution
    aspect = camera.image_width / camera.image_height
    horizontal_fov = camera.field_of_view
    vertical_fov = 2 * np.arctan(np.tan(horizontal_fov / 2) / aspect)
    distance = 1.2 * max(
        width / (2 * np.tan(horizontal_fov / 2)),
        height / (2 * np.tan(vertical_fov / 2)),
    )
    camera_position = center + np.array([0.0, -distance, distance * 0.12])
    rotation = (
        (Vector(center) - Vector(camera_position))
        .to_track_quat("-Z", "Y")
        .to_matrix()
    )
    camera.set_camera_pose(camera_position, rotation)
    camera.clip_start = max(distance / 1000, 1e-5)
    camera.clip_end = distance * 100

    scene = bpy.context.scene
    scene.world.node_tree.nodes["Background"].inputs[
        "Strength"
    ].default_value = 0.4
    for index, direction in enumerate([(1, -1, 2), (-1, -0.5, 1), (0, 1.5, 1)]):
        light_data = bpy.data.lights.new(f"Studio_{index}", type="AREA")
        light_data.energy = 120 * distance * distance
        light_data.shape = "DISK"
        light_data.size = max(width, height) / 3
        light = bpy.data.objects.new(light_data.name, light_data)
        scene.collection.objects.link(light)
        light.location = center + np.array(direction) * distance
        light.rotation_euler = (
            (Vector(center) - light.location)
            .to_track_quat("-Z", "Y")
            .to_euler()
        )
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.seed = 42
    scene.render.image_settings.file_format = "PNG"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/material_preview/heavy_duty_wheel"),
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(
            f"Model not found: {args.model}. Supply its path with --model."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = Context()
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Keep the wheel's three imported slots and their face assignments. Each
    # linked instance gets object-level material overrides, so all seven can
    # display a different material case in the same scene.
    context.add_model(
        str(args.model.resolve()),
        "wheel",
        preprocess_model=False,
        max_number_instances=7,
    )
    models = context.get_object_group("wheel")
    slot_names = models[0].get_material_slot_names()
    print("Imported material slots:", json.dumps(slot_names, indent=2))

    # Imported slot names: Stahl (steel), Alu (aluminum), Gumi (rubber).
    # Unlisted slots retain their imported materials in each selected-slot case.
    previews = [
        ("00_original", None),
        (
            "01_blue_plastic_slot",
            {
                "material_slots": {
                    "Alu": [
                        PrincipledMaterial(
                            "plastic",
                            base_color=(0.02, 0.15, 0.8),
                            roughness=(0.15, 0.3),
                        )
                    ],
                },
            },
        ),
        (
            "02_procedural_metal_slots",
            {
                "material_slots": {
                    "Stahl": [
                        PrincipledMaterial("metal", roughness=(0.05, 0.15))
                    ],
                    "Alu": [
                        PrincipledMaterial("metal", roughness=(0.35, 0.55))
                    ],
                },
            },
        ),
        (
            "03_pbr_metal_slots",
            {
                "material_slots": {
                    "Stahl": [PBRMaterial(types=("metal",))],
                    "Alu": [PBRMaterial(types=("metal",))],
                },
            },
        ),
        (
            "04_metal_and_rubber_slots",
            {
                "material_slots": {
                    "Alu": [
                        PrincipledMaterial(
                            "metal",
                            base_color=(0.65, 0.7, 0.8),
                            roughness=(0.25, 0.45),
                        )
                    ],
                    "Gumi": [
                        PrincipledMaterial(
                            "rubber",
                            base_color=(0.01, 0.01, 0.01),
                            roughness=(0.7, 0.9),
                        )
                    ],
                },
            },
        ),
        (
            "05_all_slots",
            {
                "material_slots": "all",
                "materials": [
                    PrincipledMaterial("metal"),
                    PrincipledMaterial("plastic"),
                    PrincipledMaterial("rubber"),
                ],
            },
        ),
        (
            "06_whole_object_pbr_metal",
            {"mode": "object", "types": ["metal"]},
        ),
    ]

    assignments = {}
    for model, (name, config) in zip(models, previews):
        if config is not None:
            target = MaterialTarget(objects=(model.get_name(),))
            MaterialRandomizer(
                target, context, seed=args.seed, **config
            ).randomize(context)
        obj = model.get_object().blender_obj
        assignments[name] = {
            original_name: slot.material.name if slot.material else None
            for original_name, slot in zip(slot_names, obj.material_slots)
        }

    arrange_models(models)
    if not args.no_render:
        frame_scene(context, models, args.resolution)
        bpy.context.scene.cycles.seed = args.seed
        bpy.context.scene.render.filepath = str(
            (args.output_dir / "material_variants.png").resolve()
        )
        bpy.ops.render.render(write_still=True)

    (args.output_dir / "inventory.json").write_text(
        json.dumps(
            {
                "model": str(args.model.resolve()),
                "objects": {
                    model.get_name(): model.get_material_slot_names()
                    for model in models
                },
                "centers": {
                    model.get_name(): (
                        (
                            model.get_bound_box().min(axis=0)
                            + model.get_bound_box().max(axis=0)
                        )
                        / 2
                    ).tolist()
                    for model in models
                },
                "previews": assignments,
                "left_to_right": [name for name, _ in previews],
                "seed": args.seed,
                "rendered": not args.no_render,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    context.clean_up()
    print(f"Heavy-duty wheel preview complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        # External bpy 4.2 can raise 0xC0000005 during native interpreter
        # teardown after all Python work has completed. Resources are already
        # released above, so bypass only that faulty native shutdown path.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
