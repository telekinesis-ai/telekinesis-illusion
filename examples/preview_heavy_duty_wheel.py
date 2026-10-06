"""Preview material randomization on the single-mesh heavy-duty wheel GLB.

Run: python examples/preview_heavy_duty_wheel.py
The default model is in this repository's assets. Override it with --model.
Edit the material_slots mappings below to try different materials and ranges.
"""

import argparse
import json
import random
from pathlib import Path

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import bpy
import numpy as np
from mathutils import Vector

import blenderproc as bproc
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


def frame_model(model, resolution):
    """Fit the camera and studio lights to the mesh's world-space bounds."""
    bounds = model.get_bound_box()
    center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    size = max(
        float(np.linalg.norm(bounds.max(axis=0) - bounds.min(axis=0))), 0.01
    )
    camera_position = center + size * np.array([1.15, -1.45, 0.95])
    rotation = bproc.camera.rotation_from_forward_vec(center - camera_position)
    bproc.camera.add_camera_pose(
        bproc.math.build_transformation_mat(camera_position, rotation)
    )
    camera = bpy.context.scene.camera
    camera.data.clip_start = size / 1000
    camera.data.clip_end = size * 100
    bproc.camera.set_resolution(resolution, resolution)
    scene = bpy.context.scene
    scene.world.node_tree.nodes["Background"].inputs[
        "Strength"
    ].default_value = 0.4
    for index, direction in enumerate([(1, -1, 2), (-1, -0.5, 1), (0, 1.5, 1)]):
        light_data = bpy.data.lights.new(f"Studio_{index}", type="AREA")
        light_data.energy = 120 * size * size
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(light_data.name, light_data)
        scene.collection.objects.link(light)
        light.location = center + np.array(direction) * size
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
    # Keep the wheel's three imported slots and their face assignments.
    context.add_model(
        str(args.model.resolve()), "wheel", preprocess_model=False
    )
    model = context.get_object_group("wheel")[0]
    obj = model.get_object().blender_obj
    slot_names = model.get_material_slot_names()
    print("Imported material slots:", json.dumps(slot_names, indent=2))
    target = MaterialTarget(objects=(model.get_name(),))
    originals = [(slot.link, slot.material) for slot in obj.material_slots]

    def randomizer(**config):
        return MaterialRandomizer(target, context, seed=args.seed, **config)

    # Imported slot names: Stahl (steel), Alu (aluminum), Gumi (rubber).
    # Unlisted slots retain their imported materials in each selected-slot case.
    previews = [
        ("00_original", None),
        (
            "01_blue_plastic_slot",
            randomizer(
                material_slots={
                    "Alu": [
                        PrincipledMaterial(
                            "plastic",
                            base_color=(0.02, 0.15, 0.8),
                            roughness=(0.15, 0.3),
                        )
                    ],
                }
            ),
        ),
        (
            "02_procedural_metal_slots",
            randomizer(
                material_slots={
                    "Stahl": [
                        PrincipledMaterial("metal", roughness=(0.05, 0.15))
                    ],
                    "Alu": [
                        PrincipledMaterial("metal", roughness=(0.35, 0.55))
                    ],
                }
            ),
        ),
        (
            "03_pbr_metal_slots",
            randomizer(
                material_slots={
                    "Stahl": [PBRMaterial(types=("metal",))],
                    "Alu": [PBRMaterial(types=("metal",))],
                }
            ),
        ),
        (
            "04_metal_and_rubber_slots",
            randomizer(
                material_slots={
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
                }
            ),
        ),
        (
            "05_all_slots",
            randomizer(
                material_slots="all",
                materials=[
                    PrincipledMaterial("metal"),
                    PrincipledMaterial("plastic"),
                    PrincipledMaterial("rubber"),
                ],
            ),
        ),
        (
            "06_whole_object_pbr_metal",
            randomizer(mode="object", types=["metal"]),
        ),
    ]
    if not args.no_render:
        frame_model(model, args.resolution)
        bpy.context.scene.cycles.seed = args.seed

    assignments = {}
    for name, node in previews:
        # Reset before each case so earlier randomizations do not accumulate.
        for slot, (link, material) in zip(obj.material_slots, originals):
            slot.link = link
            slot.material = material
        if node is not None:
            node.randomize(context)
        assignments[name] = {
            original_name: slot.material.name if slot.material else None
            for original_name, slot in zip(slot_names, obj.material_slots)
        }
        if not args.no_render:
            bpy.context.scene.render.filepath = str(
                (args.output_dir / f"{name}.png").resolve()
            )
            bpy.ops.render.render(write_still=True)

    (args.output_dir / "inventory.json").write_text(
        json.dumps(
            {
                "model": str(args.model.resolve()),
                "objects": {model.get_name(): slot_names},
                "previews": assignments,
                "seed": args.seed,
                "rendered": not args.no_render,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bproc.clean_up()
    print(f"Heavy-duty wheel preview complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
