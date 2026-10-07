"""Discover imperfection map sets and layer them onto a Principled shader."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import bpy

from telekinesis.illusion.types.material import SurfaceImperfections

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".exr"}


@dataclass(frozen=True)
class ImperfectionMaps:
    opacity: Path
    roughness: Path | None
    displacement: Path | None


def discover_imperfections(directory: Path) -> tuple[ImperfectionMaps, ...]:
    """Find ambientCG-style sets by *_Opacity; ignore previews and normal maps."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Surface imperfections directory not found: {directory}"
        )
    files = sorted(
        (
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        ),
        key=lambda p: p.as_posix(),
    )
    by_stem = {}
    for path in files:
        by_stem.setdefault((path.parent, path.stem.lower()), path)
    result = []
    for (parent, stem), opacity in by_stem.items():
        if stem.endswith("_opacity"):
            prefix = stem.removesuffix("_opacity")
            result.append(
                ImperfectionMaps(
                    opacity,
                    by_stem.get((parent, prefix + "_roughness")),
                    by_stem.get((parent, prefix + "_displacement")),
                )
            )
    if not result:
        raise ValueError(f"No *_Opacity image maps found in {directory}.")
    return tuple(result)


def _data_image(path: Path):
    """Reuse only our images; do not change color space on imported materials."""
    key = str(path.resolve())
    name = "IllusionImperfection_" + hashlib.sha256(key.encode()).hexdigest()
    # Blender truncates datablock names to 63 bytes.
    name = name[:63]
    existing = bpy.data.images.get(name)
    if (
        existing is not None
        and existing.get("illusion_imperfection_path") == key
    ):
        return existing
    image = bpy.data.images.load(key, check_existing=False)
    image.name = name
    image["illusion_imperfection_path"] = key
    image.colorspace_settings.name = "Non-Color"
    return image


def apply_imperfections(
    material, shader, settings: SurfaceImperfections, maps, rng
):
    """Map all imperfection channels through the mesh's rendering UV layer."""
    selected = rng.choice(maps)
    values = settings.sample(rng)
    tree = material.node_tree

    def node(kind, name, location):
        result = tree.nodes.new(kind)
        result.name = result.label = "Imperfections " + name
        result.location = location
        return result

    coordinates = node("ShaderNodeTexCoord", "Coordinates", (-1200, 0))
    mapping = node("ShaderNodeMapping", "Mapping", (-1000, 0))
    mapping.vector_type = "POINT"
    mapping.inputs["Scale"].default_value = (
        values["scale"],
        values["scale"],
        1,
    )
    mapping.inputs["Location"].default_value = (
        rng.uniform(0, 1),
        rng.uniform(0, 1),
        0,
    )
    # Rotate within the UV plane; every channel shares the same transform.
    mapping.inputs["Rotation"].default_value[2] = rng.randrange(4) * math.pi / 2
    tree.links.new(coordinates.outputs["UV"], mapping.inputs["Vector"])

    def texture(path, name, y):
        image = node("ShaderNodeTexImage", name, (-780, y))
        image.image = _data_image(path)
        image.projection = "FLAT"
        image.extension = "REPEAT"
        tree.links.new(mapping.outputs["Vector"], image.inputs["Vector"])
        return image.outputs["Color"]

    mask = texture(selected.opacity, "Mask", 360)
    strength = node("ShaderNodeMath", "Roughness Mask", (-500, 400))
    strength.operation = "MULTIPLY"
    strength.inputs[1].default_value = values["roughness_strength"]
    tree.links.new(mask, strength.inputs[0])
    roughness = node("ShaderNodeMixRGB", "Roughness", (-250, 400))
    roughness.blend_type = "MIX"
    roughness.use_clamp = True
    base = shader.inputs["Roughness"].default_value
    roughness.inputs[1].default_value = (base, base, base, 1)
    roughness.inputs[2].default_value = (1, 1, 1, 1)
    tree.links.new(strength.outputs[0], roughness.inputs[0])
    if selected.roughness:
        tree.links.new(
            texture(selected.roughness, "Roughness Map", 80),
            roughness.inputs[2],
        )
    tree.links.new(roughness.outputs[0], shader.inputs["Roughness"])

    height = mask
    if selected.displacement:
        height = texture(selected.displacement, "Height", -200)
        masked = node("ShaderNodeMath", "Masked Height", (-500, -200))
        masked.operation = "MULTIPLY"
        tree.links.new(height, masked.inputs[0])
        tree.links.new(mask, masked.inputs[1])
        height = masked.outputs[0]
    bump = node("ShaderNodeBump", "Bump", (-250, -100))
    bump.invert = selected.displacement is None
    bump.inputs["Strength"].default_value = values["bump_strength"]
    bump.inputs["Distance"].default_value = values["bump_distance"]
    tree.links.new(height, bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    material["illusion_imperfection_map"] = str(selected.opacity)


def clean_imperfection_images():
    for image in list(bpy.data.images):
        if image.get("illusion_imperfection_path") and image.users == 0:
            bpy.data.images.remove(image)
