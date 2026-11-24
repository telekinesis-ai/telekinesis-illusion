"""Layer procedural dust, edge wear and scratches over a Principled shader.

All masks are built from shader nodes (no images): noise in object space, an
Ambient Occlusion node for cavities and convexity, the world normal for
up-facing faces and a Bevel normal for edges. Bevel and Ambient Occlusion are
Cycles-only nodes; EEVEE renders them as constants (no edge wear, flat dust).
"""

import math

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import bpy  # noqa: E402,F401  (loads the Blender types used below)

_PREFIX = "Wear "
# Noise-to-mask windows: a soft threshold on Noise Texture's ~[0.2, 0.8] output.
_DUST_WINDOW = (0.4, 0.62)
# Occlusion (1 - AO) mapped to a cavity mask: open faces read 0, creases and corners 1.
_CAVITY_WINDOW = (0.1, 0.45)
_EDGE_WINDOW = (0.35, 0.6)
_SCRATCH_WINDOW = (0.6, 0.64)
_SCRATCH_STRETCH = 30.0
# 1 - cos(angle) between Bevel and surface normal at full edge strength (~25 deg).
_EDGE_GAIN = 10.0


def apply_wear(material, shader, values: dict, rng) -> None:
    """Mix wear layers into ``shader``'s Base Color and Roughness inputs.

    Args:
        material: Blender material whose node tree holds ``shader``.
        shader: Principled BSDF node; its current Base Color and Roughness
            (linked or default values) are the layers' base.
        values: A sample from MaterialWear.sample.
        rng: Random generator for the noise offsets and scratch direction.
    """
    tree = material.node_tree
    nodes, links = tree.nodes, tree.links
    x0, y0 = shader.location.x - 1400, shader.location.y - 600

    def node(kind, name, dx, dy):
        result = nodes.new(kind)
        result.name = result.label = _PREFIX + name
        result.location = (x0 + dx, y0 + dy)
        return result

    def connect(value, socket):
        if isinstance(value, bpy.types.NodeSocket):
            links.new(value, socket)
        else:
            socket.default_value = value

    def math_node(operation, a, b, name, dx, dy, clamp=True):
        result = node("ShaderNodeMath", name, dx, dy)
        result.operation = operation
        result.use_clamp = clamp
        connect(a, result.inputs[0])
        connect(b, result.inputs[1])
        return result.outputs[0]

    def map_range(value, low, high, to_low, to_high, name, dx, dy):
        """Clamped linear map; also a lerp from to_low to to_high by value in [0, 1]."""
        result = node("ShaderNodeMapRange", name, dx, dy)
        result.data_type = "FLOAT"
        result.interpolation_type = "LINEAR"
        result.clamp = True
        connect(value, result.inputs["Value"])
        result.inputs["From Min"].default_value = low
        result.inputs["From Max"].default_value = high
        connect(to_low, result.inputs["To Min"])
        connect(to_high, result.inputs["To Max"])
        return result.outputs["Result"]

    coordinates = node("ShaderNodeTexCoord", "Coordinates", 0, 0)

    def noise(name, size, dy, stretch=1.0, rotation=0.0, detail=4.0):
        mapping = node("ShaderNodeMapping", name + " Mapping", 200, dy)
        mapping.vector_type = "POINT"
        # Point mapping scales, rotates, then adds Location (a random shift in noise units).
        mapping.inputs["Scale"].default_value = (
            1.0 / size,
            1.0 / (size * stretch),
            1.0 / size,
        )
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, rotation)
        mapping.inputs["Location"].default_value = tuple(
            rng.uniform(0.0, 100.0) for _ in range(3)
        )
        links.new(coordinates.outputs["Object"], mapping.inputs["Vector"])
        texture = node("ShaderNodeTexNoise", name, 400, dy)
        texture.noise_dimensions = "3D"
        texture.inputs["Scale"].default_value = 1.0
        texture.inputs["Detail"].default_value = detail
        texture.inputs["Roughness"].default_value = 0.55
        links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
        return texture.outputs["Fac"]

    geometry = node("ShaderNodeNewGeometry", "Geometry", 0, -300)
    occlusion = node("ShaderNodeAmbientOcclusion", "Occlusion", 200, -300)
    occlusion.only_local = True
    occlusion.samples = 8
    occlusion.inputs["Distance"].default_value = values["ao_distance"]
    ao = occlusion.outputs["AO"]

    dust_mask = edge_mask = scratch_mask = None
    if values["dust"] > 0.0:
        cavity = math_node("SUBTRACT", 1.0, ao, "Occlusion Inverse", 400, -300)
        cavity = map_range(cavity, *_CAVITY_WINDOW, 0.0, 1.0, "Cavity", 500, -300)
        cavity = math_node("MULTIPLY", cavity, values["dust_cavity"], "Cavity Weight", 600, -300, clamp=False)
        up = node("ShaderNodeSeparateXYZ", "Up", 200, -500)
        links.new(geometry.outputs["Normal"], up.inputs[0])
        up_weight = math_node("MULTIPLY", up.outputs["Z"], values["dust_up"], "Up Weight", 400, -500)
        region = math_node("ADD", cavity, up_weight, "Dust Region", 800, -400)
        grime = map_range(noise("Dust Noise", values["grime_scale"], 200), *_DUST_WINDOW, 0.0, 1.0, "Dust Breakup", 600, 200)
        dust_mask = math_node("MULTIPLY", region, grime, "Dust Mask", 1000, -200)
        dust_mask = math_node("MULTIPLY", dust_mask, values["dust"], "Dust Strength", 1200, -200)
    if values["edge_wear"] > 0.0:
        bevel = node("ShaderNodeBevel", "Bevel", 200, -700)
        bevel.samples = 8
        bevel.inputs["Radius"].default_value = values["edge_width"]
        dot = node("ShaderNodeVectorMath", "Edge Dot", 400, -700)
        dot.operation = "DOT_PRODUCT"
        links.new(bevel.outputs["Normal"], dot.inputs[0])
        links.new(geometry.outputs["Normal"], dot.inputs[1])
        edge = math_node("SUBTRACT", 1.0, dot.outputs["Value"], "Edge", 600, -700)
        edge = math_node("MULTIPLY", edge, _EDGE_GAIN, "Edge Gain", 800, -700)
        # Convex edges are unoccluded; concave creases (low AO) keep their paint.
        edge = math_node("MULTIPLY", edge, ao, "Convex Edge", 1000, -700)
        breakup = map_range(
            noise("Edge Noise", values["grime_scale"] * 0.5, -900), *_EDGE_WINDOW, 0.0, 1.0, "Edge Breakup", 600, -900
        )
        edge_mask = math_node("MULTIPLY", edge, breakup, "Edge Mask", 1200, -800)
        edge_mask = math_node("MULTIPLY", edge_mask, values["edge_wear"], "Edge Strength", 1400, -800)
    if values["scratches"] > 0.0:
        direction = rng.uniform(0.0, math.pi)
        streaks = noise("Scratch Noise", values["scratch_scale"], -1100, stretch=_SCRATCH_STRETCH, rotation=direction, detail=2.0)
        scratch_mask = map_range(streaks, *_SCRATCH_WINDOW, 0.0, values["scratches"], "Scratch Mask", 600, -1100)

    def current(socket):
        return socket.links[0].from_socket if socket.links else socket.default_value

    color_socket = shader.inputs["Base Color"]
    color = current(color_socket)
    for mask, target, name, dy in (
        (dust_mask, values["dust_color"], "Dust Color", 400),
        (edge_mask, values["edge_color"], "Edge Color", 300),
    ):
        if mask is None:
            continue
        mix = node("ShaderNodeMixRGB", name, 1600, dy)
        mix.blend_type = "MIX"
        links.new(mask, mix.inputs["Fac"])
        connect(color if isinstance(color, bpy.types.NodeSocket) else tuple(color), mix.inputs["Color1"])
        mix.inputs["Color2"].default_value = (*target, 1.0)
        color = mix.outputs["Color"]
    if isinstance(color, bpy.types.NodeSocket):
        links.new(color, color_socket)

    roughness_socket = shader.inputs["Roughness"]
    roughness = current(roughness_socket)
    if dust_mask is not None:
        roughness = map_range(dust_mask, 0.0, 1.0, roughness, values["dust_roughness"], "Dust Roughness", 1600, 100)
    if edge_mask is not None:
        roughness = map_range(edge_mask, 0.0, 1.0, roughness, values["edge_roughness"], "Edge Roughness", 1600, 0)
    if scratch_mask is not None:
        roughness = math_node("ADD", roughness, scratch_mask, "Scratch Roughness", 1600, -100)
    if isinstance(roughness, bpy.types.NodeSocket):
        links.new(roughness, roughness_socket)
    material["illusion_wear"] = True
