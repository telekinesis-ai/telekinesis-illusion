"""Real Blender slot/mesh integration tests and pure material sampling tests."""

import random
from pathlib import Path

import numpy as np
import pytest

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import bpy
from blenderproc.python.types.MeshObjectUtility import MeshObject

import blenderproc as bproc
from telekinesis.illusion.randomizer.materials import (
    MaterialTarget,
)
from telekinesis.illusion.randomizer.randomizer_node import (
    MaterialRandomizer,
)
from telekinesis.illusion.types.distribution import uniform
from telekinesis.illusion.types.material import (
    MATERIAL_PRESETS,
    PARAMETERS,
    MaterialPreset,
    PrincipledMaterial,
    SurfaceImperfections,
)
from telekinesis.illusion.types.object import Object

ROOT = Path(__file__).resolve().parents[1]


def mesh(context, name="Body_INSTANCE_0", slots=("paint", "trim", "glass")):
    data = bpy.data.meshes.new(name)
    data.from_pydata(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        [],
        [(0, 1, 2), (0, 1, 3), (0, 2, 3)],
    )
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    for slot in slots:
        material = bpy.data.materials.get(slot) or bpy.data.materials.new(slot)
        data.materials.append(material)
    for index, polygon in enumerate(data.polygons):
        polygon.material_index = index % max(len(slots), 1)
    return context.add_mesh_object(MeshObject(obj), name)


def assigned(obj):
    return tuple(
        s.material for s in obj.get_object().blender_obj.material_slots
    )


def shader_values(obj):
    return [
        {
            name: tuple(socket.default_value)
            if hasattr(socket.default_value, "__len__")
            else socket.default_value
            for name, (socket_name, *_) in PARAMETERS.items()
            for socket in [
                mat.node_tree.nodes.get("Principled BSDF").inputs[socket_name]
            ]
        }
        for mat in assigned(obj)
    ]


@pytest.mark.parametrize(
    "slots",
    [
        {"paint": [PrincipledMaterial("metal")]},
        {
            "paint": [PrincipledMaterial("metal")],
            "trim": [PrincipledMaterial("rubber")],
        },
    ],
)
def test_selected_slots_preserve_others_and_repeat(context, slots):
    obj = mesh(context)
    original = assigned(obj)
    polygons = [
        p.material_index for p in obj.get_object().blender_obj.data.polygons
    ]
    node = MaterialRandomizer(["Body"], context, material_slots=slots, seed=42)
    for _ in range(3):
        node.randomize(context)
        for index, name in enumerate(obj.get_material_slot_names()):
            assert (assigned(obj)[index] == original[index]) == (
                name not in slots
            )
    assert [
        p.material_index for p in obj.get_object().blender_obj.data.polygons
    ] == polygons


@pytest.mark.parametrize("mode", ["object", "slots"])
def test_whole_object_and_all_slots(context, mode):
    obj = mesh(context)
    node = MaterialRandomizer(
        ["Body"],
        context,
        mode=mode,
        materials=[PrincipledMaterial("plastic")],
        material_slots="all" if mode == "slots" else None,
        seed=42,
    )
    node.randomize(context)
    assert len(assigned(obj)) == 3
    assert len(set(assigned(obj))) == (1 if mode == "object" else 3)
    for material in assigned(obj):
        assert {node.type for node in material.node_tree.nodes} == {
            "BSDF_PRINCIPLED",
            "OUTPUT_MATERIAL",
        }
        assert len(material.node_tree.links) == 1
    assert len(obj.get_object().blender_obj.data.uv_layers) == 0


def test_legacy_active_slot_and_pbr(context):
    obj = mesh(context)
    original = assigned(obj)
    obj.get_object().blender_obj.active_material_index = 1
    random.seed(71)
    node = MaterialRandomizer(["Body"], context, types=["metal"])
    node.randomize(context)
    assert assigned(obj)[0] == original[0]
    assert assigned(obj)[2] == original[2]
    material = assigned(obj)[1]
    assert material != original[1]
    assert any(n.type == "TEX_IMAGE" for n in material.node_tree.nodes)
    random.seed(71)
    assert (
        context.material_manager.get_random_material(["metal"]).blender_obj
        == material
    )


@pytest.mark.parametrize("policy,slots", [("replace", 1), ("preserve", 2)])
def test_preprocessing_and_linked_duplicates(context, tmp_path, policy, slots):
    path = tmp_path / "two.obj"
    path.write_text(
        "mtllib two.mtl\no test\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
        "usemtl a\nf 1 2 3\nusemtl b\nf 1 2 4\n"
    )
    path.with_suffix(".mtl").write_text(
        "newmtl a\nKd 1 0 0\nnewmtl b\nKd 0 1 0\n"
    )
    context.add_model(
        str(path), "part", max_number_instances=2, material_preprocessing=policy
    )
    obj, duplicate = context.get_object_group("part")
    assert len(assigned(obj)) == slots
    original = assigned(duplicate)
    if policy == "replace":
        assert obj.get_material_slot_names() == ("DummyMaterial",)
        with pytest.raises(ValueError, match="collapsed"):
            MaterialRandomizer(
                ["part"],
                context,
                material_slots="all",
                materials=[PrincipledMaterial()],
            ).randomize(context)
        MaterialRandomizer(["part_INSTANCE_0"], context, ["metal"]).randomize(
            context
        )
        assert assigned(obj)[0].get("is_cc_texture")
    else:
        MaterialRandomizer(
            MaterialTarget(objects=(obj.get_name(),)),
            context,
            material_slots={"a": [PrincipledMaterial("metal")]},
        ).randomize(context)
    assert assigned(duplicate) == original


@pytest.mark.parametrize(
    "slot,match", [("absent", "missing"), ("same", "ambiguous"), (5, "missing")]
)
def test_slot_errors_are_atomic(context, slot, match):
    first = mesh(context, "Body", ("ok", "same", "same"))
    before = assigned(first)
    node = MaterialRandomizer(
        ["Body"],
        context,
        material_slots={
            "ok": [PrincipledMaterial()],
            slot: [PrincipledMaterial()],
        },
    )
    with pytest.raises(ValueError, match=match):
        node.randomize(context)
    assert assigned(first) == before


def test_duplicate_names_can_be_selected_by_index(context):
    obj = mesh(context, slots=("same", "same"))
    before = assigned(obj)
    MaterialRandomizer(
        ["Body"], context, material_slots={1: [PrincipledMaterial()]}
    ).randomize(context)
    assert assigned(obj)[0] == before[0]
    assert assigned(obj)[1] != before[1]


def test_linked_mesh_and_shared_material_are_not_mutated(context):
    obj = mesh(context)
    duplicate = obj.create_linked_duplicate()
    original = assigned(duplicate)
    MaterialRandomizer(
        ["Body_INSTANCE_0"],
        context,
        material_slots={"trim": [PrincipledMaterial()]},
    ).randomize(context)
    assert (
        obj.get_object().blender_obj.data
        == duplicate.get_object().blender_obj.data
    )
    assert assigned(duplicate) == original
    assert all(mat.use_nodes is False for mat in original)


@pytest.mark.parametrize("selected", [("First",), ("First", "Second"), None])
def test_exact_registered_object_targets(context, selected):
    objects = {
        name: mesh(context, name) for name in ("First", "Second", "Third")
    }
    original = {name: assigned(obj) for name, obj in objects.items()}
    node = MaterialRandomizer(
        MaterialTarget(objects=selected),
        context,
        materials=[PrincipledMaterial("metal")],
        seed=42,
    )
    node.randomize(context)
    for name, obj in objects.items():
        assert (assigned(obj) != original[name]) == (
            selected is None or name in selected
        )


def test_different_slot_layouts_validate_before_assignment(context):
    body = mesh(context, "Body", ("paint",))
    mesh(context, "Wheels", ("rubber",))
    original = assigned(body)
    node = MaterialRandomizer(
        MaterialTarget(),
        context,
        material_slots={"paint": [PrincipledMaterial()]},
    )
    with pytest.raises(ValueError, match="missing"):
        node.randomize(context)
    assert assigned(body) == original
    MaterialRandomizer(
        MaterialTarget(),
        context,
        material_slots="all",
        materials=[PrincipledMaterial()],
    ).randomize(context)


def test_missing_targets(context):
    target = MaterialTarget(objects=("absent",))
    with pytest.raises(ValueError, match="Missing|Unknown"):
        MaterialRandomizer(
            target, context, materials=[PrincipledMaterial()]
        ).randomize(context)


@pytest.mark.parametrize(
    "config",
    [
        {"material_slots": {}},
        {"material_slots": {"paint": []}},
        {"material_slots": {True: [PrincipledMaterial()]}},
        {"mode": "bad"},
        {"mode": "slots"},
        {"materials": []},
        {"materials": "metal"},
        {"materials": [object()]},
        {"mode": "object", "material_slots": "all"},
        {"types": ["metal"], "materials": [PrincipledMaterial()]},
        {"seed": 2.5},
    ],
)
def test_invalid_randomizer_config(context, config):
    with pytest.raises((ValueError, TypeError)):
        MaterialRandomizer(["Body"], context, **config)


def test_seed_determinism_and_private_rng(context):
    obj = mesh(context)
    random.seed(9)
    global_state = random.getstate()

    def run():
        node = MaterialRandomizer(
            ["Body"],
            context,
            material_slots="all",
            materials=[
                PrincipledMaterial("metal"),
                PrincipledMaterial("plastic"),
            ],
            seed=7,
        )
        result = []
        for _ in range(3):
            node.randomize(context)
            result.append(shader_values(obj))
        return result

    assert run() == run()
    assert random.getstate() == global_state


def test_generated_material_cleanup(context):
    mesh(context)
    node = MaterialRandomizer(
        ["Body"],
        context,
        material_slots="all",
        materials=[PrincipledMaterial("rubber")],
    )
    for _ in range(12):
        node.randomize(context)
    generated = [
        m for m in bpy.data.materials if m.get("illusion_generated_material")
    ]
    assert len(generated) == 3


@pytest.mark.parametrize("preset", MATERIAL_PRESETS)
def test_presets_and_overrides(preset):
    recipe = PrincipledMaterial(preset, roughness=(0.2, 0.6), alpha=0.7)
    for _ in range(30):
        sample = recipe.sample()
        assert 0.2 <= sample["roughness"] <= 0.6
        assert sample["alpha"] == 0.7
        assert 0 <= sample["metallic"] <= 1
    assert "alpha" not in MATERIAL_PRESETS[preset].parameters


def test_color_ranges_uniform_custom_preset_and_fixed_values():
    preset = MaterialPreset("custom", {"roughness": uniform(0.2, 0.4)})
    recipe = PrincipledMaterial(
        preset,
        base_color=((0.1, 0.2, 0.3), (0.2, 0.4, 0.6)),
        subsurface_radius=(1, 2, 3),
        emission_strength=5,
    )
    for _ in range(100):
        sample = recipe.sample()
        assert 0.2 <= sample["roughness"] <= 0.4
        assert np.all(np.array(sample["base_color"]) >= [0.1, 0.2, 0.3, 1])
        assert np.all(np.array(sample["base_color"]) <= [0.2, 0.4, 0.6, 1])
        assert sample["subsurface_radius"] == (1, 2, 3)
        assert sample["emission_strength"] == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preset": "unknown"},
        {"roughness": (0.8, 0.2)},
        {"metallic": -1},
        {"alpha": 1.1},
        {"roughness": float("nan")},
        {"coat_weight": True},
        {"base_color": (0.1, 0.2)},
        {"base_color": ((0, 0, 0), (1, 1))},
        {"subsurface_radius": (-1, 1, 1)},
        {"emission_strength": float("inf")},
        {"emission_strength": 1e100},
        {"subsurface_radius": (1e100, 1, 1)},
        {"made_up_socket": 1},
        {"ior": 0.5},
    ],
)
def test_invalid_principled_ranges(kwargs):
    with pytest.raises((ValueError, TypeError)):
        PrincipledMaterial(**kwargs)


@pytest.mark.parametrize(
    "settings",
    [
        1,
        "true",
        {"enabled": "false"},
        {"typo": 1},
        {"directory": ""},
        {"roughness_strength": [-0.1, 0.2]},
        {"bump_strength": 1.1},
        {"scale": 0},
        {"scale": [3, 1]},
        {"bump_distance": float("nan")},
        {"bump_distance": 1e100},
        {"scale": True},
    ],
)
def test_invalid_surface_imperfections(settings):
    with pytest.raises((ValueError, TypeError)):
        PrincipledMaterial(surface_imperfections=settings)


@pytest.fixture
def imperfection_dir(tmp_path):
    # Small real images exercise Blender's loading and shader API, including
    # a scratches-only set and a set with separate roughness/height maps.
    for name, channels in (
        ("Scratches", ["Opacity", "NormalGL", "Color"]),
        ("Smudges", ["Opacity", "Roughness", "Displacement"]),
    ):
        folder = tmp_path / name
        folder.mkdir()
        for channel in channels:
            image = bpy.data.images.new("test_map", width=2, height=2)
            image.pixels[:] = [0.2, 0.2, 0.2, 1, 0.8, 0.8, 0.8, 1] * 2
            image.filepath_raw = str(folder / f"{name}_{channel}.png")
            image.file_format = "PNG"
            image.save()
            bpy.data.images.remove(image)
    return tmp_path


@pytest.mark.parametrize("family", ["Scratches", "Smudges"])
def test_surface_imperfections_shader_and_image_reuse(
    context, imperfection_dir, family
):
    obj = mesh(context)
    # Keep an imported image alive and verify its sRGB interpretation is intact.
    path = imperfection_dir / family / f"{family}_Opacity.png"
    imported = bpy.data.images.load(str(path), check_existing=False)
    imported.use_fake_user = True
    imported.colorspace_settings.name = "sRGB"
    config = {
        "seed": 12,
        "material_slots": {
            "paint": [
                {
                    "preset": "metal",
                    "parameters": {"roughness": 0.25, "alpha": 0.8},
                    "surface_imperfections": {
                        "directory": str(imperfection_dir / family),
                        "roughness_strength": 0.2,
                        "bump_strength": 0.1,
                        "bump_distance": 0.0001,
                        "scale": 2,
                    },
                }
            ]
        },
    }
    original = assigned(obj)
    node = MaterialRandomizer.from_config(["Body"], context, config)
    for _ in range(4):
        node.randomize(context)
        material = assigned(obj)[0]
        nodes = material.node_tree.nodes
        shader = nodes.get("Principled BSDF")
        assert assigned(obj)[1:] == original[1:]
        assert shader.inputs["Roughness"].is_linked
        assert shader.inputs["Normal"].is_linked
        assert shader.inputs["Alpha"].default_value == pytest.approx(0.8)
        assert not shader.inputs["Alpha"].is_linked
        assert not shader.inputs["Base Color"].is_linked
        assert shader.inputs["Metallic"].default_value == 1
        assert not nodes.get("Material Output").inputs["Displacement"].is_linked
        textures = [n for n in nodes if n.type == "TEX_IMAGE"]
        assert len(textures) == (3 if family == "Smudges" else 1)
        assert all(
            n.image.colorspace_settings.name == "Non-Color" for n in textures
        )
        assert all(n.projection == "FLAT" for n in textures)
        assert (
            nodes["Imperfections Mapping"]
            .inputs["Vector"]
            .links[0]
            .from_socket.name
            == "UV"
        )
        assert nodes["Imperfections Bump"].invert == (family == "Scratches")
        assert (
            nodes["Imperfections Roughness"].inputs[1].default_value[0] == 0.25
        )
        owned = [
            i for i in bpy.data.images if i.get("illusion_imperfection_path")
        ]
        assert len(owned) == len(textures)
        assert imported.colorspace_settings.name == "sRGB"
    # A later plain recipe also reclaims all imperfection image datablocks.
    MaterialRandomizer(
        ["Body"], context, materials=[PrincipledMaterial()]
    ).randomize(context)
    assert not [
        i for i in bpy.data.images if i.get("illusion_imperfection_path")
    ]
    assert imported in list(bpy.data.images)


@pytest.mark.parametrize("layer_count", [1, 2])
def test_surface_imperfections_preserve_existing_uvs(
    context, imperfection_dir, layer_count
):
    obj = mesh(context)
    data = obj.get_object().blender_obj.data
    for index in range(layer_count):
        layer = data.uv_layers.new(name=f"Authored_{index}")
        for corner, uv in enumerate(layer.data):
            uv.uv = (corner * 0.1 + index, corner * 0.05)
    data.uv_layers.active_index = layer_count - 1
    data.uv_layers[0].active_render = True

    def snapshot():
        return [
            (
                layer.name,
                layer.active_render,
                [tuple(uv.uv) for uv in layer.data],
            )
            for layer in data.uv_layers
        ]

    before = snapshot()
    node = MaterialRandomizer(
        ["Body"],
        context,
        material_slots="all",
        materials=[
            PrincipledMaterial(
                surface_imperfections={"directory": imperfection_dir}
            )
        ],
    )
    for _ in range(3):
        node.randomize(context)
        assert obj.get_object().blender_obj.data == data
        assert snapshot() == before
        assert data.uv_layers.active_index == layer_count - 1


def test_surface_imperfections_smart_uv_fallback_once_and_private(
    context, imperfection_dir, monkeypatch
):
    obj = mesh(context)
    duplicate = obj.create_linked_duplicate()
    shared = duplicate.get_object().blender_obj.data
    original_materials = assigned(duplicate)
    blender_obj = obj.get_object().blender_obj
    other = mesh(context, name="Other").get_object().blender_obj
    bpy.ops.object.select_all(action="DESELECT")
    other.select_set(True)
    bpy.context.view_layer.objects.active = other
    blender_obj.hide_viewport = True
    obj.hide(True)
    project = MeshObject.add_uv_mapping
    calls = []

    def record_projection(self, projection, overwrite=False):
        calls.append((projection, overwrite))
        return project(self, projection, overwrite=overwrite)

    monkeypatch.setattr(MeshObject, "add_uv_mapping", record_projection)
    node = MaterialRandomizer(
        ["Body"],
        context,
        material_slots="all",
        materials=[
            PrincipledMaterial(
                surface_imperfections={"directory": imperfection_dir}
            )
        ],
    )
    for _ in range(3):
        node.randomize(context)
        assert blender_obj.data != shared
        assert len(blender_obj.data.uv_layers) == 1
        coords = np.array(
            [uv.uv[:] for uv in blender_obj.data.uv_layers.active.data]
        )
        assert np.isfinite(coords).all() and np.ptp(coords) > 0
        assert calls == [("smart", False)]
        assert not shared.uv_layers
        assert assigned(duplicate) == original_materials
        assert blender_obj.hide_viewport and blender_obj.hide_get()
        assert obj.is_hidden()
        assert bpy.context.view_layer.objects.active == other
        assert bpy.context.selected_objects == [other]


def test_surface_imperfections_seed_and_asset_root(
    context, imperfection_dir, monkeypatch
):
    obj = mesh(context)
    monkeypatch.setattr(
        context.material_manager,
        "get_asset_dir",
        lambda: imperfection_dir.parent,
    )
    recipe = PrincipledMaterial(
        "plastic",
        surface_imperfections=SurfaceImperfections(
            directory=imperfection_dir.name
        ),
    )
    random.seed(45)
    state = random.getstate()

    def run():
        node = MaterialRandomizer(
            ["Body"], context, materials=[recipe], seed=93
        )
        samples = []
        for _ in range(6):
            node.randomize(context)
            material = assigned(obj)[0]
            nodes = material.node_tree.nodes
            mapping = nodes["Imperfections Mapping"]
            samples.append(
                (
                    material["illusion_imperfection_map"],
                    tuple(mapping.inputs["Scale"].default_value),
                    tuple(mapping.inputs["Location"].default_value),
                    tuple(mapping.inputs["Rotation"].default_value),
                    nodes["Imperfections Roughness Mask"]
                    .inputs[1]
                    .default_value,
                    nodes["Imperfections Bump"]
                    .inputs["Strength"]
                    .default_value,
                )
            )
        return samples

    first = run()
    assert first == run()
    assert len(set(first)) > 1
    assert random.getstate() == state


def test_surface_imperfections_default_directory_and_disabled(
    context, imperfection_dir, monkeypatch
):
    obj = mesh(context)
    monkeypatch.setattr(
        context.material_manager, "get_asset_dir", lambda: imperfection_dir
    )
    (imperfection_dir / "Scratches").rename(
        imperfection_dir / "surface_imperfections"
    )
    MaterialRandomizer.from_config(
        ["Body"],
        context,
        {"materials": [{"preset": "plastic", "surface_imperfections": True}]},
    ).randomize(context)
    assert "illusion_imperfection_map" in assigned(obj)[0]
    for settings in (False, {"enabled": False, "directory": "missing"}):
        node = MaterialRandomizer(
            ["Body"],
            context,
            materials=[PrincipledMaterial(surface_imperfections=settings)],
            seed=7,
        )
        node.randomize(context)
        assert len(assigned(obj)[0].node_tree.nodes) == 2


def test_surface_imperfections_missing_and_empty(context, tmp_path):
    for path, error in (
        (tmp_path / "missing", FileNotFoundError),
        (tmp_path, ValueError),
    ):
        with pytest.raises(error):
            MaterialRandomizer(
                ["Body"],
                context,
                materials=[
                    PrincipledMaterial(
                        surface_imperfections={"directory": str(path)}
                    )
                ],
            )


def test_config_parser(context):
    obj = mesh(context)
    config = {
        "material_slots": {
            "trim": [
                {"preset": "rubber", "parameters": {"roughness": [0.7, 0.9]}}
            ]
        },
        "seed": 5,
    }
    MaterialRandomizer.from_config(["Body"], context, config).randomize(context)
    assert (
        0.7 <= shader_values_for_material(assigned(obj)[1])["Roughness"] <= 0.9
    )
    with pytest.raises(ValueError, match="Unknown"):
        MaterialRandomizer.from_config(["Body"], context, {"typo": 5})


def shader_values_for_material(material):
    return {
        s.name: s.default_value
        for s in material.node_tree.nodes.get("Principled BSDF").inputs
        if hasattr(s, "default_value")
    }


def test_empty_mesh_material_and_linked_duplicate(context):
    obj = mesh(context, slots=())
    duplicate = obj.create_linked_duplicate()
    MaterialRandomizer(
        ["Body"], context, materials=[PrincipledMaterial()]
    ).randomize(context)
    assert len(assigned(obj)) == 1
    assert assigned(duplicate) == ()


def test_empty_mesh_does_not_pin_generated_material(context):
    obj = mesh(context, slots=())
    node = MaterialRandomizer(
        ["Body"], context, materials=[PrincipledMaterial()]
    )
    for _ in range(8):
        node.randomize(context)
    assert obj.get_object().blender_obj.data.materials[0] is None
    assert (
        len(
            [
                m
                for m in bpy.data.materials
                if m.get("illusion_generated_material")
            ]
        )
        == 1
    )


def test_legacy_duplicate_pbr_tags_and_empty_target_list(context):
    obj = mesh(context)
    before = assigned(obj)
    node = MaterialRandomizer([], context, types=["metal", "metal"])
    node.randomize(context)
    assert assigned(obj) == before


def test_missing_pbr_catalog_fails_at_construction(context):
    with pytest.raises(FileNotFoundError):
        MaterialRandomizer(["Body"], context, types=["does_not_exist"])


def test_default_preprocessing_policy_is_legacy():
    import inspect

    assert (
        inspect.signature(Object).parameters["material_preprocessing"].default
        == "replace"
    )
    assert inspect.signature(Object).parameters["uv_mapping"].default == "smart"


@pytest.mark.parametrize("projection", ["smart", "cube", "cylinder", "sphere"])
def test_uv_mapping_projection_is_configurable(
    context, tmp_path, monkeypatch, projection
):
    path = tmp_path / "triangle.obj"
    path.write_text("o test\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    calls = []
    monkeypatch.setattr(
        MeshObject,
        "add_uv_mapping",
        lambda self, method, overwrite=False: calls.append((method, overwrite)),
    )

    context.add_model(str(path), "part", uv_mapping=projection)

    assert calls == [(projection, True)]


def test_uv_mapping_projection_is_validated():
    with pytest.raises(ValueError, match="uv_mapping must be one of"):
        Object("part", "part", uv_mapping="planar")


def test_real_asset_modes(context, material_model_path):
    assert material_model_path.is_file(), (
        f"Unavailable model: {material_model_path}"
    )
    context.add_model(str(material_model_path), "model", preprocess_model=False)
    obj = context.get_object_group("model")[0]
    original = assigned(obj)
    face_indices = [
        p.material_index for p in obj.get_object().blender_obj.data.polygons
    ]
    target = MaterialTarget(objects=(obj.get_name(),))
    if original:
        indices = range(min(2, len(original)))
        node = MaterialRandomizer(
            target,
            context,
            material_slots={
                i: [PrincipledMaterial("plastic")] for i in indices
            },
            seed=42,
        )
        node.randomize(context)
        assert all(assigned(obj)[i] != original[i] for i in indices)
        assert assigned(obj)[len(indices) :] == original[len(indices) :]
        MaterialRandomizer(
            target,
            context,
            material_slots="all",
            materials=[PrincipledMaterial("rubber")],
            seed=42,
        ).randomize(context)
        assert len(assigned(obj)) == len(original)
        assert [
            p.material_index for p in obj.get_object().blender_obj.data.polygons
        ] == face_indices
    else:
        with pytest.raises(ValueError, match="No material slots"):
            MaterialRandomizer(
                target,
                context,
                material_slots="all",
                materials=[PrincipledMaterial()],
            ).randomize(context)
    MaterialRandomizer(
        target,
        context,
        mode="object",
        materials=[PrincipledMaterial("metal")],
        seed=42,
    ).randomize(context)
    assert len(set(assigned(obj))) == 1

    # The existing load/preprocess/PBR workflow still creates exactly one slot.
    context.add_model(str(material_model_path), "legacy", preprocess_model=True)
    obj = context.get_object_group("legacy")[0]
    assert len(assigned(obj)) == 1
    assert obj.get_material_slot_names()[0].startswith("DummyMaterial")
    MaterialRandomizer(["legacy"], context, types=["metal"]).randomize(context)
    assert assigned(obj)[0].get("is_cc_texture")


def test_all_principled_parameters_reach_shader(context):
    obj = mesh(context, slots=("surface",))
    values = {
        "base_color": (0.2, 0.3, 0.4, 0.9),
        "metallic": 0.3,
        "roughness": 0.4,
        "ior": 1.4,
        "specular_ior_level": 0.6,
        "transmission": 0.2,
        "alpha": 0.8,
        "coat_weight": 0.7,
        "coat_roughness": 0.3,
        "subsurface_weight": 0.1,
        "subsurface_radius": (0.1, 0.2, 0.3),
        "emission_color": (0.2, 0.4, 0.6, 1),
        "emission_strength": 2.0,
    }
    MaterialRandomizer(
        ["Body"], context, materials=[PrincipledMaterial(**values)]
    ).randomize(context)
    actual = shader_values(obj)[0]
    for name, expected in values.items():
        assert actual[name] == pytest.approx(expected)


def test_global_seed_and_parameter_order():
    a = PrincipledMaterial(roughness=(0.1, 0.5), metallic=(0.5, 1))
    b = PrincipledMaterial(metallic=(0.5, 1), roughness=(0.1, 0.5))
    random.seed(42)
    expected = a.sample()
    random.seed(42)
    assert b.sample() == expected


def test_original_presets_do_not_alias_user_configuration():
    bounds = [0.2, 0.5]
    recipe = PrincipledMaterial(roughness=bounds)
    bounds[0] = 1.0
    assert 0.2 <= recipe.sample()["roughness"] <= 0.5


def test_aliasing_selectors_fail_before_mutation(context):
    obj = mesh(context)
    original = assigned(obj)
    node = MaterialRandomizer(
        ["Body"],
        context,
        material_slots={
            0: [PrincipledMaterial()],
            "paint": [PrincipledMaterial()],
        },
    )
    with pytest.raises(ValueError, match="same slot"):
        node.randomize(context)
    assert assigned(obj) == original


def test_imported_mesh_and_linked_instances_survive_undo(context):
    from blenderproc.python.types.StructUtilityFunctions import get_instances
    from blenderproc.python.utility.Utility import UndoAfterExecution

    context.add_model(
        str(ROOT / "assets/models/mechanical_parts/gearwheel_1.glb"),
        "part",
        max_number_instances=3,
    )
    objects = context.get_object_group("part")
    for obj in objects:
        assert any(ref is obj.get_object() for _, ref in get_instances())

    for _ in range(2):
        with UndoAfterExecution():
            for obj in objects:
                obj.set_location(np.array([1.0, 2.0, 3.0]))
                obj.hide(False)
        for obj in objects:
            assert (
                obj.get_object().blender_obj == bpy.data.objects[obj.get_name()]
            )
            np.testing.assert_allclose(obj.get_location(), [0, 0, 0])
            obj.hide(True)
            assert obj.is_hidden()
            obj.hide(False)
            assert not obj.is_hidden()


def test_quickstart_models_survive_repeated_physics(context):
    # Exercise the same imports and the hide/simulate boundary as parts_in_bin.
    for path, name, count, active, shape in (
        ("mechanical_parts/gearwheel_1.glb", "part_1", 3, True, "CONVEX_HULL"),
        (
            "mechanical_parts/pipe_fixture_1.glb",
            "part_2",
            1,
            True,
            "CONVEX_HULL",
        ),
        ("bins/plastic_bin_2.glb", "crate_2", 1, False, "MESH"),
    ):
        context.add_model(
            str(ROOT / "assets/models" / path),
            name,
            max_number_instances=count,
            active_in_simulation=active,
            collision_shape=shape,
        )
    objects = list(context.get_objects().values())
    randomizer = MaterialRandomizer(["part_"], context, types=["metal"])
    for _ in range(2):
        for index, obj in enumerate(objects):
            obj.hide(True)
            obj.disable_rigid_body()
            obj.set_location(np.array([index * 0.1, 0.0, 0.15]))
            obj.hide(False)
            obj.enable_rigid_body()
        randomizer.randomize(context)
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=0.05,
            max_simulation_time=0.1,
            check_object_interval=0.05,
        )
        for obj in objects:
            assert (
                obj.get_object().blender_obj == bpy.data.objects[obj.get_name()]
            )
            assert np.isfinite(obj.get_location()).all()
            obj.hide(True)
            obj.disable_rigid_body()
            assert obj.is_hidden()
