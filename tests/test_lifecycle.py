"""Registration, capacity, activity, undo, capture and shard boundaries."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
import json
import random

import numpy as np
import pytest

from telekinesis.illusion.core.engine import (
    BlenderEngine,
    PhysicsConfig,
    RenderConfig,
)
from telekinesis.illusion.core.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from telekinesis.illusion.randomizer.instance_counts import (
    sample_counts,
    validate_counts,
)
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.randomizer.randomizer_node import (
    CameraPoseRandomizer,
    NodeConfig,
    ObjectInstanceRandomizer,
    ObjectPoseRandomizer,
    RandomizerNode,
)
from telekinesis.illusion.blender.types.mesh_object import create_primitive
from telekinesis.illusion.blender.types.material import Material
from telekinesis.illusion.blender.undo import UndoAfterExecution
from telekinesis.illusion.blender import initialization, physics
from telekinesis.illusion.blender.camera import settings
from telekinesis.illusion.blender.math import build_transformation_mat
from telekinesis.illusion.blender.rendering import renderer
from telekinesis.illusion.types.render_result import RenderResult
from telekinesis.illusion.writer.writer import CocoWriter
import bpy

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/models/mechanical_parts/gearwheel_1.glb"


def test_camera_samples_reuse_action_and_curves_without_accumulating_keys(
    context,
):
    camera = bpy.context.scene.camera
    camera_action = camera.animation_data.action
    curve_pointers = {
        (curve.data_path, curve.array_index): curve.as_pointer()
        for curve in camera_action.fcurves
    }
    other = create_primitive("CUBE").blender_obj
    other.keyframe_insert(data_path="location", frame=10)
    other_action = other.animation_data.action
    action_names = set(bpy.data.actions.keys())

    views = 2
    for sample in range(100):
        poses = iter(
            (
                np.array([sample, frame, 5.0]),
                np.array([0.01 * sample, 0.1 * frame, 0.0]),
            )
            for frame in range(views)
        )
        randomizer = CameraPoseRandomizer(
            lambda context: next(poses), number_of_views=views
        )
        randomizer.randomize(context)

        assert camera.animation_data.action == camera_action
        assert set(bpy.data.actions.keys()) == action_names
        assert all(action.users > 0 for action in bpy.data.actions)
        assert len(camera_action.fcurves) == 6
        assert {
            (curve.data_path, curve.array_index): curve.as_pointer()
            for curve in camera_action.fcurves
        } == curve_pointers
        for curve in camera_action.fcurves:
            assert [point.co.x for point in curve.keyframe_points] == list(
                range(views)
            )
        assert (bpy.context.scene.frame_start, bpy.context.scene.frame_end) == (
            0,
            views,
        )
        for frame in range(views):
            bpy.context.scene.frame_set(frame)
            np.testing.assert_allclose(camera.location, [sample, frame, 5.0])
            np.testing.assert_allclose(
                camera.rotation_euler,
                [0.01 * sample, 0.1 * frame, 0.0],
                atol=1e-6,
            )
        assert other.animation_data.action == other_action
        assert all(
            [point.co.x for point in curve.keyframe_points] == [10]
            for curve in other_action.fcurves
        )


@pytest.mark.integration
def test_catch_plane_is_reused_across_samples_and_shards(context, monkeypatch):
    container = context.add_mesh_object(
        create_primitive("CUBE", location=[0, 0, 0.5], scale=[1, 1, 0.5]), "bin"
    )
    part = context.add_mesh_object(
        create_primitive("CUBE", scale=[0.1, 0.1, 0.1]),
        "part",
        active_in_simulation=True,
        collision_shape="BOX",
    )
    engine = BlenderEngine(
        physics_config=PhysicsConfig(
            active=True,
            min_simulation_time_range=(1.0, 1.0),
            max_simulation_time_range=(2.0, 2.0),
            check_object_interval=0.5,
            catch_plane={
                "active": True,
                "container_names": ["bin"],
                "tracked_object_names": ["part"],
            },
        )
    )
    baseline_objects = len(bpy.data.objects)
    baseline_meshes = len(bpy.data.meshes)
    planes = []
    mesh_names = []

    def randomize(scene):
        scene.activate(container)
        scene.activate(part)
        part.set_location(container.get_location() + [2.5, 0, 2])

    def render(scene, **kwargs):
        plane = engine._physics_catch_plane
        assert plane.blender_obj.hide_render
        assert plane.blender_obj.hide_get()
        assert not plane.has_rigidbody_enabled()
        assert len(bpy.data.objects) == baseline_objects + 1
        assert len(bpy.data.meshes) == baseline_meshes + 1
        assert plane.blender_obj.data.users == 1
        planes.append(plane)
        mesh_names.append(plane.blender_obj.data.name)
        bottom = float(np.min(part.get_bound_box()[:, 2]))
        plane_z = plane.get_location()[2]
        assert plane_z - 1e-5 <= bottom <= plane_z + 0.05
        assert part.is_hidden()
        assert scene.get_visible_object_names() == ["bin"]
        return RenderResult({}, {}, ({},))

    randomizer = Randomizer()
    monkeypatch.setattr(randomizer, "randomize", randomize)
    monkeypatch.setattr(engine, "prepare", lambda _: None)
    monkeypatch.setattr(engine, "render", render)
    writer = SimpleNamespace(
        get_segmentation_output_config=dict, write=lambda result: len(result)
    )
    generator = SyntheticDataGenerator(context, randomizer, writer, engine)
    generator.generate(num_images=10, clean_up_scene=False)
    plane = engine._physics_catch_plane
    np.testing.assert_allclose(plane.get_location(), [0, 0, -0.25])
    np.testing.assert_allclose(plane.get_scale(), [6, 6, 1])

    container.get_object().set_location([3, 2, 1.5])
    container.get_object().set_scale([2, 3, 2])
    generator.generate(num_images=10, clean_up_scene=False)
    assert all(item is plane for item in planes)
    assert len(planes) == 20
    assert len(set(mesh_names)) == 1
    np.testing.assert_allclose(plane.get_location(), [3, 2, 0])
    np.testing.assert_allclose(plane.get_scale(), [12, 18, 1])

    def render_without_plane(scene, **kwargs):
        assert part.get_location()[2] < -1.0
        assert not plane.has_rigidbody_enabled()
        assert plane.blender_obj.hide_get()
        return RenderResult({}, {}, ({},))

    monkeypatch.setattr(engine, "render", render_without_plane)
    generator.generate(
        num_images=1,
        clean_up_scene=False,
        physics_catch_plane={"active": False},
    )
    monkeypatch.setattr(engine, "render", render)
    generator.generate(num_images=1, clean_up_scene=False)
    assert engine._physics_catch_plane is plane

    plane_name = plane.get_name()
    generator.clean_up()
    assert engine._physics_catch_plane is None
    assert plane_name not in bpy.data.objects
    assert mesh_names[0] not in bpy.data.meshes


@pytest.mark.integration
def test_catch_plane_is_inactive_after_physics_failure(context, monkeypatch):
    container = context.add_mesh_object(create_primitive("CUBE"), "bin")
    context.activate(container)
    engine = BlenderEngine(
        physics_config=PhysicsConfig(
            active=True,
            catch_plane={"active": True, "container_names": ["bin"]},
        )
    )

    def fail(*args):
        plane = engine._physics_catch_plane
        assert plane.get_rigidbody().type == "PASSIVE"
        assert not plane.blender_obj.hide_get()
        assert plane.blender_obj.hide_render
        with UndoAfterExecution():
            raise RuntimeError("physics failed")

    monkeypatch.setattr(physics, "simulate_physics_and_fix_final_poses", fail)

    object_names = set(bpy.data.objects.keys())
    mesh_names = set(bpy.data.meshes.keys())
    for _ in range(6):
        with pytest.raises(RuntimeError, match="physics failed"):
            engine.simulate(context)
        plane = engine._physics_catch_plane
        assert plane.blender_obj.hide_get()
        assert plane.blender_obj.hide_render
        assert not plane.has_rigidbody_enabled()
        assert len(bpy.data.objects) == len(object_names) + 1
        assert len(bpy.data.meshes) == len(mesh_names) + 1

    engine.clean_up()
    assert engine._physics_catch_plane is None
    assert set(bpy.data.objects.keys()) == object_names
    assert set(bpy.data.meshes.keys()) == mesh_names


def test_registration_is_metadata_and_preparation_reuses_linked_pool(context):
    before = set(bpy.data.objects.keys())
    asset = context.add_model(
        MODEL, "gear", category_id=7, scale=np.array([1, 2, 1])
    )
    assert set(bpy.data.objects.keys()) == before
    assert not context.get_objects()
    assert context.get_assets()["gear"] is asset
    with pytest.raises(FrozenInstanceError):
        asset.scale = 5
    context.prepare({"gear": 3})
    objects = context.get_object_group("gear")
    mesh = objects[0].get_object().blender_obj.data
    pointers = [obj.get_object().blender_obj.as_pointer() for obj in objects]
    assert all(obj.get_object().blender_obj.data == mesh for obj in objects)
    assert all(
        obj.get_category_id() == 7 and obj.get_category_name() == "gear"
        for obj in objects
    )
    context.prepare({"gear": 2})
    assert pointers == [
        obj.get_object().blender_obj.as_pointer() for obj in objects
    ]
    with pytest.raises(ValueError, match="new Scene"):
        context.prepare({"gear": 4})
    with pytest.raises(RuntimeError, match="before preparing"):
        context.add_model(MODEL, "extra")


def test_diamond_dag_counts_shared_node_once_and_multiple_claims_add(context):
    context.add_model(MODEL, "gear", active_in_simulation=True)
    graph = Randomizer()
    graph.add_node(ObjectInstanceRandomizer({"gear": (1, 2)}), "root")
    graph.add_node(ObjectInstanceRandomizer({"gear": (0, 1)}), "left")
    graph.add_node(
        ObjectInstanceRandomizer({"gear": (0, 9)}),
        "disabled",
        NodeConfig(enabled=False),
    )
    graph.add_node(ObjectPoseRandomizer(lambda obj: None), "join")
    for source, target in [
        ("root", "left"),
        ("root", "disabled"),
        ("left", "join"),
        ("disabled", "join"),
    ]:
        graph.add_edge(source, target)
    assert graph.required_instances() == {"gear": 3}
    context.prepare(graph.required_instances())
    for _ in range(12):
        context.begin_sample()
        # Test only composition; pose collision testing is covered elsewhere.
        graph.get_randomizer_node("root").randomize(context)
        graph.get_randomizer_node("left").randomize(context)
        names = context.get_visible_object_names()
        assert 1 <= len(names) <= 3
        assert len(names) == len(set(names))
        for obj in context.get_object_group("gear"):
            active = obj.get_name() in names
            assert obj.is_hidden() != active
            assert (
                obj.get_object().blender_obj.rigid_body is not None
            ) == active
    context.begin_sample()
    assert not context.get_visible_object_names()


def test_bad_capacity_and_cycles_fail_before_import(context):
    context.add_model(MODEL, "gear")
    with pytest.raises(ValueError, match="Unknown"):
        context.prepare({"absent": 1})
    assert not context.get_objects()
    graph = Randomizer()
    graph.add_node(ObjectInstanceRandomizer({"gear": (1, 2)}), "a")
    graph.add_node(ObjectInstanceRandomizer({"gear": (0, 1)}), "b")
    graph.add_edge("a", "b")
    graph.add_edge("b", "a")
    with pytest.raises(ValueError, match="Cycle"):
        graph.required_instances()


@pytest.mark.parametrize(
    "bounds,low,high",
    [
        ({"a": (2, 1)}, None, None),
        ({"a": (0, 2)}, 3, 4),
        ({"a": (1, 2)}, 0, 0),
        ({"a": (0, float("inf"))}, None, None),
    ],
)
def test_infeasible_bounds_are_rejected(bounds, low, high):
    with pytest.raises(ValueError):
        validate_counts(bounds, low, high)


def test_balanced_count_sampling_respects_global_and_per_model_bounds():
    rng = random.Random(23)
    for _ in range(100):
        counts = sample_counts({"a": (1, 3), "b": (0, 4)}, 2, 5, rng)
        assert 1 <= counts["a"] <= 3
        assert 0 <= counts["b"] <= 4
        assert 2 <= sum(counts.values()) <= 5


def test_undo_restores_mesh_material_light_and_background_on_exception(context):
    mesh = create_primitive("CUBE")
    material_data = bpy.data.materials.new("ReferenceMaterial")
    material_data.use_nodes = True
    material = Material(material_data)
    mesh.add_material(material)
    light = context.add_light("key", power=40)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="abort"):
            with UndoAfterExecution():
                mesh.set_location([2, 3, 4])
                light.set_location([4, 5, 6])
                material.blender_obj.diffuse_color = [1, 0, 0, 1]
                raise RuntimeError("abort")
        assert mesh.blender_obj == bpy.data.objects[mesh.get_name()]
        assert material.blender_obj == bpy.data.materials[material.get_name()]
        assert (
            material.nodes is material.blender_obj.node_tree.nodes
            or len(material.nodes) > 0
        )
        np.testing.assert_allclose(mesh.get_location(), [0, 0, 0])
        light.set_location([0, 0, 1])
        context.get_background().update_blender_ref()


def test_undo_does_not_require_an_illusion_background(monkeypatch, context):
    from telekinesis.illusion.types.background import Background

    monkeypatch.setattr(Background, "__refs__", ())
    mesh = create_primitive("CUBE")
    with UndoAfterExecution():
        mesh.set_location([1, 2, 3])
    np.testing.assert_allclose(mesh.get_location(), [0, 0, 0])


@pytest.mark.integration
def test_render_result_can_be_written_after_cleanup_with_aligned_views(
    context, tmp_path
):
    cube = context.add_mesh_object(
        create_primitive("CUBE"), "cube", category_id=9
    )
    settings.set_resolution(32, 32)
    for frame in range(2):
        settings.add_camera_pose(
            build_transformation_mat([frame * 0.2, 0, 5], [0, 0, 0]),
            frame=frame,
        )
    writer = CocoWriter(output_dir=str(tmp_path), include_camera_metadata=True)
    engine = BlenderEngine(RenderConfig(samples=1, depth=True, normals=True))
    engine.prepare(writer.get_segmentation_output_config())
    result = engine.render(context)
    assert len(result) == 2
    assert len(result.data["depth"]) == 2
    assert len(result.data["normals"]) == 2
    assert result.data["instance_segmaps"][0][16, 16] > 0
    context.clean_up()
    assert writer.write(result) == 2
    saved = json.loads((tmp_path / "coco_annotations.json").read_text())
    assert saved["images"][0]["camera_matrix_world"][0][3] == pytest.approx(0)
    assert saved["images"][1]["camera_matrix_world"][0][3] == pytest.approx(0.2)
    assert {c["id"]: c["name"] for c in saved["categories"]} == {9: "cube"}


def test_writer_rejects_conflicting_category_ids_before_writing(tmp_path):
    data = {
        "colors": [np.zeros((2, 2, 3), np.uint8)],
        "instance_segmaps": [np.ones((2, 2), np.uint8)],
        "instance_attribute_maps": [
            [{"idx": 1, "category_id": 1, "name": "gear"}]
        ],
    }
    writer = CocoWriter(output_dir=str(tmp_path))
    writer.write(RenderResult(data, {1: "gear", 2: "absent"}, ({},)))
    before = (tmp_path / "coco_annotations.json").read_bytes()
    with pytest.raises(ValueError, match="Incompatible category"):
        writer.write(RenderResult(data, {1: "different"}, ({},)))
    assert (tmp_path / "coco_annotations.json").read_bytes() == before
    assert len(list((tmp_path / "images").iterdir())) == 1


def test_undo_skips_background_wrappers_from_cleaned_scene(context):
    from telekinesis.illusion.core.scene import Scene

    old_background = context.get_background()
    context.clean_up()
    fresh = Scene()
    try:
        cube = create_primitive("CUBE")
        with UndoAfterExecution():
            cube.set_location([1, 2, 3])
        np.testing.assert_allclose(cube.get_location(), [0, 0, 0])
        with pytest.raises(ReferenceError):
            old_background.get_name()
    finally:
        fresh.clean_up()


@pytest.mark.integration
def test_procedural_worker_preserves_all_views_per_arrangement(tmp_path):
    import yaml
    from telekinesis.illusion.workers.procedural_scene_worker import (
        ProceduralSceneWorker,
    )

    spec = {
        "metadata": {
            "asset_directory": str(ROOT / "assets"),
            "dataset_name": "procedural",
            "base_output_directory": str(tmp_path),
            "num_arrangements": 2,
            "seed_start": 10,
        },
        "camera": {"image_width": 32, "image_height": 32},
        "renderer": {"samples": 1, "denoiser": None, "verbose": False},
        "camera_pose_randomizer": {
            "sampler": "shell_sampler",
            "number_of_views": 2,
            "params": {"radius_min": 5, "radius_max": 6},
        },
        "writer": {"include_camera_metadata": True},
    }
    path = tmp_path / "procedural.yaml"
    path.write_text(yaml.safe_dump(spec))

    def build(scene, seed, layout):
        scene.add_mesh_object(create_primitive("CUBE"), "cube", category_id=3)

    worker = ProceduralSceneWorker(path, build)
    output = worker.generate()
    data = json.loads((output / "coco_annotations.json").read_text())
    assert len(data["images"]) == 4
    assert [image["arrangement_seed"] for image in data["images"]] == [
        10,
        10,
        11,
        11,
    ]
    assert all("camera_matrix_world" in image for image in data["images"])


def test_worker_cleans_up_if_shard_generation_fails(tmp_path, monkeypatch):
    import yaml
    from telekinesis.illusion.workers.bin_picking_worker import BinPickingWorker

    spec = yaml.safe_load(
        (ROOT / "configs/example_bin_picking_gearwheel_2.yaml").read_text()
    )
    spec["metadata"]["asset_directory"] = str(ROOT / "assets")
    spec["metadata"]["base_output_directory"] = str(tmp_path)
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump(spec))
    worker = BinPickingWorker(path)

    def fail(**kwargs):
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(worker._data_generator, "generate", fail)
    with pytest.raises(RuntimeError, match="simulated render failure"):
        worker.generate()
    assert worker.get_context()._closed
