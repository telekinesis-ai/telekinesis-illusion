"""Repeated shard setup must reuse segmentation writers and refresh mappings."""

import pytest

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import bpy
from blenderproc.python.renderer import RendererUtility as renderer
from blenderproc.python.utility.GlobalStorage import GlobalStorage
from blenderproc.python.utility.Utility import UndoAfterExecution, Utility

import blenderproc as bproc


@pytest.fixture(autouse=True)
def isolated_render_outputs(context, monkeypatch):
    # BlenderProc scene cleanup deliberately retains render settings and outputs.
    monkeypatch.setitem(GlobalStorage._storage_dict, "output", [])
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    yield
    # Undo may replace the node tree, so fetch it again at teardown.
    tree = bpy.context.scene.node_tree
    for node in list(tree.nodes):
        if node.type in {"OUTPUT_FILE", "COMBINE_COLOR"}:
            tree.nodes.remove(node)


def test_segmentation_setup_reuses_nodes_and_updates_objects(context):
    first = bproc.object.create_primitive("CUBE").blender_obj
    renderer.enable_segmentation_output(map_by="instance")
    tree = bpy.context.scene.node_tree
    nodes = {node.as_pointer() for node in tree.nodes}
    links = len(tree.links)
    second = bproc.object.create_primitive("CUBE").blender_obj

    for _ in range(50):
        renderer.enable_segmentation_output(
            map_by=["instance", "category_id"],
            default_values={"category_id": 0},
            pass_alpha_threshold=0.1,
        )

    assert {node.as_pointer() for node in tree.nodes} == nodes
    assert len(tree.links) == links
    assert sum(node.type == "OUTPUT_FILE" for node in tree.nodes) == 1
    assert first.pass_index > 0
    assert second.pass_index > 0
    assert first.pass_index != second.pass_index
    outputs = Utility.get_registered_outputs()
    assert len(outputs) == 1
    assert outputs[0]["semantic_segmentation_mapping"] == [
        "instance",
        "category_id",
    ]
    assert outputs[0]["semantic_segmentation_default_values"] == {
        "category_id": 0
    }
    assert bpy.context.view_layer.pass_alpha_threshold == pytest.approx(0.1)


def test_segmentation_reuse_survives_physics_undo(context):
    bproc.object.create_primitive("CUBE")
    renderer.enable_segmentation_output(map_by="instance")
    for _ in range(3):
        with UndoAfterExecution():
            bpy.context.scene.frame_set(1)
        renderer.enable_segmentation_output(map_by="instance")
        nodes = bpy.context.scene.node_tree.nodes
        assert sum(node.type == "OUTPUT_FILE" for node in nodes) == 1
        assert sum(node.type == "COMBINE_COLOR" for node in nodes) == 1


def test_segmentation_outputs_remain_independent(context):
    bproc.object.create_primitive("CUBE")
    renderer.enable_segmentation_output(map_by="instance")
    tree = bpy.context.scene.node_tree
    unrelated = tree.nodes.new("CompositorNodeOutputFile")
    unrelated.name = "User output"
    unrelated.file_slots[0].path = "user_"
    for _ in range(3):
        renderer.enable_segmentation_output(
            map_by="category_id",
            default_values={"category_id": 0},
            output_key="categories",
            file_prefix="categories_",
        )
        renderer.enable_segmentation_output(map_by="instance")
    assert sum(node.type == "OUTPUT_FILE" for node in tree.nodes) == 3
    assert unrelated.file_slots[0].path == "user_"
    assert {entry["key"] for entry in Utility.get_registered_outputs()} == {
        "segmap",
        "categories",
    }


def test_repeated_segmentation_setup_renders_valid_masks(context, tmp_path):
    cube = bproc.object.create_primitive("CUBE")
    cube.set_cp("category_id", 7)
    bproc.camera.set_resolution(32, 32)
    bproc.camera.add_camera_pose(
        bproc.math.build_transformation_mat([0, 0, 5], [0, 0, 0]), frame=0
    )
    renderer.set_max_amount_of_samples(1)
    renderer.disable_all_denoiser()
    for _ in range(3):
        renderer.enable_segmentation_output(
            map_by=["instance", "category_id"],
            default_values={"category_id": 0},
            output_dir=str(tmp_path),
        )
        data = renderer.render(output_dir=str(tmp_path))
        assert data["instance_segmaps"][0].shape == (32, 32)
        assert data["instance_segmaps"][0][16, 16] > 0
        assert data["category_id_segmaps"][0][16, 16] == 7
    assert (
        sum(
            node.type == "OUTPUT_FILE"
            for node in bpy.context.scene.node_tree.nodes
        )
        == 1
    )


@pytest.mark.parametrize(
    "conflict", [{"output_key": "other"}, {"file_prefix": "other_"}]
)
def test_conflicting_segmentation_output_does_not_add_nodes(context, conflict):
    renderer.enable_segmentation_output(map_by="instance")
    tree = bpy.context.scene.node_tree
    nodes = {node.as_pointer() for node in tree.nodes}
    with pytest.raises(RuntimeError, match="same key/path"):
        renderer.enable_segmentation_output(map_by="instance", **conflict)
    assert {node.as_pointer() for node in tree.nodes} == nodes
    assert len(Utility.get_registered_outputs()) == 1
