"""Exercise retained camera math and secondary render paths with bpy."""

import numpy as np
import pytest

from telekinesis.illusion.blender.camera import (
    settings,
    projection,
    validation,
    lens_distortion,
)
from telekinesis.illusion.blender import filters, postprocessing
from telekinesis.illusion.blender.math import build_transformation_mat
from telekinesis.illusion.blender.types.mesh_object import (
    create_primitive,
    create_bvh_tree_multi_objects,
)
from telekinesis.illusion.blender.rendering import (
    renderer,
    optical_flow,
    segmentation,
)
import bpy


def camera_and_cube():
    cube = create_primitive("CUBE")
    cube.set_cp("category_id", 4)
    settings.set_resolution(32, 32)
    for frame in range(2):
        settings.add_camera_pose(
            build_transformation_mat([frame * 0.1, 0, 5], [0, 0, 0]),
            frame=frame,
        )
    renderer.set_max_amount_of_samples(1)
    renderer.disable_all_denoiser()
    return cube


def test_projection_depth_conversion_and_filters(context):
    cube = camera_and_cube()
    points = np.array([[0, 0, 0], [0.2, 0.3, 0]])
    pixels = projection.project_points(points, frame=0)
    recovered = projection.unproject_points(
        pixels, np.array([5.0, 5.0]), frame=0
    )
    np.testing.assert_allclose(recovered, points, atol=1e-5)
    depths = np.full((32, 32), 4.0)
    np.testing.assert_allclose(
        postprocessing.dist2depth(postprocessing.depth2dist(depths)),
        depths,
        atol=1e-5,
    )
    assert filters.one_by_cp([cube], "category_id", 4) is cube
    tree = create_bvh_tree_multi_objects([cube])
    ray_depth = projection.depth_via_raytracing(tree, frame=0)
    assert ray_depth[16, 16] == pytest.approx(4, abs=0.02)
    pose = build_transformation_mat([0, 0, 5], [0, 0, 0])
    assert cube in validation.visible_objects(pose)


def test_lens_distortion_identity(context):
    camera_and_cube()
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    rows, columns = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    mapping = np.array([rows.ravel(), columns.ravel()])
    distorted = lens_distortion.apply_lens_distortion(
        image, mapping_coords=mapping, orig_res_x=32, orig_res_y=32
    )
    np.testing.assert_array_equal(distorted, image)


@pytest.mark.integration
def test_flow_render_restores_objects_and_frame_interval(context, tmp_path):
    cube = camera_and_cube()
    name = cube.get_name()
    data = optical_flow.render_optical_flow(
        output_dir=str(tmp_path), temp_dir=str(tmp_path)
    )
    for key in ("forward_flow", "backward_flow"):
        assert len(data[key]) == 2
        assert data[key][0].shape == (32, 32, 2)
        assert np.isfinite(data[key][0]).all()
    assert bpy.context.scene.frame_end == 2
    assert cube.blender_obj == bpy.data.objects[name]


@pytest.mark.integration
def test_recolor_segmentation_restores_mesh_and_materials(context, tmp_path):
    cube = camera_and_cube()
    name = cube.get_name()
    data = segmentation.render_segmap(
        output_dir=str(tmp_path),
        temp_dir=str(tmp_path),
        map_by=["instance", "class"],
        default_values={"class": 0},
    )
    assert len(data["instance_segmaps"]) == 2
    assert data["instance_segmaps"][0][16, 16] > 0
    assert data["class_segmaps"][0][16, 16] == 4
    assert cube.blender_obj == bpy.data.objects[name]
    assert len(cube.blender_obj.material_slots) == 0
    assert bpy.context.scene.frame_end == 2


def test_recoloring_linked_instances_uses_object_material_slots(context):
    from telekinesis.illusion.blender.undo import UndoAfterExecution
    from telekinesis.illusion.types.object import Object

    cube = camera_and_cube()
    material = bpy.data.materials.new("original")
    material.use_nodes = True
    cube.blender_obj.data.materials.append(material)
    original = Object(
        "part_INSTANCE_0", "part", object=cube, preprocess_model=False
    )
    duplicate = original.create_linked_duplicate()
    first = original.get_object()
    second = duplicate.get_object()
    assert first.blender_obj.data == second.blender_obj.data
    with UndoAfterExecution():
        segmentation._colorize_object(first.blender_obj, [1, 0, 0], False)
        segmentation._colorize_object(second.blender_obj, [0, 1, 0], False)
        a = first.blender_obj.material_slots[0].material
        b = second.blender_obj.material_slots[0].material
        assert a != b
        assert list(
            a.node_tree.nodes.get("Emission").inputs["Color"].default_value[:3]
        ) == [1, 0, 0]
        assert list(
            b.node_tree.nodes.get("Emission").inputs["Color"].default_value[:3]
        ) == [0, 1, 0]
    assert first.blender_obj.data == second.blender_obj.data
    assert first.blender_obj.material_slots[0].material.name == "original"
    assert second.blender_obj.material_slots[0].material.name == "original"
