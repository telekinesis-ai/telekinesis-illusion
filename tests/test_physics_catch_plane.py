from types import SimpleNamespace

import numpy as np
import pytest

from telekinesis.illusion.core import (
    synthetic_data_generator as generator_module,
)
from telekinesis.illusion.core.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from telekinesis.illusion.workers.bin_picking_worker import (
    DEFAULT_PHYSICS_SIMULATOR_PARAMS,
)


def _box_bounds(lower, upper):
    return np.array(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], upper[2]],
            [lower[0], upper[1], upper[2]],
        ],
        dtype=float,
    )


class _FakeObject:
    def __init__(self, bounds):
        self._bounds = bounds
        self.hidden = False
        self.rigid_body_disabled = False

    def get_bound_box(self):
        return self._bounds

    def hide(self, hidden=True):
        self.hidden = hidden

    def disable_rigid_body(self):
        self.rigid_body_disabled = True


class _FakeContext:
    def __init__(self, objects, visible):
        self._objects = objects
        self._visible = list(visible)

    def get_objects(self):
        return self._objects

    def get_visible_object_names(self):
        return self._visible

    def set_visible_object_names(self, names):
        self._visible = list(names)


def test_bin_picking_catch_plane_is_enabled_by_default():
    assert DEFAULT_PHYSICS_SIMULATOR_PARAMS["catch_plane"]["active"] is True


def test_catch_plane_geometry_scales_with_container():
    location, scale = SyntheticDataGenerator._catch_plane_geometry(
        _box_bounds((-1.0, -2.0, 0.0), (1.0, 2.0, 4.0)),
        distance_fraction=0.25,
        minimum_distance=0.05,
        size_factor=6.0,
    )

    np.testing.assert_allclose(location, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(scale, [6.0, 12.0, 1.0])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"distance_fraction": -0.1}, "distance_fraction"),
        ({"minimum_distance": 0.0}, "minimum_distance"),
        ({"size_factor": 0.0}, "size_factor"),
    ],
)
def test_catch_plane_geometry_validates_config(kwargs, match):
    config = {
        "distance_fraction": 0.25,
        "minimum_distance": 0.05,
        "size_factor": 6.0,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        SyntheticDataGenerator._catch_plane_geometry(
            _box_bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            **config,
        )


def test_prepare_catch_plane_is_passive_and_render_invisible(monkeypatch):
    container = _FakeObject(_box_bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)))
    context = _FakeContext({"bin_INSTANCE_0": container}, ["bin_INSTANCE_0"])
    plane = SimpleNamespace(
        blender_obj=SimpleNamespace(hide_render=False),
        name=None,
        rigid_body=None,
    )
    plane.set_name = lambda name: setattr(plane, "name", name)
    plane.enable_rigidbody = lambda **kwargs: setattr(
        plane, "rigid_body", kwargs
    )
    plane.set_location = lambda value: setattr(plane, "location", value)
    plane.set_scale = lambda value: setattr(plane, "scale", value)
    plane.hide = lambda value: setattr(plane, "hidden", value)
    captured = {}

    def create_primitive(shape, **kwargs):
        captured.update(shape=shape, **kwargs)
        return plane

    monkeypatch.setattr(
        generator_module.bproc.object, "create_primitive", create_primitive
    )
    generator = SyntheticDataGenerator(context, None, None)

    created, plane_z = generator._prepare_physics_catch_plane(
        {"container_names": ["bin"]}
    )

    assert created is plane
    assert plane_z == pytest.approx(-0.5)
    assert captured["shape"] == "PLANE"
    np.testing.assert_allclose(plane.location, [0.0, 0.0, -0.5])
    np.testing.assert_allclose(plane.scale, [6.0, 6.0, 1.0])
    assert plane.rigid_body == {"active": False, "collision_shape": "BOX"}
    assert plane.hidden is False
    assert plane.blender_obj.hide_render is True


def test_objects_resting_on_catch_plane_are_hidden_and_removed():
    escaped = _FakeObject(_box_bounds((0.0, 0.0, -1.02), (0.1, 0.1, -0.8)))
    retained = _FakeObject(_box_bounds((0.0, 0.0, 0.0), (0.1, 0.1, 0.2)))
    container = _FakeObject(_box_bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)))
    context = _FakeContext(
        {
            "part_INSTANCE_0": escaped,
            "part_INSTANCE_1": retained,
            "bin_INSTANCE_0": container,
        },
        ["part_INSTANCE_0", "part_INSTANCE_1", "bin_INSTANCE_0"],
    )
    generator = SyntheticDataGenerator.__new__(SyntheticDataGenerator)
    generator._context = context

    culled = generator._cull_objects_on_catch_plane(
        -1.0,
        {"tracked_object_names": ["part"], "contact_tolerance": 0.05},
    )

    assert culled == ["part_INSTANCE_0"]
    assert escaped.hidden is True
    assert escaped.rigid_body_disabled is True
    assert retained.hidden is False
    assert context.get_visible_object_names() == [
        "part_INSTANCE_1",
        "bin_INSTANCE_0",
    ]


def test_real_catch_plane_stops_a_falling_body(context):
    bproc = generator_module.bproc
    plane_z = -0.5
    plane = bproc.object.create_primitive(
        "PLANE", location=[0.0, 0.0, plane_z], scale=[5.0, 5.0, 1.0]
    )
    plane.enable_rigidbody(active=False, collision_shape="BOX")
    plane.blender_obj.hide_render = True
    body = bproc.object.create_primitive(
        "CUBE", location=[0.0, 0.0, 0.5], scale=[0.1, 0.1, 0.1]
    )
    body.enable_rigidbody(active=True, collision_shape="BOX")

    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=1.0,
        max_simulation_time=2.0,
        check_object_interval=0.5,
    )

    body_bottom = float(np.min(body.get_bound_box()[:, 2]))
    assert plane_z - 1e-5 <= body_bottom <= plane_z + 0.05


@pytest.fixture
def catch_plane_generator(context, monkeypatch) -> SyntheticDataGenerator:
    """Build a real physics scene with an escaped part and stubbed rendering."""
    bproc = generator_module.bproc
    container = context.add_mesh_object(
        bproc.object.create_primitive(
            "CUBE", location=[0.0, 0.0, 0.5], scale=[1.0, 1.0, 0.5]
        ),
        object_name="bin",
    )
    part = context.add_mesh_object(
        bproc.object.create_primitive("CUBE", scale=[0.1, 0.1, 0.1]),
        object_name="part",
        active_in_simulation=True,
        collision_shape="BOX",
    )

    def randomize(scene) -> None:
        container.hide(False)
        part.hide(False)
        part.set_location(container.get_location() + [2.5, 0.0, 2.0])
        part.enable_rigid_body()
        scene.set_visible_object_names(["bin", "part"])

    monkeypatch.setattr(bproc.renderer, "render", lambda **_: {})
    monkeypatch.setattr(bproc.renderer, "set_render_devices", lambda **_: None)
    monkeypatch.setattr(
        bproc.renderer, "enable_segmentation_output", lambda **_: None
    )
    randomizer = SimpleNamespace(
        randomize=randomize, randomize_light_poses=lambda _: None
    )
    writer = SimpleNamespace(
        get_segmentation_output_config=dict, write=lambda **_: 1
    )
    return SyntheticDataGenerator(context, randomizer, writer)


def test_catch_plane_is_reused_across_scenes_and_shards(
    catch_plane_generator, context, monkeypatch
) -> None:
    from bpy import data

    generator = catch_plane_generator
    bproc = generator_module.bproc
    baseline_meshes = len(data.meshes)
    baseline_objects = len(data.objects)
    mesh_names = []

    def render(**kwargs) -> dict:
        plane = generator._physics_catch_plane
        assert plane.blender_obj.hide_render
        assert plane.blender_obj.hide_get()
        assert not plane.has_rigidbody_enabled()
        assert len(data.meshes) == baseline_meshes + 1
        assert len(data.objects) == baseline_objects + 1
        assert plane.blender_obj.data.users == 1
        mesh_names.append(plane.blender_obj.data.name)
        part = context.get_objects()["part"]
        body_bottom = float(np.min(part.get_bound_box()[:, 2]))
        plane_z = plane.get_location()[2]
        assert plane_z - 1e-5 <= body_bottom <= plane_z + 0.05
        assert part.is_hidden()
        assert context.get_visible_object_names() == ["bin"]
        return {}

    monkeypatch.setattr(bproc.renderer, "render", render)
    options = {
        "simulate_physics": True,
        "min_simulation_time_range": (1.0, 1.0),
        "max_simulation_time_range": (2.0, 2.0),
        "check_object_interval": 0.5,
        "clean_up_scene": False,
        "physics_catch_plane": {
            "active": True,
            "container_names": ["bin"],
            "tracked_object_names": ["part"],
        },
    }
    generator.generate(num_images=10, **options)
    plane = generator._physics_catch_plane
    np.testing.assert_allclose(plane.get_location(), [0.0, 0.0, -0.25])
    np.testing.assert_allclose(plane.get_scale(), [6.0, 6.0, 1.0])

    container = context.get_objects()["bin"].get_object()
    container.set_location([3.0, 2.0, 1.5])
    container.set_scale([2.0, 3.0, 2.0])
    generator.generate(num_images=10, **options)
    assert generator._physics_catch_plane is plane
    np.testing.assert_allclose(plane.get_location(), [3.0, 2.0, 0.0])
    np.testing.assert_allclose(plane.get_scale(), [12.0, 18.0, 1.0])
    assert len(mesh_names) == 20
    assert len(set(mesh_names)) == 1

    # A later shard can disable the catch plane without leaving a collider.
    monkeypatch.setattr(bproc.renderer, "render", lambda **_: {})
    generator.generate(
        num_images=1, **{**options, "physics_catch_plane": {"active": False}}
    )
    assert context.get_objects()["part"].get_location()[2] < -1.0
    assert not plane.has_rigidbody_enabled()
    assert len(data.meshes) == baseline_meshes + 1

    monkeypatch.setattr(bproc.renderer, "render", render)
    generator.generate(num_images=1, **options)
    assert generator._physics_catch_plane is plane

    plane_name = plane.get_name()
    generator.clean_up()
    assert generator._physics_catch_plane is None
    assert plane_name not in data.objects
    assert mesh_names[0] not in data.meshes


def test_catch_plane_is_deactivated_after_simulation_failure(
    catch_plane_generator, monkeypatch
) -> None:
    generator = catch_plane_generator

    def simulate(*args) -> None:
        plane = generator._physics_catch_plane
        assert plane.get_rigidbody().type == "PASSIVE"
        assert not plane.blender_obj.hide_get()
        raise RuntimeError("Simulation failed")

    monkeypatch.setattr(
        generator_module.bproc.object,
        "simulate_physics_and_fix_final_poses",
        simulate,
    )
    with pytest.raises(RuntimeError, match="Simulation failed"):
        generator.generate(
            num_images=1,
            simulate_physics=True,
            clean_up_scene=False,
            physics_catch_plane={"active": True, "container_names": ["bin"]},
        )

    plane = generator._physics_catch_plane
    assert not plane.has_rigidbody_enabled()
    assert plane.blender_obj.hide_get()
    assert plane.blender_obj.hide_render
