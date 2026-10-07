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


def test_create_catch_plane_is_passive_and_render_invisible(monkeypatch):
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
    captured = {}

    def create_primitive(shape, **kwargs):
        captured.update(shape=shape, **kwargs)
        return plane

    monkeypatch.setattr(
        generator_module.bproc.object, "create_primitive", create_primitive
    )
    generator = SyntheticDataGenerator.__new__(SyntheticDataGenerator)
    generator._context = context

    created, plane_z = generator._create_physics_catch_plane(
        {"container_names": ["bin"]}
    )

    assert created is plane
    assert plane_z == pytest.approx(-0.5)
    assert captured["shape"] == "PLANE"
    assert plane.rigid_body == {"active": False, "collision_shape": "BOX"}
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
