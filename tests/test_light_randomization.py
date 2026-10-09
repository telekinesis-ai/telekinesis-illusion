"""Real Blender light properties, sampling and conservative pose clearance."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

import bpy

import blenderproc as bproc
from telekinesis.illusion.core.synthetic_data_generator import (
    SyntheticDataGenerator,
)
from telekinesis.illusion.randomizer.randomizer import Randomizer
from telekinesis.illusion.randomizer.randomizer_node import (
    LightPoseRandomizer,
    LightRandomizer,
    NodeConfig,
)
from telekinesis.illusion.sampler.camera_pose_sampler import shell_sampler
from telekinesis.illusion.types.distribution import uniform
from telekinesis.illusion.types.light import Light


@pytest.mark.parametrize(
    "light_type, properties",
    [
        ("POINT", {"radius": 0.1}),
        ("SUN", {"angle": 0.2}),
        ("SPOT", {"radius": 0.1, "spot_size": 0.8, "spot_blend": 0.4}),
        ("AREA", {"shape": "RECTANGLE", "size": 0.8, "size_y": 0.4}),
    ],
)
def test_light_creation_and_registry(context, light_type, properties):
    light = context.add_light(
        "key", light_type, color=(0.4, 0.6, 0.8), power=12, **properties
    )
    assert isinstance(light, Light)
    assert context.get_light("key") is light
    assert context.get_lights() == {"key": light}
    assert context.get_objects() == {}
    assert context.get_visible_object_names() == []
    data = light.get_light().blender_obj.data
    assert data.type == light_type
    assert data.energy == 12
    assert tuple(data.color) == pytest.approx((0.4, 0.6, 0.8))
    for name, value in properties.items():
        actual = getattr(data, "shadow_soft_size" if name == "radius" else name)
        expected = value if isinstance(value, str) else pytest.approx(value)
        assert actual == expected
    snapshot = light.get_properties()
    assert snapshot["color"] == pytest.approx((0.4, 0.6, 0.8))
    assert snapshot["power"] == 12
    assert ("radius" in snapshot) == (light_type in ("POINT", "SPOT"))
    light.set_properties(**snapshot)
    snapshot["power"] = 0
    assert light.get_properties()["power"] == 12
    light.set_location((1, 2, 3))
    light.set_rotation((0.1, 0.2, 0.3))
    np.testing.assert_allclose(light.get_location(), (1, 2, 3))
    np.testing.assert_allclose(light.get_rotation(), (0.1, 0.2, 0.3))
    assert light.get_light().blender_obj.rigid_body is None
    with pytest.raises(ValueError, match="already exists"):
        context.add_light("key", light_type)
    with pytest.raises(ValueError, match="No light"):
        context.get_light("missing")


@pytest.mark.parametrize(
    "light_type, properties",
    [
        ("INVALID", {}),
        ("POINT", {"power": -1}),
        ("POINT", {"power": float("nan")}),
        ("POINT", {"power": 1e40}),
        ("POINT", {"color": (1, 2, 3)}),
        ("POINT", {"color": (1, 1)}),
        ("POINT", {"radius": -1}),
        ("POINT", {"shape": "SQUARE"}),
        ("SUN", {"radius": 1}),
        ("SUN", {"angle": 4}),
        ("SPOT", {"spot_size": 0}),
        ("SPOT", {"spot_blend": 2}),
        ("AREA", {"size": -1}),
        ("AREA", {"shape": "TRIANGLE"}),
        ("AREA", {"typo": 1}),
        ("AREA", {"location": (float("inf"), 0, 0)}),
    ],
)
def test_invalid_creation_leaves_no_orphan(context, light_type, properties):
    before = (len(bpy.data.objects), len(bpy.data.lights))
    with pytest.raises(ValueError):
        context.add_light("bad", light_type, **properties)
    assert context.get_lights() == {}
    assert before == (len(bpy.data.objects), len(bpy.data.lights))


def test_property_sampling_is_seeded_independent_and_repeated(context):
    lights = [context.add_light(name, "AREA") for name in ("a", "b")]
    properties = {
        "color": uniform((0.2, 0.3, 0.4), (0.8, 0.9, 1.0)),
        "power": uniform(10, 100),
        "shape": ["SQUARE", "RECTANGLE", "DISK", "ELLIPSE"],
        "size": uniform(0.2, 0.6),
        "size_y": 0.4,
    }

    def samples():
        node = LightRandomizer(seed=42, **properties)
        result = []
        for _ in range(8):
            node.randomize(context)
            row = []
            for light in lights:
                data = light.get_light().blender_obj.data
                assert 10 <= data.energy <= 100
                assert 0.2 <= data.size <= 0.6
                assert data.size_y == pytest.approx(0.4)
                assert np.all(np.array(data.color) >= (0.2, 0.3, 0.4))
                assert np.all(np.array(data.color) <= (0.8, 0.9, 1.0))
                row.append((data.energy, data.shape, tuple(data.color)))
            assert row[0] != row[1]
            result.append(row)
        return result

    first = samples()
    assert first == samples()
    assert len({row[0][0] for row in first}) == 8
    assert len(bpy.data.lights) == 2


@pytest.mark.parametrize(
    "properties",
    [
        {"radius": uniform(-1, 1)},
        {"power": uniform(0, float("inf"))},
        {"shape": []},
        {"color": uniform(0, 1)},
        {"spot_blend": 0.5},
    ],
)
def test_invalid_properties_leave_all_lights_unchanged(context, properties):
    point = context.add_light("point", power=11)
    context.add_light("area", "AREA", power=12)
    with pytest.raises(ValueError):
        LightRandomizer(**{"power": 15, **properties}).randomize(context)
    assert point.get_light().get_energy() == 11
    assert context.get_light("area").get_light().get_energy() == 12


def test_type_validation_and_exact_target_selection(context):
    context.add_light("key", power=10)
    context.add_light("key_fill", power=20)
    LightRandomizer(["key"], power=30).randomize(context)
    assert context.get_light("key").get_light().get_energy() == 30
    assert context.get_light("key_fill").get_light().get_energy() == 20
    for names in (["key", "missing"], [], ["key", "key"]):
        with pytest.raises(ValueError):
            LightRandomizer(names, power=40).randomize(context)
    assert context.get_light("key").get_light().get_energy() == 30
    context.add_light("sun", "SUN", power=1)
    with pytest.raises(ValueError, match="radius"):
        LightRandomizer(power=50, radius=0.1).randomize(context)
    assert context.get_light("key").get_light().get_energy() == 30


@pytest.mark.parametrize(
    "shape, size, size_y, radius",
    [
        ("SQUARE", 2, 6, math.sqrt(2)),
        ("RECTANGLE", 2, 6, math.sqrt(10)),
        ("DISK", 2, 6, 1),
        ("ELLIPSE", 2, 6, 3),
    ],
)
def test_area_clearance_includes_full_emitter(
    context, shape, size, size_y, radius
):
    light = context.add_light(
        "area", "AREA", shape=shape, size=size, size_y=size_y
    )
    assert light.get_emitter_radius() == pytest.approx(radius)


def test_pose_retries_inside_near_and_emitter_overlap(context):
    # Deliberately outside Context: scenery must also prevent unsafe placement.
    obstacle = bproc.object.create_primitive(
        "CUBE", location=(2, 0, 0), scale=(2, 1, 1)
    )
    light = context.add_light("key", "AREA", shape="DISK", size=1)
    locations = iter([(2, 0, 0), (4.1, 0, 0), (4.5, 0, 0), (5, 0, 0)])
    calls = []

    def sampler(context, marker):
        calls.append(marker)
        return next(locations), (0.1, 0.2, 0.3)

    LightPoseRandomizer(
        sampler, min_distance=0.25, max_tries=4, marker=7
    ).randomize(context)
    assert calls == [7] * 4
    np.testing.assert_allclose(light.get_location(), (5, 0, 0))
    np.testing.assert_allclose(light.get_rotation(), (0.1, 0.2, 0.3))
    obstacle.hide(True)
    LightPoseRandomizer(
        lambda **_: ((2, 0, 0), (0, 0, 0)), max_tries=1
    ).randomize(context)
    np.testing.assert_allclose(light.get_location(), (2, 0, 0))


def test_pose_failure_does_not_change_any_target(context):
    bproc.object.create_primitive("CUBE")
    lights = [context.add_light(n, location=(4, 0, 0)) for n in ("a", "b")]
    samples = iter([(5, 0, 0), (0, 0, 0), (0, 0, 0)])
    node = LightPoseRandomizer(
        lambda **_: (next(samples), (0, 0, 0)), max_tries=2
    )
    with pytest.raises(RuntimeError, match="'b'.*2 attempts"):
        node.randomize(context)
    for light in lights:
        np.testing.assert_allclose(light.get_location(), (4, 0, 0))


def test_sun_location_does_not_require_clearance(context):
    bproc.object.create_primitive("CUBE")
    light = context.add_light("sun", "SUN")
    LightPoseRandomizer(
        lambda **_: ((0, 0, 0), (0.2, 0.3, 0.4)), max_tries=1
    ).randomize(context)
    np.testing.assert_allclose(light.get_rotation(), (0.2, 0.3, 0.4))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tries": 0},
        {"max_tries": 1.5},
        {"min_distance": -1},
        {"min_distance": float("nan")},
    ],
)
def test_invalid_pose_settings(kwargs):
    with pytest.raises(ValueError):
        LightPoseRandomizer(lambda **_: None, **kwargs)


@pytest.mark.parametrize(
    "pose", [((float("nan"), 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0))]
)
def test_invalid_sample_leaves_pose_unchanged(context, pose):
    light = context.add_light("key", location=(4, 0, 0))
    with pytest.raises(ValueError):
        LightPoseRandomizer(lambda **_: pose).randomize(context)
    np.testing.assert_allclose(light.get_location(), (4, 0, 0))


def test_shell_sampler_and_randomizer_chain(context):
    light = context.add_light("spot", "SPOT")
    target = np.array([0.0, 0.0, 0.0])
    randomizer = Randomizer()
    randomizer.add_randomizer(
        LightRandomizer(power=25, radius=0.1), "properties"
    )
    randomizer.add_randomizer(
        LightPoseRandomizer(
            shell_sampler,
            center=target,
            radius_min=1.5,
            radius_max=2,
            elevation_min=45,
            elevation_max=80,
        ),
        "pose",
    )
    for _ in range(3):
        randomizer.randomize(context)
        data = light.get_light().blender_obj
        assert data.data.energy == 25
        assert light.get_location()[2] > 0
        direction = np.array(data.rotation_euler.to_matrix()) @ (0, 0, -1)
        expected = target - light.get_location()
        np.testing.assert_allclose(
            direction, expected / np.linalg.norm(expected), atol=1e-6
        )


def test_light_pose_pass_uses_moved_geometry_and_respects_disabled_nodes(
    context,
):
    obstacle = bproc.object.create_primitive("CUBE")
    light = context.add_light("key", power=10, radius=0.1)
    locations = iter([(3, 0, 0), (3, 0, 0), (6, 0, 0)])
    randomizer = Randomizer()
    randomizer.add_randomizer(LightRandomizer(power=20), "properties")
    randomizer.add_randomizer(
        LightPoseRandomizer(lambda **_: (next(locations), (0, 0, 0))),
        "pose",
    )
    randomizer.add_randomizer(
        LightPoseRandomizer(lambda **_: pytest.fail("Disabled node ran")),
        "disabled",
        node_config=NodeConfig(enabled=False),
    )
    randomizer.randomize(context)
    obstacle.set_location((3, 0, 0))
    light.set_properties(power=30)
    randomizer.randomize_light_poses(context)
    np.testing.assert_allclose(light.get_location(), (6, 0, 0))
    assert light.get_light().get_energy() == 30


def test_generation_rechecks_light_clearance_after_physics(
    context, monkeypatch
):
    obstacle = bproc.object.create_primitive("CUBE")
    light = context.add_light("key", radius=0.1)
    locations = iter([(3, 0, 0), (3, 0, 0), (6, 0, 0)])
    randomizer = Randomizer()
    randomizer.add_randomizer(
        LightPoseRandomizer(lambda **_: (next(locations), (0, 0, 0))), "pose"
    )

    def simulate(*args):
        np.testing.assert_allclose(light.get_location(), (3, 0, 0))
        obstacle.set_location((3, 0, 0))

    def render(**kwargs):
        np.testing.assert_allclose(light.get_location(), (6, 0, 0))
        return {}

    monkeypatch.setattr(
        bproc.object, "simulate_physics_and_fix_final_poses", simulate
    )
    monkeypatch.setattr(bproc.renderer, "render", render)
    monkeypatch.setattr(bproc.renderer, "set_render_devices", lambda **_: None)
    monkeypatch.setattr(
        bproc.renderer, "enable_segmentation_output", lambda **_: None
    )
    writer = SimpleNamespace(
        get_segmentation_output_config=dict, write=lambda **_: 1
    )
    SyntheticDataGenerator(context, randomizer, writer).generate(
        num_images=1, simulate_physics=True, clean_up_scene=False
    )
