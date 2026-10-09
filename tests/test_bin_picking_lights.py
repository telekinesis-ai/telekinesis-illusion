"""Bin-picking YAML lights, graph stages, live edits and a rendered shard."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from telekinesis.illusion.workers import bin_picking_worker as worker_module
from telekinesis.illusion.workers.bin_picking_worker import BinPickingWorker

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker(context, monkeypatch, tmp_path):
    spec = yaml.safe_load(
        (ROOT / "configs/example_bin_picking_gearwheel_2.yaml").read_text()
    )
    spec["metadata"].update(
        asset_directory=str(ROOT / "assets"),
        base_output_directory=str(tmp_path / "output"),
        num_images=1,
    )
    spec["shard"]["size"] = 1
    spec["models"] = spec["models"][:2]
    for model in spec["models"]:
        model["instances"] = {"min": 1, "max": 1}
    spec["distractors"] = []
    spec["instance_randomizer"] = {
        "part": {"min": 1, "max": 1},
        "container": {"min": 1, "max": 1},
    }
    spec["material_randomizer"] = {
        "part": [],
        "container": [],
        "distractor": [],
    }
    spec["background_randomizer"] = {"categories": []}
    spec["camera"].update(image_width=128, image_height=96)
    spec["camera_pose_randomizer"] = {
        "sampler": "shell_sampler",
        "number_of_views": 1,
        "params": {
            "center": ["gearwheel_2"],
            "radius_min": 0.8,
            "radius_max": 1.0,
            "elevation_min": 70,
            "elevation_max": 85,
            "inplane_rot_min": 0,
            "inplane_rot_max": 0,
        },
    }
    spec["physics_simulator"]["active"] = False
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))

    def reuse_context(camera_config, asset_dir):
        context.get_camera().apply_config(camera_config)
        return context

    monkeypatch.setattr(worker_module, "Context", reuse_context)
    return BinPickingWorker(spec_path)


def test_worker_samples_lights_and_geometry_retains_properties(worker):
    context = worker.get_context()
    randomizer = worker.get_randomizer()
    assert set(context.get_lights()) == {"bin_key", "bin_fill"}
    assert set(context.get_lights()).isdisjoint(context.get_objects())
    randomizer.randomize(context)
    key = context.get_light("bin_key")
    fill = context.get_light("bin_fill")
    assert 40 <= key.get_properties()["power"] <= 100
    assert key.get_properties()["shape"] in {"RECTANGLE", "DISK"}
    assert 10 <= fill.get_properties()["power"] <= 30
    before = {
        name: light.get_properties()
        for name, light in context.get_lights().items()
    }
    locations = {
        name: light.get_location().copy()
        for name, light in context.get_lights().items()
    }
    worker.randomize_geometry(context)
    for name, light in context.get_lights().items():
        assert light.get_properties() == before[name]
        assert np.isfinite(light.get_location()).all()
        assert light.get_location()[2] > 0
        assert not np.array_equal(light.get_location(), locations[name])


def test_live_edits_replace_nodes_without_adding_lights_or_resetting_unchanged_nodes(
    worker,
):
    randomizer = worker.get_randomizer()
    node = randomizer.get_randomizer_node("light_properties_bin_key")
    assert worker.apply_spec_updates(deepcopy(worker._specs)) == []
    assert randomizer.get_randomizer_node("light_properties_bin_key") is node
    spec = deepcopy(worker._specs)
    light = spec["light_randomizer"]["lights"][0]
    light["properties"]["power"] = 75
    light["pose"]["params"]["center"] = [0.0, 0.0, 0.0]
    assert worker.apply_spec_updates(spec) == []
    assert (
        randomizer.get_randomizer_node("light_properties_bin_key") is not node
    )
    randomizer.randomize(worker.get_context())
    assert (
        worker.get_context().get_light("bin_key").get_properties()["power"]
        == 75
    )
    assert set(worker.get_context().get_lights()) == {"bin_key", "bin_fill"}


@pytest.mark.parametrize("change", ["disable", "rename", "type"])
def test_structural_light_edits_report_reload(worker, change):
    spec = deepcopy(worker._specs)
    if change == "disable":
        spec["light_randomizer"]["active"] = False
    elif change == "rename":
        spec["light_randomizer"]["lights"][0]["name"] = "new_key"
    else:
        spec["light_randomizer"]["lights"][0]["type"] = "POINT"
        spec["light_randomizer"]["lights"][0]["properties"] = {"power": 60}
    messages = worker.apply_spec_updates(spec)
    assert any(
        "Lights" in message and "reload" in message for message in messages
    )
    assert set(worker.get_context().get_lights()) == {"bin_key", "bin_fill"}


@pytest.mark.parametrize("config", [{}, {"active": False}])
def test_omitted_or_disabled_lights_keep_existing_behavior(config):
    assert BinPickingWorker._parse_light_randomizers(config) == {}


@pytest.mark.parametrize(
    "config_name",
    ["example_bin_picking_gearwheel_2.yaml"],
)
def test_shipped_light_configs_sample_in_blender(context, config_name):
    config = yaml.safe_load((ROOT / "configs" / config_name).read_text())
    nodes = BinPickingWorker._parse_light_randomizers(
        config["light_randomizer"]
    )
    for name, (light_type, properties, _) in nodes.items():
        context.add_light(name, light_type)
        properties.randomize(context)
        assert context.get_light(name).get_properties()["power"] > 0


@pytest.mark.parametrize(
    "patch",
    [
        {"properties": {"power": {"min": -1, "max": 10}}},
        {"properties": {"power": {"low": 1, "high": 10}}},
        {"properties": {"shape": []}},
        {"type": "SUN", "properties": {"radius": 0.1}},
        {"properties": {"unknown": 1}},
        {"pose": {"sampler": "missing"}},
        {"pose": {"max_tries": 0}},
        {"typo": True},
    ],
)
def test_bad_light_config_fails_before_scene_changes(context, patch):
    entry = {
        "name": "key",
        "type": "AREA",
        "properties": {"power": 10},
        "pose": {},
    }
    entry.update(patch)
    with pytest.raises((TypeError, ValueError)):
        BinPickingWorker._parse_light_randomizers({"lights": [entry]})
    assert context.get_lights() == {}


def test_worker_renders_a_shard_with_randomized_lights(worker, monkeypatch):
    import blenderproc as bproc

    render = bproc.renderer.render

    def check_lights_and_render(**kwargs):
        lights = worker.get_context().get_lights()
        assert lights["bin_key"].get_properties()["power"] >= 40
        assert lights["bin_fill"].get_properties()["power"] >= 10
        return render(**kwargs)

    monkeypatch.setattr(bproc.renderer, "render", check_lights_and_render)
    bproc.renderer.set_max_amount_of_samples(8)
    worker.generate()
    root = Path(worker._specs["metadata"]["base_output_directory"])
    annotation = next(root.rglob("coco_annotations.json"))
    coco = json.loads(annotation.read_text())
    assert len(coco["images"]) == 1
    assert coco["annotations"]
    image = coco["images"][0]
    assert (image["width"], image["height"]) == (128, 96)
    assert (annotation.parent / image["file_name"]).is_file()
