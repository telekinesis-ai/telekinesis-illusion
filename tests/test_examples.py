"""Run the generation examples without previews using real CPU rendering."""

import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]


def run_example(name, args, log_path):
    """Run an example without a viewer and retain its output on failure."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    runner = """
import runpy
import sys

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    with log_path.open("w", encoding="utf8") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                runner,
                str(ROOT / "examples" / name),
                *map(str, args),
                "--no-preview",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    assert result.returncode == 0, log_path.read_text(encoding="utf8")[-12000:]


def validate_dataset(path, count, resolution):
    """Check rendered images and their COCO masks, then return the records."""
    coco = json.loads(path.read_text(encoding="utf8"))
    assert len(coco["images"]) == count
    ids = {image["id"] for image in coco["images"]}
    assert len(ids) == count
    assert len({image["file_name"] for image in coco["images"]}) == count
    categories = {entry["id"]: entry["name"] for entry in coco["categories"]}
    assert categories
    for image in coco["images"]:
        assert (image["height"], image["width"]) == (resolution, resolution)
        pixels = cv2.imread(str(path.parent / image["file_name"]))
        assert pixels is not None
        assert pixels.shape == (resolution, resolution, 3)
        assert pixels.std() > 1
    assert coco["annotations"], "Rendered objects must produce real annotations"
    for annotation in coco["annotations"]:
        assert annotation["image_id"] in ids
        assert annotation["category_id"] in categories
        mask = mask_utils.decode(annotation["segmentation"])
        assert mask.shape == (resolution, resolution)
        assert mask.sum() > 0
        assert annotation["area"] == int(mask.sum())
        assert np.isfinite(annotation["bbox"]).all()
    return coco


@pytest.mark.integration
@pytest.mark.parametrize(
    "name", ["quickstart_flying_things.py", "quickstart_parts_in_bin.py"]
)
def test_quickstarts_render_multiple_samples(tmp_path, name):
    """Quickstarts render the requested image count with camera metadata."""
    output = tmp_path / "dataset"
    run_example(
        name,
        [
            "--num-images",
            3,
            "--resolution",
            64,
            "--samples",
            2,
            "--device",
            "CPU",
            "--seed",
            17,
            "--output-dir",
            output,
        ],
        tmp_path / "run.log",
    )
    data = validate_dataset(output / "coco_annotations.json", 3, 64)
    assert all("camera_matrix_world" in image for image in data["images"])


@pytest.mark.integration
def test_worker_renders_and_exports_coco_shards(tmp_path):
    """The YAML example renders shards, merges them, and exports COCO splits."""
    spec = yaml.safe_load(
        (ROOT / "configs/example_bin_picking_gearwheel_2.yaml").read_text(
            encoding="utf8"
        )
    )
    spec["metadata"].update(
        asset_directory=str(ROOT / "assets"),
        base_output_directory=str(tmp_path),
        dataset_name="test_bins",
        num_images=3,
    )
    spec["shard"]["size"] = 2
    spec["camera"].update(image_width=64, image_height=64)
    spec["renderer"].update(samples=2, device="CPU")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf8")
    run_example(
        "generate_synthetic_data_with_bin_picking_worker.py",
        ["--spec_file", spec_path],
        tmp_path / "run.log",
    )
    dataset = tmp_path / "test_bins"
    shards = list(dataset.glob("shard_*/coco_annotations.json"))
    assert len(shards) == 2
    shard_counts = [
        len(json.loads(path.read_text(encoding="utf8"))["images"])
        for path in shards
    ]
    assert sorted(shard_counts) == [1, 2]
    for path, count in zip(shards, shard_counts):
        validate_dataset(path, count, 64)
    validate_dataset(dataset / "merged_coco_annotations.json", 3, 64)
    exported = list((dataset / "coco_format").glob("*/_annotations.coco.json"))
    assert len(exported) == 3
    records = [json.loads(path.read_text(encoding="utf8")) for path in exported]
    assert sum(len(data["images"]) for data in records) == 3
    for path, data in zip(exported, records):
        assert all(
            (path.parent / image["file_name"]).is_file()
            for image in data["images"]
        )
