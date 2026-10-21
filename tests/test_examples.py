"""Run the requested scripts in isolated processes with real CUDA rendering."""

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
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf8") as log:
        result = subprocess.run(
            [sys.executable, str(ROOT / "examples" / name), *map(str, args)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    assert result.returncode == 0, log_path.read_text(encoding="utf8")[-12000:]


def validate_dataset(path, count, resolution):
    coco = json.loads(path.read_text(encoding="utf8"))
    assert len(coco["images"]) == count
    ids = {image["id"] for image in coco["images"]}
    assert len(ids) == count
    assert len({image["file_name"] for image in coco["images"]}) == count
    categories = {entry["id"]: entry["name"] for entry in coco["categories"]}
    assert categories
    for image in coco["images"]:
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
    output = tmp_path / "dataset"
    run_example(
        name,
        [
            "--no-preview",
            "--num-images",
            3,
            "--resolution",
            64,
            "--samples",
            2,
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
def test_worker_cli_accumulates_compatible_shards_across_restarts(tmp_path):
    spec = yaml.safe_load(
        (ROOT / "configs/example_bin_picking_gearwheel_2.yaml").read_text()
    )
    spec["metadata"]["asset_directory"] = str(ROOT / "assets")
    spec.setdefault("output", {})["dataset_format"] = None
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf8")
    args = [
        "--spec-file",
        spec_path,
        "--no-preview",
        "--num-images",
        3,
        "--shard-size",
        2,
        "--dataset-name",
        "test_bins",
        "--output-dir",
        tmp_path,
        "--resolution",
        64,
        "--samples",
        2,
        "--seed",
        23,
    ]
    dataset = tmp_path / "test_bins"
    run_example(
        "bat_scripts/generate_synthetic_data_with_bin_picking_worker.py",
        args,
        tmp_path / "first.log",
    )
    shards_before = set(dataset.glob("shard_*"))
    assert len(shards_before) == 2
    first = validate_dataset(dataset / "merged_coco_annotations.json", 3, 64)
    run_example(
        "bat_scripts/generate_synthetic_data_with_bin_picking_worker.py",
        args,
        tmp_path / "second.log",
    )
    assert len(list(dataset.glob("shard_*"))) == 4
    assert all(path.exists() for path in shards_before)
    second = validate_dataset(dataset / "merged_coco_annotations.json", 6, 64)
    assert first["categories"] == second["categories"]


@pytest.mark.integration
def test_worker_cli_default_spec_exports_coco_splits(tmp_path):
    run_example(
        "bat_scripts/generate_synthetic_data_with_bin_picking_worker.py",
        [
            "--no-preview",
            "--num-images",
            3,
            "--shard-size",
            2,
            "--dataset-name",
            "export_bins",
            "--output-dir",
            tmp_path,
            "--resolution",
            64,
            "--samples",
            2,
            "--seed",
            23,
        ],
        tmp_path / "export.log",
    )
    dataset = tmp_path / "export_bins"
    validate_dataset(dataset / "merged_coco_annotations.json", 3, 64)
    exported = list((dataset / "coco_format").glob("*/_annotations.coco.json"))
    assert len(exported) == 3
    records = [json.loads(path.read_text()) for path in exported]
    assert sum(len(data["images"]) for data in records) == 3
    for path, data in zip(exported, records):
        assert all(
            (path.parent / image["file_name"]).is_file()
            for image in data["images"]
        )
