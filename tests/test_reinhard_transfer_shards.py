"""Color statistics, COCO layout preservation and failures without Blender."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "examples/reinhard_transfer_shards.py"
)


@pytest.fixture(scope="module")
def transfer():
    spec = importlib.util.spec_from_file_location(
        "reinhard_transfer_shards", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    encoded.tofile(path)


@pytest.fixture
def dataset(tmp_path, request):
    layout = getattr(request, "param", "shards")
    references = tmp_path / "real images"
    for name, offset in (("nested/dark.PNG", 40), ("light.jpeg", 150)):
        gray = np.arange(40, dtype=np.uint8).reshape(5, 8) + offset
        save_image(references / name, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    (references / "ignored.txt").write_text("not an image")
    synthetic = tmp_path / "synthetic"
    exported = layout in {"split_coco", "single_coco"}
    annotation_name = (
        "_annotations.coco.json" if exported else "coco_annotations.json"
    )
    directories = {
        "shards": ("shard_b", "shard_a"),
        "split_coco": ("train", "valid", "test"),
        "single_coco": ("",),
        "single_shard": ("",),
    }[layout]
    for directory_name in directories:
        directory = synthetic / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        entries = []
        names = (
            ()
            if directory_name == "test"
            else ("café.PNG", "another.jpeg", "third.jpg")
        )
        for index, name in enumerate(names):
            gray = np.arange(80, dtype=np.uint8).reshape(8, 10) + 60
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            relative = name if exported else f"images/{name}"
            save_image(directory / relative, image)
            entries.append(
                {
                    "id": index,
                    "file_name": relative,
                    "height": 8,
                    "width": 10,
                    "custom": "keep",
                }
            )
        coco = {
            "images": entries,
            "categories": [{"id": 1, "name": "part"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [1, 2, 3, 4],
                    "segmentation": [[1, 2, 3, 4, 5, 6]],
                }
            ]
            if entries
            else [],
        }
        (directory / annotation_name).write_text(
            json.dumps(coco), encoding="utf-8"
        )
        save_image(directory / "masks/0.png", gray)
        (directory / "notes.txt").write_text("keep metadata")
    if layout == "shards":
        (synthetic / "merged_coco_annotations.json").write_text(
            '{"images": [], "custom": "preserve verbatim"}'
        )
    return references, synthetic


def test_transfer_matches_lab_statistics(transfer):
    gray = np.linspace(60, 150, 400).reshape(20, 20).astype(np.uint8)
    source = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    reference = cv2.cvtColor(gray // 2 + 80, cv2.COLOR_GRAY2BGR)
    _, mean, std = transfer.lab_statistics(reference)
    output = transfer.reinhard_transfer(source, mean, std)
    _, actual_mean, actual_std = transfer.lab_statistics(output)
    np.testing.assert_allclose(actual_mean, mean, atol=0.3)
    np.testing.assert_allclose(actual_std, std, atol=0.3)
    assert output.shape == source.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, source)


def test_flat_channels_and_alpha(transfer):
    source = np.full((8, 10, 4), 128, dtype=np.uint8)
    source[..., 3] = 0
    source[:4, :, 3] = 255
    source[4:, :, :3] = 0
    reference = np.full((3, 4, 3), [120, 160, 180], dtype=np.uint8)
    _, mean, std = transfer.lab_statistics(reference)
    with np.errstate(all="raise"):
        output = transfer.reinhard_transfer(source, mean, std)
    np.testing.assert_array_equal(output[..., 3], source[..., 3])
    np.testing.assert_allclose(
        output[:4, :, :3], np.broadcast_to(reference[0, 0], (4, 10, 3)), atol=1
    )


@pytest.mark.parametrize(
    "dataset",
    ["shards", "split_coco", "single_coco", "single_shard"],
    indirect=True,
)
def test_cli_detects_layout_preserves_dataset_and_reproduces_sampling(
    transfer, dataset, tmp_path
):
    references, synthetic = dataset
    annotation_files = [
        path
        for path in synthetic.rglob("*.json")
        if path.name in {"coco_annotations.json", "_annotations.coco.json"}
    ]
    image_paths = {
        (annotation.parent / entry["file_name"]).relative_to(synthetic)
        for annotation in annotation_files
        for entry in json.loads(annotation.read_text())["images"]
    }
    original = {
        p.relative_to(synthetic): p.read_bytes()
        for p in synthetic.rglob("*")
        if p.is_file()
    }
    outputs = [tmp_path / "transformed one", tmp_path / "transformed two"]
    for output in outputs:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--reference-dir",
                str(references),
                "--synthetic-dir",
                str(synthetic),
                "--output-dir",
                str(output),
                "--seed",
                "7",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (
            f"{len(image_paths)} images in {len(annotation_files)} COCO directories"
            in result.stdout
        )
        expected_format = (
            "exported COCO"
            if annotation_files[0].name.startswith("_")
            else "native COCO"
        )
        assert f"Detected {expected_format}" in result.stdout
    assert {
        p.relative_to(outputs[0]) for p in outputs[0].rglob("*") if p.is_file()
    } == set(original)
    brightness = []
    for relative, before in original.items():
        assert (synthetic / relative).read_bytes() == before
        after = (outputs[0] / relative).read_bytes()
        assert after == (outputs[1] / relative).read_bytes()
        if relative in image_paths:
            assert after != before
            image = transfer.read_image(outputs[0] / relative)
            assert image.shape == (8, 10, 3)
            brightness.append(image.mean())
        else:
            assert after == before
    assert min(brightness) < 100 < max(brightness)


@pytest.mark.parametrize("input_kind", ["reference", "synthetic"])
def test_corrupt_image_does_not_publish_partial_dataset(
    transfer, dataset, tmp_path, input_kind
):
    references, synthetic = dataset
    broken = (
        references / "light.jpeg"
        if input_kind == "reference"
        else synthetic / "shard_b/images/third.jpg"
    )
    broken.write_bytes(b"not an image")
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="Could not read image"):
        transfer.transform_shards(references, synthetic, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-*"))


def test_refuses_existing_or_nested_output(transfer, dataset, tmp_path):
    references, synthetic = dataset
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        transfer.transform_shards(references, synthetic, existing)
    assert sentinel.read_text() == "keep"
    for output in (synthetic / "new", references / "new"):
        with pytest.raises(ValueError, match="outside both input"):
            transfer.transform_shards(references, synthetic, output)
        assert not output.exists()


@pytest.mark.parametrize(
    "dataset", ["shards", "split_coco", "single_coco"], indirect=True
)
@pytest.mark.parametrize("file_name", ["../outside.png", "missing.png"])
def test_rejects_invalid_coco_paths(transfer, dataset, tmp_path, file_name):
    references, synthetic = dataset
    annotation = next(
        path
        for path in synthetic.rglob("*.json")
        if path.name in {"coco_annotations.json", "_annotations.coco.json"}
        and json.loads(path.read_text())["images"]
    )
    data = json.loads(annotation.read_text())
    data["images"][0]["file_name"] = file_name
    annotation.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        transfer.transform_shards(references, synthetic, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


@pytest.mark.parametrize(
    "extra_annotation",
    [
        "shard_a/_annotations.coco.json",
        "train/_annotations.coco.json",
        "coco_annotations.json",
    ],
)
def test_rejects_ambiguous_layout(
    transfer, dataset, tmp_path, extra_annotation
):
    references, synthetic = dataset
    annotation = synthetic / extra_annotation
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text('{"images": []}')
    with pytest.raises(ValueError, match="Ambiguous COCO layout"):
        transfer.transform_shards(references, synthetic, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_rejects_unrecognized_layout(transfer, dataset, tmp_path):
    references, _ = dataset
    synthetic = tmp_path / "unsupported"
    synthetic.mkdir()
    with pytest.raises(ValueError, match="No COCO dataset found"):
        transfer.transform_shards(references, synthetic, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_rejects_dimension_mismatch(transfer, dataset, tmp_path):
    references, synthetic = dataset
    annotation = synthetic / "shard_a/coco_annotations.json"
    data = json.loads(annotation.read_text())
    data["images"][0]["width"] = 100
    annotation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="dimensions disagree"):
        transfer.transform_shards(references, synthetic, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_write_failure_does_not_publish(
    transfer, dataset, tmp_path, monkeypatch
):
    references, synthetic = dataset
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    with pytest.raises(OSError, match="Could not encode image"):
        transfer.transform_shards(references, synthetic, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed-*"))
