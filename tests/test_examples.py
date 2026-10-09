"""Smoke tests that run the example scripts and check they don't crash.

These invoke the real examples end-to-end. The dataset examples require
Blender rendering and a GPU. Interactive viewers are disabled by CLI flags.
The heavy-duty wheel material preview uses --no-render and runs headlessly.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

RUNS_TO_COMPLETION = {
    "quickstart_flying_things.py": (["--no-preview"], 300),
    "quickstart_parts_in_bin.py": (["--no-preview"], 300),
    "generate_synthetic_data_with_bin_picking_worker.py": (
        [
            "--no-preview",
            "--spec-file",
            str(
                EXAMPLES_DIR.parent
                / "configs/example_bin_picking_gearwheel_2.yaml"
            ),
        ],
        300,
    ),
}

# Printed by SyntheticDataGenerator.generate() right before scene clean-up,
# once rendering and writing have finished. bpy is known to occasionally
# crash the interpreter on shutdown (STATUS_ACCESS_VIOLATION) even after a
# fully successful run, so we treat this marker -- not the exit code -- as
# the signal that an example actually did its job.
COMPLETION_MARKER = "Time elapsed (hh:mm:ss.ms)"


def _run_example(
    script: str,
    args: list[str] | None = None,
    timeout: float = 300,
    cwd: Path = EXAMPLES_DIR,
) -> subprocess.CompletedProcess | None:
    """Run an example script; return the completed process, or None if it
    was still running when the timeout elapsed."""
    cmd = [sys.executable, str(EXAMPLES_DIR / script), *(args or [])]
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None


def _assert_ran_successfully(result: subprocess.CompletedProcess) -> None:
    if COMPLETION_MARKER in result.stdout or COMPLETION_MARKER in result.stderr:
        return
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("no_render", [True, False])
def test_light_preview_runs_to_completion(tmp_path, no_render):
    from PIL import Image, ImageChops

    args = ["--output-dir", ".", "--resolution", "128", "--seed", "7"]
    if no_render:
        args.append("--no-render")
    result = _run_example(
        "preview_light_randomization.py", args, timeout=120, cwd=tmp_path
    )
    assert result is not None, "Light preview did not finish within 120s"
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Light preview complete:" in result.stdout
    inventory = json.loads((tmp_path / "inventory.json").read_text())
    assert Path(inventory["model"]).name == "gearwheel_2.glb"
    assert inventory["seed"] == 7
    assert inventory["rendered"] is not no_render
    previews = inventory["previews"]
    assert inventory["row_major_order"] == list(previews)
    assert len(previews) == 9
    assert {p["type"] for p in previews.values()} == {
        "POINT",
        "SUN",
        "SPOT",
        "AREA",
    }
    for name, preview in previews.items():
        assert preview["active_lights"] == [name]
        if name != "08_area_pose":
            assert (
                preview["location"]
                == previews["00_point_reference"]["location"]
            )
    reference = previews["00_point_reference"]["properties"]
    assert (
        previews["01_point_power"]["properties"]["power"]
        >= reference["power"] * 2
    )
    assert (
        previews["02_point_radius"]["properties"]["radius"]
        > reference["radius"]
    )
    assert (
        previews["03_point_color"]["properties"]["color"] != reference["color"]
    )
    assert (
        previews["07_area_disk"]["properties"]
        == previews["08_area_pose"]["properties"]
    )
    assert (
        previews["07_area_disk"]["location"]
        != previews["08_area_pose"]["location"]
    )
    if no_render:
        assert not list(tmp_path.glob("*.png"))
        assert all(p["image"] is None for p in previews.values())
    else:
        annotations = json.loads(
            (tmp_path / "coco_annotations.json").read_text()
        )
        images = annotations["images"]
        assert len(images) == 9
        assert [image["light_variant"] for image in images] == list(previews)
        assert all(
            image["camera_matrix_world"] == images[0]["camera_matrix_world"]
            for image in images
        )
        with Image.open(tmp_path / "light_variants.png") as sheet:
            assert sheet.size == (128 * 3, (128 + 32) * 3)
            for index, preview in enumerate(previews.values()):
                with Image.open(tmp_path / preview["image"]) as panel:
                    assert panel.size == (128, 128)
                    x, y = (index % 3) * 128, (index // 3) * 160
                    assert (
                        ImageChops.difference(
                            sheet.crop((x, y, x + 128, y + 128)),
                            panel.convert("RGB"),
                        ).getbbox()
                        is None
                    )
        with (
            Image.open(
                tmp_path / previews["00_point_reference"]["image"]
            ) as base,
            Image.open(
                tmp_path / previews["01_point_power"]["image"]
            ) as brighter,
        ):
            assert (
                ImageChops.difference(
                    base.convert("RGB"), brighter.convert("RGB")
                ).getbbox()
                is not None
            )


@pytest.mark.parametrize("script,spec", RUNS_TO_COMPLETION.items())
def test_example_runs_to_completion(script, spec):
    args, timeout = spec
    result = _run_example(script, args=args, timeout=timeout)
    assert result is not None, f"{script} did not finish within {timeout}s"
    _assert_ran_successfully(result)


def test_view_dataset_importable():
    """view_dataset.py has no bpy/GPU dependency, so it can be imported
    directly and its format-detection helper exercised without running a
    full generation."""
    spec = importlib.util.spec_from_file_location(
        "view_dataset", EXAMPLES_DIR / "view_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(FileNotFoundError):
        module._detect_format(Path("/nonexistent/dataset/dir"))


def test_material_preview_runs_to_completion(tmp_path):
    result = _run_example(
        "preview_material_randomization.py",
        ["--no-render", "--output-dir", str(tmp_path)],
        timeout=60,
    )
    assert result is not None, "Material preview did not finish within 60s"
    assert "Heavy-duty wheel preview complete:" in result.stdout, (
        result.stdout + result.stderr
    )
    inventory = json.loads((tmp_path / "inventory.json").read_text())
    assert Path(inventory["model"]).name == "heavy_duty_wheel.glb"
    assert inventory["objects"] == {
        f"wheel_INSTANCE_{index}": ["Stahl", "Alu", "Gumi"]
        for index in range(7)
    }
    centers = list(inventory["centers"].values())
    assert [center[0] for center in centers] == sorted(
        center[0] for center in centers
    )
    assert len({center[0] for center in centers}) == 7
    assert all(abs(center[1]) < 1e-9 for center in centers)
    assert all(abs(center[2]) < 1e-9 for center in centers)
    assert inventory["seed"] == 42
    assert inventory["rendered"] is False
    previews = inventory["previews"]
    original = previews["00_original"]
    expected_changes = {
        "01_blue_plastic_slot": {"Alu"},
        "02_procedural_metal_slots": {"Stahl", "Alu"},
        "03_pbr_metal_slots": {"Stahl", "Alu"},
        "04_metal_and_rubber_slots": {"Alu", "Gumi"},
        "05_all_slots": set(original),
        "06_whole_object_pbr_metal": set(original),
    }
    assert set(previews) == {"00_original", *expected_changes}
    assert inventory["left_to_right"] == list(previews)
    for name, expected in expected_changes.items():
        assert set(previews[name]) == set(original)
        changed = {
            slot for slot in original if previews[name][slot] != original[slot]
        }
        assert changed == expected, name
    assert len(set(previews["06_whole_object_pbr_metal"].values())) == 1
    # Completing assignments must not mask a native crash during shutdown.
    assert result.returncode == 0, result.stdout + result.stderr
