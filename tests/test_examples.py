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
    script: str, args: list[str] | None = None, timeout: float = 300
) -> subprocess.CompletedProcess | None:
    """Run an example script; return the completed process, or None if it
    was still running when the timeout elapsed."""
    cmd = [sys.executable, str(EXAMPLES_DIR / script), *(args or [])]
    try:
        return subprocess.run(
            cmd,
            cwd=EXAMPLES_DIR,
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
