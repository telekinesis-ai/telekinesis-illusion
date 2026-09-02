"""Smoke tests that run the example scripts and check they don't crash.

These are not unit tests: they invoke the real examples end-to-end, which
means real Blender rendering (and, for the two quickstart examples, a real
GUI viewer window). They require a GPU and the bundled default assets, so
they are meant to be run locally, not in a headless CI environment.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# quickstart_flying_things.py and quickstart_parts_in_bin.py always end by
# opening a blocking Tkinter viewer once generation succeeds, so they never
# exit on their own. We give them enough time to finish rendering and reach
# the viewer, then treat "still running" as success.
BLOCKING_ON_VIEWER = {
    "quickstart_flying_things.py": 300,
    "quickstart_parts_in_bin.py": 300,
}

# This example accepts a flag to skip its interactive preview, so it can run
# to a normal, timely exit.
RUNS_TO_COMPLETION = {
    "generate_synthetic_data_with_bin_picking_worker.py": (["--no-preview"], 300),
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
            cmd, cwd=EXAMPLES_DIR, timeout=timeout, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        return None


def _assert_ran_successfully(result: subprocess.CompletedProcess) -> None:
    if COMPLETION_MARKER in result.stdout or COMPLETION_MARKER in result.stderr:
        return
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("script,timeout", BLOCKING_ON_VIEWER.items())
def test_example_reaches_viewer_without_crashing(script, timeout):
    result = _run_example(script, timeout=timeout)
    if result is not None:
        # The script exited on its own instead of blocking in the viewer.
        _assert_ran_successfully(result)


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
