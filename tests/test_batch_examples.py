"""Check Windows batch sequencing without rendering or training."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows batch script")
@pytest.mark.parametrize(
    "generation_exit,training_exit", [(0, 0), (7, 0), (0, 9)]
)
def test_batch_envs_handoff_and_failure(
    tmp_path: Path, generation_exit: int, training_exit: int
) -> None:
    """Resolve relocated paths and stop the pipeline when either stage fails."""
    repos = tmp_path / "repos with spaces"
    illusion = repos / "telekinesis-illusion"
    illusion.mkdir(parents=True)
    iris = repos / "telekinesis-iris"
    (iris / "examples").mkdir(parents=True)
    (iris / "examples" / "train_rfdetr.py").touch()
    batch = illusion / "examples/bat_scripts/generate_and_train_rfdetr.bat"
    batch.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "examples/bat_scripts" / batch.name, batch)
    conda = repos / "fake conda.bat"
    conda.write_text(
        "@echo off\n"
        'echo %*>>"%CALL_LOG%"\n'
        'echo %CD%>>"%CALL_LOG%"\n'
        'if "%~4"=="telekinesis-illusion" exit /b %GENERATION_EXIT%\n'
        "exit /b %TRAINING_EXIT%\n"
    )
    log = repos / "calls.txt"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "call", str(batch), "test run"],
        cwd=tmp_path,
        env={
            **os.environ,
            "CONDA_BAT": str(conda),
            "CALL_LOG": str(log),
            "GENERATION_EXIT": str(generation_exit),
            "TRAINING_EXIT": str(training_exit),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == (generation_exit or training_exit), (
        result.stderr
    )
    calls = log.read_text().splitlines()
    assert "-n telekinesis-illusion python" in calls[0]
    assert (
        "examples\\bat_scripts\\generate_synthetic_data_with_bin_picking_worker.py"
        in calls[0]
    )
    assert (
        f'--spec-file "{illusion}\\configs\\example_bin_picking_gearwheel_2.yaml"'
        in calls[0]
    )
    assert '--dataset-name "test run"' in calls[0]
    assert "--dataset-format coco --no-preview" in calls[0]
    assert calls[1] == str(illusion)
    if generation_exit:
        assert len(calls) == 2
    else:
        assert len(calls) == 4
        assert "-n telekinesis-iris python" in calls[2]
        assert (
            f'--dataset "{illusion}\\output\\test run\\coco_format"' in calls[2]
        )
        assert '--run-name "test run"' in calls[2]
        assert calls[3] == str(iris)
