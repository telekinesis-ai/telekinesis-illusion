"""Check CLI overrides and batch sequencing without rendering or training."""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from runpy import run_path
from types import ModuleType
from unittest.mock import Mock, call

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN = run_path(
    str(ROOT / "examples/generate_synthetic_data_with_bin_picking_worker.py")
)["main"]


def test_generation_overrides_preserve_yaml_and_asset_paths(
    tmp_path, monkeypatch
):
    spec = tmp_path / "configs" / "example.yaml"
    spec.parent.mkdir()
    original = (
        "metadata:\n  asset_directory: ../assets\n  dataset_name: original\n"
        "  num_images: 10\n  base_output_directory: output\n"
        "shard:\n  size: 5\noutput:\n  dataset_format: yolo\n"
    )
    spec.write_text(original)
    captured = {}
    worker = Mock()

    def make_worker(path):
        captured.update(yaml.safe_load(path.read_text()))
        return worker

    module = ModuleType("telekinesis.illusion.workers.bin_picking_worker")
    module.BinPickingWorker = Mock(side_effect=make_worker)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = tmp_path / "output with spaces"
    args = [
        "--spec-file",
        str(spec),
        "--dataset-name",
        "new run",
        "--output-dir",
        str(output),
        "--num-images",
        "21",
        "--shard-size",
        "7",
        "--dataset-format",
        "coco",
        "--no-preview",
    ]
    MAIN(args)

    assert captured["metadata"]["asset_directory"] == str(tmp_path / "assets")
    assert captured["metadata"]["dataset_name"] == "new run"
    assert captured["metadata"]["base_output_directory"] == str(output)
    assert captured["metadata"]["num_images"] == 21
    assert captured["shard"]["size"] == 7
    assert captured["output"]["dataset_format"] == "coco"
    assert spec.read_text() == original
    assert worker.mock_calls == [
        call.generate(),
        call.merge_shards(preview=False),
    ]

    (output / "new run" / "coco_format").mkdir(parents=True)
    with pytest.raises(SystemExit) as error:
        MAIN(args)
    assert error.value.code == 2
    assert module.BinPickingWorker.call_count == 1


@pytest.mark.parametrize(
    "args",
    [
        ["--num-images", "0"],
        ["--shard-size", "-1"],
        ["--dataset-name", ".."],
        ["--dataset-name", "../outside"],
    ],
)
def test_generation_rejects_invalid_overrides(args):
    with pytest.raises(SystemExit) as error:
        MAIN(args)
    assert error.value.code == 2


@pytest.mark.skipif(sys.platform != "win32", reason="Windows bpy shutdown")
@pytest.mark.parametrize("failure", ["none", "generate", "merge"])
def test_successful_export_bypasses_shutdown_but_errors_do_not(
    tmp_path, failure
):
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "metadata:\n  dataset_name: shutdown_test\n"
        f"  base_output_directory: {tmp_path.as_posix()}\n"
    )
    cleanup_marker = tmp_path / "cleaned.txt"
    harness = tmp_path / "run_generator.py"
    harness.write_text(
        textwrap.dedent("""\
        import atexit
        import os
        import runpy
        import sys
        from pathlib import Path
        from types import ModuleType
        from unittest.mock import Mock

        script, spec, cleanup_marker, failure = sys.argv[1:]
        worker = Mock()
        if failure != "none":
            method = worker.generate if failure == "generate" else worker.merge_shards
            method.side_effect = RuntimeError("generation/export failed")
        module = ModuleType("telekinesis.illusion.workers.bin_picking_worker")
        module.BinPickingWorker = Mock(return_value=worker)
        sys.modules[module.__name__] = module
        initializer = ModuleType("blenderproc.python.utility.Initializer")
        initializer.clean_temp_dir = lambda: Path(cleanup_marker).touch()
        sys.modules[initializer.__name__] = initializer
        # Simulate native shutdown failure, after the script has returned.
        atexit.register(lambda: os._exit(99))
        sys.argv = [script, "--spec-file", spec, "--no-preview"]
        runpy.run_path(script, run_name="__main__")
    """)
    )
    result = subprocess.run(
        [
            sys.executable,
            str(harness),
            str(
                ROOT
                / "examples"
                / "generate_synthetic_data_with_bin_picking_worker.py"
            ),
            str(spec),
            str(cleanup_marker),
            failure,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if failure == "none":
        assert result.returncode == 0, result.stderr
        assert cleanup_marker.is_file()
        assert "Dataset ready:" in result.stdout
    else:
        assert result.returncode != 0
        assert not cleanup_marker.exists()
        assert "generation/export failed" in result.stderr
        assert "Dataset ready:" not in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows batch script")
@pytest.mark.parametrize(
    "generation_exit,training_exit", [(0, 0), (7, 0), (0, 9)]
)
def test_batch_envs_handoff_and_failure(
    tmp_path, generation_exit, training_exit
):
    repos = tmp_path / "repos with spaces"
    illusion = repos / "telekinesis-illusion"
    illusion.mkdir(parents=True)
    iris = repos / "telekinesis-iris"
    (iris / "examples").mkdir(parents=True)
    (iris / "examples" / "train_rfdetr.py").touch()
    batch = illusion / "generate_and_train_rfdetr.bat"
    shutil.copyfile(ROOT / batch.name, batch)
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
        ["cmd.exe", "/d", "/c", batch.name, "test run"],
        cwd=illusion,
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
