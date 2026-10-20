"""Check CLI overrides and normal process shutdown without rendering."""

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
    str(
        ROOT
        / "examples/bat_scripts/generate_synthetic_data_with_bin_picking_worker.py"
    )
)["main"]


def test_normal_example_uses_original_yaml(monkeypatch):
    module = ModuleType("telekinesis.illusion.workers.bin_picking_worker")
    module.BinPickingWorker = Mock()
    monkeypatch.setitem(sys.modules, module.__name__, module)

    main = run_path(
        str(
            ROOT / "examples/generate_synthetic_data_with_bin_picking_worker.py"
        )
    )["main"]
    main()

    module.BinPickingWorker.assert_called_once_with(
        ROOT / "configs/example_bin_picking_gearwheel_2.yaml"
    )
    assert module.BinPickingWorker.return_value.mock_calls == [
        call.generate(),
        call.merge_shards(preview=False),
    ]


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


@pytest.mark.parametrize("failure", ["none", "generate", "merge"])
@pytest.mark.parametrize("batch_wrapper", [False, True])
def test_generation_preserves_exit_status_and_cleanup(
    tmp_path: Path, failure: str, batch_wrapper: bool
) -> None:
    """Limit forced shutdown to successful Windows batch generation."""
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "metadata:\n  dataset_name: shutdown_test\n"
        f"  base_output_directory: {tmp_path.as_posix()}\n"
    )
    cleanup_marker = tmp_path / "cleaned.txt"
    temporary_cleanup_marker = tmp_path / "temporary_cleaned.txt"
    harness = tmp_path / "run_generator.py"
    harness.write_text(
        textwrap.dedent("""\
        import atexit
        import runpy
        import sys
        from pathlib import Path
        from types import ModuleType
        from unittest.mock import Mock

        script, spec, cleanup_marker, temporary_cleanup_marker, failure = sys.argv[1:]
        worker = Mock()
        worker.merge_shards.side_effect = lambda **kwargs: print(
            "Dataset merged", flush=True
        )
        if failure != "none":
            method = worker.generate if failure == "generate" else worker.merge_shards
            method.side_effect = RuntimeError("generation/export failed")
        module = ModuleType("telekinesis.illusion.workers.bin_picking_worker")
        module.BinPickingWorker = Mock(return_value=worker)
        sys.modules[module.__name__] = module
        initializer = ModuleType("blenderproc.python.utility.Initializer")
        initializer.clean_temp_dir = lambda: Path(temporary_cleanup_marker).touch()
        sys.modules[initializer.__name__] = initializer
        atexit.register(lambda: Path(cleanup_marker).touch())
        sys.path.insert(0, str(Path(script).parent))
        sys.argv = [script]
        if Path(script).parent.name == "bat_scripts":
            sys.argv.extend(["--spec-file", spec, "--no-preview"])
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
                / ("bat_scripts" if batch_wrapper else ".")
                / "generate_synthetic_data_with_bin_picking_worker.py"
            ),
            str(spec),
            str(cleanup_marker),
            str(temporary_cleanup_marker),
            failure,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    bypassed = batch_wrapper and sys.platform == "win32" and failure == "none"
    assert cleanup_marker.is_file() is not bypassed
    assert temporary_cleanup_marker.is_file() is bypassed
    if failure == "none":
        assert result.returncode == 0, result.stderr
        assert "Dataset merged" in result.stdout
    else:
        assert result.returncode != 0
        assert "generation/export failed" in result.stderr
        assert "Dataset merged" not in result.stdout
