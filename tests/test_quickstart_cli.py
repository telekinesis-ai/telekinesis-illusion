"""Check quickstart process cleanup without rendering."""

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_name",
    ["quickstart_flying_things.py", "quickstart_parts_in_bin.py"],
)
@pytest.mark.parametrize("batch_wrapper", [False, True])
def test_quickstart_runs_exit_handlers(
    tmp_path: Path, script_name: str, batch_wrapper: bool
) -> None:
    """Only the batch entrypoints bypass normal cleanup after generation."""
    cleanup_marker = tmp_path / "cleaned.txt"
    temporary_cleanup_marker = tmp_path / "temporary_cleaned.txt"
    harness = textwrap.dedent("""\
        import atexit
        from pathlib import Path
        import runpy
        import sys
        from types import ModuleType
        from unittest.mock import Mock

        script, cleanup_marker, temporary_cleanup_marker = sys.argv[1:]
        module_exports = {
            "core.scene": "Scene",
            "core.engine": "BlenderEngine RenderConfig PhysicsConfig",
            "types.camera": "CameraConfig",
            "core.synthetic_data_generator": "SyntheticDataGenerator",
            "randomizer.randomizer": "Randomizer",
            "randomizer.randomizer_node": (
                "BackgroundRandomizer CameraPoseRandomizer LightPoseRandomizer "
                "LightRandomizer MaterialRandomizer ObjectInstanceRandomizer "
                "ObjectPoseRandomizer"
            ),
            "sampler.camera_pose_sampler": "shell_sampler volume_sampler",
            "types.distribution": "uniform",
            "types.object": "Object",
            "utils.assets": "resolve_asset_dir",
            "writer.writer": "CocoWriter",
        }
        for suffix, exports in module_exports.items():
            name = "telekinesis.illusion." + suffix
            module = ModuleType(name)
            for export in exports.split():
                setattr(module, export, Mock())
            sys.modules[name] = module

        assets = sys.modules["telekinesis.illusion.utils.assets"]
        assets.resolve_asset_dir.return_value = Path.cwd()
        generator = sys.modules[
            "telekinesis.illusion.core.synthetic_data_generator"
        ].SyntheticDataGenerator.return_value
        generator.generate.side_effect = lambda **kwargs: print(
            "Generation completed", flush=True
        )
        initializer = ModuleType("telekinesis.illusion.blender.initialization")
        initializer.clean_temp_dir = lambda: Path(temporary_cleanup_marker).touch()
        sys.modules[initializer.__name__] = initializer
        atexit.register(lambda: Path(cleanup_marker).touch())
        sys.path.insert(0, str(Path(script).parent))
        sys.argv = [script, "--no-preview"]
        runpy.run_path(script, run_name="__main__")
    """)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            harness,
            str(
                ROOT
                / "examples"
                / ("bat_scripts" if batch_wrapper else ".")
                / script_name
            ),
            str(cleanup_marker),
            str(temporary_cleanup_marker),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Generation completed" in result.stdout
    bypassed = batch_wrapper and sys.platform == "win32"
    assert cleanup_marker.is_file() is not bypassed
    assert temporary_cleanup_marker.is_file() is bypassed
