"""Run pytest and preserve its exit status across bpy's Windows teardown."""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest

if __name__ == "__main__":
    code = int(pytest.main(sys.argv[1:]))
    if sys.platform == "win32" and "bpy" in sys.modules:
        from telekinesis.illusion.blender.initialization import clean_temp_dir

        clean_temp_dir()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)
