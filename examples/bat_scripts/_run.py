"""Run standard examples with the Windows batch shutdown workaround."""

import os
from pathlib import Path
import runpy
import sys


def run_example(filename: str) -> None:
    """Run an example and bypass Windows native teardown after success.

    Args:
        filename: Name of the script in the parent examples directory.
    """
    script = Path(__file__).resolve().parents[1] / filename
    runpy.run_path(str(script), run_name="__main__")
    finish_batch_run()


def finish_batch_run() -> None:
    """Clean temporary Blender files and bypass Windows native teardown."""
    if sys.platform == "win32":
        from telekinesis.illusion.blender.initialization import clean_temp_dir

        clean_temp_dir()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
