"""Fresh bpy scenes; BlenderProc imports are prohibited in all tests."""

import gc
import sys
from pathlib import Path
import importlib.abc
import pytest


class RejectBlenderProc(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "blenderproc" or fullname.startswith("blenderproc."):
            raise RuntimeError(
                "The new implementation must not import BlenderProc"
            )


sys.meta_path.insert(0, RejectBlenderProc())


@pytest.fixture
def context():
    from telekinesis.illusion.core.scene import Scene

    gc.collect()
    scene = Scene()
    yield scene
    scene.clean_up()
    gc.collect()


def pytest_addoption(parser):
    parser.addoption("--material-model", action="append", default=[])


def pytest_generate_tests(metafunc):
    if "material_model_path" in metafunc.fixturenames:
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "assets/models/mechanical_parts/gearwheel_2.glb",
            root / "assets/models/bins/plastic_bin_3.glb",
            *map(Path, metafunc.config.getoption("--material-model")),
        ]
        metafunc.parametrize("material_model_path", paths, ids=lambda p: p.name)
