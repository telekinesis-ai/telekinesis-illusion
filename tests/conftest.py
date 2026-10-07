"""Shared Blender lifecycle and optional model paths for integration tests."""

import gc
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def initialized_blender():
    """Initialize BlenderProc once across all in-process integration tests."""
    from telekinesis.illusion.utils.blender_env import isolate_user_extensions

    isolate_user_extensions()
    import blenderproc as bproc

    bproc.init()
    yield bproc
    bproc.clean_up()


@pytest.fixture
def context(initialized_blender, monkeypatch):
    """Provide a fresh scene without repeating BlenderProc initialization."""
    from telekinesis.illusion.core.context import Context

    bproc = initialized_blender
    # Release cycles from earlier tests before BlenderProc scans its weak
    # references during physics undo operations.
    gc.collect()
    # Production Context owns process initialization. Within pytest, the
    # session fixture owns it and each Context only needs a fresh scene.
    monkeypatch.setattr(
        bproc, "init", lambda: bproc.clean_up(clean_up_camera=True)
    )
    try:
        yield Context()
    finally:
        # Retain BlenderProc's process resources until session teardown.
        bproc.clean_up(clean_up_camera=True)


def pytest_addoption(parser):
    parser.addoption(
        "--material-model",
        action="append",
        default=[],
        help="Additional OBJ/GLB path for material integration checks (repeatable).",
    )


def pytest_generate_tests(metafunc):
    if "material_model_path" in metafunc.fixturenames:
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "assets/models/mechanical_parts/gearwheel_2.glb",
            root / "assets/models/bins/plastic_bin_3.glb",
            *map(Path, metafunc.config.getoption("--material-model")),
        ]
        metafunc.parametrize("material_model_path", paths, ids=lambda p: p.name)
