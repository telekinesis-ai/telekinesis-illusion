"""Optional model paths for material integration checks."""

from pathlib import Path


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
