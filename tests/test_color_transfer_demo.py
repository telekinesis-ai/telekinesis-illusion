"""Exercise the demo with real images and transfers, without opening a GUI."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock, call

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("python_color_transfer")


@pytest.fixture
def demo():
    path = Path(__file__).resolve().parents[1] / "examples" / "demo.py"
    spec = importlib.util.spec_from_file_location("color_transfer_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target_shape", [(8, 10, 3), (32, 40, 3)])
def test_cli_previews_original_sizes(demo, tmp_path, monkeypatch, target_shape):
    rng = np.random.default_rng(42)
    source = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
    target = rng.integers(0, 256, target_shape, dtype=np.uint8)
    source_path = tmp_path / "source image.png"
    target_path = tmp_path / "target image.png"
    assert cv2.imwrite(str(source_path), source)
    assert cv2.imwrite(str(target_path), target)
    monkeypatch.setattr(
        sys, "argv", ["demo.py", str(source_path), str(target_path)]
    )
    for name in ("namedWindow", "imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, Mock())

    demo.demo()

    titles = ["Source", "Target", "Mean/std", "Lab", "PDF (regrain)"]
    assert cv2.namedWindow.call_args_list == [
        call(title, cv2.WINDOW_AUTOSIZE) for title in titles
    ]
    assert [args.args[0] for args in cv2.imshow.call_args_list] == titles
    previews = [args.args[1] for args in cv2.imshow.call_args_list]
    np.testing.assert_array_equal(previews[0], source)
    np.testing.assert_array_equal(previews[1], target)
    for result in previews[2:]:
        assert result.shape == source.shape
        assert result.dtype == np.uint8
    assert not np.array_equal(previews[-1], source)
    cv2.waitKey.assert_called_once_with(0)
    cv2.destroyAllWindows.assert_called_once_with()


@pytest.mark.parametrize("invalid_role", ["source", "target"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_cli_rejects_unreadable_images(
    demo, tmp_path, monkeypatch, capsys, invalid_role, corrupt
):
    valid = tmp_path / "valid.png"
    invalid = tmp_path / "invalid.png"
    assert cv2.imwrite(str(valid), np.zeros((8, 8, 3), dtype=np.uint8))
    if corrupt:
        invalid.write_text("not an image")
    paths = [
        invalid if role == invalid_role else valid
        for role in ("source", "target")
    ]
    monkeypatch.setattr(sys, "argv", ["demo.py", *map(str, paths)])

    with pytest.raises(SystemExit) as error:
        demo.demo()

    assert error.value.code == 2
    assert f"Could not read {invalid_role} image" in capsys.readouterr().err
