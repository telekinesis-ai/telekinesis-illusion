"""Render failures must preserve camera frames and the process output streams."""

import errno
import os
import sys
from contextlib import ExitStack, contextmanager, redirect_stdout
from types import SimpleNamespace

import pytest

from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()

from blenderproc.python.renderer import RendererUtility as renderer
from blenderproc.python.utility import Utility as utility_module
from blenderproc.python.utility.Utility import stdout_redirected


@pytest.mark.parametrize("views", [1, 2])
@pytest.mark.parametrize("failure_stage", ["render", "stdout_cleanup"])
def test_render_failure_preserves_camera_frames(
    context, monkeypatch, tmp_path, views, failure_stage
):
    scene = renderer.bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = views
    attempts = []

    def render(**kwargs):
        attempts.append((scene.frame_start, scene.frame_end))
        if len(attempts) <= 2 and failure_stage == "render":
            raise RuntimeError("Error: run out of memory!")

    @contextmanager
    def redirect(**kwargs):
        yield
        if len(attempts) <= 2 and failure_stage == "stdout_cleanup":
            raise OSError("Output flush failed")

    monkeypatch.setattr(renderer, "get_all_blender_mesh_objects", lambda: [1])
    monkeypatch.setattr(renderer, "stdout_redirected", redirect)
    monkeypatch.setattr(
        renderer,
        "bpy",
        SimpleNamespace(
            context=renderer.bpy.context,
            ops=SimpleNamespace(render=SimpleNamespace(render=render)),
        ),
    )

    for _ in range(2):
        error = RuntimeError if failure_stage == "render" else OSError
        with pytest.raises(error):
            renderer.render(
                output_dir=str(tmp_path), output_key=None, return_data=False
            )
        assert (scene.frame_start, scene.frame_end) == (0, views)

    assert (
        renderer.render(
            output_dir=str(tmp_path), output_key=None, return_data=False
        )
        == {}
    )
    assert attempts == [(0, views - 1)] * 3
    assert (scene.frame_start, scene.frame_end) == (0, views)


@pytest.mark.parametrize("fail", [False, True])
def test_redirect_restores_python_and_native_output(tmp_path, fail):
    original = tmp_path / "original.txt"
    redirected = tmp_path / "redirected.txt"
    with (
        original.open("w", encoding="utf-8") as stream,
        redirect_stdout(stream),
    ):
        print("before", flush=True)
        try:
            with stdout_redirected(str(redirected)) as saved:
                print("python", flush=True)
                os.write(stream.fileno(), b"native\n")
                print("saved", file=saved, flush=True)
                if fail:
                    raise RuntimeError("render failed")
        except RuntimeError as error:
            assert str(error) == "render failed"
        assert sys.stdout is stream
        print("after", flush=True)
        os.write(stream.fileno(), b"native after\n")
    assert original.read_text().splitlines() == [
        "before",
        "saved",
        "after",
        "native after",
    ]
    assert redirected.read_text().splitlines() == ["python", "native"]


def test_redirect_restores_descriptor_when_flush_fails(tmp_path):
    original = tmp_path / "original.txt"
    redirected = tmp_path / "redirected.txt"

    def fail_flush():
        raise OSError("Output flush failed")

    with (
        original.open("w", encoding="utf-8") as stream,
        redirect_stdout(stream),
    ):
        with (
            pytest.raises(OSError, match="Output flush failed"),
            stdout_redirected(str(redirected)),
        ):
            # Fail the stream's final flush, after rendering returned.
            sys.stdout.flush = fail_flush
        assert sys.stdout is stream
        print("after", flush=True)
        os.write(stream.fileno(), b"native after\n")
    assert original.read_text().splitlines() == ["after", "native after"]


@pytest.mark.parametrize("failure_stage", [None, "render", "flush", "open"])
def test_redirect_releases_owned_descriptors(
    tmp_path, monkeypatch, failure_stage
):
    duplicates = []
    duplicate = os.dup

    def track_duplicate(fd):
        result = duplicate(fd)
        duplicates.append(result)
        return result

    def fail_flush():
        raise OSError("injected failure")

    monkeypatch.setattr(utility_module.os, "dup", track_duplicate)
    destination = (
        str(tmp_path / "missing" / "output.txt")
        if failure_stage == "open"
        else os.devnull
    )
    with (
        (tmp_path / "stdout.txt").open("w") as original,
        redirect_stdout(original),
    ):
        try:
            with stdout_redirected(destination):
                if failure_stage == "render":
                    raise RuntimeError("injected failure")
                if failure_stage == "flush":
                    sys.stdout.flush = fail_flush
        except (RuntimeError, OSError):
            assert failure_stage is not None
        else:
            assert failure_stage is None
        assert sys.stdout is original
        assert duplicates
        for fd in duplicates:
            with pytest.raises(OSError) as error:
                os.fstat(fd)
            assert error.value.errno == errno.EBADF
        print("restored", flush=True)
    assert (tmp_path / "stdout.txt").read_text() == "restored\n"


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows console regression"
)
def test_windows_console_redirect_does_not_leak():
    with ExitStack() as stack:
        try:
            console = stack.enter_context(
                open("CONOUT$", "w", encoding="utf-8")
            )
        except OSError:
            pytest.skip(
                "Run with an attached Windows console to exercise this leak."
            )
        stack.enter_context(redirect_stdout(console))
        # Exceed the 8192-descriptor limit that aborted long generation runs.
        for _ in range(10000):
            with stdout_redirected() as saved:
                saved_fd = saved.fileno()
                redirected_fd = sys.stdout.fileno()
                print("suppressed", flush=True)
            for fd in (saved_fd, redirected_fd):
                with pytest.raises(OSError) as error:
                    os.fstat(fd)
                assert error.value.errno == errno.EBADF
            assert sys.stdout is console
