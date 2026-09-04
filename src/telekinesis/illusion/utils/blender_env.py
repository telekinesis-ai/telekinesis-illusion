"""Keep Blender's user extensions off ``sys.path`` when using standalone ``bpy``.

Importing the pip ``bpy`` module makes Blender prepend the *user* extensions
site-packages -- on Windows that is::

    %APPDATA%/Blender Foundation/Blender/4.2/extensions/.local/lib/python3.11/site-packages

-- ahead of the active environment's ``site-packages``. That directory is where
Blender unpacks the wheels bundled with installed extensions, including our own
``illusion_randomizer`` add-on (see ``blender_extension/``). Those wheels are
pinned for Blender's own interpreter, which still ships NumPy 1.x, so its
``scipy==1.11.4`` is built against the NumPy 1 C ABI. Once it shadows the
environment's ``scipy``, any import of ``trimesh``/``skimage`` under our
NumPy 2 environment dies with::

    ValueError: numpy.dtype size changed, may indicate binary incompatibility.

Both pins are correct for their own interpreter, so the fix is isolation rather
than alignment: point ``BLENDER_USER_EXTENSIONS`` at a scratch directory before
``bpy`` is imported. Blender then finds no user extensions to expose and the
environment's packages win.

The same shadowing also hits *this package*. That extensions directory is where
the ``illusion_randomizer`` add-on unpacks its bundled ``telekinesis-illusion``
wheel, i.e. a second, older copy of the very code you are running.
``telekinesis`` and ``telekinesis.illusion`` are namespace packages, so their
``__path__`` is recomputed from ``sys.path`` and the extension copy lands first;
``telekinesis.illusion.utils`` is a *regular* package, so that first match wins
outright with no namespace merging. Any module the stale copy lacks then fails
with ``ModuleNotFoundError``, and any module it does have silently runs the old
version.

**Therefore: every module that imports ``bpy`` (or ``blenderproc``, which pulls
``bpy`` in) must call** ``isolate_user_extensions()`` **above that import.** The
call is idempotent and imports only the standard library, so it is safe and
cheap to repeat. Doing it after ``bpy`` is already imported is too late: by then
``telekinesis.illusion.utils`` is bound to the stale directory. This convention
cannot be enforced from a package ``__init__`` because ``telekinesis.illusion``
is intentionally a namespace package.
"""

import os
import sys
import tempfile
from pathlib import Path

# Stable per-machine scratch root, so repeated runs reuse the same (empty)
# extensions directory instead of littering the temp dir.
_ISOLATED_EXTENSIONS_DIR = (
    Path(tempfile.gettempdir()) / "illusion_blender_extensions"
)


def _is_inside_blender() -> bool:
    """True when running under a real Blender process rather than pip ``bpy``.

    The standalone module reports an empty ``binary_path``; a genuine Blender
    build reports the path to its executable. Inside Blender the user
    extensions *must* stay on ``sys.path`` -- that is how our own add-on
    resolves its bundled wheels.
    """
    bpy = sys.modules.get("bpy")
    return bool(bpy is not None and getattr(bpy.app, "binary_path", ""))


def isolate_user_extensions() -> None:
    """Redirect Blender's user extensions directory away from the real one.

    Idempotent, cheap (never imports ``bpy``) and a no-op when ``bpy`` has
    already been imported or ``BLENDER_USER_EXTENSIONS`` is already set, so an
    explicit override by the caller always wins.
    """
    if _is_inside_blender() or "BLENDER_USER_EXTENSIONS" in os.environ:
        return

    _ISOLATED_EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["BLENDER_USER_EXTENSIONS"] = str(_ISOLATED_EXTENSIONS_DIR)

    # If bpy slipped in first, the env var comes too late -- drop the offending
    # entries so the environment's packages are still the ones that resolve.
    if "bpy" in sys.modules:
        drop_user_extensions_from_path()


def drop_user_extensions_from_path() -> None:
    """Remove Blender user-resource directories that ``bpy`` added to ``sys.path``."""
    bpy = sys.modules.get("bpy")
    if bpy is None or _is_inside_blender():
        return

    user_root = os.path.normcase(
        os.path.abspath(bpy.utils.resource_path("USER"))
    )
    sys.path[:] = [
        p
        for p in sys.path
        if not os.path.normcase(os.path.abspath(p)).startswith(user_root)
    ]
