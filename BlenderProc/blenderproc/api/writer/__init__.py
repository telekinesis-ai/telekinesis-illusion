"""Writer API - lazily imported per-function, see api/loader/__init__.py's
docstring for why (BopWriterUtility needs trimesh, WriterUtility needs h5py,
GifWriterUtility needs imageio/progressbar - none needed unless that specific
writer is actually called)."""

import importlib

_LAZY_ATTRS = {
    "write_gif_animation": "blenderproc.python.writer.GifWriterUtility",
    "write_bop": "blenderproc.python.writer.BopWriterUtility",
    "write_coco_annotations": "blenderproc.python.writer.CocoWriterUtility",
    "write_hdf5": "blenderproc.python.writer.WriterUtility",
}


def __getattr__(name):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))
