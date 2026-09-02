"""Constructor API - lazily imported, see api/loader/__init__.py's docstring
for why (RandomRoomConstructor pulls in FaceSlicer -> sklearn/scipy)."""

import importlib

_LAZY_ATTRS = {
    "construct_random_room": "blenderproc.python.constructor.RandomRoomConstructor",
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
