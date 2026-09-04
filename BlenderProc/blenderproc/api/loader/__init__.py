"""Loader API - lazily imported per-function.

Each loader submodule below is independent and some (AMASS, Matterport3D via
FaceSlicer, URDF, ...) pull in heavy/exotic third-party deps (sklearn, etc.)
that most callers never touch. Importing all of them eagerly here means every
`import blenderproc` pays for every loader's dependencies, even when only one
is ever used - this bit us running inside actual Blender, which ships its own
fixed numpy version incompatible with newer scipy/sklearn releases that only
AMASS/Matterport3D-style loaders need. Deferring via module __getattr__
(PEP 562) keeps `bproc.loader.load_x(...)` working identically, just resolved
on first access instead of at import time.
"""

import importlib

_LAZY_ATTRS = {
    "load_AMASS": "blenderproc.python.loader.AMASSLoader",
    "load_blend": "blenderproc.python.loader.BlendLoader",
    "load_bop_objs": "blenderproc.python.loader.BopLoader",
    "load_bop_scene": "blenderproc.python.loader.BopLoader",
    "load_bop_intrinsics": "blenderproc.python.loader.BopLoader",
    "load_ccmaterials": "blenderproc.python.loader.CCMaterialLoader",
    "load_front3d": "blenderproc.python.loader.Front3DLoader",
    "load_haven_mat": "blenderproc.python.loader.HavenMaterialLoader",
    "load_ikea": "blenderproc.python.loader.IKEALoader",
    "load_matterport3d": "blenderproc.python.loader.Matterport3DLoader",
    "load_obj": "blenderproc.python.loader.ObjectLoader",
    "load_pix3d": "blenderproc.python.loader.Pix3DLoader",
    "load_replica": "blenderproc.python.loader.ReplicaLoader",
    "load_replica_segmented_mesh": "blenderproc.python.loader.ReplicaLoader",
    "load_scenenet": "blenderproc.python.loader.SceneNetLoader",
    "load_shapenet": "blenderproc.python.loader.ShapeNetLoader",
    "load_suncg": "blenderproc.python.loader.SuncgLoader",
    "load_texture": "blenderproc.python.loader.TextureLoader",
    "get_random_world_background_hdr_img_path_from_haven": "blenderproc.python.loader.HavenEnvironmentLoader",
    "load_urdf": "blenderproc.python.loader.URDFLoader",
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
