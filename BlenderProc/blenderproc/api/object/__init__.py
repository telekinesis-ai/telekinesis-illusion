"""Object API - lazily imported per-function, see api/loader/__init__.py's
docstring for why (FaceSlicer needs sklearn/scipy, which conflicts with the
numpy version actual Blender ships internally - not needed unless
extract_floor/slice_faces_with_normals is actually called)."""

import importlib

_LAZY_ATTRS = {
    "extract_floor": "blenderproc.python.object.FaceSlicer",
    "slice_faces_with_normals": "blenderproc.python.object.FaceSlicer",
    "sample_poses": "blenderproc.python.object.ObjectPoseSampler",
    "merge_objects": "blenderproc.python.object.ObjectMerging",
    "replace_objects": "blenderproc.python.object.ObjectReplacer",
    "sample_poses_on_surface": "blenderproc.python.object.OnSurfaceSampler",
    "simulate_physics_and_fix_final_poses": "blenderproc.python.object.PhysicsSimulation",
    "simulate_physics": "blenderproc.python.object.PhysicsSimulation",
    "simulate_physics_and_persist_all_frames": "blenderproc.python.object.PhysicsSimulation",
    "get_all_mesh_objects": "blenderproc.python.types.MeshObjectUtility",
    "convert_to_meshes": "blenderproc.python.types.MeshObjectUtility",
    "create_from_blender_mesh": "blenderproc.python.types.MeshObjectUtility",
    "create_with_empty_mesh": "blenderproc.python.types.MeshObjectUtility",
    "create_primitive": "blenderproc.python.types.MeshObjectUtility",
    "disable_all_rigid_bodies": "blenderproc.python.types.MeshObjectUtility",
    "create_bvh_tree_multi_objects": "blenderproc.python.types.MeshObjectUtility",
    "compute_poi": "blenderproc.python.types.MeshObjectUtility",
    "scene_ray_cast": "blenderproc.python.types.MeshObjectUtility",
    "create_from_point_cloud": "blenderproc.python.types.MeshObjectUtility",
    "create_empty": "blenderproc.python.types.EntityUtility",
    "delete_multiple": "blenderproc.python.types.EntityUtility",
    "convert_to_entities": "blenderproc.python.types.EntityUtility",
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
