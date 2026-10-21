# BlenderProc attribution

The Blender helpers in this directory are adapted from a modified
BlenderProc distribution in telekinesis-illusion, originally developed by DLR
and BlenderProc contributors. See [LICENSE.upstream](LICENSE.upstream) for the
upstream license.
The algorithms for wrappers, reference restoration, undo, rendering, physics,
camera operations, filters and postprocessing are retained. Imports and resource
paths are local; environment installation/launching is intentionally omitted.
V-HACD retains its separate license notice in external/vhacd/decompose.py.
Its upstream permission note states: "We specifically asked for the permission
to use this inside BlenderProc. All rights are still with Khaled Mamou."

Source snapshot: telekinesis-illusion commit
`43d2eb487aac4c1fd264f9c65968274ea66ccaba`, including its tracked BlenderProc
directory. This is an adapted copy, not an unmodified upstream release.
Reference tracking still uses the Struct weak-reference set and name-based
rebinding. Undo additionally handles zero or multiple Background wrappers and
skips wrappers whose worlds were already removed by scene cleanup. The denoiser
accepts omitted settings used by the flow and segmentation helpers. Recoloring
segmentation uses object-level material overrides for linked mesh instances.

Upstream references:

- [BlenderProc documentation](https://dlr-rm.github.io/BlenderProc/)
- [BlenderProc repository](https://github.com/DLR-RM/BlenderProc)
- `../sampler/upper_region_sampler.py` is adapted from
  [UpperRegionSampler.py](https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/sampler/UpperRegionSampler.py).
- `../randomizer/randomizer_node.py` pose sampling is adapted from
  [ObjectPoseSampler.py](https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/object/ObjectPoseSampler.py)
  and [OnSurfaceSampler.py](https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/object/OnSurfaceSampler.py).

## Runtime names

The adapted helpers use Illusion names. Update external callers and saved
scene properties that use the former names; compatibility aliases are not
provided.

| Purpose | Former name | Current name |
| --- | --- | --- |
| Random seed environment variable | `BLENDER_PROC_RANDOM_SEED` | `ILLUSION_RANDOM_SEED` |
| HDF5 version dataset | `blender_proc_version` | `illusion_version` |
| Scene cleanup preservation property | `bproc_preserve` | `illusion_preserve` |
| Segmentation node ownership property | `blenderproc_segmentation_output_key` | `illusion_segmentation_output_key` |
| Default convex decomposition cache | `blenderproc_resources/decomposition_cache` | `illusion_resources/decomposition_cache` |
| `select_upper_region` mesh keyword argument | `bproc_mesh_object` | `mesh_object` |

Existing HDF5 files and cache directories are not rewritten or moved. New
HDF5 files use the current version key. To reuse a previous decomposition
cache, pass its path explicitly as `cache_dir` or move it to the new default
location.
