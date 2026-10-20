# BlenderProc attribution

The Blender helpers in this directory are adapted from the locally modified
BlenderProc distribution in telekinesis-illusion, originally developed by DLR
and BlenderProc contributors. See LICENSE.BlenderProc for the upstream license.
The algorithms for wrappers, reference restoration, undo, rendering, physics,
camera operations, filters and postprocessing are retained. Imports and resource
paths are local; environment installation/launching is intentionally omitted.
V-HACD retains its separate license notice in external/vhacd/decompose.py.

Source snapshot: telekinesis-illusion commit
`43d2eb487aac4c1fd264f9c65968274ea66ccaba`, including its tracked BlenderProc
directory. This is an adapted copy, not an unmodified upstream release.
Reference tracking still uses the Struct weak-reference set and name-based
rebinding. Undo additionally handles zero or multiple Background wrappers and
skips wrappers whose worlds were already removed by scene cleanup. The denoiser
accepts omitted settings used by the flow and segmentation helpers. Recoloring
segmentation uses object-level material overrides for linked mesh instances.
