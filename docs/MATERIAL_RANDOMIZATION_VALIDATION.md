# Material randomization validation

Validated on 2026-10-06 with Python 3.11 and the repository's external
`bpy 4.2.17 LTS` / BlenderProc environment.

## Current scope

Input models must contain **one mesh object without empties**. The experimental
tree importer, node selection, extra mesh wrappers and tree copying code have
been removed, along with their tests and examples. Previous reports describing
those features are superseded.

The original mesh loader and single-mesh Object lifecycle are restored. Material
slot preservation, procedural recipes, existing PBR sampling and independent
materials on linked instances remain supported. No separate asset registry or
registration API is introduced.

## Results

- **67 tests passed; one subprocess smoke test failed.** The passing tests cover
  material behavior, COCO masks, all four supplied models and two new regressions
  for imported/linked mesh references across repeated undo and physics cycles.
- `quickstart_parts_in_bin.py` completed three scene iterations with its original
  physics settings and wrote **six images and 14 annotations**. Validation used
  128 x 128 rendering at eight samples and seed 42. The viewer was replaced by
  checks on the generated COCO JSON and image files. The scene generation,
  physics, material randomization, rendering and writing code ran normally.
  The reported `StructRNA of type Object has been removed` error did not recur.
- The preview produced its inventory and completion marker, but its process
  exited with **3221225477 (`0xC0000005`, access violation)**. The subprocess test
  now requires a zero exit code, so this is recorded as a failure.
- The main pytest process and quickstart process also crashed during shutdown
  after recording their results. This is not a clean end-to-end pass.
- A separate process performing only BlenderProc import reproduced the same
  exit status, without model loading, material assignment or physics. Python
  alone and `import bpy` alone exited normally. The native shutdown failure
  therefore persists independently of the removed model functionality; its
  precise native cause remains unresolved.
- Ruff checks pass for the new material, test and preview modules. Formatting,
  Python compilation and `git diff --check` pass for the revised files.

## Supplied models

| Model | Imported slots |
| --- | --- |
| panel_mount_pilot_light.obj | 9 |
| panel_mount_pilot_light.glb | 9 |
| gearwheel_2.glb | 0 |
| plastic_bin_3.glb | 0 |

These are single-mesh inputs. Tests verify selected-slot and all-slot assignment
where slots exist, whole-object assignment, and the legacy preprocessing/PBR
workflow. Slot-less models receive a slot in whole-object mode.

## Local evidence

Artifacts are under the ignored directory
`output/material_preview/single_mesh_validation/`:

- `tests.log`, `pytest_result.txt`, `process_status.json`: 67 passed, one failed;
  pytest returned 1, and the native process subsequently crashed.
- `quickstart.log`, `quickstart_result.json`, `quickstart_process_status.json`:
  completed scene generation and subsequent process-exit status.
- `quickstart_dataset/`: COCO annotations and six rendered images.
- `baseline_process_status.json`, `blenderproc_baseline_status.json`: isolated
  Python, bpy and BlenderProc process results.

## Test environment

A local `.venv` reuses the installed Blender environment and provides its missing
`loguru` dependency. The Windows sandbox cannot access pytest's owner-only
scratch directories, so an invocation-only fixture creates temporary paths
under the workspace with inherited permissions. Tests themselves use the normal
pytest `tmp_path` interface; none were skipped.

Windows crash dialogs were disabled only for the validation processes using
`SetErrorMode`, so native crashes return an exit status instead of displaying a
blocking dialog. No application shutdown workaround or forced successful exit
was added. One existing pycocotools/NumPy deprecation warning remains.
