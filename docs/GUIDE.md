# `telekinesis-illusion` — Guide

Detailed reference for assets, project structure, the worker pipeline, dataset formats, and the viewer. See the main [README](../README.md) for installation and quickstart.

## Assets

`telekinesis-illusion` ships with a small collection of default assets and models, so no separate download is required to run the examples. If you want to use your own assets, organize them under a directory with the same layout (`hdris/`, `materials/`, `models/`) and point `telekinesis-illusion` at it - see [`telekinesis.illusion.utils.assets.resolve_asset_dir`](../src/telekinesis/illusion/utils/assets.py) for how the asset directory is resolved (explicit path, `metadata.asset_directory` in a spec, or the bundled default).

## Project Structure

```
└─ illusion/
   ├── BlenderProc/
   │
   ├── assets/
   │   ├── hdris/
   │   ├── materials/
   │   └── models/
   │
   ├── configs/                      # spec YAMLs for the workers
   │   └── example_bin_picking_gearwheel_2.yaml
   │
   ├── examples/                     # runnable entry points
   │   ├── quickstart_flying_things.py
   │   ├── quickstart_parts_in_bin.py
   │   ├── generate_synthetic_data_with_bin_picking_worker.py
   │   └── view_dataset.py
   │
   ├── src/telekinesis/illusion/
   │   ├── core/
   │   │   ├── __init__.py
   │   │   ├── context.py
   │   │   └── synthetic_data_generator.py
        .
        .
        .
   ├── pyproject.toml
   └── README.md
```

## Worker
Illusion-workers generate data shards for a specific use case. A shard is a "batch" of the final dataset - a fixed number of images plus their annotation JSON - and shards are merged into a single annotation file at the end of the run.

Each use case is driven by a spec YAML under [`configs/`](../configs/) and a runnable entry script under [`examples/`](../examples/). `telekinesis-illusion` currently supports the following workers:
 - `bin_picking_worker`: generates a dataset for bin-picking use cases (parts in bins). IDs and category names must be consistent across datasets that are combined later.

### Running a worker

1. Pick or write a spec YAML, e.g. [`configs/example_bin_picking_gearwheel_2.yaml`](../configs/example_bin_picking_gearwheel_2.yaml). Each YAML is a self-contained dataset definition.
2. Run:

   ```bash
   python examples/generate_synthetic_data_with_bin_picking_worker.py --spec-file example_bin_picking_gearwheel_2.yaml
   ```

   `--spec-file` defaults to `example_bin_picking_gearwheel_2.yaml`; bare filenames resolve against `configs/`, or pass an absolute/relative path to a spec elsewhere. Add `--no-preview` to skip opening the merged dataset in the viewer afterward.

   This calls `BinPickingWorker(spec_path).generate()` followed by `worker.merge_shards(preview=...)`. The merged COCO annotations land at `<metadata.base_output_directory>/<metadata.dataset_name>/merged_coco_annotations.json`. If the spec sets `output.dataset_format` (`coco` or `yolo`), `merge_shards()` also runs a `DatasetConverter` split automatically - see [Dataset](#dataset) below.
3. (Optional) For a quick smoke test, lower `metadata.num_images` and `shard.size` in the chosen YAML (e.g. both to `10`) so a full generate + merge cycle finishes in minutes.

To add a new use case, add a new spec YAML under `configs/` and a corresponding entry script under `examples/` that instantiates the appropriate worker.

### Overview of the Worker

What `BinPickingWorker.__init__` and `generate()` do, in order:

1. Load the spec YAML; compute `num_shards = ceil(metadata.num_images / shard.size)`; read `output.max_size_gb` (default `10`).
2. Build a `CameraConfig` from `camera:` and create a `Context`.
3. Register every entry in `models:` and `distractors:` on the context (name, category id/name, instance min/max, simulation flags, scale, preprocess flag). If you want to exclude a model from the COCO annotations, specify the `id` as `None` and the `category_name` as `distractor`. For the bins, the worker uses the supercategory for applying the randomization.
4. Wire up randomizers in fixed order: object-instance, container-instance (always 1), object-pose (sampled on the visible container's upper face via `upper_region_sampler`, or on a grid via `pose_sampling.strategy: grid`; parameters come from `pose_sampling.params` in the spec), distractor-instance, distractor-pose, material randomizers (per supercategory, configurable via `material_randomizer`), background (configurable via `background_randomizer`), camera-pose (sampler and view count are spec-driven via `camera_pose_randomizer.*`).
5. Per shard: create `<base_output_directory>/<dataset_name>/<shard_name>/` and run `SyntheticDataGenerator.generate(num_images=<shard size>, simulate_physics=<physics_simulator.active>, clean_up_scene=False)`. Each shard requests `min(shard.size, images still owed)`, so the total across all shards always equals `metadata.num_images` exactly.
6. After each shard, sum the dataset directory size and stop early if it exceeds `output.max_size_gb`. Once the loop ends, `merge_shards()` runs `CocoShardMerger` to produce a single merged COCO annotation file.

### Output dataset format and shard layout

- Output format is **COCO with RLE-encoded instance-segmentation masks** (`metadata.annotation_format: coco_instances_rle`). Written by `CocoWriter` in [src/telekinesis/illusion/writer/writer.py](../src/telekinesis/illusion/writer/writer.py), which delegates to `bproc.writer.write_coco_annotations(..., mask_encoding_format="rle", color_file_format="PNG")`.
- A **shard** is a self-contained mini-COCO dataset on disk: one `coco_annotations.json` plus an `images/` subdirectory of rendered PNGs. Multiple shards let the worker checkpoint progress and stop early on size limits without losing finished work.
- `num_shards = ceil(metadata.num_images / shard.size)`; the total number of scenes generated across all shards always equals `metadata.num_images` exactly. Each scene yields `camera_pose_randomizer.number_of_views` rendered images; `shard.size` counts scenes, not images.
- Output root: `metadata.base_output_directory / metadata.dataset_name / <shard_name>`. If `base_output_directory` is empty/null, falls back to `./output/<dataset_name>` relative to the directory you run from.
- Shard directory name comes from `output.shard_name_template`; placeholders are `{date}` (`YYYYMMDD_HHMMSS`) and `{uuid}` (8-hex). Default: `shard_{date}_{uuid}`.
- Generation halts after the first shard whose total dataset-dir size is `>= output.max_size_gb` (GiB).
- **Shards must be merged at the end.** `merge_shards()` runs `CocoShardMerger` (in [src/telekinesis/illusion/dataset/dataset_generator.py](../src/telekinesis/illusion/dataset/dataset_generator.py)) which discovers every sub-directory containing a `coco_annotations.json`, validates that `categories` (id + name) match across shards, re-indexes image and annotation IDs to be globally unique, rewrites each `file_name` to `<shard_dirname>/images/<frame>.png`, and writes a single `merged_coco_annotations.json` at the dataset root. The original per-shard files are left in place.
- Consumers should load `merged_coco_annotations.json`; image paths inside it resolve relative to the dataset root.

Final directory layout after `generate()` + `merge_shards()`:

```
<base_output_directory>/
└── <metadata.dataset_name>/
    ├── merged_coco_annotations.json        # written by merge_shards()
    ├── shard_<date>_<uuid>/                # one per shard
    │   ├── coco_annotations.json           # per-shard COCO file
    │   └── images/
    │       ├── 000000.png
    │       ├── 000001.png
    │       └── ...
    ├── shard_<date>_<uuid>/
    │   ├── coco_annotations.json
    │   └── images/
    │       └── ...
    └── ...
```

### Spec YAML reference

Reference shape: [`configs/example_bin_picking_gearwheel_2.yaml`](../configs/example_bin_picking_gearwheel_2.yaml).

| Key | Type | Purpose / allowed values |
|---|---|---|
| `metadata.dataset_name` | string | Dataset directory name under the output root. |
| `metadata.package_version` | string | Recorded in the manifest, no runtime effect. |
| `metadata.use_case` | string | Recorded only (e.g. `instance_segmentation`). |
| `metadata.annotation_format` | string | Recorded only (e.g. `coco_instances_rle`). |
| `metadata.num_images` | int | Total scenes generated across all shards. |
| `metadata.base_output_directory` | string \| null | Output root; null/empty falls back to `./output/<dataset_name>` relative to the run directory. |
| `metadata.asset_directory` | string \| null | Directory holding `models/`, `hdris/`, `materials/`. See [Assets](#assets). |
| `shard.size` | int | Scenes per shard. |
| `min_number_visible_models` / `max_number_visible_models` | int | Bounds on visible target objects (supercategory `object`) per scene. |
| `models[].name` | string | Unique asset name. |
| `models[].id` | int | COCO `category_id`. Multiple entries may share an `id` to map several meshes to one category. |
| `models[].supercategory` | `object` \| `container` | Drives randomizer routing. |
| `models[].category_name` | string | COCO category name; must be consistent across entries that share an `id`. |
| `models[].path` | string | Model path relative to `metadata.asset_directory`. |
| `models[].instances.min` / `.max` | int | Per-scene instance count range. |
| `models[].simulation.active` | bool | Whether the asset participates in physics. Containers typically `false`. |
| `models[].simulation.collision_shape` | `CONVEX_HULL` \| `MESH` | Collider type. |
| `models[].scale` | float | Uniform scale. |
| `models[].preprocess_model` | bool | Run import-time preprocessing. |
| `pose_sampling.strategy` | `random` \| `grid` | Placement strategy for target objects on the container's upper face. |
| `pose_sampling.params.min_height` / `max_height` | float | Vertical band above the container's upper face within which object centres are sampled. |
| `pose_sampling.params.face_sample_range` | `[float, float]` | Fractional inset on the container's upper face used by `upper_region_sampler` (default `[0.25, 0.75]`). |
| `pose_sampling.params.grid.rows` / `.cols` / `.layers` | int | Grid dimensions, used when `strategy: grid`. |
| `pose_sampling.params.grid.layer_spacing` | float | Vertical spacing between grid layers. |
| `pose_sampling.params.grid.xy_jitter` | float | Random XY offset applied to each grid cell. |
| `pose_sampling.params.grid.z_rotation_range` | `[float, float]` | Bounds on per-instance Z rotation (degrees). |
| `pose_sampling.params.grid.shuffle` | bool | Whether grid cell assignment is shuffled. |
| `camera.field_of_view` | float | Radians. |
| `camera.clip_start` / `camera.clip_end` | float | Near/far clip planes. |
| `camera.pixel_aspect_x` / `pixel_aspect_y` | float | Pixel aspect ratio. |
| `camera.shift_x` / `shift_y` | float | Lens shift. |
| `camera.image_width` / `image_height` | int | Output resolution. |
| `renderer.image_format` | string | E.g. `PNG`. |
| `camera_pose_randomizer.sampler` | string | Camera-pose sampler key (e.g. `volume_sampler`). |
| `camera_pose_randomizer.number_of_views` | int | Rendered images per scene. |
| `camera_pose_randomizer.params.distance_range` | `[float, float]` | Min/max camera distance from the point of interest. |
| `camera_pose_randomizer.params.inplane_rot_min` / `inplane_rot_max` | float | Bounds on in-plane camera rotation (degrees). |
| `camera_pose_randomizer.params.point_of_interst` | `[float, float, float]` | World-space point the camera looks at. (Key spelled as in the specs; copy verbatim.) |
| `min_number_visible_distractors` / `max_number_visible_distractors` | int | Bounds on visible distractors per scene. |
| `distractors[]` | list | Same shape as `models[]`; `supercategory: distractor` and `id` may be `None`. Distractors are optional - omit the list to disable. |
| `instance_randomizer.<role>.min` / `.max` | int | Per-supercategory override of visible-instance bounds (`part`, `container`, `distractor`). |
| `material_randomizer.<role>` | list of string | Material type tags per supercategory (default: `part: [metal]`, `container: [plastic]`, `distractor: [metal]`). |
| `background_randomizer.categories` | list of string | HDRI categories to sample the background from. |
| `physics_simulator.active` | bool | Whether to run a physics simulation before rendering each scene. |
| `physics_simulator.min_simulation_time_range` / `max_simulation_time_range` | `[float, float]` | Simulated-seconds range sampled per scene. |
| `output.shard_name_template` | string | Supports `{date}` and `{uuid}`. |
| `output.write_manifest` | bool | Whether to write a manifest. |
| `output.max_size_gb` | number | Early-stop threshold for total dataset-dir size (default `10`). |
| `output.dataset_format` | `coco` \| `yolo` \| null | If set, `merge_shards()` automatically runs a `DatasetConverter` split into this format. |
| `output.train_val_tes_ratio` | `[float, float, float]` | Train/valid/test split ratios, used when `output.dataset_format` is set (default `[0.7, 0.2, 0.1]`). |
| `output.stratify` | bool | Whether the split is stratified by dominant category (default `true`). |
| `output.seed` | int | RNG seed for the split (default `42`). |

### Common modifications

- **Add a new target part** - append an entry to `models:` with `supercategory: object`, a unique `name`, a `.glb` `path`, an `id` + `category_name` (reuse an existing id to fold into an existing class, or pick a new id for a new class), and the `instances`, `simulation`, `scale`, `preprocess_model` fields.
- **Map several meshes to one category** - give every entry the same `id` and `category_name`; the worker registers them as separate assets but COCO groups them under one category.
- **Swap or extend the bin** - add/replace an entry with `supercategory: container`, `simulation.active: false`, `collision_shape: MESH`. The worker always picks exactly one container per scene.
- **Distractors vs targets** - distractors live in the top-level `distractors:` list, use `supercategory: distractor`, and may set `id: None`. Distractor counts use `min/max_number_visible_distractors`; target counts use `min/max_number_visible_models`.
- **More clutter per scene** - raise `max_number_visible_models` and/or `max_number_visible_distractors`; raise per-asset `instances.max` so the instance pool is large enough.
- **Resize the dataset** - change `metadata.num_images` and `shard.size`. Use `output.max_size_gb` as a safety cap.
- **Change output location** - set `metadata.base_output_directory` to an absolute path; leave it null/empty to use `./output/<dataset_name>` relative to the run directory.
- **Change image resolution / FOV** - edit `camera.image_width`, `camera.image_height`, `camera.field_of_view` (radians).

## Dataset

A worker run produces COCO shards under `<base_output_directory>/<dataset_name>/`. `BinPickingWorker.merge_shards()` consolidates them into a single `merged_coco_annotations.json` (see [Output dataset format and shard layout](#output-dataset-format-and-shard-layout) for the full directory layout).

Both classes are re-exported from `illusion.dataset`:

```python
from telekinesis.illusion.dataset import CocoShardMerger, DatasetConverter
```

### CocoShardMerger

Lives in [src/telekinesis/illusion/dataset/dataset_generator.py](../src/telekinesis/illusion/dataset/dataset_generator.py) and is invoked automatically by `BinPickingWorker.merge_shards()`. It validates that all shards share identical `categories` (id + name pairs), re-indexes image and annotation IDs to be globally unique, rewrites each `file_name` to `<shard_dirname>/images/<frame>.png`, and writes `merged_coco_annotations.json` at the dataset root. Per-shard `coco_annotations.json` files are left in place. See [Output dataset format and shard layout](#output-dataset-format-and-shard-layout) for the resulting directory tree.

### DatasetConverter

Lives in [src/telekinesis/illusion/dataset/converter.py](../src/telekinesis/illusion/dataset/converter.py). Reads `<merged_dataset_dir>/merged_coco_annotations.json` and writes a stratified train/valid/test split to `output_dir` in either `coco` or `yolo` format. Stratification is by each image's dominant `category_id`; if the requested ratio yields a stratum that `sklearn.model_selection.train_test_split` cannot split (too few samples per class), the converter logs a warning and falls back to a random shuffle for that step.

Constructor parameters:

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `merged_dataset_dir` | `str \| Path` | required | Directory containing `merged_coco_annotations.json`. |
| `output_dir` | `str \| Path` | required | Destination for the split dataset. Created if missing. |
| `dataset_format` | `"coco" \| "yolo"` | required | Output layout. |
| `ratios` | `(float, float, float)` | `(0.7, 0.2, 0.1)` | Train / valid / test fractions. Must be non-negative and sum to 1.0. |
| `stratify` | `bool` | `True` | Stratify by dominant annotation category. `False` = pure random shuffle. |
| `seed` | `int` | `42` | RNG seed for reproducibility. |
| `merged_filename` | `str` | `"merged_coco_annotations.json"` | Override if the merged file uses a non-default name. |

Run with `.split()`, which returns the output directory path:

```python
from telekinesis.illusion.dataset import DatasetConverter

DatasetConverter(
    merged_dataset_dir="<path-to-merged-dataset>",
    output_dir="<path-to-split-output>",
    dataset_format="coco",  # or "yolo"
    ratios=(0.7, 0.2, 0.1),
    stratify=True,
    seed=42,
).split()
```

This is also what `BinPickingWorker.merge_shards()` runs automatically when the spec sets `output.dataset_format` — see [Running a worker](#running-a-worker) above.

**COCO output** (`dataset_format="coco"`, RF-DETR-compatible):

```
<output_dir>/
├── train/
│   ├── _annotations.coco.json
│   ├── 000001.png
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   └── ...
└── test/
    ├── _annotations.coco.json
    └── ...
```

Images are renamed to `{img_id:06d}.<ext>`. Each split's `_annotations.coco.json` keeps the source `info`, `licenses`, and the full `categories` list, so all three splits share identical category definitions.

**YOLO output** (`dataset_format="yolo"`, Ultralytics YOLO-seg):

```
<output_dir>/
├── data.yaml
├── train/
│   ├── images/000001.png ...
│   └── labels/000001.txt ...
├── valid/
│   ├── images/ ...
│   └── labels/ ...
└── test/
    ├── images/ ...
    └── labels/ ...
```

Each `<id>.txt` line is `class x1 y1 x2 y2 ... xn yn` with coordinates normalised to `[0, 1]`. RLE segmentations are decoded and the largest external contour is kept; multi-polygon annotations also keep only the largest polygon. `data.yaml` follows the Ultralytics convention (`path`, `train`, `val`, `test`, `nc`, `names`) - note the key is `val`, not `valid`, while the on-disk directory is still named `valid/`.

## Viewer

Final datasets should be inspected in [FiftyOne](https://docs.voxel51.com/) using [examples/view_dataset.py](../examples/view_dataset.py). Edit `DATASET_DIR` (and optionally `DATASET_NAME`) at the top of the script, then run:

```bash
python examples/view_dataset.py
```

The script auto-detects the layout, imports the dataset into FiftyOne, prints per-split sample counts and a class-distribution table to the console, then opens the interactive FiftyOne app. `fiftyone` is already a project dependency, so no extra install step is needed.

Supported layouts (detected by the contents of `DATASET_DIR`):

| Layout | Detected by | `DATASET_DIR` should be |
|---|---|---|
| merged | `merged_coco_annotations.json` at the root | the dataset root produced by `merge_shards()`, e.g. `.../<dataset_name>` |
| shard  | `coco_annotations.json` + `images/` | a single `shard_*` directory inside a merged dataset |
| coco   | `_annotations.coco.json` at the root, or in any of `train/valid/test` | the `DatasetConverter` COCO output root, or a single split |
| yolo   | `data.yaml` at the root | the `DatasetConverter` YOLO output root |

Detection precedence is `merged > shard > yolo > coco`. The console table reports, per class: count and share of each split (each column sums to 100%), total instances across all splits, the number of distinct images containing the class, and that class's share of the grand total. For interactive in-app charts, click `+` next to the **Samples** tab in the FiftyOne app and add a **Histograms** panel.

### Legacy tkinter viewer

The tkinter viewer in [src/telekinesis/illusion/viewer/shard_viewer.py](../src/telekinesis/illusion/viewer/shard_viewer.py) is **deprecated** and retained only as a lightweight preview used by the quickstart examples through `CocoShardMerger.preview()`. For final dataset inspection, use FiftyOne via `view_dataset.py`.

CLI invocation (still works, mainly for the quickstarts):

```bash
python -m illusion.viewer.shard_viewer --dataset <path>
```

Layouts it can detect (no YOLO support):

| Layout | Detected by | `<path>` should be |
|---|---|---|
| shard  | `coco_annotations.json` + `images/` | a single shard directory, e.g. `.../<dataset_name>/shard_<date>_<uuid>` |
| merged | `merged_coco_annotations.json` | the dataset root produced by `merge_shards()` |
| split  | `_annotations.coco.json` | a split subdirectory produced by `DatasetConverter` (`coco` format) |

Detection precedence is `merged > shard > split`. In-window controls: `←`/`→` (or `k`/`j`) to navigate, `b` toggles bboxes, `l` labels, `m` masks, `space` toggles all, `Ctrl+S` saves the composed image, `Ctrl+Q` exits. `CocoShardMerger.preview()` is the in-process entry point invoked by the quickstart examples after merging.

## Efficiency

Consider disabling energy-saving and sleep mode on your Windows machine to speed up the generation process.

## Troubleshooting

### Downgrading FiftyOne
Follow the procedure describing [here](https://docs.voxel51.com/installation/index.html) to downgrade, e.g. donwgrading to version 1.14.2:

```bash
pip install "fiftyone>=1.15.0,<1.16"
fiftyone migrate --all --version 1.14.2
pip install --force-reinstall fiftyone==1.14.2
```
