# Development guide

This guide covers local development of `telekinesis-illusion`: setup, assets,
running examples, changing code, validation, and package builds. For usage, see
the [README](README.md); for issues and pull requests, see
[CONTRIBUTING.md](CONTRIBUTING.md).

- [Get started](#get-started)
- [Assets and paths](#assets-and-paths)
- [Run examples](#run-examples)
- [Find the relevant code](#find-the-relevant-code)
- [Coding conventions](#coding-conventions)
- [Validate changes](#validate-changes)
- [Build and release](#build-and-release)
- [Troubleshooting](#troubleshooting)

## Get started

### Requirements

[pyproject.toml](pyproject.toml) is the source of truth for dependencies and tool
configuration. The current requirements are:

| Component | Requirement |
| --- | --- |
| Python | `>=3.11.14,<3.12`, expressed as `~=3.11.14` in package metadata. |
| Blender runtime | `bpy==4.2.17`, installed by pip with the package. |
| Platform | An OS and architecture with a compatible [bpy wheel](https://pypi.org/project/bpy/4.2.17/#files). |
| Rendering | The default is CUDA. CPU rendering can be selected explicitly. |
| Assets | The repository's `assets/` tree, a compatible external asset directory, or the public example assets downloaded on first use. |

Run scripts with the environment's `python`. The standalone `bpy` package
provides the Blender runtime; the examples do not need the Blender desktop
application. Blender utilities live inside this repository.

### Clone and create an environment

```bash
git clone https://github.com/telekinesis-ai/telekinesis-illusion.git
cd telekinesis-illusion
```

Use a dedicated environment. Conda is one option, consistent with the README:

```bash
conda create -n telekinesis-illusion python=3.11.14
conda activate telekinesis-illusion
```

If a compatible Python is already installed, `venv` also works. On Windows
PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS (not tested yet):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Check that the selected interpreter is at least Python 3.11.14 and below 3.12.
Run the remaining commands from the repository root with this environment
active:

```bash
python --version
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
python -m pip show telekinesis-illusion bpy
```

An
[editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)
uses source files from this checkout. Restart the running Python process after
edits; reinstall when dependencies or package metadata change. Point your IDE
at the same environment and use the repository root as its working directory.

The interactive dataset viewer is optional:

```bash
python -m pip install -e ".[viewer]"
```

The setup uses public package dependencies. It does not require GitLab tokens,
camera vendor SDKs, or a custom global pip index.

## Assets and paths

The checkout includes example models, HDRIs, materials, and surface textures:

```text
assets/
  models/
  hdris/
  materials/
  surface_imperfections/
```

The top-level `assets/` directory is not included in the Python wheel. Even a
basic `Environment` needs an asset directory for relative paths such as surface
imperfections. Models, materials, and HDRIs are not registered automatically:
add them with `Environment.add()`, for example
`environment.add(config.MaterialConfig.discover(assets_dir / "materials"))`.

Asset lookup is implemented in
[utils/assets.py](src/telekinesis/illusion/utils/assets.py):

1. An explicit asset directory takes precedence. A relative
   `Environment(asset_dir=...)` path is relative to the working directory;
   a worker's `dataset.asset_directory` is relative to its YAML file.
2. Without an explicit directory, lookup checks `./assets` in the working
   directory, then the repository's `assets/` next to an editable checkout.
3. If neither exists, the public example assets are used from
   `~/.cache/telekinesis/telekinesis-illusion/assets-<VERSION>/assets`. The first such lookup
   downloads `https://assets.telekinesis.ai/telekinesis-illusion/assets-<VERSION>.zip`
   and extracts it there; members with paths outside the archive are
   rejected. An interrupted download leaves no partial cache. Delete the
   directory to download the assets again.
4. Model paths in worker YAML are relative to the resolved asset directory;
   absolute model paths are used directly.

For example, `dataset.asset_directory: ../assets` in a file under `configs/`
selects the repository assets. If you move the YAML elsewhere, update that path.
Keep model names, material categories, and HDRI categories consistent with the
files referenced by the example.

Use `output/` for local datasets; it is ignored by Git. Choose a fresh output
directory when comparing runs so earlier images do not affect the result.

## Run examples

### Quickstarts

Start with a small render and disable the viewer:

```bash
python examples/quickstart_flying_things.py --no-preview --device CPU --num-images 2 --resolution 128 --samples 4 --seed 17 --output-dir output/dev-flying
python examples/quickstart_parts_in_bin.py --no-preview --device CPU --num-images 2 --resolution 128 --samples 4 --seed 17 --output-dir output/dev-bin
```

These commands use CPU rendering. The first generates models without physics; the
second simulates parts in a bin. Each output directory contains
`coco_annotations.json` and the image files referenced by it. Low resolution
and sample counts are useful for checking the pipeline, not judging final
image quality. The quickstarts enable preview by default, so retain
`--no-preview` unless the `viewer` extra is installed and a viewer is wanted.

The quickstarts default to CUDA; pass `--device CPU` for CPU rendering. In a
script that constructs its own engine:

```python
from telekinesis.illusion import config
from telekinesis.illusion.core import engine

render_engine = engine.BlenderEngine(
    config.RenderConfig(device="CPU", samples=4),
    config.PhysicsConfig(active=False),
)
```

Pass that engine to `SyntheticDataGenerator`. Set
`config.PhysicsConfig(active=True)` when the environment requires simulation.

### YAML-driven bin picking

The bin-picking worker builds a complete generation run from one YAML file,
such as the tracked
[gearwheel configuration](configs/bin_picking_example.yaml):

```bash
python examples/generate_synthetic_data_with_bin_picking_worker.py --config-file configs/bin_picking_example.yaml --no-preview
```

This writes shards and merged annotations under
`<dataset.output_directory>/<dataset.name>/`. The example YAML requests a COCO
export in `coco_format/`. Use a new dataset name for another exported run: the
worker asks before replacing an existing export directory. For deliberate
shard accumulation, set `dataset.writer.dataset_format: null` in a local YAML copy and
keep category IDs consistent across runs. The script defaults to
`configs/bin_picking_example.yaml`; use `--config-file` to select another file
and `--preview` or `--no-preview` to control the viewer.

#### Configuration structure

Every section is converted into a config from `telekinesis.illusion.config`
before anything is built. The keys of a section are the fields of its config,
and omitted optional fields use the config defaults. Distances are in meters
and angles in radians, except where a field states degrees.

| Section | Config | Content |
| --- | --- | --- |
| `dataset` | `DatasetConfig` | Name, number of images, and asset and output directories. |
| `dataset.shard` | `ShardConfig` | Images per shard and the shard directory name template. |
| `dataset.writer` | `WriterConfig` | Image format, JPEG quality, output size limit, and the COCO or YOLO export split. |
| `dataset.writer.coco` | `CocoConfig` | Info and licenses written into the COCO annotations. |
| `renderer` | `RenderConfig` | Render device, samples and noise threshold. |
| `camera` | `CameraConfig` | Image size and lens. |
| `physics` | `PhysicsConfig` | Rigid-body simulation that settles the parts. |
| `models` | `ModelConfig` per entry | Model files, annotation categories, physics and import settings, and groups. |
| `lights` | `LightConfig` per entry | Scene lights and their fixed properties. |
| `pose_sampling` | `PoseSamplingConfig` | Random or grid placement of parts above the visible container. |
| `randomizers.instance.<role>` | `InstanceRandomizerConfig` | Visible instances per model and in total. |
| `randomizers.model_pose` | `ModelPoseRandomizerConfig` | Placement attempts and an optional fixed surface. |
| `randomizers.material.<name>` | `MaterialRandomizerConfig` | Target models or model groups, and the materials sampled for them. |
| `randomizers.background` | `BackgroundRandomizerConfig` | HDRI groups or names. Omit it to keep the default background. |
| `randomizers.light.<light>` | `LightRandomizerConfig` | Sampled light properties. |
| `randomizers.light_pose.<light>` | `LightPoseRandomizerConfig` | Light pose sampler, its `params`, and clearance. |
| `randomizers.camera_pose` | `CameraPoseRandomizerConfig` | Camera pose sampler, its `params`, and views per scene. |

`dataset`, `dataset.shard`, `models` and `randomizers.instance` are required. The
`params` of a pose sampler are a `ShellSamplerConfig` for `shell_sampler` and a
`VolumeSamplerConfig` for `volume_sampler`.

Each model has exactly one role group: `part`, `container` or `distractor`.
The role decides which randomizers place the model and is written as the COCO
supercategory. `randomizers.instance` configures exactly the roles that the
models use; its `counts` name models of that role. Material rules select their
targets with `target_groups`, such as `[part]`, or `target_models`. Lights are
declared under `lights`, and `randomizers.light` and `randomizers.light_pose`
refer to them by name.

Relative paths resolve as follows: `dataset.asset_directory` against the YAML
file, `models[].model_path` against the asset directory, and
`dataset.output_directory` against the working directory.

#### Validation

Unknown sections or keys, missing required keys, invalid values, and
contradictions between sections raise `ValueError` or `TypeError` before any
asset is loaded. The message names the file and the key path of the value:

```text
ValueError: Invalid configuration file 'configs/my_config.yaml':
randomizers.light_pose.bin_key.params.radius_min: must be a number >= 0.0, got -1.
```

To check one section without starting Blender, build its config directly:

```python
import yaml

from telekinesis.illusion import config

with open("configs/bin_picking_example.yaml", encoding="utf-8") as file:
    raw = yaml.safe_load(file)
camera_pose = config.CameraPoseRandomizerConfig.from_dict(
    raw["randomizers"]["camera_pose"]
)
```

Every config converts back with `to_dict()`, and `replace()` returns a
validated copy with changed fields.

For CPU rendering, set the renderer in a local YAML copy:

```yaml
renderer:
  device: CPU
  samples: 4
```

## Find the relevant code

Paths below are relative to `src/telekinesis/illusion/`:

| Path | Responsibility |
| --- | --- |
| `config.py` | Validated configs for assets, scene, randomizers, rendering, physics, and output. |
| `core/` | Asset managers, environment coordination, generation lifecycle, rendering and physics. |
| `types/` | Models, cameras, lights, backgrounds, materials, and render results. |
| `randomizer/`, `sampler/` | Randomization DAG, instance counts, property and pose sampling. |
| `blender/` | Local Blender wrappers, loading, rendering, physics, and camera utilities. |
| `writer/` | Writing images and COCO annotations from render results. |
| `workers/` | YAML-driven generation and shard orchestration. |
| `dataset/` | Shard merging and dataset format conversion. |
| `utils/`, `viewer/` | Asset resolution, environment isolation, and optional previews. |

The generation sequence is owned by
[SyntheticDataGenerator](src/telekinesis/illusion/core/synthetic_data_generator.py):
prepare, reset sample activity, randomize, optionally simulate and update light
poses, render, write, and clean up. When changing this flow, preserve these
constraints:

- Register assets before preparation. Instance-count bounds determine the pool
  capacity; imports and pool growth do not belong in the sample loop.
- A process has one active Blender scene. Constructing another `Environment`
  resets Blender state; wrappers from a cleaned environment must not be reused.
  Each randomized scene is a realization within the same prepared environment.
- Use environment activation/deactivation to keep the current scene's
  visibility and physics participation consistent. Hiding a Blender object
  alone is not equivalent.
- `num_images` counts camera views; `num_samples` counts randomized scenes.
  Keep images, masks, category IDs, and capture metadata aligned.
- Normal generation cleans up in `finally`. Code retaining an environment
  between shards must explicitly clean up at the end, including on failure.

Review callers and the relevant examples before changing a public
signature, default, unit, return value, or side effect. Adapted Blender helpers
carry provenance in [blender/NOTICE.md](src/telekinesis/illusion/blender/NOTICE.md).
Use the [Blender 4.2 API](https://docs.blender.org/api/4.2/)
and [4.2 API release notes](https://developer.blender.org/docs/release_notes/4.2/python_api/).

## Coding conventions

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
and the surrounding code, with the repository's configured 80-character line
length. In particular:

- Fix the source of incorrect state or assumptions. Keep changes focused and
  reuse existing components before introducing a new abstraction or dependency.
- Prefer module imports where practical, descriptive names, `PascalCase` class
  names without underscores, and `snake_case` functions and variables.
- Use Python 3.11 annotations: `list[str]`, `X | None` when `None` is meaningful,
  and abstract collection types from `collections.abc`.
- Catch exceptions only where they can be handled meaningfully. Preserve useful
  error information and resource cleanup; avoid silent fallbacks.
- Keep configuration near its use. Preserve calibrated parameters and document
  changes to physical units, coordinate conventions, or defaults.
- Document the complete public contract, including relevant inherited behavior,
  units, mutation, validation, and errors. Give classes a small valid `Example:`;
  add method examples only when useful. Document private methods for developers.
- Comments should explain non-obvious reasoning rather than narrate the code.
- Preserve the existing environment-isolation setup before importing `bpy`.
  See [utils/blender_env.py](src/telekinesis/illusion/utils/blender_env.py).

## Validate changes

### Lint and formatting

Install Ruff separately; it reads [pyproject.toml](pyproject.toml). Check the
changed Python files; replace the example path below with the files in your
change:

```bash
python -m pip install "ruff>=0.15.9"
python -m ruff check src/telekinesis/illusion/core/engine.py
python -m ruff format --check src/telekinesis/illusion/core/engine.py
git diff --check
```

Fix issues introduced by your change. Existing files may have lint or format
debt; avoid repository-wide autofixes or unrelated formatting. There is no
configured Pylint score gate or pre-commit workflow in this checkout. Ruff's
configuration and behavior are described in its
[official documentation](https://docs.astral.sh/ruff/configuration/).

### Documentation and reproducibility

For documentation-only changes, check relative links, Markdown rendering, and
commands against the current configuration and CLI arguments. The `docs` extra
declares MkDocs dependencies, but this checkout has no `mkdocs.yml` or `docs/`
source tree, so it does not provide a working `mkdocs build` workflow.

For a reproducible rendering report, record the commit, Python and `bpy`
versions, resolved dependencies, GPU/driver or CPU device, configuration, assets,
and seed. `--seed` on the example CLIs seeds Python and NumPy; it does not promise
identical images across hardware or Blender versions. Worker `dataset.writer.split_seed`
controls dataset splitting, not all scene randomization.

The repository has no committed dependency lockfile. Capture the environment
when sharing a reproduction. Create `output/` first if no generation run has
created it yet:

```bash
python -m pip freeze > output/development-environment.txt
```

Review the file for private URLs or local paths before sharing it.

## Build and release

Build only when checking packaging or preparing a release. Start with a clean
`dist/` directory so old artifacts are not mistaken for the current build.
Install the build tools separately:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

These commands create and inspect a source distribution and wheel; they do not
publish them. See the
[Python packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
for the underlying workflow.

For packaging changes, inspect the archives for package modules, Blender
resources, and license notices, then install the selected wheel into a separate
compatible environment. Run from outside the checkout with an explicit asset
directory to verify the installed package. Editable installs can otherwise
hide missing package files.

Publishing requires the maintainer's registry access and agreed release process;
ordinary contributions do not require publishing credentials or a version bump.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Installation rejects Python or cannot find `bpy` | Check `python --version`, the active environment, and wheel availability for `bpy==4.2.17`. Python 3.12+ does not satisfy this project's metadata. |
| Imports use old code or fail after an edit | Check the IDE interpreter and `python -m pip show telekinesis-illusion`. Reinstall editable metadata when needed and restart Python. |
| Missing asset directory, model, HDRI, or material | Check the working directory, YAML-relative asset root, and referenced files. A wheel does not supply the top-level example assets; they are downloaded to `~/.cache/telekinesis/illusion/assets-<VERSION>/assets` when no local `assets/` directory exists. |
| Downloading the public assets fails | Check network access to `assets.telekinesis.ai`, or provide a local asset directory. |
| CUDA device selection fails | Check the GPU and driver, or select CPU explicitly as described above. |
| Viewer import fails or a preview starts unexpectedly | Install the `viewer` extra, or pass `--no-preview` to the quickstarts. |
| NumPy/SciPy ABI error or imports resolve into Blender extensions | Check environment isolation and any `BLENDER_USER_EXTENSIONS` override. Blender extension wheels can shadow environment packages; see `utils/blender_env.py`. |
| Windows process crashes after rendering | Use the documented batch wrappers for generation. Standard examples use normal Python shutdown and can encounter native `bpy` teardown failures. Preserve logs and verify output rather than assuming success from the last message. |
| Bin-picking CLI reports an existing export | Use a new dataset name/output directory, or deliberately accumulate shards with `dataset.writer.dataset_format: null` in your YAML. |

If the problem persists, follow the
[bug report checklist](CONTRIBUTING.md#report-a-bug) with the smallest
reproduction and its environment details.
