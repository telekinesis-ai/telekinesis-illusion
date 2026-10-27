<div align="center">
  <p>
    <a align="center" href="https://telekinesis.ai/" target="_blank">
      <img
        width="100%"
        src="https://telekinesis-public-assets.s3.us-east-1.amazonaws.com/Telekinesis+Banner.png"
      >
    </a>
  </p>

  <p align="center">
    <a href="https://docs.telekinesis.ai">
      <img src="https://img.shields.io/badge/docs-telekinesis.ai-blue" />
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/license-GPL--3.0--or--later-green" />
    </a>
  </p>

  <p>
    <a href="https://docs.telekinesis.ai/">Docs</a>
    &nbsp;•&nbsp;
    <a href="https://github.com/telekinesis-ai/telekinesis-illusion">GitHub</a>
    &nbsp;•&nbsp;
    <a href="https://discord.gg/S5v8bYAnc6">Discord</a>
    &nbsp;•&nbsp;
    <a href="https://www.linkedin.com/company/telekinesis-ai/">LinkedIn</a>
    &nbsp;•&nbsp;
    <a href="https://x.com/telekinesis_ai">X</a>
    &nbsp;•&nbsp;
    <a href="https://telekinesis.ai/">Website</a>
  </p>
</div>

#  `telekinesis-illusion`

`telekinesis-illusion` is a python package for procedurally generating physically simulated synthetic datasets for training computer vision models in robotics applications.

Open source under [GPL-3.0-or-later](LICENSE).

Full documentation: [Telekinesis Agentic OS: Illusion](https://docs.telekinesis.ai/data-engine/synthetic-datasets/overview.html).

## Installation

1. Create the enviornment;
    ```bash
    conda create -n telekinesis-illusion python=3.11
    ```
    ```bash
    conda activate telekinesis-illusion
    ```
2. Install the package:

    Via pip: The simplest way to install `telekinesis-illusion`.
    ```bash
    python -m pip install telekinesis-illusion
    ```
    Via GitHub: Alternatively, if you need to make changes to `telekinesis-illusion` or use a different branch.
    ```bash
    git clone https://github.com/telekinesis-ai/telekinesis-illusion.git
    ```
    ```bash
    cd telekinesis-illusion
    ```
    ```bash
    python -m pip install -e .
    ```

## Tested configurations

The following configurations have been manually tested with
`telekinesis-illusion` 0.2.2 and Python 3.11. This list records known-working
setups; other configurations may also work.

| Operating system | Hardware |
|---|---|
| Windows 11 Pro | NVIDIA GeForce RTX 3090 |
| Windows 11 Pro | NVIDIA GeForce RTX 2060 |
| macOS Tahoe 26.3 | Apple MacBook Pro with M5 Pro (CPU only) |
| Ubuntu 20.04.5 LTS | NVIDIA GeForce RTX 3080 |

## Quickstart

The example scripts are run from the Git checkout and are not included in the
package installed from PyPI. Follow the GitHub installation above, then run
these commands from the repository root. The examples rely on the checkout's
`assets/` directory, and the worker example also uses `configs/`.

The examples default to CUDA (`--device CUDA`); pass `--device CPU` to select
CPU rendering:

> [!NOTE]
> On macOS, launching an example for the first time can take a long time.

```bash
python examples/quickstart_flying_things.py --preview
```
```bash
python examples/quickstart_parts_in_bin.py --preview
```
```bash
python examples/generate_synthetic_data_with_bin_picking_worker.py --config-file configs/bin_picking_example.yaml --preview
```

The bin-picking worker defaults to this example YAML with preview disabled.
Use `--config-file config.yaml` to select another config file and `--preview`
or `--no-preview` to control the viewer.

For another asset tree, pass `asset_dir` to `Environment`, or set
`dataset.asset_directory` in the worker YAML. Relative YAML asset paths
resolve against that YAML file.

Every section of the worker YAML is validated as a typed config from
`telekinesis.illusion.config` before anything is built. Unknown keys and
invalid values raise an error that names the file and the key path, such as
`randomizers.camera_pose.params.distance_range`. See
[DEVELOPMENT.md](DEVELOPMENT.md#yaml-driven-bin-picking) for the file
structure.

### Register local assets

`config.ModelConfig`, `config.MaterialConfig`, and `config.BackgroundConfig`
describe files without loading Blender data. `from_path()` creates one config;
`discover()` recursively returns configs in deterministic order.
`Environment.add()` registers one config or an iterable of configs, such as the
list returned by `discover()`:

```python
from telekinesis.illusion import config

environment.add(
    config.ModelConfig.from_path("extra_assets/gear.glb", model_name="gear")
)
environment.add(
    config.MaterialConfig.from_path("extra_assets/Steel_1K", groups="metal/steel")
)
environment.add(
    config.BackgroundConfig.discover(
        "extra_assets/hdris/industrial", group_prefix="indoor/industrial"
    )
)
```

Material directories use `<directory-name>_Color.png`, `.jpg`, or `.jpeg`;
other texture maps are optional. Backgrounds accept `.hdr` and `.exr` files.
Names default to the material directory name or file stem and must be unique
within each asset type. Model selection groups do not change annotation
categories.

`environment.model_manager` owns model registration, categories, instance pools,
activation, and sample resets. Existing `Environment` model methods remain
available. Use `environment.prepare()` and `environment.clean_up()` to coordinate
the model lifecycle with shared Blender scene setup and teardown.

Material and background configs require groups. `indoor/industrial`
includes membership in both `indoor` and `indoor/industrial`. Discovery derives
groups from relative parent directories. Supply `group_prefix` when the root
itself contains assets or represents part of an existing hierarchy.

The environment registers nothing implicitly: add every model, material, and
background that a randomizer should be able to choose. Randomizers only select
from the registered assets, by group or by name. Each randomizer takes its
config:
`MaterialRandomizer(config.MaterialRandomizerConfig(target_models=("gear",), groups=("metal/steel",)))`
and
`BackgroundRandomizer(config.BackgroundRandomizerConfig(groups=("indoor/industrial",)))`,
or `names=(...)` for individual assets. Without a selection, every registered
asset is eligible.
Registration does not load a material or activate a background; randomizers
apply them later. URL download and archive support are deferred.

### View generated datasets

Illusion writes generated datasets in a native sharded format. The shards can
then be merged and exported to common formats such as COCO or YOLO. Generated
images and annotations can always be inspected directly under `output/`, and
COCO or YOLO exports can be opened with any compatible dataset viewer.

Passing `--preview` to a quickstart opens Illusion's built-in COCO viewer after
generation. This viewer uses Tkinter and Pillow for its interface.

For interactive exploration with FiftyOne, install the optional dependencies
and run the provided script on the final dataset directory:

```bash
python -m pip install -e ".[viewer]"
```

```bash
python examples/view_dataset.py output/bin_picking_example/coco_format
```

The script detects native shards and merged datasets as well as exported COCO
and YOLO datasets automatically.

### Optional dependencies

Optional dependencies are grouped by use case and are not required for basic
dataset generation.

| Extra | Purpose |
|---|---|
| `viewer` | Installs FiftyOne for interactive dataset previews. Generation with `--no-preview` does not require it. |
| `dev` | Installs pytest, coverage, benchmarking, Ruff, and Pylint for developing and checking the package. |
| `docs` | Installs MkDocs and the plugins used to build the project documentation. |

Install an extra with the published package:

```bash
python -m pip install "telekinesis-illusion[viewer]"
```

For an editable source checkout, include the extra in the local install:

```bash
python -m pip install -e ".[dev]"
```

Multiple extras can be installed together:

```bash
python -m pip install -e ".[dev,docs,viewer]"
python -m pip install -e ".[dev,docs,viewer]"
```

The quoted commands work in Windows PowerShell and Command Prompt, as well as
Linux shells such as Bash.

## Tested configurations

The following configurations have been manually tested with. This list records known-working
setups; other configurations may also work.

| Operating system | Hardware |
|---|---|
| Windows 11 Pro | CPU; NVIDIA GeForce RTX 3090; RTX 2060 |
| macOS Tahoe 26.3 | Apple MacBook Pro with M5 Pro (CPU only) |
| Ubuntu 20.04.5 LTS | CPU; NVIDIA GeForce RTX 3080 |

## Documentation

Find the documentation for Illusion at: [Telekinesis Agentic OS: Illusion](https://docs.telekinesis.ai/).

## Citation

```bibtex
@software{telekinesis_illusion,
  author = {Telekinesis GmbH},
  title  = {Telekinesis-Illusion: Synthetic Data Generation for Robotics},
  year   = {2026},
  url    = {https://gitlab.com/telekinesis/illusion},
  note   = {GPL-3.0-or-later}
}
```
