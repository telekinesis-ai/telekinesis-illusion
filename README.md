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
python examples/generate_synthetic_data_with_bin_picking_worker.py --config-file configs/example_bin_picking_gearwheel_2.yaml --preview
```

The bin-picking worker defaults to this example YAML with preview disabled.
Use `--config-file config.yaml` to select another config file and `--preview`
or `--no-preview` to control the viewer.

For another asset tree, pass `asset_dir` to `Environment`, or set
`metadata.asset_directory` in the worker YAML. Relative YAML asset paths
resolve against that YAML file.

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
python examples/view_dataset.py output/example_bin_picking_gearwheel_2/coco_format
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
