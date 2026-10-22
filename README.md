<div align="center">
  <p>
    <a href="https://gitlab.com/telekinesis/illusion">
      <img width="100%" src="media/images/telekinesis_banner.png" />
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
The examples default to CUDA (`--device CUDA`); pass `--device CPU` to select CPU rendering:

Run the exmamples:
```bash
python examples/quickstart_flying_things.py --preview
```
```bash
python examples/quickstart_parts_in_bin.py --preview
```
```bash
python examples/generate_synthetic_data_with_bin_picking_worker.py --spec_file configs/example_bin_picking_gearwheel_2.yaml --preview
```

The bin-picking worker defaults to this example YAML with preview disabled.
Use `--spec_file PATH` to select another YAML and `--preview` or `--no-preview`
to control the viewer.

The checkout includes the original example assets. For another asset tree,
pass `asset_dir` to `Scene`, or set `metadata.asset_directory` in the worker
YAML. Relative YAML asset paths resolve against that YAML file. Interactive
FiftyOne previews are optional: 
```bash
python -m pip install -e ".[viewer]"
```

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
