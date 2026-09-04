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

It is highly recommended to install a Miniconda environment before setting up the project. You can install Miniconda by following instructions from [here](https://docs.conda.io/en/latest/miniconda.html#installing).

```bash
conda create -n telekinesis-illusion python=3.11
conda activate telekinesis-illusion
```

Clone the repository:

```bash
git clone -b develop https://gitlab.com/telekinesis/illusion.git
cd illusion
```

This repository comes with a modified BlenderProc 2.8.0 package that uses an external `bpy` package (4.2.17 (LTS)) - see [BlenderProc/NOTICE.md](BlenderProc/NOTICE.md) for the list of modifications. Install it in editable mode:

```bash
cd BlenderProc
pip install -e .
```

Then install `telekinesis-illusion` from the repository root:

```bash
cd ..
pip install -e .
```

`telekinesis-illusion` comes with a small collection of default assets and models for running the examples.

In case you want to use your own assets, make sure to organize them in the expected layout as described in [docs/GUIDE.md](docs/GUIDE.md#assets).

## Blender Extension

To ease the parameter tuning of the `telekinesis-illusion` randomizer tree, we have additionally developed a Blender node-graph editor, with a live preview. 

You can install the `telekinesis-illusion` Blender extension from our [GitHub](https://github.com/telekinesis-ai/telekinesis-illusion-blender-extension).

## Quickstart

Run the examples from the `examples` folder. Note: running an example for the first time might take a few minutes and you might see `Loading render kernels (may take a few minutes the first time)` in the terminal. The quickstart examples demonstarte the following use cases:

```bash
python examples/quickstart_flying_things.py
```

Runs the low-level API to build a "flying things" scene: gearwheels and pipes are scattered in mid-air (no physics simulation) against randomized industrial/studio backgrounds. Generates a COCO instance-segmentation dataset of 5 images and opens it in the interactive dataset viewer.

```bash
python examples/quickstart_parts_in_bin.py
```

Runs the low-level API to build a bin-picking scene: gearwheels and a pipe fixture are dropped into a bin and settled with physics simulation. Generates a COCO instance-segmentation dataset of 5 images and opens it in the interactive dataset viewer.

```bash
python examples/generate_synthetic_data_with_bin_picking_worker.py
```
Launches synthetic data generation for a bin-picking use case with the default assets. Generates a COCO instance-segmentation dataset of 20 images (gearwheels in an industrial bin, with pipes and pipe fixtures as distractors) and opens the result in the interactive dataset viewer.

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
