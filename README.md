<div align="center">
  <p>
    <a href="https://gitlab.com/telekinesis/illusion">
      <img width="100%" src="https://assets.telekinesis.ai/logo/telekinesis-banner.png" />
    </a>
  </p>

  <p align="center">
    <a href="https://docs.telekinesis.ai">
      <img src="https://img.shields.io/badge/docs-telekinesis.ai-blue" />
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/license-GPL--3.0--or--later-green" />
    </a>
    <a href="https://www.blender.org/download/releases/4-2/">
      <img src="https://img.shields.io/badge/Blender-4.2%20LTS-orange" />
    </a>
    <a href="#requirements">
      <img src="https://img.shields.io/badge/platform-Windows%20x64-lightgrey" />
    </a>
  </p>

  <p>
    <a href="https://docs.telekinesis.ai/data-engine/synthetic-data-generation/overview.html">Docs</a>
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
git clone https://github.com/telekinesis-ai/telekinesis-illusion
cd telekinesis-illusion
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

Runs the low-level API to build a "flying things" scene: gearwheels and pipes are scattered in mid-air (no physics simulation) against randomized industrial/studio backgrounds. Generates a COCO instance-segmentation dataset of 5 images and opens it in the interactive dataset viewer. Pass `--no-preview` to suppress the viewer.

```bash
python examples/quickstart_parts_in_bin.py
```

Runs the low-level API to build a bin-picking scene: gearwheels and a pipe fixture are dropped into a bin and settled with physics simulation. Generates a COCO instance-segmentation dataset of 5 images and opens it in the interactive dataset viewer. Pass `--no-preview` to suppress the viewer.

```bash
python examples/generate_synthetic_data_with_bin_picking_worker.py
```
Launches synthetic data generation for a bin-picking use case with the default assets. Generates a COCO instance-segmentation dataset of 20 images (gearwheels in an industrial bin, with pipes and pipe fixtures as distractors) and opens the result in the interactive dataset viewer.

## Color transfer on existing COCO datasets

Apply Reinhard-style Lab color transfer after generation, using a folder of
real images as references:

```powershell
python examples/reinhard_transfer_shards.py `
  --reference-dir "data/real_images/train" `
  --synthetic-dir "output/bin_picking" `
  --output-dir "output/bin_picking_reinhard" `
  --seed 42
```

The reference folder is searched recursively for `.png`, `.jpg`, and `.jpeg`
(case insensitive). Each synthetic image receives the Lab mean and standard
deviation of one randomly selected reference. Reference statistics are cached;
the same inputs and seed reproduce the selection. Use training references only.

The input format is detected automatically; no format flag is needed:

- Native shards: subdirectories containing `coco_annotations.json`.
- Exported COCO: split directories such as `train`, `valid`, and `test`, each
  containing `_annotations.coco.json` and its images.
- An individual shard or exported split: either annotation file directly
  inside `--synthetic-dir`.

For an export, point `--synthetic-dir` at its dataset root or one split folder.
The output retains that layout, including empty splits. Only images listed in
the detected annotation files are transformed; image names, dimensions,
annotations, masks, and other files inside those directories are preserved.
An existing root `merged_coco_annotations.json` is copied too. When processing
multiple shards/splits, other root files and unrelated directories are not
copied. Mixed or ambiguous layouts are rejected.

The destination must **not exist** and must be outside both input directories.
The script publishes it only after all images finish successfully, leaving
the inputs untouched. It requires space for a complete copy of the dataset.
Uses the existing NumPy, OpenCV and tqdm dependencies; no Blender or
`python-color-transfer` is needed.

Inputs must be 8-bit images. Grayscale is converted to color; PNG alpha is
preserved and fully transparent pixels are excluded from statistics. PNGs
remain PNGs; JPEGs are re-encoded at quality 95. The transform matches global
color statistics, so reference composition and clipping can affect results.

## Generate a dataset and train RF-DETR on Windows

Run `generate_and_train_rfdetr.bat` from this repository, or double-click it.
It generates and exports a COCO dataset in the `telekinesis-illusion` conda
environment, then trains using the sibling `telekinesis-iris` checkout and
its `telekinesis-iris` environment. Both environments must already be installed.

```bat
generate_and_train_rfdetr.bat gearwheel_experiment_01
```

The optional argument sets both the dataset name and training run name. Without
it, a random suffix gives each invocation a fresh name. Edit the settings at the
top of the batch file for the YAML spec, image count, shard size, model variant,
epochs, batch size, accumulation steps, learning rates, and checkpoint intervals.
The current settings generate 20 scenes and train Seg Medium for 10 epochs.
`NUM_IMAGES` uses the worker's existing scene count; multiple camera views can
produce more image files. The YAML's output size limit still applies.

Set `IRIS_REPO` if Iris is not next to this repository. The default conda path is
`%USERPROFILE%\miniconda3\condabin\conda.bat`; edit `CONDA_BAT` or set it in your
terminal if conda is installed elsewhere. The batch file uses `conda run` with
live console output, and stops on either stage's nonzero exit code.
After a successful export on Windows, the generator explicitly cleans its
temporary files and exits before `bpy`'s faulty native interpreter shutdown.
Generation and export exceptions still stop the pipeline.

The dataset is written to `output/<run-name>/coco_format` and passed directly to
training. Checkpoints and logs go to `telekinesis-iris/results/<run-name>`.
Choose a fresh name for each pipeline run; existing exports/runs are rejected.
The source YAML is never modified. Training prints the complete TensorBoard
command at startup; run it in another terminal and open `http://localhost:6006`.

Both scripts can also run independently:

```bat
conda run --no-capture-output -n telekinesis-illusion python examples/generate_synthetic_data_with_bin_picking_worker.py --spec-file example_bin_picking_gearwheel_2.yaml --dataset-name gearwheel_experiment_01 --output-dir output --num-images 100 --shard-size 10 --dataset-format coco --no-preview
conda run --no-capture-output -n telekinesis-iris python ../telekinesis-iris/examples/train_rfdetr.py --dataset output/gearwheel_experiment_01/coco_format --run-name gearwheel_experiment_01 --epochs 50 --batch-size 4
```

Use `--help` on either script for all options. Training also supports
`--output-dir`, `--resolution`, `--device`, `--no-mixed-precision`,
`--no-pretrained`, `--seed`, and `--resume-from CHECKPOINT`. When resuming,
`--epochs` is the total target epoch count, not additional epochs.

## Documentation

See [Material randomization](docs/MATERIAL_RANDOMIZATION.md) for whole-object,
object and material-slot selection, procedural Principled materials and presets.
See [Light randomization](docs/LIGHT_RANDOMIZATION.md) for point, sun, spot,
and area lights, sampled properties, and placement with clearance from meshes.
Both quickstarts demonstrate light randomization alongside HDRI backgrounds.
Run `python examples/preview_material_randomization.py`
to save reproducible material previews without generating a dataset.

Run `python examples/preview_light_randomization.py` to compare nine lighting
setups on a fixed steel gearwheel. The labeled comparison, individual PNGs,
and sampled settings are saved under `output/light_preview/gearwheel/`.
Use `--no-render` to check scene setup, or `--seed` to try different samples.

To compare clean, scratched and worn plastic bins using only the public
Illusion API:

```bash
python examples/preview_bin_surface_imperfections.py --asset-dir E:/telekinesis-illusion/assets
```

The comparison PNG and COCO annotations are written under
`output/material_preview/plastic_bin_imperfections/`. Use `--no-render` to
check scene setup without rendering.

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
