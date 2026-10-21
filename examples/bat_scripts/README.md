# Windows batch examples

The bin-picking entrypoint provides CLI overrides and writes the effective
configuration to a temporary YAML file, leaving the source YAML unchanged.
The quickstart entrypoints reuse the standard examples. After a successful run
on Windows, these entrypoints remove Blender's temporary directory, flush
console output, and call `os._exit(0)` to bypass the native shutdown crash.
Generation and export exceptions propagate normally.

This deliberately retains the previous workaround: Blender may print memory
leak diagnostics, and Python exit handlers do not run after success. The
standard examples in the parent directory use normal Python shutdown.

## Generate a dataset and train RF-DETR

From the repository root:

```bat
examples\bat_scripts\generate_and_train_rfdetr.bat gearwheel_experiment_01
```

The batch file also works from another directory or by double-clicking it.
The optional argument sets the dataset and training run name; otherwise the
launcher generates a name with a random suffix.

Edit the settings at the top of the batch file to select the Iris checkout,
generation counts, model, epochs, batch size, training workers, and other
training parameters. The default environments are `telekinesis-illusion` and
`telekinesis-iris`; both must already be installed. Set `CONDA_BAT` if Conda is
not installed at `%USERPROFILE%\miniconda3\condabin\conda.bat`.

Generation exports `output/<run-name>/coco_format`, then Iris trains from that
directory and writes to its `results/<run-name>` directory. The pipeline stops
on either stage's nonzero exit code. Use a fresh name; existing exports and
training runs are rejected. The source YAML is never modified.

`NUM_IMAGES` sets the requested image count. The YAML's output size limit
still applies.

## Run a batch-specific example directly

```bat
python examples\bat_scripts\generate_synthetic_data_with_bin_picking_worker.py --help
python examples\bat_scripts\quickstart_flying_things.py --no-preview
python examples\bat_scripts\quickstart_parts_in_bin.py --no-preview
```

The quickstarts only generate data; the pipeline uses the bin-picking worker
entrypoint. These wrappers do not change Iris's multiprocessing behavior.
