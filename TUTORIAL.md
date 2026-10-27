# Generate your first dataset

Create six images of a gear from randomized camera positions, with COCO
instance-segmentation annotations. This tutorial introduces the environment,
randomizer, engine, writer, and generator through one small dataset run.

The environment owns reusable assets and represents the active Blender scene.
Each execution of the randomizer tree realizes a scene within that environment;
the generator renders that scene from the configured camera views.

## 1. Prepare your environment

Use a source checkout with its `assets/` directory and install the package as
described in [Getting started](DEVELOPMENT.md#get-started). Activate that
environment and run the following steps from the repository root.

The example uses CPU rendering, a 128-by-128 image size, and eight Cycles samples
to keep the first run small. The scene uses the supplied model and default
background assets. No interactive viewer is needed.

## 2. Create the generation script

Save this as `generate_gears.py` in the repository root:

```python
from telekinesis.illusion import config
from telekinesis.illusion.core import (
    engine,
    environment,
    synthetic_data_generator,
)
from telekinesis.illusion.randomizer import randomizer, randomizer_node
from telekinesis.illusion.writer import writer


dataset_environment = environment.Environment()
dataset_environment.add(config.CameraConfig(image_width=128, image_height=128))
dataset_environment.add(
    config.ModelConfig.from_path(
        "assets/models/mechanical_parts/gearwheel_1.glb",
        model_name="gear",
        category_name="gear",
        category_id=1,
    )
)
dataset_environment.add(
    config.LightConfig(
        "key",
        "POINT",
        location=(0.0, 0.0, 1.0),
        properties={"power": 40.0},
    )
)

randomization = randomizer.Randomizer()
randomization.add_randomizer(
    randomizer_node.ModelInstanceRandomizer(
        config.InstanceRandomizerConfig(counts={"gear": (1, 1)})
    ),
    "counts",
)
randomization.add_randomizer(
    randomizer_node.CameraPoseRandomizer(
        config.CameraPoseRandomizerConfig(
            sampler="shell_sampler",
            number_of_views=2,
            params={"radius_min": 0.5, "radius_max": 0.7},
        )
    ),
    "camera",
)

generator = synthetic_data_generator.SyntheticDataGenerator(
    environment=dataset_environment,
    randomizer=randomization,
    engine=engine.BlenderEngine(
        config.RenderConfig(device="CPU", samples=8),
    ),
    writer=writer.CocoWriter(output_dir="output/tutorial-gears"),
)
generator.generate(num_images=6)
```

The script builds the run in this order:

1. **Register a model and light.** `model_name="gear"` identifies the model to
   the randomizer. The category name and ID identify it in the annotations.
   Create the environment before constructing its lights.
2. **Choose the active instances.** The `(1, 1)` bounds activate exactly one
   gear per sample. Counts belong to the randomizer rather than model metadata.
3. **Choose camera views.** The camera node runs after the count node, sampling
   two positions 0.5 to 0.7 meters from the active model's point of interest.
   The gear stays fixed while the camera changes position.
4. **Render and write.** The engine renders RGB and segmentation data; the
   writer saves images and COCO annotations. Physics is disabled by default.

`generate()` prepares the environment automatically, runs the randomizers,
renders each scene's views, writes the results, and cleans up the environment.
With two views per scene, `num_images=6` produces three scene realizations.
`num_samples=6` would instead produce six scenes and twelve images; pass only
one of these count arguments.

## 3. Run and inspect the result

```bash
python generate_gears.py
```

After a successful run, inspect:

```text
output/tutorial-gears/
  coco_annotations.json
  images/
```

Open an image to see the gear and background. The JSON contains six image
records, the `gear` category with ID `1`, and annotations for visible instances.
Each annotation links an image to a category and contains its segmentation,
bounding box, and area. Camera positions are random, so the images vary between
runs.

Use a fresh `output_dir` when repeating the tutorial: the COCO writer appends to
existing output. The low sample count can produce noisy images. Increase the
resolution and render samples when you move on to image-quality experiments.
For CUDA-capable hardware, change the render device to `"CUDA"`.

If a run fails, consult [Troubleshooting](DEVELOPMENT.md#troubleshooting),
including the known Windows `bpy` shutdown issue.

## Continue with the examples

- [Flying things](examples/quickstart_flying_things.py) adds randomized model
  counts, poses, materials, backgrounds, and lights.
- [Parts in a bin](examples/quickstart_parts_in_bin.py) adds physics and shows
  explicit preparation before setting a fixed container transform.
- [YAML-driven bin picking](DEVELOPMENT.md#yaml-driven-bin-picking) covers
  configuration, shards, and export.
- [Architecture](ARCHITECTURE.md) explains ownership, preparation, instance
  pools, and the generation lifecycle.
