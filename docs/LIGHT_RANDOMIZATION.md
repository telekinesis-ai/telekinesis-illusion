# Light randomization

Create scene lights with `Context.add_light(name, light_type, **properties)`.
It returns a `telekinesis.illusion.types.light.Light`; retrieve it later with
`context.get_light(name)` or inspect `context.get_lights()`. Names are exact
and unique. Lights are separate from mesh objects, instance counts, rigid
bodies, and annotation categories.

| Property | Light types | Value / units |
|---|---|---|
| `color` | All | Linear RGB, each channel in `[0, 1]` |
| `power` | All | Watts; `SUN` uses watts per square meter |
| `radius` | `POINT`, `SPOT` | Emitter radius, meters |
| `angle` | `SUN` | Angular diameter, radians, `[0, pi]` |
| `spot_size` | `SPOT` | Cone angle, radians, `[pi/180, pi]` |
| `spot_blend` | `SPOT` | Edge softness, `[0, 1]` |
| `shape` | `AREA` | `SQUARE`, `RECTANGLE`, `DISK`, `ELLIPSE` |
| `size` | `AREA` | Side length / width / diameter, meters |
| `size_y` | `AREA` | Height for `RECTANGLE` and `ELLIPSE`, meters |

Power and dimensions must be non-negative and finite. Omitted properties keep
BlenderProc's defaults. Update them with `light.set_properties(power=100, ...)`.
Unsupported properties raise instead of being silently ignored.
`light.get_properties()` returns current settings using those same property
names, without exposing renderer internals.

```python
from telekinesis.illusion.randomizer.randomizer_node import (
    LightRandomizer,
    LightPoseRandomizer,
)
from telekinesis.illusion.sampler.camera_pose_sampler import shell_sampler
from telekinesis.illusion.types.distribution import uniform

# Assuming context already contains your models and randomizer is a Randomizer.
context.add_light("key", "AREA", power=80, size=0.4)
randomizer.add_randomizer(
    LightRandomizer(
        target_lights=["key"],
        seed=42,
        color=uniform((0.8, 0.85, 0.9), (1.0, 1.0, 1.0)),
        power=uniform(40, 100),
        shape=["RECTANGLE", "DISK"],
        size=uniform(0.3, 0.6),
        size_y=uniform(0.2, 0.4),
    ),
    node_name="light_properties",
)
randomizer.add_randomizer(
    LightPoseRandomizer(
        shell_sampler,
        target_lights=["key"],
        min_distance=0.3,
        max_tries=100,
        radius_min=1.2,
        radius_max=1.6,
        elevation_min=40,
        elevation_max=80,
    ),
    node_name="light_pose",
)
```

`LightRandomizer` accepts fixed values or the existing `uniform(min, max)`
distribution. Area `shape` also accepts a list/tuple of choices. Each light
gets an independent sample each time the node runs. A seed controls this node's
property sampling; it does not seed the pose sampler. Without a seed, properties
use Python's global `random` stream; the existing shell sampler uses NumPy's
global stream. `target_lights=None` selects all registered lights. Use separate
property nodes for different light types when setting type-specific properties.

Add `LightPoseRandomizer` after object placement and light property sampling.
Its callable has the same signature as a camera pose sampler:
`sampler(context=context, **kwargs)` returns `(location, rotation)` with XYZ
Euler rotation in **radians**. `set_location` and `set_rotation` use these same
units. The existing `shell_sampler` points local `-Z` toward its center, making
it useful for spot, area, and sun lights. Shell elevation, azimuth, and in-plane
rotation arguments are in **degrees**, just as for cameras. Positive elevations
provide overhead lighting.

Clearance includes the light's emitter radius (or a sphere enclosing an area
light) and the world bounding boxes of **all visible scene meshes**, including
containers and scenery outside Context. Hidden meshes are ignored. This is
conservative: hollow containers and empty corners around rotated meshes can
reject otherwise valid positions. A sun's location has no physical effect, so
only its orientation matters and positional clearance does not apply.

If no valid pose is found within `max_tries`, the node raises `RuntimeError`
and leaves all targeted light poses unchanged. Invalid sampled coordinates
raise immediately. Clearance applies to geometry at the time the node runs.
When physics is enabled, `SyntheticDataGenerator` reruns enabled light pose
nodes after simulation, in graph order, to check the settled geometry and aim
at the final targets. It retains the sampled light properties. Outside that
pipeline, call `randomizer.randomize_light_poses(context)` after moving objects.

Both [flying things](../examples/quickstart_flying_things.py) (point and sun)
and [parts in a bin](../examples/quickstart_parts_in_bin.py) (area and spot)
show property and pose randomization alongside HDRI backgrounds. Run either
with `--no-preview` to generate its dataset without opening the viewer.

## Bin-picking worker YAML

`BinPickingWorker` accepts an optional `light_randomizer` block. The example
gearwheel spec enables an area key light and a spotlight fill, aimed
at the visible bin and combined with the existing HDRI lighting. Omit this
block or set `active: false` to create no worker-managed lights.

```yaml
light_randomizer:
  active: true
  lights:
    - name: bin_key
      type: AREA
      seed: 42
      properties:
        power: { min: 40, max: 100 }
        color: { min: [0.85, 0.85, 0.85], max: [1, 1, 1] }
        shape: [RECTANGLE, DISK]
        size: { min: 0.3, max: 0.6 }
        size_y: 0.3
      pose:
        sampler: shell_sampler
        min_distance: 0.3
        max_tries: 100
        params:
          center: [plastic_bin_]
          radius_min: 1.2
          radius_max: 1.6
          elevation_min: 50
          elevation_max: 80
```

Each light requires a unique `name`, a non-empty `properties` mapping, and a
`pose` mapping. `type` accepts `POINT` (default), `SUN`, `SPOT`, or `AREA`.
Numeric properties accept fixed values or `{min: ..., max: ...}` ranges; color
uses a fixed RGB vector or vector bounds, and area shapes can be a list of
choices. Property names and units are those in the table above. The optional
per-light `seed` controls property sampling only.

Pose `sampler` names use the existing camera sampler registry (default:
`shell_sampler`). `params` are forwarded to that sampler; shell `center` can
be numeric coordinates or object name prefixes. Omit the center to aim at the
visible scene objects. Placement retains emitter clearance checks and is
rechecked after physics. Light properties run in the appearance stage; poses
run in the pose stage so `randomize_geometry()` also updates light placement.

`apply_spec_updates()` replaces property and pose nodes for existing lights
without importing assets again. Adding/removing/renaming lights, changing
their types, or toggling the whole block requires a worker reload and is
reported in the method's return value. Unchanged settings retain their random
sampling state. Adjust these example powers to your camera exposure and HDRIs.

## Mechanical part comparison

```bash
python examples/preview_light_randomization.py
```

The [light preview](../examples/preview_light_randomization.py) renders a
3-by-3 comparison of a steel gearwheel under nine lighting setups. The model,
material, camera, exposure, matte floor, and dim ambient fill remain fixed.
Each panel renders with one active light, so changes in highlights and shadows
come from that light alone. All scene operations use Illusion's public API;
the example imports neither BlenderProc nor bpy. It renders through
`SyntheticDataGenerator` and `CocoWriter`, as the quickstarts do.

Read the panels from left to right, then top to bottom:

1. White point reference, sampled higher power, and sampled larger radius.
2. Sampled warm color, sun angular diameter, and spotlight cone/blend.
3. Sampled area shape/dimensions, disk area reference, and that same disk with
   a sampled pose constrained by mesh clearance.

The two references are fixed; other cases use `LightRandomizer` or
`LightPoseRandomizer`. Edit `lighting_variants()` to change property ranges.
The last two panels share identical light properties to isolate pose changes.

Outputs under `output/light_preview/gearwheel/`:

- `light_variants.png`: labeled comparison.
- `images/`: the individual PNG panels, also referenced by `inventory.json`.
- `coco_annotations.json`: annotations and each image's light variant/camera pose.
- `inventory.json`: sampled properties, poses, active lights, panel order, seed,
  and the fixed camera pose.
- `scene_assets/`: a generated floor mesh and constant environment texture,
  loaded through the same asset APIs as the mechanical part.

Use `--model path/to/part.glb` for another single-mesh part,
`--resolution 256` for smaller panels, `--seed 7` for different samples,
or `--output-dir path/to/output`. Rendering uses the generator's normal device
selection. `--no-render` exercises the same scene setup and sampling, saving
the inventory and scene assets without rendering frames. Like the material
preview, the Windows entry point bypasses the
known native Blender teardown failure after successful work and scene cleanup.

## Validation

The mechanical-part preview rendered all nine 512-pixel panels and exited
successfully. Both preview smoke tests pass, covering rendering, `--no-render`,
relative output paths, isolated lights, fixed camera poses, and grid assembly.

The light tests, physics regression tests, and both rendering quickstarts
reported **53 passed** on 2026-10-09 with the repository's Blender 4.2.17
environment. Each quickstart wrote six images and COCO annotations. The
Windows process still returned exit status 1 after the passing report, with
the native Blender import/shutdown issue also documented in
[the existing validation notes](MATERIAL_RANDOMIZATION_VALIDATION.md).

```powershell
.venv/Scripts/python.exe -m pytest tests/test_light_randomization.py tests/test_physics_catch_plane.py tests/test_examples.py -q -k 'not worker and not view_dataset and not material_preview'
```
