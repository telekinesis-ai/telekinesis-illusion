# Material randomization

See [validation results](MATERIAL_RANDOMIZATION_VALIDATION.md) for tests,
supplied-asset inventories, rendered previews and environment limitations.

`MaterialRandomizer` supports texture-based PBR materials and procedural
Principled BSDF recipes, with optional surface-imperfection textures. Both use
the same selection and assignment pipeline.
The examples below use the public Python API; the workers also accept the YAML
mapping shown below.

## Compatibility and preprocessing

Existing calls retain their behavior:

```python
context.add_model("part.glb", "part", preprocess_model=True)
MaterialRandomizer(["part"], context, types=["metal"]).randomize(context)
```

Preprocessing defaults to `material_preprocessing="replace"`: remove the old
slots, add one dummy material, then randomize the active slot. Without
preprocessing, legacy randomizers still change **only the active slot**.
Their target lists still match object-name prefixes; an unmatched prefix is a
no-op. Empty worker role lists still disable randomization for that role.

For slot selection, preserve materials **when loading**, before they can be
discarded:

```python
context.add_model(
    "part.glb", "part", preprocess_model=True,
    material_preprocessing="preserve",
)
```

This keeps material assignments while retaining the existing UV, shading,
scaling and visibility preprocessing. `preprocess_model=False` also preserves
materials. A slot randomizer on an object that was already collapsed raises a
clear error asking you to reload with `material_preprocessing="preserve"`.
Original materials cannot be reconstructed after destructive preprocessing.

## Object and slot selection

```python
from telekinesis.illusion.core.context import Context
from telekinesis.illusion.randomizer.randomizer_node import MaterialRandomizer
from telekinesis.illusion.randomizer.materials import MaterialTarget, PBRMaterial
from telekinesis.illusion.types.material import MaterialPreset, PrincipledMaterial

context = Context()
context.add_model("part.glb", "part", material_preprocessing="preserve")
metal = PrincipledMaterial("metal", roughness=(0.2, 0.6))
plastic = PrincipledMaterial("plastic")
rubber = PrincipledMaterial("rubber")

# One sample for every slot of each selected mesh; creates a slot if absent.
MaterialRandomizer(
    ["part"], context, mode="object", materials=[metal], seed=42,
).randomize(context)

# Sample every existing slot independently.
MaterialRandomizer(
    ["part"], context, material_slots="all",
    materials=[plastic, rubber], seed=42,
).randomize(context)

# Mapping keys are original material names or zero-based indices.
MaterialRandomizer(
    ["part"], context,
    material_slots={"cavity.002": [plastic, metal], "metal_part": [metal]},
    seed=42,
).randomize(context)
```

These are independent examples: replace the slot names with the names in your
asset. Inspect them using `obj.get_material_slot_names()`.

| Configuration | Sampling/assignment |
| --- | --- |
| Existing `types=...` call without a mode | One PBR sample per mesh, assigned to its active slot |
| `mode="object"` | One sample per mesh assigned to all its slots |
| `materials=...`, mode omitted | Whole-object mode |
| `material_slots="all"` | Independent sample for every slot |
| `material_slots={name_or_index: choices}` | Independent sample for each selected slot |

Whole-object assignment preserves existing slots and polygon indices. It does
not change mesh topology. Non-selected slots and objects retain their material
references and shader graphs. Assignment uses object-level slots, allowing
linked instances to have different materials without copying their geometry.
An empty linked mesh is copied only if a new material slot must be added.

Slot names are captured after import, since Blender normally changes a slot's
name when its assigned material changes. Repeated randomization therefore
continues to address the original slot. If you deliberately edit the slot
layout yourself, call `obj.refresh_material_slot_names()` afterwards. Name
matching is exact and case-sensitive, including Blender's `.001` suffixes.

- Missing slots raise `ValueError`, including selecting all slots on a mesh
  with none. Use whole-object mode to create a slot on such meshes.
- Duplicate names are ambiguous and raise; select them by integer index.
- Overlapping selectors (e.g. a name and its index) raise.
- Every named slot must exist on **each** selected mesh. Use separate nodes
  for parts with different layouts, or `material_slots="all"`.
- Target/slot resolution happens before sampling or assignment, so selection
  errors do not partially change the selected objects.

## Supported model input and exact object selection

Each input model must contain **one mesh object without empties**. Models with
multiple objects or parent/child structures are a known limitation and are
reserved for future work. Material slots on the supported single mesh can be
randomized independently.

Use the existing `Context.add_model()` API to load a mesh and create linked
instances. `MaterialTarget` selects exact registered context names:

```python
# One registered instance.
MaterialRandomizer(
    MaterialTarget(objects=("part_INSTANCE_0",)), context,
    materials=[metal], seed=42,
).randomize(context)

# All registered objects, each backed by one mesh.
MaterialRandomizer(
    MaterialTarget(), context, materials=[plastic], seed=42,
).randomize(context)
```

Inspect `context.get_objects()` for registered names. Lists such as `["part"]`
retain legacy prefix matching and select the model's matching instances.
`get_object()` returns the single BlenderProc mesh wrapper. Loading, transforms,
visibility, physics and linked duplication use the existing single-mesh path.

## Principled recipes and presets

The supported parameters correspond to Blender 4.2 LTS's
[Principled BSDF](https://docs.blender.org/manual/en/4.2/render/shader_nodes/shader/principled.html).
By default materials contain one Principled node connected to Material Output,
without image textures. Render-engine settings still govern transparency/refraction.

| Parameter | Values |
| --- | --- |
| `base_color`, `emission_color` | Linear RGB/RGBA in [0, 1], or `(lower_color, upper_color)`; RGB gets alpha 1 |
| `metallic`, `roughness`, `specular_ior_level`, `transmission`, `alpha` | Number in [0, 1] or `(min, max)` |
| `coat_weight`, `coat_roughness`, `subsurface_weight` | Number in [0, 1] or `(min, max)` |
| `ior` | Number in [1, 1000] or `(min, max)` |
| `subsurface_radius` | Three non-negative distances or a pair of lower/upper vectors |
| `emission_strength` | Finite non-negative number or `(min, max)` |

The existing `uniform(min, max)` distribution also works. Vector components
sample independently. Omitted sockets keep Blender's defaults. Invalid keys,
non-finite values, reversed ranges and out-of-bounds values fail at recipe
construction, instead of relying on Blender's silent clamping.
Distances and emission strength must also fit Blender's 32-bit float sockets.

Built-in `metal`, `plastic` and `rubber` presets are immutable parameter
collections, not branches in the randomizer. Overrides are merged into a copy:

```python
blue_plastic = PrincipledMaterial(
    "plastic", base_color=(0.05, 0.2, 0.8), roughness=(0.25, 0.4),
)
painted = MaterialPreset("painted_metal", {
    "metallic": 0.0, "roughness": (0.2, 0.4), "coat_weight": 0.7,
})
painted_recipe = PrincipledMaterial(painted, coat_roughness=0.2)
```

Unknown preset names raise. Recipes do not mutate presets or caller-owned
lists/distributions. Future presets can be supplied as `MaterialPreset`
instances without modifying the randomizer.

Pools may mix recipes, BlenderProc `Material` instances, PBR strings, and
`PBRMaterial(types=("metal", "plastic"))`. Strings always mean existing PBR
catalog categories, **not procedural preset names**. Custom sources implement
`prepare(manager)` and `generate(manager, rng)` and return a BlenderProc
material. Generation is separate from assignment. Only unused generated
procedural materials are reclaimed; imported, user-supplied and PBR materials
are not deleted or edited.

## Surface imperfections

Enable imperfections on individual procedural recipes with
`surface_imperfections=True`. The default is **off**. Both whole-object and
material-slot randomization support it; PBR catalog strings keep their existing
behavior. Assets are read from `<asset_directory>/surface_imperfections`, so
`metadata.asset_directory: E:/telekinesis-illusion/assets` uses the supplied
folder without copying it. Python callers can set the root via
`Context(asset_dir="E:/telekinesis-illusion/assets")`.

```python
from telekinesis.illusion.types.material import (
    PrincipledMaterial, SurfaceImperfections,
)

scratched_metal = PrincipledMaterial("metal", surface_imperfections=True)
subtle_plastic = PrincipledMaterial(
    "plastic",
    surface_imperfections=SurfaceImperfections(
        directory="surface_imperfections",  # Relative to the context asset root.
        roughness_strength=(0.1, 0.2),
        bump_strength=(0.05, 0.1),
        bump_distance=0.0001,
        scale=(1.0, 3.0),
    ),
)
```

The YAML option is a sibling of `preset` and `parameters`:

```yaml
metadata:
  asset_directory: E:/telekinesis-illusion/assets

material_randomizer:
  container:
    mode: object
    seed: 42
    materials:
      - preset: plastic
        surface_imperfections: true
```

For explicit worker `rules`, put the same option on each recipe in `materials`
or `material_slots`. Use a mapping for custom settings:

```yaml
- preset: metal
  surface_imperfections:
    enabled: true
    directory: E:/telekinesis-illusion/assets/surface_imperfections
    roughness_strength: [0.1, 0.3]
    bump_strength: [0.05, 0.2]
    bump_distance: 0.0001
    scale: [1.0, 3.0]
  parameters:
    roughness: [0.15, 0.35]
```

These are the default numeric settings. Each accepts a fixed number or a
uniform `[minimum, maximum]` range. Strengths must be in [0, 1], bump distance
non-negative, and scale positive. `enabled: false` skips asset discovery and
leaves the original procedural shader and random sequence intact. To limit
sampling to one set, point `directory` at that set's subfolder. A missing folder
or a folder without `*_Opacity` maps fails when constructing the randomizer.

### Shader setup

One map set is selected per generated material, recursively and uniformly from
the folder's `*_Opacity` image sets. The five supplied `Scratches` sets use
their opacity masks; the three `SurfaceImperfections` sets also supply
roughness and displacement maps. Preview thumbnails, color textures, and
NormalDX/NormalGL maps are not used.

1. **Texture Coordinate (UV) -> Mapping -> Image Textures (Flat).** Mapping
   samples a uniform UV scale, U/V offsets in [0, 1], and a rotation of 0, 90,
   180, or 270 degrees within the UV plane. All maps share this transform and
   repeat across the mesh's rendering UV layout. Scale is relative to UV space,
   so the unwrap determines the size and orientation of marks on the surface.
   When assigning an imperfection material to a mesh without any UV layer,
   Illusion creates one using the same **smart projection** used by default
   in PBR model preprocessing. This also works with `preprocess_model=False`.
   Existing UV layers are preserved by material assignment, including multiple
   layers; the model-loading UV preprocessing policy is unchanged. Missing
   UVs on a linked mesh are generated on a private copy, leaving untargeted
   instances intact. Repeated randomization reuses the resulting layout.
   Data textures use Non-Color, following Blender's
   [image texture guidance](https://docs.blender.org/manual/en/4.2/render/shader_nodes/textures/image.html).
2. **Opacity mask -> Multiply by roughness strength -> Mix -> Principled
   Roughness.** With `m = mask * roughness_strength`, the result is
   `clamp((1 - m) * sampled_base_roughness + m * texture_roughness, 0, 1)`.
   When a roughness map is absent, `texture_roughness = 1`. Unmasked areas
   retain the sampled recipe roughness.
3. **Masked displacement -> Bump -> Principled Normal.** Where displacement is
   available, its product with opacity supplies height. Otherwise opacity
   supplies height with Bump's Invert enabled, approximating recessed scratches.
   Bump strength and distance control the relief; distance is in Blender units
   (0.0001 is 0.1 mm when one unit represents one meter). NormalDX/NormalGL
   textures remain unused; detail comes from the scalar bump height.
4. **Principled BSDF -> Material Output Surface.** Base color, metallic,
   transmission, emission and alpha retain their recipe values. Opacity is
   used only as an imperfection mask. Vertex positions and output Displacement
   are unchanged, so silhouette, physical simulation and geometry-based annotations
   are unaffected.

Map selection and all sampled controls use the material randomizer's existing
seeded RNG. Keep the asset catalog unchanged for reproducibility. Images are
reused across live generated materials; unused generated materials and their
unused imperfection images are reclaimed. Imported image color spaces are not
modified.

### Plastic bin example

```bash
python examples/preview_bin_surface_imperfections.py --asset-dir E:/telekinesis-illusion/assets
```

This example uses only public `telekinesis.illusion` scene APIs, with no
`blenderproc`, `bpy`, or `mathutils` imports. It renders three identical blue
plastic bins in one row: **clean, scratches, patchy wear**, left to right.
The base plastic color and roughness are fixed; only imperfections change.
Scratch and wear families are pinned to `Scratches003_1K-JPG` and
`SurfaceImperfections015_1K-JPG`, using stronger settings than the defaults to
make the effects visible. Change the recipe to `surface_imperfections=True`
to sample across the whole catalog with default strengths.

The output directory defaults to
`output/material_preview/plastic_bin_imperfections/` and contains a PNG under
`images/`, COCO annotations, and `inventory.json` describing the comparison.
Reruns append another image to the existing dataset. `--output-dir` chooses a
separate directory, `--seed` changes texture placement, `--resolution` sets
image height (width is three times this), and `--no-render` checks scene setup
and writes only the inventory. `--model`, `--imperfections-dir`, and `--hdri`
accept absolute paths or paths relative to `--asset-dir`. The default bin is
`models/bins/plastic_bin_3.glb`; lighting defaults to the first studio HDRI.

## YAML worker configuration

The legacy `material_randomizer.part: [metal]` form still works. Each role can
instead use an explicit mapping parsed by `MaterialRandomizer.from_config()`:

```yaml
models:
  # Other required model fields are omitted here.
  - name: part
    preprocess_model: true
    material_preprocessing: preserve

material_randomizer:
  part:
    seed: 42
    material_slots:
      cavity.002:
        - preset: plastic
          parameters:
            roughness: [0.2, 0.6]
        - metal  # PBR texture category
      metal_part:
        - preset: metal
  container:
    mode: object
    seed: 43
    materials:
      - preset: plastic
  distractor: []
```

Use `material_slots: all` with a top-level `materials` list for all slots.
`types` and `materials` are mutually exclusive. A slot mapping contains its
own choices and cannot also specify either top-level pool. Unknown mapping
keys, invalid source types and empty pools fail early. Unavailable PBR folders
raise `FileNotFoundError`; empty catalogs raise `ValueError`. Wrong configuration
types raise `TypeError`. Procedural worker rules support these same keys alongside
their existing `name` and `target_objects` fields.

## Seeding and preview

`seed=42` creates a private Python RNG stream per node. Each call advances it;
construct a new node with the same seed to replay the sequence. Without a seed,
sampling uses the existing global `random` state (`random.seed(...)` or
`BLENDER_PROC_RANDOM_SEED`). NumPy seeds alone do not control material sampling.
`NodeConfig.seed_key` remains unused by the existing graph executor. Exact
all-object targets use sorted names; explicit target/slot lists keep their
configuration order. Recipe parameter sampling uses canonical socket order.
Keep object names and configuration ordering fixed to reproduce results.

```bash
python examples/preview_material_randomization.py
python examples/preview_material_randomization.py --no-render
pytest tests/test_material_randomization.py --material-model path/to/model.glb
```

The example defaults to `assets/models/mechanical_parts/heavy_duty_wheel.glb`;
use `--model` if that file is elsewhere. Its three slots are `Stahl` (steel),
`Alu` (aluminum), and `Gumi`
(rubber). The seven side-by-side models show the original, blue plastic on
`Alu`, procedural and PBR metal on `Stahl`/`Alu`, metal/rubber on `Alu`/`Gumi`,
all-slot randomization, and whole-object PBR metal. The combined render is
written to
`output/material_preview/heavy_duty_wheel/material_variants.png`;
the inventory records the material assigned to each original slot in every case.
Use `--seed 7` to change the random choices, `--output-dir` for a separate output
directory, or `--no-render` to check assignments without producing images.
The example expects the wheel's exact slot names. Edit the example's
`material_slots` mappings and recipe parameters to explore different effects.

## Implementation map

- `randomizer_node.MaterialRandomizer` orchestrates validated assignment plans.
- `randomizer.materials` contains target/slot selectors, material source adapters,
  pool selection and Blender assignment/cleanup. `SlotSelector.resolve()` is the
  extension point for further selection strategies.
- `types.material` contains Blender-independent recipes, range validation and
  composable presets. PBR uses the existing manager/texture loader.
- `randomizer.surface_imperfections` discovers mask sets and builds the optional
  roughness/bump layer; `types.material.SurfaceImperfections` holds its settings.
- `types.object` wraps one mesh and owns preprocessing and stable original-slot
  identities.
- `core.context` owns the existing model instance pools and PBR catalog.
- Worker parsers adapt existing YAML roles/rules to the same public API.
- `tests/test_material_randomization.py` covers real Blender assignments,
  generated shader graphs, imported models, compatibility and pure sampling.
