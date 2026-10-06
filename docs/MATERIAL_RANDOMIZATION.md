# Material randomization

See [validation results](MATERIAL_RANDOMIZATION_VALIDATION.md) for tests,
supplied-asset inventories, rendered previews and environment limitations.

`MaterialRandomizer` supports texture-based PBR materials and texture-free
Principled BSDF recipes. Both use the same selection and assignment pipeline.
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
Materials contain one Principled node connected to Material Output, without
image textures. Render-engine settings still govern transparency/refraction.

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
python examples/preview_heavy_duty_wheel.py
python examples/preview_heavy_duty_wheel.py --no-render
pytest tests/test_material_randomization.py --material-model path/to/model.glb
```

The example defaults to `assets/models/mechanical_parts/heavy_duty_wheel.glb`;
use `--model` if that file is elsewhere. Its three slots are `Stahl` (steel),
`Alu` (aluminum), and `Gumi`
(rubber). The seven previews show the original, blue plastic on `Alu`, procedural
and PBR metal on `Stahl`/`Alu`, metal/rubber on `Alu`/`Gumi`, all-slot
randomization, and whole-object PBR metal. Each case restores the original
materials first. Outputs go to `output/material_preview/heavy_duty_wheel/`;
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
- `types.object` wraps one mesh and owns preprocessing and stable original-slot
  identities.
- `core.context` owns the existing model instance pools and PBR catalog.
- Worker parsers adapt existing YAML roles/rules to the same public API.
- `tests/test_material_randomization.py` covers real Blender assignments,
  generated shader graphs, imported models, compatibility and pure sampling.
