# Changelog

Notable changes to `telekinesis-illusion` are recorded here, newest first.

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

## Unreleased

### Added

- Added `ModelManager` in `telekinesis.illusion.core.model_manager`, exposed as
  `environment.model_manager`. It owns model registration, annotation
  categories, fixed-capacity instance pools, activation, sample resets, and
  model cleanup state. Existing `Environment` model methods delegate to it;
  the environment coordinates shared Blender scene initialization and teardown.
  Preparation and cleanup state are available as `is_prepared` and `is_closed`
  on the manager. Worker capacity updates use `is_prepared` instead of the
  removed private `Environment._prepared` field.
- Added Blender-independent, immutable `MaterialAsset` and `BackgroundAsset`
  descriptors, exported alongside `ModelAsset` from `telekinesis.illusion.types`.
  All three support local `from_path()` and deterministic recursive `discover()`.
  Material discovery recognizes the existing color-texture convention and stops
  descending at an asset directory; model formats match the existing loader.
  Background discovery accepts `.hdr` and `.exr`. URL support is deferred.
- Added normalized selection groups with forward-slash ancestor membership.
  Discovery derives groups from relative parent directories and accepts
  `group_prefix`; material/background assets directly in the collection root
  require this prefix. Direct material/background creation requires groups.
  `ModelAsset` adds optional groups after all existing positional fields and
  read-only `path`/`name` properties; annotation categories are unchanged.
- Added `MaterialManager.add()` and `BackgroundManager`, with unique-name
  descriptor storage and ordered group indexes. `Environment.add()` accepts a
  model, material, background, light, or an iterable of them, such as the list
  returned by `discover()`. Registration does not load textures or activate
  HDRIs; randomizers select registered assets later.
- Added `AssetManager`, the shared base of `ModelManager`, `MaterialManager`,
  and `BackgroundManager`. `get_names(groups, names)` selects the union of the
  requested groups and names, or every asset when neither is given, and
  raises `ValueError` for unknown groups or names. `ModelManager` now indexes
  `ModelAsset.groups` through it.

### Changed

- Moved `MaterialManager` to `telekinesis.illusion.core.material_manager`,
  alongside `BackgroundManager` in its own module. The existing
  `telekinesis.illusion.core.environment.MaterialManager` import remains
  supported; manager behavior is unchanged.
- **Breaking:** `Environment` and `MaterialManager` no longer discover the
  `materials/` and `hdris/` collections on construction. Register them
  explicitly, for example
  `environment.add(MaterialAsset.discover(assets_dir / "materials"))` and
  `environment.add(BackgroundAsset.discover(assets_dir / "hdris"))`. Randomizers
  only select from registered assets. Duplicate names within a type raise
  `ValueError`, even across folders; give colliding descriptors distinct names.
  Discovery no longer drops materials with an opacity texture; filter the
  discovered list if transparent sets should be excluded. The bin-picking
  worker keeps excluding them. `MaterialManager(asset_dir=...)` now only
  resolves relative surface-imperfection directories.
- **Breaking:** `MaterialRandomizer(types=...)` is now `groups=...` and accepts
  `names=...` for individual materials; `PBRMaterial(types)` takes
  `groups, names`, `MaterialManager.update_materials()` and
  `get_random_material()` take `groups, names`, and `MaterialRandomizer.types`
  is replaced by `groups` and `names`. Selection is the union of groups and
  names; with only groups, a group is drawn first, then a material within it.
  YAML configurations keep the `types` key. Procedural `PrincipledMaterial`
  recipes and their value ranges are unchanged.
- `MaterialManager.get_random_material()` selects registered assets and loads
  an uncached selection on demand. With no selection, it considers all
  registered top-level groups. Callers that previously relied on loading only a
  subset with `update_materials()` must also pass that subset when sampling.
- `Background()` now only creates the live world shader setup. Construct it and
  call `set_background_texture(path)`, or use `BackgroundRandomizer` with
  registered backgrounds. Asset lookup belongs to the environment;
  `Background.assets_path` and `Background.DEFAULT_HDRI_CATEGORY` are removed,
  and no default HDRI is selected.
- **Breaking:** `BackgroundRandomizer(categories=..., hdris_root=...)` is now
  `BackgroundRandomizer(groups=..., names=...)`. It selects from the
  environment's registered backgrounds by exact group (such as
  `indoor/industrial` or `indoor`) or name, and no longer accepts a root, keeps
  a private catalog, or discovers files. Replace `hdris_root` with
  `Environment.add(BackgroundAsset.discover(root))`. `get_random_hdri()`,
  `catalog`, and `categories` are removed; the requested selection is exposed
  as `groups` and `names`. YAML configurations keep the `categories` key.
- Removed the unused `Environment.get_materials()` and `set_materials()`.
- Updated quickstarts, the tutorial, and asset documentation for local creation
  and registration. Deterministic material ordering may change previously
  seeded material choices; category-first sampling remains unchanged.
- Renamed the live scene-instance API from `Object` to `Model`. Import `Model`
  and `compute_poi` from `telekinesis.illusion.types.model`. `ModelAsset` remains
  the frozen reference and import configuration; one asset can produce multiple
  linked `Model` instances with mutable transforms, materials, visibility, and
  physics. Models still contain a single mesh; multi-mesh support is not added.
- Renamed `object_name` to `model_name` in `ModelAsset` and model APIs. The
  `Model` constructor accepts `mesh_object=` instead of `object=`, its mesh
  accessor is `get_mesh_object()` instead of `get_object()`, and `hide()` accepts
  `hide_model=` instead of `hide_object=`. `compute_poi()` now accepts `models=`
  instead of `objects=`.
- Renamed environment accessors to `get_models()`, `get_models_by_name()`,
  `get_model_group()`, `get_visible_model_names()`, and
  `set_visible_model_names()`. Replace the corresponding `object`/`objects`
  names in existing calls. `activate()` and `deactivate()` now accept `model=`
  instead of `obj=`; visibility setters use `visible_model_names=`.
- Renamed `ObjectPoseRandomizer` and `ObjectInstanceRandomizer` to
  `ModelPoseRandomizer` and `ModelInstanceRandomizer`. Pose sampling uses
  `target_models`, `models_to_sample`, and `models_to_check_collisions` instead
  of their `objects` counterparts, and `surface_model` instead of `surface`.
  Instance limits are now `min_num_total_models` and `max_num_total_models`.
- Renamed material targeting to `MaterialTarget.models` and
  `MaterialRandomizer(target_models=...)`; material mode `"object"` is now
  `"model"`. Worker material rules use the `target_models` key. Camera bounding
  box sampling also uses `target_models`; region sampling uses
  `model_names_to_sample_on`, and grid sampling uses `target_model_names`.
- Renamed physics settings to `check_model_interval`,
  `model_stopped_location_threshold`, and `model_stopped_rotation_threshold`
  in `PhysicsConfig`, generator keyword arguments, worker defaults, and YAML
  configuration. Catch-plane configuration uses `tracked_model_names` instead
  of `tracked_object_names`. Defaults and simulation behavior are unchanged.
- Renamed the bin-picking worker's `instance_randomizer_objects` node to
  `instance_randomizer_models`; update custom node lookups accordingly.
- Updated examples, configuration, documentation, and model-related messages
  to use the model API. Native Blender APIs, Python's `object`, and dataset
  object-detection terminology remain unchanged.
- Renamed `Scene` to `Environment`. Import `Environment` and `MaterialManager`
  from `telekinesis.illusion.core.environment`. The environment represents the
  active Blender scene and owns reusable resources; a scene is one realization
  produced by the randomizer tree.
- Renamed `SyntheticDataGenerator`'s `scene` constructor argument and attribute
  to `environment`. Its constructor now explicitly requires `environment`,
  `randomizer`, and `writer` arguments.
- Renamed `clean_up_scene` to `clean_up_environment` in
  `SyntheticDataGenerator.generate()`, `blender.initialization.init()`, and
  the bin-picking worker's defaults and generator calls.
- Renamed the `context` argument to `environment` throughout randomizers,
  nodes, material targets, samplers, and worker APIs. Custom camera and light
  pose samplers must accept `environment=`. The bin-picking worker now exposes
  `get_environment()` instead of `get_context()`.
- Updated quickstart examples, the tutorial, architecture and development
  documentation, and API docstrings to use the environment API and distinguish
  reusable environments from scene realizations. A realization lives in the
  environment's mutable Blender state; there is no separate `Scene` class.

### Fixed

- Saving a Blender scene with `save_blender_scene=True` now writes
  `scene_<uuid>.blend` with a fresh UUID in the writer's output directory,
  avoiding overwrites of the previous fixed `scene.blend` filename.

### Removed

- Removed the `Object` class and `telekinesis.illusion.types.object` module.
  The model API renames above are breaking changes without compatibility
  aliases; migrate external imports, keyword arguments, node lookups, and YAML
  keys to the new names.
- Removed the unused `ProceduralSceneWorker` and its
  `workers.procedural_scene_worker` module.
- Removed the deprecated `Context` alias and `core.context` module, and the old
  `core.scene` module. Pass `environment=` to `SyntheticDataGenerator`;
  its old `scene=` and deprecated `context=` keywords are no longer accepted.
- Removed the unused `auto_config` constructor argument when renaming `Scene`
  to `Environment`. Remove this argument from existing calls; `camera_config`
  is now the first constructor parameter.

## 0.2.0 - 2026-10-21

### Changed

- Introduced the BlenderProc-free version of Illusion: rendering uses
  Blender's Python API (`bpy`) without requiring the `blenderproc` package.
