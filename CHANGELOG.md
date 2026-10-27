# Changelog

Notable changes to `telekinesis-illusion` are recorded here, newest first.

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

## Unreleased

### Added

- `resolve_asset_dir()`, and therefore `Environment()` and the quickstarts,
  fall back to the public example assets when no explicit directory,
  `./assets`, or checkout `assets/` exists. The first lookup downloads
  `https://assets.telekinesis.ai/telekinesis-illusion/assets-<VERSION>.zip` and
  extracts it to `~/.cache/telekinesis//telekinesis-illusion/assets-<VERSION>/assets`; later
  lookups reuse the cache. Archive members with paths outside the extraction
  directory are rejected, and an interrupted download leaves no partial
  cache. `resolve_asset_dir()` no longer raises `FileNotFoundError` when no
  local directory is found; a failed download raises `httpx.HTTPError`.
- Added the `telekinesis.illusion.config` module. Every config derives from
  `BaseConfig`, is immutable, validates its values on construction, converts to
  and from plain dictionaries with `to_dict()` and `from_dict()`, and returns a
  validated copy with `replace()`. `from_dict()` rejects unknown and missing
  keys. Invalid values raise `ValueError` or `TypeError` with the key path of
  the value, for example `params.radius_min: must be a number >= 0.0, got -1.`.
  The module does not import Blender.
- Added randomizer configs `InstanceRandomizerConfig`,
  `ModelPoseRandomizerConfig`, `MaterialRandomizerConfig`,
  `PrincipledMaterialConfig`, `BackgroundRandomizerConfig`,
  `LightRandomizerConfig`, `LightPoseRandomizerConfig` and
  `CameraPoseRandomizerConfig`, and the pose sampler parameter configs
  `ShellSamplerConfig` and `VolumeSamplerConfig`.
- Added the worker section configs `DatasetConfig`, `CocoConfig`,
  `CocoLicenseConfig`, `ShardConfig`, `WriterConfig`, `PoseSamplingConfig` and
  `GridSamplingConfig`. `DatasetConfig` holds the `shard` and `writer`
  configs; `WriterConfig` holds the COCO info and licenses as `coco`.
- Added `configs/bin_picking_example.yaml`, the gearwheel bin-picking example in
  the new configuration format. The worker example uses it by default.
- `shell_sampler` accepts a `center` list of three numbers as coordinates.
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
  Background discovery accepts `.hdr` and `.exr`.
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

- Added `LightManager` and `CameraManager` in
  `telekinesis.illusion.core`, exposed as `environment.light_manager` and
  `environment.camera_manager`. They own the lifecycle of the lights and the
  camera. `Environment.get_light()`, `get_lights()` and `get_camera()` delegate
  to them.
- Added `LightConfig` in `telekinesis.illusion.types.light`, a validated
  description of a light that creates nothing in Blender until it is registered.

### Changed

- **Breaking:** All configs live in `telekinesis.illusion.config` and derive
  from `BaseConfig`. Import the module and use `config.<Name>`:

  | Old | New |
  | --- | --- |
  | `types.model_asset.ModelAsset` | `config.ModelConfig` |
  | `types.material_asset.MaterialAsset` | `config.MaterialConfig` |
  | `types.background_asset.BackgroundAsset` | `config.BackgroundConfig` |
  | `types.asset.Asset` | `config.AssetConfig` |
  | `types.material_asset.find_color_texture` | `config.find_color_texture` |
  | `types.camera.CameraConfig` | `config.CameraConfig` |
  | `types.light.LightConfig` | `config.LightConfig` |
  | `types.light.LIGHT_TYPES`, `AREA_SHAPES` | `config.LIGHT_TYPES`, `config.AREA_SHAPES` |
  | `types.model.UV_MAPPING_METHODS` | `config.UV_MAPPING_METHODS` |
  | `core.engine.RenderConfig`, `PhysicsConfig` | `config.RenderConfig`, `config.PhysicsConfig` |

  `telekinesis.illusion.types` no longer exports asset classes. `CameraConfig`
  is now frozen; use `replace()` to change it. Model configs validate the
  category ID, collision shape, shading, material preprocessing, UV mapping and
  scale on construction instead of on registration. Renamed parameters:
  `Camera(camera_config)`, `Camera.apply_config(camera_config)`,
  `CameraManager.add(camera_config)`, `LightManager.add(light_config)` and
  `BlenderEngine.simulate(scene, physics_config)`.
- **Breaking:** Randomizer nodes take their config as the first argument.
  Sampler functions are selected by name in the config; their keyword arguments
  move into the config's `params`:

  | Old | New |
  | --- | --- |
  | `ModelInstanceRandomizer(counts, min_num_total_models, max_num_total_models)` | `ModelInstanceRandomizer(config.InstanceRandomizerConfig(counts, ...))` |
  | `ModelPoseRandomizer(fn, target_models, max_tries, sample_on_surface, resolver)` | `ModelPoseRandomizer(config.ModelPoseRandomizerConfig(max_tries, sample_on_surface), fn, target_models, resolver)` |
  | `MaterialRandomizer(target_models, environment, groups, names, ...)` | `MaterialRandomizer(config.MaterialRandomizerConfig(target_models=..., groups=...), environment)` |
  | `BackgroundRandomizer(groups, names)` | `BackgroundRandomizer(config.BackgroundRandomizerConfig(groups, names))` |
  | `LightRandomizer(target_lights, seed=..., power=uniform(1, 2))` | `LightRandomizer(config.LightRandomizerConfig(properties={"power": {"min": 1, "max": 2}}, seed=...), target_lights)` |
  | `LightPoseRandomizer(shell_sampler, target_lights, min_distance, max_tries, **kwargs)` | `LightPoseRandomizer(config.LightPoseRandomizerConfig(sampler="shell_sampler", min_distance=..., max_tries=..., params=kwargs), target_lights)` |
  | `CameraPoseRandomizer(volume_sampler, number_of_views, **kwargs)` | `CameraPoseRandomizer(config.CameraPoseRandomizerConfig(sampler="volume_sampler", number_of_views=..., params=kwargs))` |

  `MaterialRandomizer` targets the instances of the models named in
  `target_models` or carrying a group in `target_groups`, by exact registered
  model name instead of instance-name prefix. `materials` and
  `material_slots` choices are PBR group names or `PrincipledMaterialConfig`
  recipes. `CameraPoseRandomizerConfig.number_of_views` defaults to 1.
- **Breaking:** `BinPickingWorker` reads a new configuration format and rejects
  the previous one; `configs/example_bin_picking_gearwheel_2.yaml` is kept but
  no longer loads. Each section is validated as its config before anything is
  built. Migrate configuration files as follows (see
  `configs/bin_picking_example.yaml`):

  | Old key | New key |
  | --- | --- |
  | `metadata.dataset_name`, `num_images`, `asset_directory` | `dataset.name`, `num_images`, `asset_directory` |
  | `metadata.info.*`, `metadata.licenses` | `dataset.writer.coco.*`, `dataset.writer.coco.licenses` |
  | `metadata.base_output_directory` | `dataset.output_directory` |
  | `shard.size`, `output.shard_name_template` | `dataset.shard.size`, `dataset.shard.name_template` |
  | `renderer.image_format`, `renderer.jpg_quality` | `dataset.writer.image_format`, `dataset.writer.jpg_quality` |
  | `output.max_size_gb`, `output.dataset_format`, `output.stratify` | `dataset.writer.max_size_gb`, `dataset.writer.dataset_format`, `dataset.writer.stratify` |
  | `output.train_val_tes_ratio`, `output.seed` | `dataset.writer.split_ratios`, `dataset.writer.split_seed` |
  | `physics_simulator` | `physics` |
  | `models[]` and `distractors[]` | `models[]` with one role group: `groups: [part]`, `[container]` or `[distractor]` |
  | `models[].name`, `path`, `id` | `models[].model_name`, `model_path`, `category_id` |
  | `models[].simulation.active`, `simulation.collision_shape` | `models[].active_in_simulation`, `collision_shape` |
  | `models[].supercategory` | the model's role group |
  | `models[].instances` and `instance_randomizer.<role>` | `randomizers.instance.<role>.counts`, `min_num_total_models`, `max_num_total_models` |
  | `min/max_number_visible_models`, `min/max_number_visible_distractors` | `randomizers.instance.part` and `.distractor` totals |
  | `pose_sampling.params.*` | `pose_sampling.*` |
  | `pose_sampling.sample_on_surface` | `randomizers.model_pose.sample_on_surface` |
  | `material_randomizer.<role>: [groups]` | `randomizers.material.<name>: {target_groups: [<role>], groups: [...]}` |
  | `material_randomizer.rules[]` with `target_models` | `randomizers.material.<name>` with `target_models` |
  | `material_randomizer` `types` | `groups` |
  | `background_randomizer.categories` | `randomizers.background.groups` |
  | `light_randomizer.lights[].name`, `type` | `lights[].light_name`, `light_type` |
  | `light_randomizer.lights[].properties`, `seed` | `randomizers.light.<light>.properties`, `seed` |
  | `light_randomizer.lights[].pose` | `randomizers.light_pose.<light>` |
  | `camera_pose_randomizer` | `randomizers.camera_pose` |

  Required keys no longer have implicit defaults, so `dataset`, `dataset.shard`,
  `models` and `randomizers.instance` must be present. Material and background
  randomization run only when configured. `apply_config_updates()` takes the
  edited configuration dictionary in the same format.
- **Breaking:** `Environment.add()` no longer accepts a constructed `Light`; it
  accepts a `LightConfig`, and the light manager creates the `Light`. Replace
  `environment.add(Light("key", "POINT", power=40.0))` with
  `environment.add(LightConfig("key", "POINT", properties={"power": 40.0}))`.
  Light properties move into the `properties` mapping; `location` and `rotation`
  stay keyword fields. `add()` now returns a copy of the config; use
  `get_light()` to obtain the created `Light`. `Light` itself takes a
  `LightConfig`, like `Camera` takes a `CameraConfig`: replace
  `Light("key", "POINT", power=40.0)` with
  `Light(LightConfig("key", "POINT", properties={"power": 40.0}))`.
- **Breaking:** `Environment` no longer takes `camera_config`. Add the camera
  like any other resource: replace `Environment(camera_config=cfg)` with
  `environment = Environment()` followed by `environment.add(cfg)`. The
  environment has no camera until a `CameraConfig` is added; `get_camera()`
  raises `RuntimeError` before that, and adding a second camera raises
  `ValueError`. An environment that previously relied on the default camera
  must add `CameraConfig()`. `asset_dir` is now the first positional
  parameter.
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
- The physics stop check now detects movement in every direction. Before, only
  movement toward negative coordinates and positive rotation changes counted,
  so a simulation could stop while models were still moving.
- Models that fall below the lowest point of every static model, such as parts
  that miss the container, no longer keep the physics simulation running until
  its maximum time. `BlenderEngine.simulate()` deactivates them afterwards, so
  they are neither rendered nor annotated. No configuration is needed.

### Removed

- **Breaking:** Removed the physics catch plane: the `PhysicsConfig.catch_plane`
  field and the `physics_simulator.catch_plane` YAML section. Remove the section
  from configuration files; models that fall below the container are now
  detected and deactivated automatically.
- **Breaking:** Removed the legacy keyword arguments of
  `SyntheticDataGenerator.generate()` (`simulate_physics`,
  `physics_catch_plane`, `render_max_retries`, `render_verbose` and the
  `PhysicsConfig` field names). Set them on the engine's `physics_config` and
  `render_config` instead.
- Removed `MaterialRandomizer.from_config()` and its `configuration` attribute,
  `randomizer.materials.material_choices_from_config()` and `MaterialTarget`;
  build a `config.MaterialRandomizerConfig` instead. Material choices that are
  Blender `Material` objects or custom material sources are no longer accepted.
- Removed `Light._validate_properties()`; use
  `config.validate_light_properties()`.
- Removed the unused YAML keys `metadata.package_version`, `metadata.use_case`,
  `metadata.annotation_format`, `output.write_manifest` and
  `physics_simulator.clean_up_scene`.
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
