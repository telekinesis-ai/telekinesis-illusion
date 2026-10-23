# Changelog

Notable changes to `telekinesis-illusion` are recorded here, newest first.

This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

## Unreleased

### Changed

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
