# Notice

This directory contains a modified copy of [BlenderProc](https://github.com/DLR-RM/BlenderProc)
2.8.0, originally authored by Maximilian Denninger, Dominik Winkelbauer, and Martin
Sundermeyer (DLR-RM), and licensed under the GNU General Public License v3.0. See
[LICENSE](LICENSE) for the full license text.

## Modifications

- `2026-01-23` — `blenderproc/python/utility/SetupUtility.py`: removed an environment-variable
  setting.
- `2026-02-25` — `blenderproc/python/object/PhysicsSimulation.py`,
  `blenderproc/python/utility/Utility.py`: removed scale normalization in physics
  simulation; utility changes to support illusion's background/camera proxy structs.
- `2026-03-05` — `blenderproc/python/types/EntityUtility.py`: minor addition.
- `2026-03-06` — `blenderproc/python/renderer/RendererUtility.py`,
  `blenderproc/python/utility/DefaultConfig.py`,
  `blenderproc/python/utility/Initializer.py`: adjusted renderer defaults.
- `2026-03-09` — `blenderproc/python/object/PhysicsSimulation.py`,
  `blenderproc/python/renderer/RendererUtility.py`,
  `blenderproc/python/utility/Utility.py`: disabled stdout redirection (diagnosing
  "Too many files open").
- `2026-05-19` — `blenderproc/python/object/PhysicsSimulation.py`,
  `blenderproc/python/renderer/RendererUtility.py`,
  `blenderproc/python/utility/Utility.py`: removed unnecessary prints, improved logging.
- `2026-08-31` — `blenderproc/__init__.py`, `blenderproc/api/constructor/__init__.py`,
  `blenderproc/api/loader/__init__.py`, `blenderproc/api/object/__init__.py`,
  `blenderproc/api/writer/__init__.py`, `blenderproc/python/utility/Initializer.py`:
  added support for running as a Blender extension.
- `2026-09-02` — `setup.py`: relaxed the `Pillow` pin to `>=10.3.0` to resolve a
  conflict with `fiftyone`'s `Pillow>=12.2` requirement.
