"""
Reusable Blender environment resources and the current scene realization.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path

from telekinesis.illusion import config
from telekinesis.illusion.core.background_manager import BackgroundManager
from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported. See blender_env.py for why.
isolate_user_extensions()

# pylint: disable=wrong-import-order
from telekinesis.illusion.blender import initialization as initialization
from telekinesis.illusion.blender.labels import LabelIdMapping
from telekinesis.illusion.blender.initialization import clean_temp_dir
from telekinesis.illusion.blender.setup import is_using_external_bpy_module

from telekinesis.illusion.core.camera_manager import CameraManager
from telekinesis.illusion.core.light_manager import LightManager
from telekinesis.illusion.core.material_manager import MaterialManager
from telekinesis.illusion.core.model_manager import ModelManager
from telekinesis.illusion.types.model import Model
from telekinesis.illusion.types.background import Background
from telekinesis.illusion.types.camera import Camera
from telekinesis.illusion.types.light import Light
from telekinesis.illusion.utils.assets import (
    resolve_asset_dir,
)


# pylint: disable=too-many-instance-attributes
class Environment:
    """
    Represent the active Blender scene and its reusable generation resources.

    The environment coordinates model, material, background, light, and camera
    managers and owns the live background. ModelManager owns model
    registration, categories, instance pools, and activity. LightManager
    creates registered lights from their configs, and CameraManager creates
    the camera from its config. Existing model, light, and camera accessors and
    lifecycle methods remain available on Environment.

    A scene is one realization of these resources produced by the randomizer
    tree: active instances, poses, appearance, and camera views. Successive
    scenes reuse the same prepared environment.

    The current realization lives in the environment's mutable Blender state.
    Constructing another environment resets the shared Blender scene; only
    one environment can be active per process.

    Example:
        environment = Environment()
        environment.add(
            config.CameraConfig(image_width=128, image_height=128)
        )
        environment.add(config.LightConfig("key", properties={"power": 40.0}))
    """

    def __init__(
        self,
        asset_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize reusable resources and reset the active Blender scene.

        Args:
            asset_dir: str, Path or None
                Directory against which relative asset paths, such as
                surface-imperfection directories, are resolved. Assets are
                not discovered from it; register them with add(). If 'None',
                it is resolved by
                telekinesis.illusion.utils.assets.resolve_asset_dir().
        """
        initialization.init()

        # Resolve the asset directory once and share it with everything in the
        # environment that reads assets
        self._asset_dir = resolve_asset_dir(asset_dir)

        self._closed = False

        self.light_manager = LightManager()

        self.model_manager = ModelManager()

        self.background_manager = BackgroundManager()
        self._background = Background()

        self.material_manager = MaterialManager(asset_dir=self._asset_dir)

        self.camera_manager = CameraManager()

    def get_asset_dir(self) -> Path:
        """
        Get the directory against which relative asset paths are resolved.
        """
        return self._asset_dir

    def clean_up(self) -> None:
        """Reset shared Blender data, close model pools, and remove temp files.

        Repeated calls have no effect. Registered model definitions and category
        metadata remain readable, but the environment cannot be prepared again.
        """
        if self._closed:
            return
        initialization.clean_up()
        self.model_manager.clean_up()
        self._closed = True
        if is_using_external_bpy_module():
            clean_temp_dir()

    def get_camera(self) -> Camera:
        """
        Get the camera used for rendering.
        """
        return self.camera_manager.get_camera()

    def get_lights(self) -> dict[str, Light]:
        """Return lights by their exact registered names, separate from meshes."""
        return self.light_manager.get_lights()

    def get_light(self, light_name: str) -> Light:
        """Return a registered light; raise ValueError for an unknown name."""
        return self.light_manager.get_light(light_name)

    def get_categories(self) -> LabelIdMapping:
        """
        Get the categories registered in the environment.
        """
        return self.model_manager.get_categories()

    def get_models(self) -> dict[str, Model]:
        """
        Get all pooled models, including those inactive in the current scene.
        """
        return self.model_manager.get_models()

    def get_models_by_name(
        self,
        name: str,
    ) -> list[Model] | Model:
        """Return pooled instances whose names contain the supplied substring.

        Return a Model for one match or a list for multiple matches. Raise
        ValueError if no instance matches.
        """
        return self.model_manager.get_models_by_name(name)

    def get_model_group(
        self,
        model_name: str,
    ) -> list[Model]:
        """
        Get every instance of a model, keyed by the 'model_name' it was added
        under (i.e. the config's model name, not the '<name>_INSTANCE_<n>' keys
        used by get_models()).

        Args:
            model_name: str
                The model_name of the ModelConfig registered with add().

        Returns:
            The live list of instances, including inactive models. Return an
            empty list if no instances were prepared for this name.
        """
        return self.model_manager.get_model_group(model_name)

    def get_background(self) -> Background:
        """
        Return the background of the environment.
        """
        return self._background

    def get_visible_model_names(self) -> list[str]:
        """
        Return the names of pooled models active in the current scene.
        """
        return self.model_manager.get_visible_model_names()

    def set_visible_model_names(self, visible_model_names: list[str]) -> None:
        """Compatibility setter; keep visibility and physics consistent."""
        self.model_manager.set_visible_model_names(visible_model_names)

    def add(
        self,
        asset: config.ModelConfig
        | config.MaterialConfig
        | config.BackgroundConfig
        | config.LightConfig
        | config.CameraConfig
        | Iterable[
            config.ModelConfig
            | config.MaterialConfig
            | config.BackgroundConfig
            | config.LightConfig
            | config.CameraConfig
        ],
    ) -> (
        config.ModelConfig
        | config.MaterialConfig
        | config.BackgroundConfig
        | config.LightConfig
        | config.CameraConfig
        | list
    ):
        """Register a config or an iterable of configs.

        Models, materials, and backgrounds are all registered here and are
        selected afterwards by the randomizers, by group or name. Nothing is
        registered implicitly, so add every collection that a randomizer should
        be able to select, for example the result of
        config.MaterialConfig.discover().

        Models are validated and assigned an annotation category without
        importing meshes. Register them before preparation; instance counts
        belong to ModelInstanceRandomizer.counts. A light config creates its
        light immediately; lights are registered separately from models and
        mesh instances. A camera config creates the scene camera and applies
        its image size, lens, and pose; add it before calling get_camera().

        Args:
            asset: A model, material, background, light, or camera config,
                or an iterable of these, such as the list returned by
                discover(). Items are registered in order; an error leaves the
                earlier items registered.

        Returns:
            For a model, the registered copy with an absolute Path and
            resolved category metadata. The input config is unchanged.
            Materials and backgrounds return a copy without loading or
            activating them. A light
            returns a copy of its config; get_light() returns the created
            light. A camera returns a copy of its config. For an iterable,
            return the list of these results.

        Raises:
            TypeError: The asset is not a supported config or iterable of
                them.
            RuntimeError: A model is added after preparation or cleanup.
            ValueError: A model, material, background, or light name is
                already registered, a camera was already added, or category
                metadata conflicts.
            FileNotFoundError: An asset path is missing.
        """
        if isinstance(asset, Iterable) and not isinstance(asset, str):
            return [self.add(item) for item in asset]
        if isinstance(asset, config.ModelConfig):
            return self.model_manager.add(asset)
        if isinstance(asset, config.MaterialConfig):
            return self.material_manager.add(asset)
        if isinstance(asset, config.BackgroundConfig):
            return self.background_manager.add(asset)
        if isinstance(asset, config.LightConfig):
            return self.light_manager.add(asset)
        if isinstance(asset, config.CameraConfig):
            return self.camera_manager.add(asset)
        raise TypeError(
            "asset must be a ModelConfig, MaterialConfig, BackgroundConfig, "
            "LightConfig, or CameraConfig."
        )

    def get_assets(self) -> Mapping[str, config.ModelConfig]:
        """Return a read-only view of definitions keyed by exact model name."""
        return self.model_manager.get_assets()

    def prepare(self, required_instances: dict[str, int]) -> None:
        """Import each requested asset once, then create linked duplicates.

        The pool is reused across scene realizations. Repeated preparation
        validates its capacity; increasing capacity requires a new Environment.
        No geometry is imported while sampling scenes.

        Invalid requests leave existing pools usable. Import or duplication
        failures clean up the entire environment before propagating the error.

        Args:
            required_instances (dict[str, int]):
                Exact registered names mapped to nonnegative integer capacities.
                Zero and omitted entries create no pool, e.g.
                `{'part_1': 3, 'part_2': 1}`.
        """
        try:
            self.model_manager.prepare(required_instances)
        except Exception:
            # Validation leaves pools usable; failed imports require a scene reset.
            if self.model_manager.is_closed:
                self.clean_up()
            raise

    def activate(self, model: Model) -> None:
        """Activate a pooled model for the current scene's render and physics."""
        self.model_manager.activate(model)

    def deactivate(self, model: Model) -> None:
        """Hide a model and remove its rigid body and annotation eligibility."""
        self.model_manager.deactivate(model)

    def begin_sample(self) -> None:
        """Deactivate pooled models before randomizing the next scene.

        Registered assets and instance pools remain available. Poses and
        materials are not restored to defaults.
        """
        self.model_manager.begin_sample()
