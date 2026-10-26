"""
Reusable Blender environment resources and the current scene realization.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path

from telekinesis.illusion.types.model_asset import ModelAsset
from telekinesis.illusion.types.material_asset import MaterialAsset
from telekinesis.illusion.types.background_asset import BackgroundAsset
from telekinesis.illusion.core.background_manager import BackgroundManager
from telekinesis.illusion.utils.blender_env import isolate_user_extensions

# Must run before bpy is imported. See blender_env.py for why.
isolate_user_extensions()

# pylint: disable=wrong-import-order
from telekinesis.illusion.blender import initialization as initialization
from telekinesis.illusion.blender.labels import LabelIdMapping
from telekinesis.illusion.blender.initialization import clean_temp_dir
from telekinesis.illusion.blender.setup import is_using_external_bpy_module

from telekinesis.illusion.core.material_manager import MaterialManager
from telekinesis.illusion.core.model_manager import ModelManager
from telekinesis.illusion.types.model import Model
from telekinesis.illusion.types.background import Background
from telekinesis.illusion.types.camera import Camera, CameraConfig
from telekinesis.illusion.types.light import Light
from telekinesis.illusion.utils.assets import (
    resolve_asset_dir,
)


# pylint: disable=too-many-instance-attributes
class Environment:
    """
    Represent the active Blender scene and its reusable generation resources.

    The environment coordinates model, material, and background managers and
    owns a camera, lights, and the live background. ModelManager owns model
    registration, categories, instance pools, and activity. Existing model
    accessors and lifecycle methods remain available on Environment.

    A scene is one realization of these resources produced by the randomizer
    tree: active instances, poses, appearance, and camera views. Successive
    scenes reuse the same prepared environment.

    The current realization lives in the environment's mutable Blender state.
    Constructing another environment resets the shared Blender scene; only
    one environment can be active per process.

    Example:
        environment = Environment()
        environment.add(Light("key", power=40.0))
    """

    def __init__(
        self,
        camera_config: CameraConfig = None,
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
        # TODO: Add LightManager
        self._lights: dict[str, Light] = {}

        self.model_manager = ModelManager()

        self.background_manager = BackgroundManager()
        self._background = Background()

        self.material_manager = MaterialManager(asset_dir=self._asset_dir)

        # TODO: Add CameraManager
        self._camera = Camera(camera_config)

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
        return self._camera

    def get_lights(self) -> dict[str, Light]:
        """Return lights by their exact registered names, separate from meshes."""
        return self._lights

    def get_light(self, light_name: str) -> Light:
        """Return a registered light; raise ValueError for an unknown name."""
        if light_name not in self._lights:
            raise ValueError(f"No light with the name {light_name!r}.")
        return self._lights[light_name]

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
                The model_name of the ModelAsset registered with add().

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
        asset: ModelAsset
        | MaterialAsset
        | BackgroundAsset
        | Light
        | Iterable[ModelAsset | MaterialAsset | BackgroundAsset | Light],
    ) -> ModelAsset | MaterialAsset | BackgroundAsset | Light | list:
        """Register a descriptor, an existing light, or an iterable of them.

        Models, materials, and backgrounds are all registered here and are
        selected afterwards by the randomizers, by group or name. Nothing is
        registered implicitly, so add every collection that a randomizer should
        be able to select, for example the result of MaterialAsset.discover().

        Models are validated and assigned an annotation category without
        importing meshes. Register them before preparation; instance counts
        belong to ModelInstanceRandomizer.counts. Lights are constructed after
        the Environment and are registered separately from models and mesh
        instances.

        Args:
            asset: A model, material, or background descriptor, a live Light,
                or an iterable of these, such as the list returned by
                discover(). Items are registered in order; an error leaves
                the earlier items registered.

        Returns:
            For a model, the registered copy with an absolute Path, a float or
            three-float tuple scale, and resolved category metadata. The input
            specification is unchanged. Materials and backgrounds return a
            normalized descriptor without loading or activating it. For a
            light, return the supplied light. For an iterable, return the list
            of these results.

        Raises:
            TypeError: The asset is not a supported descriptor, Light, or
                iterable of them.
            RuntimeError: A model is added after preparation or cleanup.
            ValueError: A model name is empty or already registered, a light
                name is already registered, scale is not a positive finite
                scalar or three-vector, category metadata conflicts, or a
                material/background descriptor is invalid or duplicates a name.
            FileNotFoundError: An asset path is missing.
        """
        if isinstance(asset, Iterable) and not isinstance(asset, str):
            return [self.add(item) for item in asset]
        if isinstance(asset, ModelAsset):
            return self.model_manager.add(asset)
        if isinstance(asset, MaterialAsset):
            return self.material_manager.add(asset)
        if isinstance(asset, BackgroundAsset):
            return self.background_manager.add(asset)
        if isinstance(asset, Light):
            return self._add_light(
                asset
            )  # TODO: Needs to be replaced with LightManager
        raise TypeError(
            "asset must be a ModelAsset, MaterialAsset, BackgroundAsset, or Light."
        )

    def _add_light(
        self, asset: Light
    ) -> Light:  # TODO: Needs to be deprecated by a LightManager
        """Register an existing light once under its exact Blender name."""
        light_name = asset.get_name()
        if light_name in self._lights:
            raise ValueError(f"Light {light_name!r} already exists.")
        self._lights[light_name] = asset
        return asset

    def get_assets(self) -> Mapping[str, ModelAsset]:
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
