"""Model registration, annotation categories, and reusable instance pools."""

from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
from loguru import logger

from telekinesis.illusion import config
from telekinesis.illusion.blender.labels import LabelIdMapping
from telekinesis.illusion.core.asset_manager import AssetManager
from telekinesis.illusion.types.model import Model


class ModelManager(AssetManager[config.ModelConfig]):
    """Own model definitions, categories, instance pools, and active models.

    add() registers one ModelConfig without importing geometry. get_names()
    selects registered models by selection group or exact model name. prepare() imports
    each requested model once and creates a fixed pool of linked instances.
    Registration closes after preparation, and later preparation can only reuse
    existing capacity. Annotation categories are independent of selection groups.
    activate(), deactivate(), and begin_sample() manage visibility and physics
    while retaining pooled instances between samples.

    clean_up() releases the instance references and permanently closes the
    manager; definitions and category metadata remain readable. Environment
    coordinates the shared Blender scene reset, including after an import error.
    Use Environment.prepare() and Environment.clean_up() for those lifecycle
    operations on an environment's manager.

    Example:
        manager = ModelManager()
        model = manager.add(
            config.ModelConfig(
                "assets/models/mechanical_parts/gearwheel_1.glb", "gear"
            )
        )
    """

    _asset_type = config.ModelConfig

    def __init__(self) -> None:
        """Create the owned model registry, category mapping, and empty pools."""
        super().__init__()
        self._models: dict[str, Model] = {}
        self._model_groups: dict[str, list[Model]] = {}
        self._categories = LabelIdMapping()
        self._next_category_id = 1
        self._visible_model_names: list[str] = []
        self._prepared = False
        self._closed = False

    @property
    def is_prepared(self) -> bool:
        """Whether initial pool preparation completed successfully."""
        return self._prepared

    @property
    def is_closed(self) -> bool:
        """Whether cleanup or a failed import permanently closed the manager."""
        return self._closed

    def add(self, asset: config.ModelConfig) -> config.ModelConfig:
        """Register one model without importing geometry.

        Args:
            asset: Model definition with a unique, nonempty model name.

        Returns:
            A copy with an absolute path and resolved annotation category. Missing categories default to
            the model name; existing names or IDs reuse their mapping.
            The category name "distractor" receives no ID. The supplied
            definition is unchanged.

        Raises:
            TypeError: The asset is not a ModelConfig.
            RuntimeError: Preparation or cleanup has closed registration.
            FileNotFoundError: The model path does not point to a file.
            ValueError: The model name is already registered or categories
                conflict.
        """
        if not isinstance(asset, config.ModelConfig):
            raise TypeError("asset must be a ModelConfig.")
        if self._prepared or self._closed:
            raise RuntimeError(
                "Register models before preparing the environment."
            )
        if asset.model_name in self.assets_by_name:
            raise ValueError(
                f"Model name {asset.model_name!r} is already registered."
            )
        path = Path(asset.model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        category_name, category_id = self._resolve_category_id_and_name(
            asset.model_name, asset.category_name, asset.category_id
        )
        return super().add(
            replace(
                asset,
                model_path=path,
                category_name=category_name,
                category_id=category_id,
            )
        )

    def get_assets(self) -> Mapping[str, config.ModelConfig]:
        """Return a read-only view of definitions keyed by exact model name."""
        return MappingProxyType(self.assets_by_name)

    def get_categories(self) -> LabelIdMapping:
        """Return the live mapping of annotation category names and IDs."""
        return self._categories

    def get_models(self) -> dict[str, Model]:
        """Return the live dictionary of all pooled instances by instance name."""
        return self._models

    def get_models_by_name(
        self,
        name: str,
    ) -> list[Model] | Model:
        """Return pooled instances whose names contain the supplied substring.

        A single match returns a Model; multiple matches return a list in pool
        insertion order. Raise ValueError if no instance matches.
        """
        matches = [value for key, value in self._models.items() if name in key]

        if not matches:
            raise ValueError(f"No model with the name {name}")
        if len(matches) == 1:
            return matches[0]
        return matches

    def get_model_group(
        self,
        model_name: str,
    ) -> list[Model]:
        """Return the pool for one exact registered model name.

        Args:
            model_name: The model_name of the ModelConfig registered with add().

        Returns:
            The live list of instances, including inactive models. Return an
            empty list if no instances were prepared for this name.
        """
        return self._model_groups.get(model_name, [])

    def prepare(self, required_instances: dict[str, int]) -> None:
        """Import each requested asset once, then create linked duplicates.

        Instances start inactive and share mesh data. Initial preparation
        closes registration, even when no instances are requested. Subsequent
        calls reuse the pools; they cannot increase capacity.

        Import or duplication errors propagate and close the manager. For an
        environment-owned manager, call Environment.prepare() so it also
        resets shared Blender data after such failures.

        Args:
            required_instances (dict[str, int]): Exact registered names mapped to nonnegative
                integer capacities. Zero and omitted entries create no pool, e.g.
                `{'part_1': 3, 'part_2': 1}`.

        Raises:
            RuntimeError: The manager has been cleaned up.
            ValueError: A name is unknown, a capacity is invalid, or a request
                exceeds an already prepared pool. These errors leave the
                existing pools usable.
        """
        if self._closed:
            raise RuntimeError(
                "This environment was cleaned up; construct a new Environment."
            )
        for name, count in required_instances.items():
            if name not in self.assets_by_name:
                raise ValueError(f"Unknown registered model: {name!r}")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError(
                    f"Capacity for {name!r} must be a nonnegative integer."
                )
        if self._prepared:
            for name, count in required_instances.items():
                if count > len(self._model_groups.get(name, ())):
                    raise ValueError(
                        f"{name!r} now needs {count} instances; "
                        "build a new Environment."
                    )
            return
        try:
            for name, count in required_instances.items():
                if count == 0:
                    continue
                kwargs = asdict(self.assets_by_name[name])
                kwargs.pop("groups")
                kwargs["model_path"] = str(kwargs["model_path"])
                if isinstance(kwargs["scale"], tuple):
                    kwargs["scale"] = np.asarray(kwargs["scale"])
                model = Model(**kwargs)
                self._model_groups[name] = [model]
                self._models[model.get_name()] = model
                self.deactivate(model)
                for _ in range(count - 1):
                    duplicate = model.create_linked_duplicate()
                    self._model_groups[name].append(duplicate)
                    self._models[duplicate.get_name()] = duplicate
                    self.deactivate(duplicate)
        except Exception:
            # A failed import can leave partial Blender data; don't reuse it.
            self.clean_up()
            raise
        self._prepared = True

    def activate(self, model: Model) -> None:
        """Enable rendering, physics, and annotations for a pooled instance.

        Repeated activation has no additional effect. Raise ValueError if the
        model does not belong to this manager's pool.
        """
        name = model.get_name()
        if self._models.get(name) is not model:
            raise ValueError("Model does not belong to this environment.")
        if name not in self._visible_model_names:
            model.hide(False)
            model.enable_rigid_body()
            self._visible_model_names.append(name)

    def deactivate(self, model: Model) -> None:
        """Hide a model and remove its rigid body and annotation eligibility."""
        model.hide(True)
        model.disable_rigid_body()
        name = model.get_name()
        if name in self._visible_model_names:
            self._visible_model_names.remove(name)

    def get_visible_model_names(self) -> list[str]:
        """Return a copy of active instance names in activation order."""
        return list(self._visible_model_names)

    def set_visible_model_names(self, visible_model_names: list[str]) -> None:
        """Activate the named instances and deactivate all others.

        Names must exactly match pooled instance names. Repeated names have
        no additional effect. Raise ValueError before changing activity if any
        name is unknown. Visibility and rigid-body participation change together.
        """
        names = set(visible_model_names)
        unknown = names - self._models.keys()
        if unknown:
            raise ValueError(f"Unknown environment models: {unknown}")
        for name in tuple(self._visible_model_names):
            if name not in names:
                self.deactivate(self._models[name])
        for name in dict.fromkeys(visible_model_names):
            self.activate(self._models[name])

    def begin_sample(self) -> None:
        """Deactivate pooled models before randomizing the next scene.

        Registered assets and instance pools remain available. Poses and
        materials are not restored to defaults.
        """
        for name in tuple(self._visible_model_names):
            self.deactivate(self._models[name])

    def _get_next_available_category_id(self) -> int:
        """Advance the category counter past used IDs and return a free ID."""
        while self._categories.has_id(self._next_category_id):
            self._next_category_id += 1
        return self._next_category_id

    def _resolve_category_id_and_name(
        self,
        model_name: str,
        category_name: str | None,
        category_id: int | None = None,
    ) -> tuple[str, int | None]:
        """Resolve and register annotation metadata, returning its name and ID.

        Reuse existing names or IDs, rejecting conflicting pairs. Missing names
        default to model_name when the ID has no mapping; missing IDs are
        allocated. The name "distractor" returns None as its ID without adding
        a category mapping.
        """
        cats = self._categories

        def bump_next(start: int) -> None:
            """Set _next_category_id to the next free id >= start."""
            self._next_category_id = max(self._next_category_id, start)
            while cats.has_id(self._next_category_id):
                self._next_category_id += 1

        def alloc_id() -> int:
            """Get next available id and advance _next_category_id."""
            cid = self._get_next_available_category_id()
            bump_next(cid + 1)
            return cid

        def add_mapping(name: str, cid: int) -> tuple[str, int]:
            """Register a category and advance the counter past its ID."""
            cats.add(name, cid)
            bump_next(cid + 1)
            return name, cid

        if category_name is None and category_id is None:
            category_name = model_name
            logger.info(
                f"No category name provided. Using model name '{model_name}' as category name."
            )

        # Case A: name provided (id optional)
        if category_name is not None:
            if category_name == "distractor":
                logger.info("Skipping category name update for distractor...")
                return "distractor", None
            name_exists = cats.has_label(category_name)
            id_exists = category_id is not None and cats.has_id(category_id)

            # If name exists, it wins unless the user also provided a conflicting id
            if name_exists:
                existing_id = cats.id_from_label(category_name)
                if category_id is not None and category_id != existing_id:
                    raise ValueError(
                        f"Category combination conflict: category name '{category_name}' is already mapped "
                        f"to ID {existing_id}, but you provided category_id={category_id}. "
                        f"Please provide only category_name or category_id, not both."
                    )
                logger.info(
                    f"Category name '{category_name}' already exists with ID {existing_id}. Reusing existing mapping."
                )
                return category_name, existing_id

            # Name doesn't exist
            if category_id is None:
                cid = alloc_id()
                logger.info(
                    f"Created new category '{category_name}' with auto-assigned ID {cid}."
                )
                return add_mapping(category_name, cid)

            # Both provided and name is new:
            if id_exists:
                existing_name = cats.label_from_id(category_id)
                raise ValueError(
                    f"Category combination conflict: category ID {category_id} is already mapped to "
                    f"'{existing_name}', but you provided category_name='{category_name}'. "
                    f"Please provide only category_id or category_name, not both category_id and category_name."
                )

            logger.info(
                f"Created new category '{category_name}' with ID {category_id}."
            )
            return add_mapping(category_name, category_id)

        # Case B: only id provided
        assert category_id is not None  # (since name is None here)

        if cats.has_id(category_id):
            name = cats.label_from_id(category_id)
            logger.info(
                f"Category ID {category_id} already exists with name '{name}'. Reusing existing mapping."
            )
            return name, category_id

        # ID doesn't exist: create mapping using model_name
        name = model_name
        logger.info(
            f"Category ID {category_id} does not exist. Using model name '{model_name}' as category name "
            f"and creating new mapping."
        )

        return add_mapping(name, category_id)

    def clean_up(self) -> None:
        """Release pooled references and permanently close model registration.

        Definitions and categories remain available for inspection. Blender
        data is shared with the other managers; Environment resets that data
        before calling this method during normal cleanup. Repeated calls are
        harmless. An import failure also closes the manager before Environment
        removes the partially imported scene data.
        """
        if self._closed:
            return
        self._models.clear()
        self._model_groups.clear()
        self._visible_model_names.clear()
        self._closed = True
