"""Registered PBR descriptors and cached Blender materials."""

import random
from collections.abc import Sequence
from pathlib import Path

from telekinesis.illusion.blender.loaders.material_loader import load_material_asset
from telekinesis.illusion.blender.types.material import Material
from telekinesis.illusion.core.asset_manager import AssetManager
from telekinesis.illusion.types.material_asset import MaterialAsset
from telekinesis.illusion.utils.assets import resolve_asset_dir


class MaterialManager(AssetManager[MaterialAsset]):
    """Register PBR descriptors and cache loaded materials separately by name.

    assets_by_name holds descriptors and names_by_group indexes their normalized
    groups in registration order. add() validates a copy, rejects duplicate names
    with ValueError, and never loads textures. Nothing is registered
    implicitly; register materials with add(), for example from
    MaterialAsset.discover(). update_materials() loads selected materials, and
    get_random_material() samples a group uniformly, then one material within
    it. Without groups, top-level groups are used.

    Example:
        manager = MaterialManager("assets")
        manager.add(MaterialAsset.from_path("assets/materials/metal/Metal001", groups="metal"))
        manager.update_materials(groups=["metal"])
        material = manager.get_random_material(["metal"])
    """

    _asset_type = MaterialAsset

    def __init__(
        self,
        asset_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            asset_dir: str, Path or None
                Directory against which relative surface-imperfection
                directories are resolved. If 'None', it is resolved by
                telekinesis.illusion.utils.assets.resolve_asset_dir().
        """
        super().__init__()
        self._materials: dict[str, Material] = {}
        self._asset_dir = resolve_asset_dir(asset_dir)

    def get_asset_dir(self) -> Path:
        """Return the root that relative surface-imperfection paths use."""
        return self._asset_dir

    def update_materials(
        self,
        groups: Sequence[str] | None = None,
        names: Sequence[str] | None = None,
    ) -> None:
        """Load registered materials selected by group or name, once per name.

        No selection loads all registered assets. Unknown groups or names, or
        no registered materials, raise ValueError.
        """
        for name in self.get_names(groups, names):
            if name not in self._materials:
                self._materials[name] = load_material_asset(
                    self.assets_by_name[name]
                )

    def get_random_material(
        self,
        groups: Sequence[str] | None = None,
        names: Sequence[str] | None = None,
        rng=None,
    ) -> Material:
        """Sample a registered material using the supplied RNG.

        With names, the material is drawn uniformly from the union of the
        named materials and the materials in groups. Without names, a group
        from groups is drawn first (all top-level groups if None or empty),
        then one material within it. Unknown groups or names, or an empty
        selection, raise ValueError. Unloaded selections are loaded once and
        cached; update_materials() can preload them before sampling.
        """
        rng = random if rng is None else rng
        if names:
            name = rng.choice(self.get_names(groups, names))
        else:
            available = groups or [
                group for group in self.names_by_group if "/" not in group
            ]
            self.get_names(available)
            random_group = rng.choice(available)
            name = rng.choice(self.names_by_group[random_group])
        if name not in self._materials:
            self._materials[name] = load_material_asset(
                self.assets_by_name[name]
            )
        return self._materials[name]
