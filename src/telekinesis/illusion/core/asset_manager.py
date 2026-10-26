"""Shared registration, group indexing and selection for assets."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Generic, TypeVar

from telekinesis.illusion.types.background_asset import BackgroundAsset
from telekinesis.illusion.types.material_asset import MaterialAsset
from telekinesis.illusion.types.model_asset import ModelAsset


AssetType = TypeVar("AssetType", MaterialAsset, BackgroundAsset, ModelAsset)


class AssetManager(Generic[AssetType]):
    """Store descriptors by unique name and index every selection group.

    Registration never loads Blender data. Index lists retain registration
    order. Treat the exposed dictionaries as read-only; register through add().

    Example:
        from telekinesis.illusion.core.background_manager import BackgroundManager

        manager = BackgroundManager()
        manager.add(BackgroundAsset.from_path("studio.exr", groups="indoor"))
    """

    _asset_type: type[AssetType]

    def __init__(self) -> None:
        """Create owned descriptor and group indexes."""
        self.assets_by_name: dict[str, AssetType] = {}
        self.names_by_group: dict[str, list[str]] = {}

    def add(self, asset: AssetType) -> AssetType:
        """Validate and register one descriptor, returning a normalized copy.

        Raises TypeError for the wrong descriptor type, ValueError for a
        duplicate name or invalid descriptor, and FileNotFoundError for a
        missing path. Existing registrations are never overwritten.
        """
        if not isinstance(asset, self._asset_type):
            raise TypeError(f"asset must be a {self._asset_type.__name__}.")
        asset = replace(asset)
        if asset.name in self.assets_by_name:
            raise ValueError(
                f"Asset name {asset.name!r} is already registered."
            )
        self.assets_by_name[asset.name] = asset
        for group in asset.groups:
            self.names_by_group.setdefault(group, []).append(asset.name)
        return asset

    def get_names(
        self,
        groups: Sequence[str] | None = None,
        names: Sequence[str] | None = None,
    ) -> list[str]:
        """Return registered names in order, selecting by group and by name.

        The result is the union of every asset in the requested groups and the
        requested names. None or empty groups and names together select all
        assets. Overlaps do not duplicate names. Unknown groups or names, or an
        empty result, raise ValueError.
        """
        if not groups and not names:
            selected = list(self.assets_by_name)
        else:
            missing_groups = [
                group
                for group in groups or ()
                if group not in self.names_by_group
            ]
            if missing_groups:
                raise ValueError(f"Unknown asset groups: {missing_groups}.")
            missing_names = [
                name for name in names or () if name not in self.assets_by_name
            ]
            if missing_names:
                raise ValueError(f"Unknown asset names: {missing_names}.")
            requested = set(names or ())
            for group in groups or ():
                requested.update(self.names_by_group[group])
            selected = [
                name for name in self.assets_by_name if name in requested
            ]
        if not selected:
            raise ValueError(
                f"No {self._asset_type.__name__} assets are registered."
            )
        return selected
