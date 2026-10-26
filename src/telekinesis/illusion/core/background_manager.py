"""Registered HDRI descriptors and their selection groups."""

from telekinesis.illusion.core.asset_manager import AssetManager
from telekinesis.illusion.types.background_asset import BackgroundAsset


class BackgroundManager(AssetManager[BackgroundAsset]):
    """Register uniquely named HDRIs without loading or activating them.

    assets_by_name stores descriptors; names_by_group indexes every normalized
    group in registration order. add() validates a normalized copy and rejects
    duplicate names with ValueError. get_names() selects assets matching any
    requested group, without repeating names for overlapping groups.

    Example:
        manager = BackgroundManager()
        manager.add(BackgroundAsset.from_path("studio.exr", groups="indoor"))
    """

    _asset_type = BackgroundAsset

