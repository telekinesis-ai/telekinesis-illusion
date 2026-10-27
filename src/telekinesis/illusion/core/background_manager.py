"""Registered HDRI descriptors and their selection groups."""

from telekinesis.illusion.core.asset_manager import AssetManager
from telekinesis.illusion import config


class BackgroundManager(AssetManager[config.BackgroundConfig]):
    """Register uniquely named HDRIs without loading or activating them.

    assets_by_name stores background configs; names_by_group indexes every normalized
    group in registration order. add() validates a normalized copy and rejects
    duplicate names with ValueError. get_names() selects assets matching any
    requested group, without repeating names for overlapping groups.

    Example:
        manager = BackgroundManager()
        manager.add(
            config.BackgroundConfig.from_path("studio.exr", groups="indoor")
        )
    """

    _asset_type = config.BackgroundConfig

