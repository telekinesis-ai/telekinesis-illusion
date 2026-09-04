"""
Resolution of the asset directory and of asset-relative paths.

This is the single place in the library that knows where assets live. Nothing
else should derive asset locations from __file__ or from environment variables.
"""

from pathlib import Path
from typing import Optional, Union

from loguru import logger

# <repo>/assets when the package is used from a src-layout checkout. A wheel
# ships no assets (see pyproject.toml, which defines no package_data), so this
# path does not exist for an installed package - hence the is_dir() guard
# wherever it is used.
_REPO_ASSETS = Path(__file__).resolve().parents[4] / "assets"


def resolve_asset_dir(
    explicit: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Determine the directory that holds the models, hdris and materials.

    Lookup order:

    1. 'explicit', if given. An absolute path is used as is, a relative one is
       resolved against 'base_dir' (or the current working directory when no
       'base_dir' is given).
    2. './assets' relative to the current working directory, if it exists.
    3. '<repo>/assets' next to the source tree, if it exists. Only available
       for a src-layout checkout.

    Args:
        explicit: str, Path or None
            An explicitly configured asset directory, e.g. the spec's
            'metadata.asset_directory'.
        base_dir: str, Path or None
            Directory a relative 'explicit' is anchored on. Typically the
            directory containing the spec file.

    Returns:
        The asset directory as a pathlib.Path() object.

    Raises:
        FileNotFoundError:
            - If 'explicit' does not exist.
            - If no explicit directory was given and no default could be found.
        NotADirectoryError:
            - If the resolved path is not a directory.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_absolute():
            anchor = Path(base_dir) if base_dir is not None else Path.cwd()
            path = anchor / path
        return _validate_asset_dir(path.resolve())

    cwd_assets = Path.cwd() / "assets"
    if cwd_assets.is_dir():
        logger.info(f"Using the asset directory '{cwd_assets}'.")
        return cwd_assets

    if _REPO_ASSETS.is_dir():
        logger.info(f"Using the asset directory '{_REPO_ASSETS}'.")
        return _REPO_ASSETS

    raise FileNotFoundError(
        "No asset directory found. Set 'metadata.asset_directory' in your "
        f"spec, or run from a directory containing an 'assets' folder "
        f"(looked for '{cwd_assets}' and '{_REPO_ASSETS}')."
    )


def _validate_asset_dir(
    path: Path,
) -> Path:
    """
    Ensure the given asset directory exists and is a directory.
    """
    if not path.exists():
        raise FileNotFoundError(f"Asset directory '{path}' does not exist.")

    if not path.is_dir():
        raise NotADirectoryError(
            f"Asset directory '{path}' is not a directory."
        )

    logger.info(f"Using the asset directory '{path}'.")
    return path


def pick_default_hdri(
    hdris_dir: Union[str, Path],
    preferred_category: Optional[str] = None,
) -> Path:
    """
    Pick a deterministic default HDRI from an hdris directory.

    Asset trees differ in which HDRIs they ship, so the default is discovered
    rather than hardcoded to a file name.

    Args:
        hdris_dir: str or Path
            Directory scanned recursively for '.exr' files.
        preferred_category: str or None
            A sub-directory, e.g. 'indoor/industrial', to prefer. If it holds
            no '.exr' files, the whole tree is used instead.

    Returns:
        The first '.exr' in sorted order as a pathlib.Path() object.

    Raises:
        FileNotFoundError:
            - If no '.exr' file is found under 'hdris_dir'.
    """
    hdris_dir = Path(hdris_dir)

    if preferred_category:
        preferred = sorted((hdris_dir / preferred_category).rglob("*.exr"))
        if preferred:
            return preferred[0]

    candidates = sorted(hdris_dir.rglob("*.exr"))
    if not candidates:
        raise FileNotFoundError(
            f"No .exr files found under the HDRI directory '{hdris_dir}'."
        )
    return candidates[0]


def resolve_asset_path(
    path: Union[str, Path],
    asset_dir: Union[str, Path],
) -> Path:
    """
    Resolve a single asset path, e.g. a spec's 'models[].path'.

    Absolute paths are returned unchanged, relative ones are resolved against
    'asset_dir'.

    Args:
        path: str or Path
            The asset path as configured, e.g.
            'models/mechanical_parts/gearwheel_2.glb'.
        asset_dir: str or Path
            The directory relative paths are anchored on.

    Returns:
        The resolved path as a pathlib.Path() object.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(asset_dir) / candidate
