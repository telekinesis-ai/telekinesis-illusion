"""
Resolution of the asset directory and of asset-relative paths.

This is the single place in the library that knows where assets live. Nothing
else should derive asset locations from __file__ or from environment variables.
"""

import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional, Union

import httpx
from loguru import logger
from tqdm import tqdm

# <repo>/assets when the package is used from a src-layout checkout. A wheel
# ships no assets (see pyproject.toml, which defines no package_data), so this
# path does not exist for an installed package - hence the is_dir() guard
# wherever it is used.
_REPO_ASSETS = Path(__file__).resolve().parents[4] / "assets"

# Public copy of <repo>/assets for installs without a checkout. The archive
# holds a top-level 'assets/' folder. Its cache directory is named after the
# archive, so publishing a new archive version never reuses an older cache.
_PUBLIC_ASSETS_URL = (
    "https://assets.telekinesis.ai/telekinesis-illusion/assets-v1.zip"
)
_CACHED_ASSETS = (
    Path("~/.cache/telekinesis/telekinesis-illusion").expanduser()
    / PurePosixPath(_PUBLIC_ASSETS_URL).stem
    / "assets"
)


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
    4. The public assets cached in
       '~/.cache/telekinesis/illusion/assets-<VERSION>/assets' by an earlier call.
    5. Otherwise the public assets are downloaded once, extracted into that
       cache directory, and reused by later calls. Delete the cache
       directory to download them again.

    Args:
        explicit: str, Path or None
            An explicitly configured asset directory, e.g. the config's
            'dataset.asset_directory'.
        base_dir: str, Path or None
            Directory a relative 'explicit' is anchored on. Typically the
            directory containing the config file.

    Returns:
        The asset directory as a pathlib.Path() object.

    Raises:
        FileNotFoundError:
            - If 'explicit' does not exist.
        NotADirectoryError:
            - If the resolved path is not a directory.
        httpx.HTTPError:
            - If downloading the public assets fails.
        ValueError:
            - If the downloaded archive contains a path outside its extraction
              directory.
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

    if not _CACHED_ASSETS.is_dir():
        _download_public_assets()
    logger.info(f"Using the asset directory '{_CACHED_ASSETS}'.")
    return _CACHED_ASSETS


def _download_public_assets() -> None:
    """
    Download the public assets archive and extract it into the cache.

    The archive is downloaded and extracted inside a temporary directory next
    to the cache, and only the complete 'assets' folder is moved into place.
    An interrupted download therefore never leaves a partial cache behind.

    Raises:
        httpx.HTTPError:
            - If the download fails.
        ValueError:
            - If an archive member would be extracted outside the extraction
              directory, e.g. through '..' or an absolute path.
    """
    _CACHED_ASSETS.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Downloading the public assets from '{_PUBLIC_ASSETS_URL}' to "
        f"'{_CACHED_ASSETS}'."
    )
    with tempfile.TemporaryDirectory(dir=_CACHED_ASSETS.parent) as temp_dir:
        archive_path = Path(temp_dir) / "assets.zip"
        with httpx.stream(
            "GET", _PUBLIC_ASSETS_URL, follow_redirects=True
        ) as response:
            response.raise_for_status()
            size = int(response.headers.get("Content-Length", 0)) or None
            with (
                archive_path.open("wb") as archive_file,
                tqdm(
                    total=size, unit="B", unit_scale=True, desc="Assets"
                ) as progress,
            ):
                for chunk in response.iter_bytes():
                    archive_file.write(chunk)
                    progress.update(len(chunk))

        extract_dir = (Path(temp_dir) / "extracted").resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                member_path = (extract_dir / name).resolve()
                if not member_path.is_relative_to(extract_dir):
                    raise ValueError(
                        f"The asset archive member '{name}' points outside "
                        "the extraction directory."
                    )
            archive.extractall(extract_dir)
        os.replace(extract_dir / "assets", _CACHED_ASSETS)


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
    Resolve a single asset path, e.g. a config's 'models[].model_path'.

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
