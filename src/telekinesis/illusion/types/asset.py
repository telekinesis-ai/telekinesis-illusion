"""Blender-independent local asset creation and discovery."""

import abc
from collections.abc import Sequence
from pathlib import Path
from typing import Self


class Asset(abc.ABC):
    """Shared local discovery flow for immutable asset descriptors.

    Concrete descriptors recognize paths and construct discovered assets.

    Example:
        from telekinesis.illusion.types import ModelAsset

        assets = ModelAsset.discover("assets/models")
    """

    __slots__ = ()

    @classmethod
    def discover(
        cls, root: str | Path, *, group_prefix: str = ""
    ) -> list[Self]:
        """Discover assets in sorted depth-first order beneath a directory.

        Groups come from each asset's parent relative to root, prefixed with
        group_prefix. Recognized directories are not traversed further.
        Material and background assets directly in root need a group_prefix.
        Directory symlinks are not traversed, preventing cycles.

        Raises:
            FileNotFoundError: The root is missing or no assets are found.
            NotADirectoryError: The root is not a directory.
            TypeError: group_prefix is not a string.
            ValueError: A group is invalid or a required group is missing.
        """
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        if not root.is_dir():
            raise NotADirectoryError(root)
        if not isinstance(group_prefix, str):
            raise TypeError("group_prefix must be a string.")
        prefix = cls._normalize_groups(group_prefix) if group_prefix else ()
        pending = [root]
        assets = []
        while pending:
            path = pending.pop()
            if cls._recognizes_path(path):
                parents = (
                    () if path == root else path.parent.relative_to(root).parts
                )
                group = "/".join((*prefix[-1:], *parents))
                groups = cls._normalize_groups(group) if group else ()
                assets.append(cls._from_discovered_path(path, groups))
            elif path.is_dir() and not path.is_symlink():
                pending.extend(sorted(path.iterdir(), reverse=True))
        if not assets:
            raise FileNotFoundError(
                f"No {cls.__name__} assets found under '{root}'."
            )
        return assets

    @classmethod
    def _validate_path(cls, path: str | Path) -> Path:
        """Resolve one local path and require this descriptor's asset type."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if not cls._recognizes_path(path):
            raise ValueError(f"'{path}' is not a valid {cls.__name__} path.")
        return path

    @staticmethod
    def _normalize_groups(groups: str | Sequence[str]) -> tuple[str, ...]:
        """Validate slash-separated groups and include each ancestor once."""
        if isinstance(groups, str):
            groups = (groups,)
        if not isinstance(groups, Sequence):
            raise TypeError("groups must be a string or a sequence of strings.")
        normalized = {}
        for group in groups:
            if not isinstance(group, str):
                raise TypeError("Each group must be a string.")
            parts = group.split("/")
            if "\\" in group or any(
                not part.strip() or part in (".", "..") for part in parts
            ):
                raise ValueError(
                    f"Invalid group {group!r}; use forward-slash paths."
                )
            for index in range(1, len(parts) + 1):
                normalized["/".join(parts[:index])] = None
        return tuple(normalized)

    @classmethod
    @abc.abstractmethod
    def _recognizes_path(cls, path: Path) -> bool:
        """Return whether path is one supported asset during local traversal."""

    @classmethod
    @abc.abstractmethod
    def _from_discovered_path(cls, path: Path, groups: tuple[str, ...]) -> Self:
        """Construct one descriptor after recognition, using derived groups."""
