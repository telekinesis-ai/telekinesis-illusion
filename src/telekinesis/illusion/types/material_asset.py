"""Immutable PBR texture-directory descriptors, without Blender state."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from telekinesis.illusion.types.asset import Asset


def find_color_texture(path: Path) -> Path | None:
    """Return the first color texture matching the existing PBR convention."""
    return next(
        (
            candidate
            for suffix in (".png", ".jpg", ".jpeg")
            if (candidate := path / f"{path.name}_Color{suffix}").is_file()
        ),
        None,
    )


@dataclass(frozen=True, slots=True, init=False)
class MaterialAsset(Asset):
    """One PBR texture directory with a unique name and selection groups.

    Paths are stored as absolute Paths. The directory must contain
    <directory-name>_Color.png, .jpg, or .jpeg; other maps are optional.
    Groups accept a string or sequence and become a tuple including ancestors:
    "metal/steel" gives ("metal", "metal/steel"). At least one is required.
    Construction checks file names without loading textures into Blender.
    Missing paths raise FileNotFoundError; invalid assets, names, or groups
    raise ValueError; non-string group entries raise TypeError.
    discover() derives groups from relative parent folders;
    use group_prefix when starting inside a group or at an asset directory.

    Example:
        asset = MaterialAsset.from_path(
            "assets/materials/metal/Metal019_1K-PNG", groups="metal")
    """

    path: Path
    name: str
    groups: tuple[str, ...]

    def __init__(
        self, path: str | Path, name: str, groups: str | Sequence[str]
    ) -> None:
        """Validate one directory, its registration name, and required groups."""
        path = self._validate_path(path)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        groups = self._normalize_groups(groups)
        if not groups:
            raise ValueError(
                "MaterialAsset requires groups; supply group_prefix "
                "when discovering root assets."
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "groups", groups)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        groups: str | Sequence[str],
        name: str | None = None,
    ) -> Self:
        """Create one descriptor; default its name to the directory name."""
        path = cls._validate_path(path)
        return cls(path, path.name if name is None else name, groups)

    @classmethod
    def _recognizes_path(cls, path: Path) -> bool:
        """Recognize directories with a color texture before descending."""
        return path.is_dir() and find_color_texture(path) is not None

    @classmethod
    def _from_discovered_path(cls, path: Path, groups: tuple[str, ...]) -> Self:
        """Construct a recognized material with its directory name and groups."""
        return cls(path, path.name, groups)
