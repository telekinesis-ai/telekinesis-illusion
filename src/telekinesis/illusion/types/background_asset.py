"""Immutable HDRI file descriptors, without Blender state."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from telekinesis.illusion.types.asset import Asset


@dataclass(frozen=True, slots=True, init=False)
class BackgroundAsset(Asset):
    """One .hdr or .exr file with a unique name and selection groups.

    Paths are stored as absolute Paths. Groups accept a string or sequence and
    become a tuple including ancestors: "indoor/industrial" gives
    ("indoor", "indoor/industrial"). At least one group is required.
    Construction checks the extension and file existence without loading it.
    Missing paths raise FileNotFoundError; invalid assets, names, or groups
    raise ValueError; non-string group entries raise TypeError.
    discover() derives groups from relative parent folders;
    use group_prefix when starting inside a group containing HDRI files.

    Example:
        asset = BackgroundAsset.from_path(
            "assets/hdris/indoor/industrial/boiler_room_1k.exr",
            groups="indoor/industrial")
    """

    path: Path
    name: str
    groups: tuple[str, ...]

    def __init__(
        self, path: str | Path, name: str, groups: str | Sequence[str]
    ) -> None:
        """Validate one HDRI file, its registration name, and required groups."""
        path = self._validate_path(path)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        groups = self._normalize_groups(groups)
        if not groups:
            raise ValueError(
                "BackgroundAsset requires groups; supply group_prefix "
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
        """Create one descriptor; default its name to the file stem."""
        path = cls._validate_path(path)
        return cls(path, path.stem if name is None else name, groups)

    @classmethod
    def _recognizes_path(cls, path: Path) -> bool:
        """Recognize HDR and EXR files during validation and discovery."""
        return path.is_file() and path.suffix.lower() in (".hdr", ".exr")

    @classmethod
    def _from_discovered_path(cls, path: Path, groups: tuple[str, ...]) -> Self:
        """Construct a recognized HDRI with its file stem and derived groups."""
        return cls(path, path.stem, groups)
