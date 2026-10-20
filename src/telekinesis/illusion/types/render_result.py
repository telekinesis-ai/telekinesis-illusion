"""A captured batch, independent of the lifetime of its Blender scene."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RenderResult:
    data: dict[str, Any]
    categories: dict[int, str]
    captures: tuple[dict[str, Any], ...]

    def __len__(self) -> int:
        return len(self.captures)
