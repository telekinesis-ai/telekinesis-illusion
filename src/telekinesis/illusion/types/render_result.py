"""A captured batch, independent of the lifetime of its Blender scene."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Rendered passes, category names, per-view capture metadata and model poses.

    ``objects`` lists every active model with a category id > 0 at render time:
    ``{"name", "category_id", "matrix_world" (4x4, metres, rigid: the load scale
    is baked into the mesh), "scale"}``. Pose writers (BOP) use it; it is empty
    when the engine did not record it.
    """

    data: dict[str, Any]
    categories: dict[int, str]
    captures: tuple[dict[str, Any], ...]
    objects: tuple[dict[str, Any], ...] = ()

    def __len__(self) -> int:
        return len(self.captures)
