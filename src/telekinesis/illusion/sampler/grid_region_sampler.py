"""
Defines a stateful sampler that hands out grid-cell centers on the upper face
of a target container, for structured bin picking scene generation.
"""

from typing import List, Optional, Tuple
import random

import numpy as np
from loguru import logger

from telekinesis.illusion.core.context import Context
from telekinesis.illusion.sampler.upper_region_sampler import (
    Region2D,
    select_upper_region,
)


class GridRegionSampler:
    """
    Stateful sampler that returns the next grid cell center on the upper face
    of the currently visible container. Used for structured (grid) part
    placement in bin picking scenes.

    The grid is computed lazily against the bin's top face the first time it
    is requested, and re-computed when the visible container changes (e.g.
    across scenes that randomize between several bin geometries) or when all
    slots have been consumed for the current scene.
    """

    def __init__(
        self,
        context: Context,
        rows: int,
        cols: int,
        layers: int = 1,
        layer_spacing: float = 0.03,
        face_sample_range: Tuple[float, float] = (0.05, 0.95),
        min_height: float = 0.0,
        max_height: float = 0.0,
        shuffle: bool = True,
        xy_jitter: float = 0.0,
        upper_dir: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            context: Context
                The scene context for resolving the currently visible container.
            rows: int
                Number of grid rows along the first in-plane face vector.
            cols: int
                Number of grid columns along the second in-plane face vector.
            layers: int
                Number of vertically stacked layers above the bin floor.
            layer_spacing: float
                Distance between successive layers along the upper direction.
            face_sample_range: Tuple[float, float]
                Relative bounds [u_min, u_max] within the top face used for the
                grid. Margins keep cells away from bin walls.
            min_height: float
                Minimum extra height above the top face for the lowest layer.
            max_height: float
                Maximum extra height above the top face for the lowest layer.
                If equal to min_height, no jitter is applied.
            shuffle: bool
                If True, the order in which cells are handed out is shuffled
                per scene.
            xy_jitter: float
                Maximum positional offset as a fraction of cell spacing
                (0.0 – 0.5). Each cell center is randomly displaced along the
                two in-plane face vectors by up to this fraction of the
                cell-to-cell distance.
            upper_dir: Optional[np.ndarray]
                The 'up' direction. Defaults to +Z.
        """
        if rows < 1 or cols < 1 or layers < 1:
            raise ValueError(
                f"rows/cols/layers must be >= 1, got {rows}/{cols}/{layers}"
            )
        if not 0.0 <= xy_jitter <= 0.5:
            raise ValueError(
                f"xy_jitter must be in [0.0, 0.5], got {xy_jitter}"
            )
        if max_height < min_height:
            raise ValueError(
                f"max_height ({max_height}) must be >= min_height ({min_height})"
            )

        self._context = context
        self._rows = rows
        self._cols = cols
        self._layers = layers
        self._layer_spacing = layer_spacing
        self._face_sample_range = face_sample_range
        self._min_height = min_height
        self._max_height = max_height
        self._shuffle = shuffle
        self._xy_jitter = xy_jitter
        self._upper_dir = (
            np.array([0.0, 0.0, 1.0])
            if upper_dir is None
            else np.array(upper_dir)
        )
        self._upper_dir = self._upper_dir / np.linalg.norm(self._upper_dir)

        self._cached_bin_name: Optional[str] = None
        self._cells: List[np.ndarray] = []
        self._next_index: int = 0

    def capacity(self) -> int:
        """Total number of grid slots."""
        return self._rows * self._cols * self._layers

    def update_grid_params(self, **params) -> bool:
        """
        Re-tune the grid in place, discarding the cached cells so the next
        next_location() rebuilds with the new values.

        Intended for interactive tools (e.g. the Blender spec editor) that
        adjust the grid between scenes - the cells are only rebuilt lazily, so
        an unchanged call costs nothing.

        Args:
            **params:
                Any of 'rows', 'cols', 'layers', 'layer_spacing',
                'face_sample_range', 'min_height', 'max_height', 'shuffle',
                'xy_jitter'. Omitted parameters keep their current value.

        Raises:
            TypeError: When an unknown parameter name is given.
            ValueError: When the resulting configuration is invalid (same
                constraints as __init__).

        Returns:
            True if any value actually changed, False otherwise.
        """
        tunable = (
            "rows",
            "cols",
            "layers",
            "layer_spacing",
            "face_sample_range",
            "min_height",
            "max_height",
            "shuffle",
            "xy_jitter",
        )
        unknown = set(params) - set(tunable)
        if unknown:
            raise TypeError(
                f"Unknown grid parameter(s) {sorted(unknown)}. "
                f"Expected any of {list(tunable)}."
            )

        pending = {}
        for name in tunable:
            if name not in params or params[name] is None:
                continue
            value = params[name]
            if name == "face_sample_range":
                value = tuple(value)
            pending[name] = value

        merged = {name: getattr(self, f"_{name}") for name in tunable}
        merged.update(pending)

        # Same guards as __init__ - a bad interactive edit must not leave the
        # sampler in a state that only blows up later during a rebuild.
        if merged["rows"] < 1 or merged["cols"] < 1 or merged["layers"] < 1:
            raise ValueError(
                f"rows/cols/layers must be >= 1, got "
                f"{merged['rows']}/{merged['cols']}/{merged['layers']}"
            )
        if not 0.0 <= merged["xy_jitter"] <= 0.5:
            raise ValueError(
                f"xy_jitter must be in [0.0, 0.5], got {merged['xy_jitter']}"
            )
        if merged["max_height"] < merged["min_height"]:
            raise ValueError(
                f"max_height ({merged['max_height']}) must be >= "
                f"min_height ({merged['min_height']})"
            )

        changed = any(
            merged[name] != getattr(self, f"_{name}") for name in tunable
        )
        if not changed:
            return False

        for name in tunable:
            setattr(self, f"_{name}", merged[name])

        # Force a rebuild on the next next_location() call.
        self._cells = []
        self._next_index = 0
        self._cached_bin_name = None
        return True

    def next_location(self, target_object_names: List[str]) -> np.ndarray:
        """
        Return the next grid cell center (world coordinates) on the upper face
        of the visible container.

        Args:
            target_object_names: List[str]
                Container name prefixes to match against the currently visible
                objects. The first match is used as the bin.

        Returns:
            np.ndarray: world-space location for the next grid cell.
        """
        bin_name = self._resolve_visible_bin_name(target_object_names)
        if bin_name != self._cached_bin_name or self._next_index >= len(
            self._cells
        ):
            self._rebuild_grid(bin_name)

        location = self._cells[self._next_index]
        self._next_index += 1
        return location

    def _resolve_visible_bin_name(self, target_object_names: List[str]) -> str:
        visible = self._context.get_visible_object_names()
        bin_name = next(
            (
                name
                for name in visible
                if any(t in name for t in target_object_names)
            ),
            None,
        )
        if bin_name is None:
            raise RuntimeError(
                f"No visible container matching {target_object_names} found in "
                "the context."
            )
        return bin_name

    def _rebuild_grid(self, bin_name: str) -> None:
        illusion_object = self._context.get_objects_by_name(bin_name)
        bproc_mesh_object = illusion_object.get_object()
        region: Region2D = select_upper_region(
            bproc_mesh_object, self._upper_dir
        )

        base_point = region.base_point()
        vec1, vec2 = region.vectors()

        cells: List[np.ndarray] = []
        u_min, u_max = self._face_sample_range
        for layer in range(self._layers):
            layer_offset = self._upper_dir * (
                self._min_height + layer * self._layer_spacing
            )
            for i in range(self._rows):
                u = u_min + (u_max - u_min) * (i + 0.5) / self._rows
                for j in range(self._cols):
                    v = u_min + (u_max - u_min) * (j + 0.5) / self._cols
                    cell = (
                        np.array(base_point, dtype=float)
                        + u * np.array(vec1, dtype=float)
                        + v * np.array(vec2, dtype=float)
                        + layer_offset
                    )
                    if self._xy_jitter > 0.0:
                        du = (u_max - u_min) / self._rows
                        dv = (u_max - u_min) / self._cols
                        jitter_u = random.uniform(
                            -self._xy_jitter * du, self._xy_jitter * du
                        )
                        jitter_v = random.uniform(
                            -self._xy_jitter * dv, self._xy_jitter * dv
                        )
                        cell = (
                            cell
                            + jitter_u * np.array(vec1, dtype=float)
                            + jitter_v * np.array(vec2, dtype=float)
                        )
                    if self._max_height > self._min_height:
                        jitter = random.uniform(
                            0.0, self._max_height - self._min_height
                        )
                        cell = cell + self._upper_dir * jitter
                    cells.append(cell)

        if self._shuffle:
            random.shuffle(cells)

        self._cells = cells
        self._next_index = 0
        self._cached_bin_name = bin_name
        logger.info(
            f"Built grid of {len(self._cells)} cells "
            f"({self._rows}x{self._cols}x{self._layers}) on '{bin_name}'."
        )
