"""Blender-independent bounds validation and the existing balanced allocation."""

from collections.abc import Mapping
from numbers import Integral
import random


def validate_counts(
    counts: Mapping[str, tuple[int, int]], minimum=None, maximum=None
):
    if not counts:
        raise ValueError(
            "Provide instance counts keyed by registered model name."
        )
    for name, bounds in counts.items():
        if len(bounds) != 2 or any(
            isinstance(v, bool) or not isinstance(v, Integral) for v in bounds
        ):
            raise ValueError(
                f"{name}: provide integer (minimum, maximum) bounds."
            )
        low, high = bounds
        if not name or low < 0 or high < low:
            raise ValueError(f"Invalid instance bounds for {name!r}: {bounds}")
    lower = sum(v[0] for v in counts.values())
    upper = sum(v[1] for v in counts.values())
    for value in (minimum, maximum):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 0
        ):
            raise ValueError(
                "Total instance bounds must be nonnegative integers."
            )
    lower = max(lower, minimum) if minimum is not None else lower
    upper = min(upper, maximum) if maximum is not None else upper
    if lower > upper:
        raise ValueError(
            "Total instance bounds conflict with per-model bounds."
        )
    return int(lower), int(upper)


def sample_counts(counts, minimum=None, maximum=None, rng=random):
    low, high = validate_counts(counts, minimum, maximum)
    total = rng.randint(low, high)
    selected = {name: bounds[0] for name, bounds in counts.items()}
    for _ in range(total - sum(selected.values())):
        candidates = [
            name
            for name, bounds in counts.items()
            if selected[name] < bounds[1]
        ]
        rng.shuffle(candidates)
        name = min(candidates, key=lambda n: selected[n] / counts[n][1])
        selected[name] += 1
    return selected
