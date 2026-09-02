from dataclasses import dataclass
import numpy as np
from typing import Union, Tuple

Number = Union[int, float]
Vector = Union[Number, Tuple[Number, ...]]


@dataclass(
    frozen=True
)  # frozen=True makes the instance immutable, dataclass adds __init__, __repr__ and __eq__ automatically
class Uniform:
    min: np.ndarray
    max: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "min", np.asarray(self.min))
        object.__setattr__(self, "max", np.asarray(self.max))

        if self.min.shape != self.max.shape:
            raise ValueError("min and max must have the same shape")

        if not (self.min <= self.max).all():
            raise ValueError("min must be <= max for all dimensions")


def uniform(min_value: Vector, max_value: Vector) -> Uniform:
    return Uniform(min_value, max_value)
