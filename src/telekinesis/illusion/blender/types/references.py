# Adapted code; see NOTICE.md for attribution and licensing.
"""Struct utility functions, like providing all current struct instances."""

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from telekinesis.illusion.blender.types.struct import Struct


def get_instances() -> List[Tuple[str, "Struct"]]:
    """Returns a list containing all existing struct instances.

    :return: A list of tuples, each containing a struct and its name.
    """
    # this can only be imported here, else it causes a circle import
    # pylint: disable=import-outside-toplevel,cyclic-import
    from telekinesis.illusion.blender.types.struct import Struct
    # pylint: enable=import-outside-toplevel,cyclic-import

    instances = []
    # Iterate over all still existing instances
    for instance in Struct.__refs__:
        # Check that the referenced blender_obj inside is valid
        if instance.is_valid():
            # Collect instance and its name (its unique identifier)
            instances.append((instance.get_name(), instance))
    return instances
