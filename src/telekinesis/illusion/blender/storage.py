"""Internal state shared by the retained Blender rendering helpers.

Cleared when initializing a new Scene. See NOTICE.md for attribution.
"""

from typing import Any


class GlobalStorage:
    _storage_dict: dict[str, Any] = {}

    @staticmethod
    def add(key: str, value: Any):
        """
        Adds a key to the GlobalStorage this is independent of the global config, this can be used to store values
        over Module boundaries. Adding only works if there is not already a key like this in the GlobalStorage.

        For example the distance renderer sets the value "distance_end" during the rendering process, a module which is
        executed afterwards can then with get() access this value.

        These values can be added before the global config was inited as they do not depend on each other.

        :param key: which is added to the GlobalStorage
        :param value: which can be accessed by this key over the get() fct.
        """
        if key not in GlobalStorage._storage_dict:
            GlobalStorage._storage_dict[key] = value
        else:
            raise RuntimeError(
                f"The key: {key} was already set before with "
                f"this value: {GlobalStorage._storage_dict[key]}"
            )

    @staticmethod
    def set(key: str, value: Any):
        """
        Sets a key in the GlobalStorage this is independent of the global config, this can be used to store values
        over Module boundaries. Setting always works and overwrites existing keys

        For example the distance renderer sets the value "renderer_distance_end" during the rendering process, a module
        which is executed afterwards can then with get() access this value.

        These values can be added before the global config was inited as they do not depend on each other.

        :param key: which is added to the GlobalStorage
        :param value: which can be accessed by this key over the get() fct.
        """
        GlobalStorage._storage_dict[key] = value

    @staticmethod
    def get(key: str) -> Any:
        """
        Returns a value from the GlobalStorage, please check add() and set() for more information

        :param key: for which a value is searched
        :return: value for the key
        """
        if key in GlobalStorage._storage_dict:
            return GlobalStorage._storage_dict[key]
        raise RuntimeError(f"The key: {key} is not in the global storage!")

    @staticmethod
    def is_in_storage(key: str) -> bool:
        """
        Checks if a key is in the GlobalStorage

        :param key: for which a value is searched
        :return: True if the key is in the storage
        """
        return key in GlobalStorage._storage_dict
