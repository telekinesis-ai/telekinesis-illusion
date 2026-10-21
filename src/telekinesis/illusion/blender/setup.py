"""Temporary paths for the standalone bpy runtime. No package installation."""

import os
import tempfile


def is_using_external_bpy_module() -> bool:
    return True


class SetupUtility:
    @staticmethod
    def determine_temp_dir(temp_dir=None):
        # Always own a child directory; cleanup must never remove the caller's
        # existing temporary root and its unrelated files.
        if temp_dir is not None:
            os.makedirs(temp_dir, exist_ok=True)
        return tempfile.mkdtemp(prefix="bpy-illusion-", dir=temp_dir)

    @staticmethod
    def setup_utility_paths(temp_dir):
        from telekinesis.illusion.blender.utility import Utility

        Utility.temp_dir = os.path.abspath(temp_dir)
        os.makedirs(Utility.temp_dir, exist_ok=True)
