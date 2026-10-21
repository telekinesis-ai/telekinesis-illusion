"""Local bpy helpers"""

import os
from telekinesis.illusion.utils.blender_env import isolate_user_extensions

isolate_user_extensions()
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
