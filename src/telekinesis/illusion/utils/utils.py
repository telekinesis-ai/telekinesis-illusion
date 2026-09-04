# import bpy
from datetime import datetime
import secrets


def make_run_name(prefix: str = "training") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(4)
    return f"{prefix}_{ts}_{suffix}"


# def save_scene(
#    filepath: str = "scene.blend"
# ) -> None:
#    bpy.ops.wm.save_as_mainfile(
#        filepath=filepath
#    )
