"""Blender undo with restoration of pre-existing Python wrappers.

Adapted from the locally modified BlenderProc Utility.py (see NOTICE.md).
Wrappers created inside a reverted block must not escape the block.
"""

import inspect
from types import TracebackType
from typing import Optional, Type, List, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telekinesis.illusion.blender.types.struct import Struct

import bpy
from telekinesis.illusion.blender.types.references import get_instances
from telekinesis.illusion.types.background import Background


class UndoAfterExecution:
    """Reverts all changes done to the blender project inside this block.

    Usage: with UndoAfterExecution():
    """

    def __init__(
        self,
        check_point_name: Optional[str] = None,
        perform_undo_op: bool = True,
    ):
        if check_point_name is None:
            check_point_name = (
                inspect.stack()[1].filename
                + " - "
                + inspect.stack()[1].function
            )
        self.check_point_name = check_point_name
        self._perform_undo_op = perform_undo_op
        self.struct_instances: List[Tuple[str, "Struct"]] = []

    def __enter__(self):
        if self._perform_undo_op:
            # Collect all existing struct instances
            self.struct_instances = get_instances()
            self.background_instances = []
            for instance in Background.__refs__:
                # Like get_instances(), skip wrappers from an already-cleaned
                # scene which Python may still retain (e.g. in a traceback).
                try:
                    name = instance.get_name()
                except ReferenceError:
                    continue
                self.background_instances.append((name, instance))
            bpy.ops.ed.undo_push(message="before " + self.check_point_name)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        if self._perform_undo_op:
            bpy.ops.ed.undo_push(message="after " + self.check_point_name)
            # The current state points to "after", now by calling undo we go back to "before"
            bpy.ops.ed.undo()
            # After applying undo, all references to blender objects are invalid.
            # Therefore, we now go over all instances and update their references using their name as unique identifier.
            for name, struct in self.struct_instances:
                struct.update_blender_ref(name)

            for name, background in self.background_instances:
                background.update_blender_ref()
