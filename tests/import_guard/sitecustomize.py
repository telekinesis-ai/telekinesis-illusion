"""Reject accidental use of an installed BlenderProc in example subprocesses."""

import importlib.abc
import sys


class RejectBlenderProc(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "blenderproc" or fullname.startswith("blenderproc."):
            raise RuntimeError("BlenderProc import attempted by example")


sys.meta_path.insert(0, RejectBlenderProc())
