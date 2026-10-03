"""Compatibility shim: moved to methods.protgnn.adapter. Old imports and `python3 -m` keep working."""
import sys
from importlib import import_module

_module = import_module('protgnn.adapter')
if __name__ == '__main__':
    import runpy
    runpy.run_module(_module.__name__, run_name='__main__', alter_sys=True)
else:
    sys.modules[__name__] = _module
