"""Module shim: moved to core.repair_metadata (two frozen files still import this path)."""
import sys
from importlib import import_module

_module = import_module('core.repair_metadata')
if __name__ == '__main__':
    import runpy
    runpy.run_module(_module.__name__, run_name='__main__', alter_sys=True)
else:
    sys.modules[__name__] = _module
