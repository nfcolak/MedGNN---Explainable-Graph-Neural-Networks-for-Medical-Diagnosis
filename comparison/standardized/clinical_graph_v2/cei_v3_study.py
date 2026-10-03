"""Compatibility shim: moved to studies.cei.cei_v3_study. Old imports and `python3 -m` keep working."""
import sys
from importlib import import_module

_module = import_module('comparison.standardized.clinical_graph_v2.studies.cei.cei_v3_study')
if __name__ == '__main__':
    import runpy
    runpy.run_module(_module.__name__, run_name='__main__', alter_sys=True)
else:
    sys.modules[__name__] = _module
