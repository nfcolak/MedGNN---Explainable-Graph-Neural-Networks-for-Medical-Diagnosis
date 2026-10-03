"""ProtGNN clinical method implementation."""
import sys
from . import adapter as _module

# The old module name now names a package; preserve discovery while aliasing it.
_module.__path__ = __path__
setattr(_module, 'adapter', _module)
sys.modules[__name__] = _module
