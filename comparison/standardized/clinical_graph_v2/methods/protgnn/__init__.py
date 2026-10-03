"""ProtGNN clinical method implementation."""
from . import adapter as _adapter
from .adapter import *


def __getattr__(name):
    """Preserve legacy module attributes while keeping the package's own path."""
    return getattr(_adapter, name)
