# -- Make all important functions/classes from module available here
from .base_class import *  # noqa: F401
from .calc_utils import *  # noqa: F401
from .network import *  # noqa: F401
from .distances import *  # noqa: F401


__doc__ = """
A subpackage of `gw_signal_tools`, containing a Fisher matrix
implementation along with various funcions based on this (for instance
to calculate statistical and systematic errors).
"""
