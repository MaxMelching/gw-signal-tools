# -- Make all important functions/classes from module available here
from .fisher_utils import (  # noqa: F401
    num_diff, fisher_matrix
)
from .fisher import FisherMatrix  # noqa: F401
from .fisher_network import FisherMatrixNetwork  # noqa: F401
from .distances import distance, linearized_distance  # noqa: F401


__doc__ = """
A subpackage of `gw_signal_tools`, containing a Fisher matrix
implementation along with various funcions based on this (for instance
to calculate statistical and systematic errors).
"""
