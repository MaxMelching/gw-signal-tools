# -- Make all important functions/classes from module available here
from .deriv import *
from .inner_product import *
from .utils import *
from .ft import *


__doc__ = """
A subpackage of `gw_signal_tools`, containing various useful tools
related to waveforms. This includes an implementation of the commonly
used noise-weighted inner product (with the possibility to optimize over
time and phase), an routine to optimize the overlap between two waveform
models over arbitrary input parameters and several classes for
numerical derivative calculation of waveforms.
"""
