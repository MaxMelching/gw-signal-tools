# -- Make all important functions/classes from module available here
from .deriv import (  # noqa: F401
    WaveformDerivativeGWSignaltools, WaveformDerivative
)
from .inner_product import (  # noqa: F401
    inner_product, norm, overlap, optimize_overlap, _INNER_PROD_ARGS,
    get_default_opt_params, test_hm, test_precessing
)
from .nd_deriv import (  # noqa: F401
    WaveformDerivativeNumdifftools, WaveformDerivativeAmplitudePhase
)
from .utils import *  # noqa: F401

# TODO: Really add inner_prod_args here?
# TODO: definitely only import from .waveform in other files. Safer in case something is renamed


__doc__ = """
A subpackage of `gw_signal_tools`, containing various useful tools
related to waveforms. This includes an implementaiton of the commonly
used noise-weighted inner product (with the possibility to optimize over
time and phase), an routine to optimize the overlap between two waveform
models over arbitrary input parameters and several classes for
numerical derivative calculation of waveforms.
"""
