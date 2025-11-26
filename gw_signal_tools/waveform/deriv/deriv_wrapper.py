# -- Standard Lib Imports
# from functools import cached_property  # TODO: use for some stuff?
from typing import Optional, Literal, Any

# -- Third Party Imports
import numpy as np
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import matplotlib as mpl

# -- Local Package Imports
from .deriv_gw_signal_tools import WaveformDerivativeGWSignaltools
from .deriv_nd import WaveformDerivativeNumdifftools
from .deriv_amp_phase import WaveformDerivativeAmplitudePhase


__doc__ = """Module for the ``WaveformDerivative`` class."""

__all__ = ('WaveformDerivative',)


deriv_routine_class_map = {
    'gw_signal_tools': 'WaveformDerivativeGWSignaltools',
    'numdifftools': 'WaveformDerivativeNumdifftools',
    'amplitude_phase': 'WaveformDerivativeAmplitudePhase',
}


class WaveformDerivative:
    r"""
    Constructor class for numerical derivative of waveforms. This class
    allows to choose between different implementations by passing the
    .code:`deriv_routine` argument. All other arguments are passed on to
    the selected derivative class.

    Parameters
    ----------
    deriv_routine : Literal or Callable, optional
        Available routines, i.e. they keys of
        `~gw_signal_tools.waveform.deriv_routine_class_map`. Default
        is `'numdifftools'`.

        TODO: of not a string, it is assumed to be a class that is callable
        in the same manner as WaveformDerivativeBase (i.e. ideally, is
        inherited from it)
    *args, **kw_args :
        Arguments passed on to the selected derivative class.

    Returns
    -------
    Instance of requested class.

    Notes
    -----
    Here we compare the different available derivative routines.

        - 'gw_signal_tools': usually the fastest method, but can can
        lack accuracy for certain configurations (since it only refines
        estimate for whole frequency range, not parts of it)

        - 'numdifftools': can do adaptive refinement only for certain
        frequencies where convergence is slower, making it potentially
        more reliable than the previous routine. However, this also
        requires more waveform calls, making the calculation slower.

        - 'amplitude_phase': may be beneficial for accuracy in case
        strain oscillates fast and thus has steep derivative. Then,
        looking at amplitude and phase separately should yield much more
        well-posed functions. For usual applications though, it may be
        significantly slower than the other routines. After all, two
        derivatives have to be calculated, which means it involves the
        waveform calls. But in case other routines fail, it might be
        worth a try. Moreover, this issue depends on whether waveform
        caching is activated or not.
    """

    # TODO: make deriv_routine_class_map an attribute of this class?

    def __new__(cls, *args, **kw_args):
        # TODO: we could have deriv_routine as a class!!! May be better than plugin
        try:
            deriv_routine = kw_args.pop('deriv_routine', 'numdifftools')
            if isinstance(deriv_routine, str):
                # deriv_routine = deriv_routine.lower()
                deriv_routine_class = deriv_routine_class_map[deriv_routine]
            else:
                # deriv_routine_class = deriv_routine
                return deriv_routine(*args, **kw_args)
        except KeyError:
            raise ValueError(
                f"Invalid deriv_routine '{deriv_routine}', it is not registered "
                "in `~gw_signal_tools.waveform.deriv_routine_class_map`."
            )
        try:
            deriv = globals()[deriv_routine_class]
        except KeyError:
            raise ValueError(
                f"Cannot find '{deriv_routine_class}'."
            )
        return deriv(*args, **kw_args)  # Do NOT do inside try. Otherwise errors are potentially messed
