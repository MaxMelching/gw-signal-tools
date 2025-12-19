# -- Local Package Imports
from .base import WaveformDerivativeBase
from .custom import WaveformDerivativeGWSignaltools  # noqa: F401
from .nd import WaveformDerivativeNumdifftools  # noqa: F401
from .nd_amp_phase import WaveformDerivativeAmplitudePhase  # noqa: F401


__doc__ = """Module for the ``WaveformDerivative`` class."""

__all__ = ('WaveformDerivative',)


class WaveformDerivative:
    r"""
    Constructor class for derivatives of waveforms. This class allows to
    choose between different implementations by passing the
    :code:`deriv_routine` argument. All other arguments are passed on to
    the selected derivative class.

    Parameters
    ----------
    deriv_routine : string or Callable, optional
        Available routines, i.e. they keys of
        The routine that could be used. Can be either a callable object
        (ideally, a class inherited from ``~gw_signal_tools.waveform.
        deriv.WaveformDerivativeBase`` because many users of the
        ``WaveformDerivative`` class expect certain properties to be
        defined) or a string that is registered in `~gw_signal_tools.
        waveform.WaveformDerivative.deriv_routine_class_map`.
        Default is `'numdifftools'`.
    *args, **kw_args :
        Arguments passed on to the selected derivative class.

    Returns
    -------
    Instance of requested class.

    Notes
    -----
    Here we compare the different derivative routines available from
    the `~gw_signal_tools.waveform.deriv` module (i.e. those that can
    be passed as strings to `deriv_routine`).

        - 'numdifftools': can do adaptive refinement only for certain
        frequencies where convergence is slower, making it potentially
        more reliable than the previous routine. However, this also
        requires more waveform calls, making the calculation slower.
        This routine is the most robust one for general use cases.

        - 'gw_signal_tools': usually the fastest method, but can lack
        accuracy for certain configurations (since it only refines
        estimate for whole frequency range, not parts of it).

        - 'amplitude_phase': may be beneficial for accuracy in case
        strain oscillates fast and thus has steep derivative. Then,
        looking at amplitude and phase separately should yield much more
        well-posed functions. For usual applications though, it may be
        significantly slower than the other routines. After all, two
        derivatives have to be calculated, which means it involves more
        waveform calls (though that depends on whether waveform caching
        is activated or not). But in case other routines fail, it might be
        worth a try.
    """

    deriv_routine_class_map = {
        'gw_signal_tools': WaveformDerivativeGWSignaltools,
        'numdifftools': WaveformDerivativeNumdifftools,
        'amplitude_phase': WaveformDerivativeAmplitudePhase,
    }
    # -- Idea: one can easily add entries here for custom derivative routines.
    # -- Instances created after creation will have those entries.

    def __new__(cls, *args, **kw_args) -> WaveformDerivativeBase:
        deriv_routine = kw_args.pop('deriv_routine', 'numdifftools')

        if isinstance(deriv_routine, str):
            try:
                deriv_routine_class = cls.deriv_routine_class_map[deriv_routine]
            except KeyError as e:
                raise ValueError(
                    f"Invalid deriv_routine '{deriv_routine}', it is not registered "
                    'in `~gw_signal_tools.waveform.WaveformDerivative.deriv_routine_class_map`.'
                ) from e

            return deriv_routine_class(*args, **kw_args)
            # -- Do NOT do inside try. Otherwise errors are potentially messed
        else:
            try:
                return deriv_routine(*args, **kw_args)
            except Exception as e:
                raise RuntimeError(
                    'Either the `deriv_routine` you provided is not callable '
                    'or one of the arguments passed to it is invalid.'
                ) from e
