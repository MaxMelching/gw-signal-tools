# -- Standard Lib Imports
from typing import Optional, Literal, Final
from inspect import signature

# -- Third Party Imports
import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import astropy.units as u
from gwpy.types import Index

# -- Local Package Imports
from ..units import preferred_unit_system
from ..logging import logger
from .utils import (
    signal_at_xindex,
    apply_time_phase_shift,
    fill_x_range,
    adjust_x_range,
)
from .ft import td_to_fd
from ._error_helpers import _q_convert, _compare_series_xindex, _assert_ft_compatible
from ..types import FDWFGen


__doc__ = """
Implementation of noise-weighted inner product that is
common in GW data analysis and helpers for computation.
"""

__all__ = (
    'inner_product',
    'norm',
    'overlap',
    'inner_product_computation',
    'optimized_inner_product',
    '_INNER_PROD_ARGS',
    'optimize_overlap',
    'get_default_opt_params',
    'test_hm',
    'test_precessing',
)


def _determine_x_range(x_range, *s):
    """Inner product helper to determine the range of x-values."""
    x_unit = s[0].xunit
    x_lower, x_upper = (
        max([_s.xindex[0] for _s in s]),
        min([_s.xindex[-1] for _s in s]),
    )

    # -- If bounds are given, check that they fit the input data
    if x_range is not None:
        if len(x_range) != 2:  # pragma: no cover
            raise ValueError(
                '`f_range` must contain lower and upper frequency bounds for '
                'integration. One of them or both can be `None`, but both '
                'have to be specified if `f_range` is given.'
            )

        # -- Check if both lower and upper are given or one of them is None
        if x_range[0] is not None:
            x_lower_new = _q_convert(
                x_range[0], x_unit, 'f_range[0]', 'signal.frequencies'
            )
        else:
            x_lower_new = x_lower

        if x_range[1] is not None:
            x_upper_new = _q_convert(
                x_range[1], x_unit, 'f_range[1]', 'signal.frequencies'
            )
        else:
            x_upper_new = x_upper

        # -- New lower bound must be greater than current lower bound,
        # -- otherwise no values for the range are available in signals
        if x_lower_new >= x_lower:
            x_lower = x_lower_new
        else:
            # -- Leave lower bound at x_lower, no update
            logger.info(
                f'Given lower bound of {x_lower_new} is smaller than '
                'values available from given signals. Taking a lower '
                f'bound of {x_lower} instead.'
            )

        # -- New upper bound must be smaller than current upper bound,
        # -- otherwise no values for the range are available in signals
        if x_upper_new <= x_upper:
            x_upper = x_upper_new
        else:
            # -- Leave upper bound at x_upper, no update
            logger.info(
                f'Given upper bound of {x_upper_new} is larger than '
                'values available from given signals. Taking an upper '
                f'bound of {x_upper} instead.'
            )

    return x_lower, x_upper


def inner_product(
    signal1: TimeSeries | FrequencySeries,
    signal2: TimeSeries | FrequencySeries,
    psd: Optional[FrequencySeries] = None,
    signal_interpolation: bool = True,
    f_range: Optional[list[float] | list[u.Quantity]] = None,
    df: Optional[float | u.Quantity] = None,
    optimize_time_and_phase: bool = False,
    optimize_time: bool = False,
    optimize_phase: bool = False,
    min_dt_prec: Optional[float] = None,
    return_opt_info: bool = False,
) -> (
    u.Quantity
    | tuple[
        u.Quantity,
        dict[
            Literal['match_series', 'peak_phase', 'peak_time'], u.Quantity | TimeSeries
        ],
    ]
):
    r"""
    Calculates the noise-weighted inner product

    .. math:: \langle a, b \rangle = 2 \Re \int_{-\infty}^{\infty}
        \frac{\tilde{a}(f) \tilde{b}^*(f)}{S_n(f)} \, df

    of two signals using their representations
    :math:`\tilde{a}(f), \tilde{b}(f)` in frequency domain.

    In case of a :code:`psd` :math:`S_n(f)` that is equal to 1 at all
    frequencies (the default case), this corresponds to the standard
    :math:`L^2` inner product.

    Parameters
    ----------
    signal1 : ~gwpy.timeseries.TimeSeries or ~gwpy.frequencyseries.FrequencySeries
        First signal.
    signal2 : ~gwpy.timeseries.TimeSeries or ~gwpy.frequencyseries.FrequencySeries
        Second signal.
    psd : ~gwpy.frequencyseries.FrequencySeries, optional, default = None
        Power spectral density to use in inner product. If None, it is
        taken to be 1 1/Hz at all frequencies. The frequency range of
        this default PSD is [-2048 Hz, 2048 Hz], so in case larger
        ranges shall be used, a custom PSD with suitable frequencies
        has to be provided.

        Note that this inner product is designed for one-sided PSDs.
    signal_interpolation : boolean, optional, default = True
        Determines whether or not it is ensured that signals have the
        same frequency range and spacing, with a potential interpolation
        happening. If you do not want this to happen, set this argument
        to ``False``, whence all arguments will be directly passed on to
        the inner product calculators (resulting in a potential, though
        hopefully small, speedup), allowing very precise control over
        the involved frequency ranges. This can be achieved via a
        specific way of generating the input waveforms/data or by
        a proper call to a function like
        `~gwsignal_tools.waveform.signal_at_index` that yields signals
        at specific frequencies, right before this function. Providing
        a similar functionality inside this function turned out to be
        very error-prone, particularly the interplay of evaluating at
        the frequencies and managing when to interpolate, in combination
        with having to distinguish equal/unequal sampling. Hence,
        we have resorted to keeping just `signal_interpolation`.

        In principle, one could just resort to
        `inner_product_calculation`, `optimized_inner_product` to get
        the same results, without having to go through `inner_product`.
        However, this would mean convenient wrappers such as `norm` or
        `overlap` would have to redefined, which serves as justification
        for this argument.

        Additionally, this argument is compatible with giving different
        `f_range` arguments, i.e. restricting is still supported (this
        would have to be done manually for `inner_product_computation`).
    f_range : list[float] or list[~astropy.units.Quantity], optional, default = None
        Frequency range to compute inner product over. Is potentially
        cropped if bounds are greater than frequency range of one of the
        input signals.

        The whole argument can be None, otherwise it must have length 2.
        However, one of the bounds (or both) can still be None, which
        indicates that no boundary shall be set. If no bound is given,
        automatic bounds are computed from the frequency ranges of the
        input signals and :code:`psd`. Note that conditioned waveforms
        might have a larger range than the one specified during waveform
        generation. For this reason, giving :code:`f_range` may be very
        important.
    df : float or ~astropy.units.Quantity, optional, default = None
        Distance df between samples in frequency domain to use in
        integration.
        If None, it is set to 0.0625 Hz (or whatever frequency unit is
        used in the signals), which is the default df of frequency
        domain waveforms generated by :code:`~lalsimulation.gwsignal`.
    optimize_time_and_phase : bool, optional, default = False
        Determines if a match is computed or just a "regular" inner
        product. The match will be optimized over relative time and
        phase shifts between :code:`signal1` and :code:`signal2`.

        It is also possible to optimize separately over time or phase
        shifts by using the arguments :code:`optimize_time`,
        :code:`optimize_phase`. Note, however, that
        :code:`optimize_time_and_phase=True` will override those two.
    return_opt_info : boolean, optional, default = False
        Whether or not to return a dictionary with additional
        information about the optimization results. Contains the full
        time series of match values at all time shifts that are not
        optimized over phase yet.

        Has no effect if no optimization is not carried out.

    Returns
    -------
    ~astropy.units.Quantity or tuple[~astropy.units.Quantity, dict[str, ~gwpy.timeseries.TimeSeries | ~astropy.units.Quantity]]
        Inner product value with :code:`signal1`, :code:`signal2`
        inserted. Additional information if optimization is carried out
        and :code:`return_opt_info=True`. More details on the latter are
        provided in the documentaiton of the
        :code:`optimized_inner_product` function.

    Raises
    ------
    TypeError
        In case either one of :code:`signal1`, :code:`signal2`,
        :code:`psd` has wrong type.
    ValueError
        In case format of :code:`f_range` parameter is not correct.

    See Also
    --------
    gw_signal_tools.waveform.ft.td_to_fd :
        Used to convert ``TimeSeries`` input to a ``FrequencySeries``.
    gwpy.frequencyseries.frequencyseries.interpolate :
        Function used to get signals to same sampling rate.

    Notes
    -----
    Some tips regarding the :code:`df` parameter:
    (i) It should not be too large, otherwise results of the inner
    product may be erroneous.
    (ii) If chosen very small (e.g. in range of 0.001Hz), you should
    consider selecting only powers of two like 2**-10 Hz because these
    work best with certain function calls that utilize the Fourier
    transform of the involved signals. An indicator this might be
    necessary is a shape mismatch error.
    """
    # -- If necessary, do Fourier transform
    if isinstance(signal1, TimeSeries):
        logger.info(
            '`signal1` is a ``TimeSeries``, performing an automatic FFT. '
            'Due to potential issues with conventions and resolution of '
            'the result, this is discouraged, consider doing it manually.'
        )

        signal1 = td_to_fd(signal1, convention='unwrap')

    if isinstance(signal2, TimeSeries):
        logger.info(
            '`signal2` is a ``TimeSeries``, performing an automatic FFT. '
            'Due to potential issues with conventions and resolution of '
            'the result, this is discouraged, consider doing it manually.'
        )

        signal2 = td_to_fd(signal2, convention='unwrap')

    # -- Store frequently accessed, quite lengthy boolean
    _optimize = optimize_time_and_phase or optimize_time or optimize_phase

    # -- Handling of units
    if isinstance(signal1, FrequencySeries):
        frequ_unit = signal1.frequencies.unit
    else:
        raise TypeError(
            '`signal1` has to be a GWpy ``TimeSeries`` or ``FrequencySeries``.'
        )

    # -- NOTE: we do not check for equal units of signals, since there
    # -- are usecases for inner products between signals with different
    # -- units (e.g. Fisher matrix). Thus this is not a physical
    # -- requirement, so we do not enforce it.

    if isinstance(signal2, FrequencySeries):
        assert signal2.frequencies.unit._is_equivalent(
            frequ_unit
        ), 'Need consistent frequency/time units for `signal1` and `signal2`.'
    else:
        raise TypeError(
            '`signal2` has to be a GWpy ``TimeSeries`` or ``FrequencySeries``.'
        )

    # -- Handling PSD
    if psd is None:
        if not signal_interpolation:
            # -- We know frequencies on which to evaluate
            psd = FrequencySeries(
                np.ones(signal1.frequencies.size),
                frequencies=signal1.frequencies,
                unit=u.strain**2 / frequ_unit,
            )
        else:
            from ..PSDs import psd_no_noise
            psd = psd_no_noise.copy()

            # -- Make sure units are consistent with input. PSD is always a
            # -- density, i.e. some unit per frequency
            if (psd_frequ_unit := psd.frequencies.unit) != frequ_unit:
                psd.frequencies *= frequ_unit / psd_frequ_unit
                psd /= frequ_unit / psd_frequ_unit
                # -- Rescale density that it represents, psd is per frequ_unit

    if isinstance(psd, FrequencySeries):
        assert psd.frequencies.unit._is_equivalent(
            frequ_unit
        ), 'Need consistent frequency/time units for `psd` and other signals.'
    else:
        raise TypeError('`psd` has to be a GWpy ``FrequencySeries`` or None.')

    # -- Handling frequency range, needed for every case of return
    f_lower, f_upper = _determine_x_range(f_range, signal1, signal2, psd)

    if not signal_interpolation:
        # -- Signals are assumed to be on correct frequencies already,
        # -- thus the only things left to do are taking care of
        # -- frequency ranges and returning.
        if not _optimize:
            eval_range = (f_lower, f_upper)
            # eval_range = (f_lower - 0.5 * df, f_upper + 0.5 * df)  # Ensure all signals are non-zero on same range
            # -- Filling is done UP TO THIS frequency, but we want it included
            # TODO: do we need this?

            # -- Returning views of signals is fine (done due to
            # -- copy=False), inner_product_computation does not edit
            # -- the signals in any way.
            signal1 = adjust_x_range(
                signal1,
                x_range=eval_range,
                copy=False,
            )
            signal2 = adjust_x_range(
                signal2,
                x_range=eval_range,
                copy=False,
            )
            psd = adjust_x_range(
                psd,
                x_range=eval_range,
                copy=False,
            )

            return inner_product_computation(signal1, signal2, psd)
        else:
            non_zero_range = (f_lower, f_upper)

            if f_lower >= 0.0 * frequ_unit:
                eval_range = 0.0 * frequ_unit, f_upper
            else:
                f_limit = max(abs(f_lower), abs(f_upper))
                eval_range = -f_limit, f_limit

            signal1 = adjust_x_range(
                signal1,
                x_range=eval_range,
                fill_val=0.0 * signal1.unit,
                fill_range=non_zero_range,
                copy=True,
            )
            signal2 = adjust_x_range(
                signal2,
                x_range=eval_range,
                fill_val=0.0 * signal2.unit,
                fill_range=non_zero_range,
                copy=True,
            )
            psd = adjust_x_range(
                psd,
                x_range=eval_range,
                fill_val=1.0 * psd.unit,
                fill_range=non_zero_range,
                copy=True,
            )

            if optimize_time_and_phase:
                # -- Overwrite
                optimize_time = True
                optimize_phase = True

            return optimized_inner_product(
                signal1=signal1,
                signal2=signal2,
                psd=psd,
                optimize_time=optimize_time,
                optimize_phase=optimize_phase,
                min_dt_prec=min_dt_prec,
                return_opt_info=return_opt_info,
            )

    # -- Frequency range needs to be constructed, we need df for that
    if df is None:
        # df = 0.0625 * frequ_unit
        # -- Choose default value of output of FDWaveform (2**-4*u.Hz)
        # df = _q_convert(0.0625 * u.Hz, frequ_unit, 'df', 'signal.frequencies')
        # if frequ_unit._is_equivalent(u.Hz):
        #     df = _q_convert(0.0625 * u.Hz, frequ_unit, 'df', 'signal.frequencies')
        # else:
        #     df = 0.0625 * frequ_unit
        # if signal1.frequencies.regular and signal2.frequencies.regular:
        #     df = _q_convert(max(signal1.df, signal2.df), frequ_unit, 'df', 'signal.frequencies')
        # else:
        #     df = _q_convert(max(signal1.frequencies.diff(), signal2.frequencies.diff()), frequ_unit, 'df', 'signal.frequencies')
        # TODO: check if .regular is as expensive as taking diff directly -> could also be that it is immediately set True if FrequencySeries is initialized with df
        # df = _q_convert(max(signal1.frequencies.diff().max(), signal2.frequencies.diff().max()), frequ_unit, 'df', 'signal.frequencies')

        try:
            df = _q_convert(max(signal1.df, signal2.df), frequ_unit, 'df', 'signal.frequencies')
        except AttributeError:
            # -- No df attribute, i.e. unequal sampled signal(s). Choosing default value.
            if frequ_unit._is_equivalent(u.Hz):
                df = _q_convert(0.0625 * u.Hz, frequ_unit, 'df', 'signal.frequencies')
            else:
                # -- We have no idea what frequ_unit is, just set to some number
                df = 0.0625 * frequ_unit
    else:
        df = _q_convert(df, frequ_unit, 'df', 'signal.frequencies')

    # -- Get signals to same frequencies, i.e. make df
    # -- equal (if necessary) and then restrict range
    if not _optimize:
        target_range = (
            np.arange(
                f_lower.to_value(frequ_unit),
                f_upper.to_value(frequ_unit) + 0.5 * df.to_value(frequ_unit),
                step=df.to_value(frequ_unit),
            )
            << frequ_unit
        )
        fill_bounds = None

        signal1 = signal_at_xindex(
            signal1,
            target_range,
            fill_val=0.0 * signal1.unit,
            fill_bounds=fill_bounds,
        )
        signal2 = signal_at_xindex(
            signal2,
            target_range,
            fill_val=0.0 * signal2.unit,
            fill_bounds=fill_bounds,
        )
        psd = signal_at_xindex(
            psd,
            target_range,
            fill_val=1.0 * psd.unit,
            fill_bounds=fill_bounds,
        )

        return inner_product_computation(signal1, signal2, psd)
    else:
        # -- Ensure all signals are non-zero on same range and on IFT-compatible range
        if f_lower >= 0.0 * frequ_unit:
            eval_range = 0.0 * frequ_unit, f_upper
        else:
            f_limit = max(abs(f_lower), abs(f_upper))
            eval_range = -f_limit, f_limit

        target_range = (
            np.arange(
                eval_range[0].to_value(frequ_unit),
                eval_range[1].to_value(frequ_unit) + 0.5 * df.to_value(frequ_unit),
                step=df.to_value(frequ_unit),
            )
            << frequ_unit
        )
        non_zero_range = (f_lower, f_upper)
        # non_zero_range = (f_lower - 0.5*df, f_upper + 0.5*df)  # Ensure all signals are non-zero on same range
        # -- Filling is done UP TO THIS frequency, but we want it included
        # TODO: do we need this?

        signal1 = signal_at_xindex(
            signal1,
            target_range,
            fill_val=0.0 * signal1.unit,
            fill_bounds=non_zero_range,
        )
        signal2 = signal_at_xindex(
            signal2,
            target_range,
            fill_val=0.0 * signal2.unit,
            fill_bounds=non_zero_range,
        )
        psd = signal_at_xindex(
            psd,
            target_range,
            fill_val=1.0 * psd.unit,
            fill_bounds=non_zero_range,
        )

        if optimize_time_and_phase:
            # -- Overwrite
            optimize_time = True
            optimize_phase = True

        return optimized_inner_product(
            signal1=signal1,
            signal2=signal2,
            psd=psd,
            optimize_time=optimize_time,
            optimize_phase=optimize_phase,
            min_dt_prec=min_dt_prec,
            return_opt_info=return_opt_info,
        )


# -- The following is needed frequently in other files. Makes most sense
# -- to calculate here already, can then be imported by other files.
_INNER_PROD_ARGS: Final[list[str]] = list(signature(inner_product).parameters)


def inner_product_computation(
    signal1: FrequencySeries, signal2: FrequencySeries, psd: FrequencySeries
) -> u.Quantity:
    """
    Lower level function for inner product calculation. Only accepts
    signals at identical frequency ranges and then carries out the
    actual integral calcutation.

    Parameters
    ----------
    signal1 : ~gwpy.frequencyseries.FrequencySeries
        First signal to put into inner product.
    signal2 : ~gwpy.frequencyseries.FrequencySeries
        Second signal to put into inner product.
    psd : ~gwpy.frequencyseries.FrequencySeries
        Power spectral density to use in inner product.

    Returns
    -------
    ~astropy.units.Quantity
        Inner product value with :code:`signal1`, :code:`signal2`,
        :code:`psd` inserted.

    See Also
    --------
    scipy.integrate.simpson : Used for evaluation of inner product.
    """
    # -- Assure input signals are compatible
    _compare_series_xindex(signal1, signal2, psd, enforce_dx=False)

    output_unit = (
        signal1.unit * signal2.unit / psd.unit * signal1.frequencies.unit
    ).decompose(bases=preferred_unit_system.bases)
    # -- Resets scale only for units, not for value. Best we can do

    # -- To determine factor in front of integral, check if one-sided
    one_sided = (signal1.frequencies[0].value >= 0.0) or (
        signal1.frequencies[-1].value <= 0.0
    )
    return (
        (4.0 if one_sided else 2.0)
        * np.real(simpson(y=signal1 * signal2.conjugate() / psd, x=signal1.frequencies))
        * output_unit
    )


def optimized_inner_product(
    signal1: FrequencySeries,
    signal2: FrequencySeries,
    psd: FrequencySeries,
    optimize_time: bool,
    optimize_phase: bool,
    min_dt_prec: Optional[float] = None,
    return_opt_info: bool = False,
) -> (
    u.Quantity
    | tuple[
        u.Quantity,
        dict[
            Literal['match_series', 'peak_phase', 'peak_time'], u.Quantity | TimeSeries
        ],
    ]
):
    """
    Lower level function for inner product calculation. Only accepts
    signals at identical frequency ranges and then carries out the
    actual integral calcutation via an IFFT.

    In contrast to :code:`inner_product_computation`, this function
    optimizes the inner product over time and phase shifts.

    Parameters
    ----------
    signal1 : ~gwpy.frequencyseries.FrequencySeries
        First signal to put into inner product.
    signal2 : ~gwpy.frequencyseries.FrequencySeries
        Second signal to put into inner product.
    psd : ~gwpy.frequencyseries.FrequencySeries
        Power spectral density to use in inner product.
    optimize_time : boolean
        Whether or not optimization over time shall be carried out.
    optimize_phase : boolean
        Whether or not optimization over phase shall be carried out.
    min_dt_prec : float, optional, default = None
        Maximum time spacing allowed in the time series that is
        calculated via the inverse Fourier transform. Not only does this
        determine the accuracy of the inferred optimal time shift, but
        it also has substantial influence of the inferred phase because
        this quantity can be very sensitive to even little changes in
        the optimal time (of course, the match value is also affected by
        all of this, but typically not as much as the other quantities
        that have just been mentioned).
    return_opt_info : boolean, optional, default = False
        Whether or not to return a dictionary with additional
        information about the optimization results.

    Returns
    -------
    ~astropy.units.Quantity or tuple[~astropy.units.Quantity, dict[Literal['match_series', 'peak_phase', 'peak_time'], ~astropy.units.Quantity | ~gwpy.timeseries.TimeSeries]]
        Output depends on the value of :code:`return_opt_info`. If False
        (the default), only the optimized value of the inner product is
        returned. If True, a tuple of this value and a dictionary is
        returned. This dictionary contains:
        (i) a ``TimeSeries`` where values of the inner product for
        different relative time shifts between :code:`signal1`,
        :code:`signal2` are stored. Optimization over phase is not
        carried out in this return, but it can easily be done by taking
        the :code:`.abs()` of the ``TimeSeries``. To get inner product
        values that are not optimized over phase, take :code:`.real` of
        the ``TimeSeries``.
        (ii) time at which maximum value occurs in (i) if
        :code:`optimize_time=True` or
        :code:`optimize_time_and_phase=True`. Otherwise it is set to
        zero, which corresponds to no optimization over time shift. This
        number represents a shift :math:`t_0` from :code:`signal1` to
        :code:`signal2`, i.e. :code:`signal1` is "ahead in time"
        in the sense that :math:`signal1(t) = signal2(t+t0)`.
        (iii) phase shift needed to get maximum of (i) at time (ii) if
        :code:`optimize_phase=True` or
        :code:`optimize_time_and_phase=True`, i.e. it is the phase that
        the complex match series returned in (i) has at the time
        returned in (ii). Otherwise it is set to zero, which corresponds
        to no optimization over phase (the real part of (i) is taken).

        In other words, one can obtain the same inner product value by
        calculating the non-optimized inner product between
        :code:`signal1` and :code:`signal2*np.exp(-2.j*np.pi*signal2.
        frequencies*time_shift + 1.j*phase_shift)`. Here,
        :code:`time_shift` is the value returned in (ii), i.e. with key
        :code:`'peak_time'`, and :code:`phase_shift` the value returned
        in (iii), i.e. with key :code:`'peak_phase'`.
    """
    # -- First step: ensuring input signals are consistent
    frequ_unit = signal1.frequencies.unit

    _compare_series_xindex(signal1, signal2, psd)

    # -- Second step: make sure all signals start at valid frequencies
    _assert_ft_compatible(signal1, signal2, psd)

    # -- Third step: computations
    dft_vals = signal1 * signal2.conjugate() / psd

    if min_dt_prec is not None:
        min_dt_prec = _q_convert(
            min_dt_prec,
            1.0 / frequ_unit,
            'min_dt_prec',
            'signal',
            err_msg='Need consistent (i.e. convertible) units for `%s` (%s)'
            ' and inverse frequency unit of `%s` (%s).',
        )

    # -- Append zeros or bring into correct format so that ifft can be
    # -- used. The prefactor is added here already because it depends on
    # -- the given frequency range
    if np.isclose(0.0, dft_vals.f0.value, atol=0.5 * dft_vals.df.value, rtol=0.0):
        n_append = dft_vals.size - 1
        n_total = dft_vals.size + n_append
        current_dt = 1.0 / (n_total * signal1.df)

        if min_dt_prec is None:
            min_dt_prec = current_dt

        if current_dt > min_dt_prec:
            n_required = np.ceil(1.0 / (min_dt_prec * signal1.df))
        else:
            n_required = n_total

        n_required = next_power_of_two(n_required)

        n_append = n_required - dft_vals.size

        full_dft_vals = 4.0 * np.append(dft_vals.value, np.zeros(n_append))
        # -- Note: ifft function expects positive frequency values first
        # -- and then negative. Since we add at least as many zeros as
        # -- len(dft_vals), we adhere to this and things are consistent
    else:
        n_append = 0
        n_total = dft_vals.size

        current_dt = 1.0 / (n_total * signal1.df)

        if min_dt_prec is None:
            min_dt_prec = current_dt

        if current_dt > min_dt_prec:
            n_required = np.ceil(1.0 / (min_dt_prec * signal1.df))
        else:
            n_required = n_total

        n_required = next_power_of_two(n_required)

        n_append = n_required - dft_vals.size

        if n_append % 2 == 0:  # Equivalent to dft_vals.size % 2 == 0
            # -- Symmetric appending
            n_append_lower = n_append_upper = n_append // 2
        else:
            # -- Via checks at beginning of function, we know that less
            # -- values at positive frequencies
            n_append_lower = n_append // 2 + 1
            n_append_upper = n_append // 2

        n_split = dft_vals.size // 2
        # -- Should actually work for all cases. For odd, rounds down
        # -- and thus takes only until f=0. For even, the positive ones
        # -- are expected to be one less than in number than negative

        full_dft_vals = 2.0 * np.concatenate(
            (
                dft_vals.value[n_split:],
                np.zeros(n_append_upper),
                np.zeros(n_append_lower),
                dft_vals.value[:n_split],
            )
        )

    assert (
        next_power_of_two(full_dft_vals.size) == full_dft_vals.size
    ), 'Consistency check, not your fault if it fails.'

    dt = (1.0 / (full_dft_vals.size * signal1.df)).decompose(
        bases=preferred_unit_system.bases
    )

    output_unit = (
        signal1.unit * signal2.unit / psd.unit * signal1.frequencies.unit
    ).decompose(bases=preferred_unit_system.bases)
    # -- Resets scale only for units, not for value. Best we can do

    match_series = TimeSeries(
        np.fft.ifft(full_dft_vals / dt.value),  # Discrete -> continuous
        unit=output_unit,
        t0=0.0 * dt.unit,
        dt=dt,
    )

    # -- Handle wrap-around of signal
    number_to_roll = match_series.size // 2  # Arbitrary value, no deep meaning
    # TODO: can we do better with rolling? The "starting time" of IFT signal
    # is usually chosen to be the epoch. So couldn't we use that epoch of
    # h1*conj(h2) should be h1.epoch-h2.epoch? Would then also have to adjust
    # starting time when setting match series, be consistent there
    # -> or maybe it does not play role because we can argue via periodicity
    #    in signal length? Starting time zero should remain
    # -> also, we only care about relative (!) time shift that we have
    #    to introduce, not the actual difference in starting time

    match_series = np.roll(match_series, shift=number_to_roll)
    match_series.shift(-match_series.times[number_to_roll])

    if optimize_phase:
        _match_series = match_series.abs()
    else:
        _match_series = match_series.real

    if optimize_time:
        peak_index = np.argmax(_match_series)
        match_result = _match_series[peak_index]
        peak_time = _match_series.times[peak_index]
    else:
        # -- Evaluation at t0=0 corresponds to usual inner product
        peak_time = 0.0 * match_series.times.unit
        peak_index = np.searchsorted(
            match_series.xindex.value, peak_time.value, side="left"
        )
        match_result = _match_series[peak_index]

    if return_opt_info:
        peak_phase = (
            np.angle(match_series)[peak_index] if optimize_phase else 0.0 * u.rad
        )

        return match_result, {
            'match_series': match_series,
            'peak_phase': peak_phase,
            'peak_time': peak_time,
        }
    else:
        return match_result


def next_power_of_two(x):
    """Calculate next power of two of the input."""
    return 1 if x == 0 else int(2 ** np.ceil(np.log2(x)))


def norm(signal: TimeSeries | FrequencySeries, *args, **kwargs) -> (
    u.Quantity
    | tuple[
        u.Quantity,
        dict[
            Literal['match_series', 'peak_phase', 'peak_time'], u.Quantity | TimeSeries
        ],
    ]
):
    """
    Wrapper function for calculation of the norm of the given
    :code:`signal` (i.e. square root of inner product between
    :code:`signal` and :code:`signal`, its SNR) as measured by the
    noise-weighted inner product implemented in :code:`inner_product`.

    Parameters
    ----------
    signal : ~gwpy.timeseries.TimeSeries or ~gwpy.frequencyseries.FrequencySeries
        Signal to compute norm for.
    *args, **kwargs
        Additional arguments, passed to :code:`inner_product` function.

    Returns
    -------
    ~astropy.units.Quantity or tuple[~gwpy.timeseries.TimeSeries, ~astropy.units.Quantity, ~astropy.units.Quantity]
        Norm of :code:`signal`.

        If :code:`optimize_time_and_phase = True`, a tuple is returned
        that contains a ``TimeSeries``, the aforementioned norm and a
        time. See :code:`optimized_inner_product` for details on the
        return.

    See Also
    --------
    gw_signal_tools.inner_product.inner_product :
        Arguments are passed to this function for calculations.
    """
    out = inner_product(signal, signal, *args, **kwargs)

    if isinstance(out, u.Quantity):
        return np.sqrt(out)
    else:
        out[1]['match_series'] = np.sqrt(out[1]['match_series'])
        return np.sqrt(out[0]), out[1]


def overlap(
    signal1: TimeSeries | FrequencySeries,
    signal2: TimeSeries | FrequencySeries,
    *args,
    **kwargs,
) -> (
    u.Quantity
    | tuple[
        u.Quantity,
        dict[
            Literal['match_series', 'peak_phase', 'peak_time'], u.Quantity | TimeSeries
        ],
    ]
):
    """
    Wrapper for calculation of the overlap of two given signals as
    measured by the noise-weighted inner product implemented in
    :code:`inner_product`. This means the signals are normalized to have
    unit norm (with respect to this inner product) and then inserted
    into :code:`inner_product`.

    Parameters
    ----------
    signal1 : ~gwpy.timeseries.TimeSeries or ~gwpy.frequencyseries.FrequencySeries
        First signal.
    signal2 : ~gwpy.timeseries.TimeSeries or ~gwpy.frequencyseries.FrequencySeries
        Second signal.
    *args, **kwargs
        Additional arguments, passed to :code:`inner_product` function.

    Returns
    -------
    ~astropy.units.Quantity or tuple[~gwpy.timeseries.TimeSeries, ~astropy.units.Quantity, ~astropy.units.Quantity]
        Overlap of :code:`signal1` and :code:`signal2`.

        If :code:`optimize_time_and_phase = True`, a tuple is returned
        that contains a ``TimeSeries``, the aforementioned norm and a
        time. See :code:`optimized_inner_product` for details on the
        return.

    See Also
    --------
    gw_signal_tools.inner_product.inner_product :
        Arguments are passed to this function for calculations.
    """
    out = inner_product(signal1, signal2, *args, **kwargs)

    normalization = 1.0  # Default value
    if isinstance(norm1 := norm(signal1, *args, **kwargs), u.Quantity):
        normalization *= norm1
    else:
        normalization *= norm1[0]
    if isinstance(norm2 := norm(signal2, *args, **kwargs), u.Quantity):
        normalization *= norm2
    else:
        normalization *= norm2[0]

    if isinstance(out, u.Quantity):
        return out / normalization
    else:
        out[1]['match_series'] /= normalization
        return out[0] / normalization, out[1]


# -- Optimization of Inner Product over Arbitrary Parameters ------------------
# TODO: put this into waveform_utils?
def test_hm(wf_params: dict[str, u.Quantity], wf_generator: FDWFGen) -> bool:
    """
    Perform test whether or not higher modes are relevant for the chosen
    point in parameter space and waveform model. This is done by
    comparing the similarity of two waveforms, one with a certain value
    for the reference phase 'phi_ref' and the other one with 'phi_ref'
    equal to zero, but phase shifted by two times this value. If higher
    modes are relevant, these waveforms will have an overlap not equal
    to 1, which is the test that is performed.

    Parameters
    ----------
    wf_params : dict[str, ~astropy.units.Quantity]
        Point in parameter space that waveform is generated at.
    wf_generator : ~gw_signal_tools.types.FDWFGen
        Routine to generate waveforms.

    Returns
    -------
    boolean
        Whether higher modes have a significant impact for the selected
        configuration.
    """
    phi_val = 0.76 * u.rad  # Some arbitrary shift
    wf_phizero_shifted = wf_generator(wf_params | {'phi_ref': 0.0 * u.rad}) * np.exp(
        1.0j * 2 * phi_val
    )
    wf_phinonzero = wf_generator(wf_params | {'phi_ref': phi_val})

    if overlap(wf_phizero_shifted, wf_phinonzero) > 0.999:
        # -- Arbitrary value, but should be sufficient to assess
        # -- influence of HMs
        return False
    else:
        return True


def test_precessing(wf_params: dict[str, u.Quantity]) -> bool:
    """
    Perform test whether or not the given binary system is precessing.
    This is done by looking at the x-, y-values of the component spins
    of each binary and if one of them is non-zero, the system is taken
    to be precessing. While this might not cover all cases correctly, it
    is sufficient to serve the main purpose of this function, namely its
    use in the optimization of the overlap between waveforms, where the
    return is used to select the default parameters to optimize over.

    Note that this function is compatible with all input accepted by
    the :code:`lalsimulation.gwsignal` module. That means (i) there is
    no need to actually specify the spin components (default values of
    zero for all are assumed) and (ii) components need not be specified
    in cartesian coordinates, can also be spherical.

    Parameters
    ----------
    wf_params : dict[str, ~astropy.units.Quantity]
        Point in parameter space that waveform is generated at.

    Returns
    -------
    boolean
        Whether the system is precessing.
    """
    # TODO: maybe test for valid spin config?
    for i in [1, 2]:
        # -- Check for cartesian components first
        for index in ['x', 'y']:
            try:
                # if wf_params[f'spin{i}{index}'] != 0.*u.dimensionless_unscaled:
                if (
                    wf_params[f'spin{i}{index}'] != 0.0 * u.dimensionless_unscaled
                    and wf_params[f'spin{i}z'] != 0.0 * u.dimensionless_unscaled
                ):
                    # -- Spins are not parallel to L and not in orbital plane
                    return True
            except KeyError:
                pass

        # TODO: spins might not be parallel to L, but can still cancel
        # and in that case, no precession!!!
        # -> but that also depends on mass, very specific... Just neglect?

        # -- No precession in cartesian components, but spherical ones
        # -- might be given
        try:
            if wf_params[f'spin{i}_tilt'] % (np.pi * u.rad) != 0.0 * u.rad:
                return True
        except KeyError:
            pass

    return False


# -- Preparing optimization function. Some parameters have physical
# -- bounds that may not be crossed, otherwise waveform error.
param_bounds: dict[str, tuple[float, float]] = {
    'total_mass': (0.0, np.inf),
    'mass1': (0.0, np.inf),
    'mass2': (0.0, np.inf),
    'mass_ratio': (0.0, 1.0),
    'inverse_mass_ratio': (1.0, np.inf),
    'sym_mass_ratio': (0.0, 0.25),
    'distance': (0.0, np.inf),
    'spin1x': (0.0, 1.0),
    'spin1y': (0.0, 1.0),
    'spin1z': (0.0, 1.0),
    'spin2x': (0.0, 1.0),
    'spin2y': (0.0, 1.0),
    'spin2z': (0.0, 1.0),
    'f_ref': (0.0, np.inf),
}
# -- Note: we do not add angles here because the waveform will not fail
# -- to generate if we cross some boundary (periodicity).


def get_default_opt_params(
    wf_params: dict[str, u.Quantity], wf_generator: FDWFGen
) -> list[str]:
    """
    Determine external parameters to optimize over for the given
    waveform configuration and generator. The "base set" consists of
    :code:`'time'` and :code:`'phase'`, but more might be added
    depending on the outputs of :code:`test_precessing` and
    :code:`test_hm`.

    Parameters
    ----------
    wf_params : dict[str, ~astropy.units.Quantity]
        Point in parameter space that waveform is generated at.
    wf_generator : ~gw_signal_tools.types.FDWFGen
        Routine to generate waveforms.

    Returns
    -------
    list[str]
        List of waveform parameters to optimize over.
    """
    default_opt_params = ['time', 'phase']

    if test_precessing(wf_params):
        default_opt_params += ['phi_ref', 'phi_jl']
        # TODO: find good definition of phi_jl. Bilby seems to use it, but in
        # "wrong order": https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/source.py#L649
    elif test_hm(wf_params, wf_generator):
        # -- Higher modes still be relevant for non-precessing
        default_opt_params += ['phi_ref']

    return default_opt_params


def optimize_overlap(
    wf_params: dict[str, u.Quantity],
    fixed_wf_generator: FDWFGen,
    vary_wf_generator: FDWFGen,
    opt_params: Optional[str | list[str]] = None,
    **inner_prod_kwargs,
) -> tuple[FrequencySeries, FrequencySeries, dict[str, u.Quantity]]:
    r"""
    Maximize the overlap between two waveforms at the given point in the
    parameter space. For the majority of parameters, this is done by
    varying certain parameters for one of the waveform generators while
    keeping them fixed for the other waveform generator and then
    minimizing the mismatch (:math:`1-overlap`).

    Two exceptions to this rule exist: relative time and phase shifts
    (corresponding parameters are mentioned in description of the
    :code:`opt_params` argument). These are optimized using the inner
    product itself, which allows to infer the required shifts for these
    parameters simply by evaluating it.

    Parameters
    ----------
    wf_params : dict[str, ~astropy.units.Quantity]
        Set of parameters, some of which will be fixed while others are
        varied.
    fixed_wf_generator : ~gw_signal_tools.types.FDWFGen
        Waveform generator for the first waveform that is fixed, i.e. it
        is only generated once.
    vary_wf_generator : ~gw_signal_tools.types.FDWFGen
        Waveform generator for which the parameters will be varied, i.e.
        it will be called many times.
    opt_params : Optional[str | list[str]], optional, default = None
        Set of parameters to optimize the overlap over. If None, a set
        of external parameters is automatically determined by calling
        :code:`get_default_opt_params`. However, internal parameters can
        also be part of the list.

        Must be in :code:`wf_params` or :code:`'time'`, :code:`phase`.
        The latter two are global time and phase shifts, they enter the
        waveform only via a factor
        :math:`\exp(i \cdot phase - i \cdot 2 \pi \cdot f \cdot time)`.
        Beware that the polarization angle :code:`psi` might be
        degenerate with :code:`phase`, if you are using the complex
        strain combination :math:`h = h_+ \pm i \, h_{\times}`.

        Note that it is also possible to optimize over phase and/or time
        by enabling optimization over phase and time in the inner
        product function (by passing keyword arguments like
        :code:`optimize_time_and_phase=True`, will be given to inner
        product; in fact, this is how things are handled anyway for
        these two parameters). The corresponding result will also be
        part of the output, despite potentially not being in
        :code:`opt_params`.
    inner_prod_kw_args :
        All additional keyword arguments are passed to the
        :code:`~gw_signal_tools.inner_product.overlap` function. Can be
        used as an alternative way to enable optimization over the
        parameters :code:`'time'` or :code:`'phase'` by passing
        :code:`optimize_time=True` or :code:`optimize_phase=True`,
        respectively (passing :code:`optimize_time_and_phase=True`
        enables optimization for both).

    Returns
    -------
    tuple[~gwpy.frequencyseries.FrequencySeries, ~gwpy.frequencyseries. FrequencySeries, dict[str, ~astropy.units.Quantity]]
        A three-tuple consisting of (i) the waveform generated by
        :code:`fixed_wf_generator` at :code:`wf_params`, (ii) the
        optimized waveform generated by :code:`vary_wf_generator` and
        (iii) a dictionary where the optimal values for all elements of
        :code:`opt_params` are stored.

    Notes
    -----
    When conducting tests using this function, it turned out to be
    beneficial to use it with :code:`optimize_time_and_phase=True` even
    if no optimization over time and phase was desired. If no shifts in
    those parameters were present, the function recovers this, but the
    behaviour seems to be much more benefitial for the minimization
    routine because the convergence worked for many cases where "raw"
    optimizing over certain parameters (e.g. 'mass1') did not work.
    """
    wf1 = fixed_wf_generator(wf_params)

    if opt_params is None:
        _opt_params = get_default_opt_params(wf_params, vary_wf_generator)

        return optimize_overlap(
            wf_params=wf_params,
            fixed_wf_generator=fixed_wf_generator,
            vary_wf_generator=vary_wf_generator,
            opt_params=_opt_params,
            **inner_prod_kwargs,
        )
    elif isinstance(opt_params, str):
        _opt_params = [opt_params]

        return optimize_overlap(
            wf_params=wf_params,
            fixed_wf_generator=fixed_wf_generator,
            vary_wf_generator=vary_wf_generator,
            opt_params=_opt_params,
            **inner_prod_kwargs,
        )
    else:
        _opt_params = np.array(opt_params)

        # -- time and phase indices are accessed frequently, thus store
        if 'time' in _opt_params:
            time_index = np.argwhere(_opt_params == 'time')[0, 0]
            inner_prod_kwargs['optimize_time'] = True
        else:
            time_index = None

        if 'phase' in _opt_params:
            phase_index = np.argwhere(_opt_params == 'phase')[0, 0]
            inner_prod_kwargs['optimize_phase'] = True
        else:
            phase_index = None

        # -- If only time and/or phase shall be optimized, we can take shortcut
        if (
            len(_opt_params) == 2 and time_index is not None and phase_index is not None
        ) or (
            len(_opt_params) == 1
            and (time_index is not None or phase_index is not None)
        ):
            inner_prod_kwargs['return_opt_info'] = True
            wf2 = vary_wf_generator(wf_params)

            _match_val, opt_info = overlap(wf1, wf2, **inner_prod_kwargs)

            logger.info(
                'Optimization was conducted successfully. Remaining '
                f'waveform mismatch is {1.-_match_val:.5f}.'
            )

            opt_params_results = {}
            time = opt_info['peak_time']
            phase = opt_info['peak_phase']

            if time_index is not None:
                opt_params_results['time'] = time + wf_params.get('time', 0.0 * u.s)

            if phase_index is not None:
                opt_params_results['phase'] = phase + wf_params.get(
                    'phase', 0.0 * u.rad
                )
            # -- Important distinction: results contain the "absolute"
            # -- time and phase shifts, that have to be put into the
            # -- waveform generator. For the shift we apply now, part
            # -- of this absolute shift is already applied and we only
            # -- want to add the relative shift on top!

            return wf1, apply_time_phase_shift(wf2, time, phase), opt_params_results
            # -- Note: redefining wf2 with the phase factor using *=
            # -- is a very bad idea. That is because the result of the
            # -- call is potentially already cached, and in that case
            # -- the cached result would be overwritten (bad)

        # -- Check if optimization in inner product is carried out
        # -- (can be given as equivalent input to )
        if (
            inner_prod_kwargs.get('optimize_time_and_phase', False)
            or inner_prod_kwargs.get('optimize_time', False)
            or inner_prod_kwargs.get('optimize_phase', False)
        ):
            _inner_prod_is_optimized = True

            _opt_params = opt_params.copy()  # Otherwise removing bad

            if time_index is not None:
                _opt_params.remove(opt_params[time_index])

            if phase_index is not None:
                _opt_params.remove(opt_params[phase_index])

            _opt_params = np.array(_opt_params)
        else:
            _inner_prod_is_optimized = False

    # -- We only get to this point if opt_params is list of parameters
    def wf2_shifted(args):
        wf_args = wf_params | {
            param: args[i] * wf_params[param].unit
            for i, param in enumerate(_opt_params)
        }
        return vary_wf_generator(wf_args)

    inner_prod_kwargs['return_opt_info'] = False

    # -- This setting ensures overlap returns a number, which allows for
    # -- a more convenient loss definition
    def loss(args):
        _match = overlap(wf1, wf2_shifted(args), **inner_prod_kwargs)
        return 1.0 - _match

    init_guess = [wf_params[param].value for param in _opt_params]

    bounds = len(_opt_params) * [(None, None)]
    for i in range(len(_opt_params)):
        bounds[i] = param_bounds.get(_opt_params[i], (None, None))
        if _opt_params[i] == 'mass_ratio' and wf_params['mass_ratio'] > 1:
            bounds[i] = param_bounds.get('inverse_mass_ratio', (None, None))

    result = minimize(fun=loss, x0=init_guess, bounds=bounds, method='Nelder-Mead')
    # -- If possible, will find parameters so that mismatch <= 1e-5

    opt_params_results = {param: result.x[i] for i, param in enumerate(_opt_params)}
    for param, param_val in wf_params.items():
        if param in opt_params_results:
            opt_params_results[param] *= param_val.unit

    logger.info(result.message + f' Remaining waveform mismatch is {result.fun:e}.')

    wf2 = wf2_shifted(result.x)

    if _inner_prod_is_optimized:
        # -- For return of optimized waveform, time and phase have to be
        # -- evaluated in values of optimized inner product result
        _, opt_info = inner_product(
            wf1, wf2, **(inner_prod_kwargs | {'return_opt_info': True})
        )
        time = opt_info['peak_time']
        phase = opt_info['peak_phase']

        if time_index is not None:
            opt_params_results['time'] = time + wf_params.get('time', 0.0 * u.s)

        if phase_index is not None:
            opt_params_results['phase'] = phase + wf_params.get('phase', 0.0 * u.rad)
        # -- Important distinction: results contain the "absolute"
        # -- time and phase shifts, that have to be put into the
        # -- waveform generator. For the shift we apply now, part
        # -- of this absolute shift is already applied and we only
        # -- want to add the relative shift on top!

        wf2 = apply_time_phase_shift(wf2, time, phase)
        # -- Using *= here would be bad because the result of the call
        # -- is potentially already cached, and in that case the cached
        # -- result would be overwritten (bad)

    return wf1, wf2, opt_params_results
