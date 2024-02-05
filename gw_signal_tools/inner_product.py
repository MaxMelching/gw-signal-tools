from typing import Optional
import logging

import numpy as np
from scipy.integrate import simpson

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import astropy.units as u

from .waveform_utils import (
    td_to_fd_waveform, fd_to_td_waveform,
    pad_to_get_target_df, restrict_f_range
)


__doc__ = """
Implementation of noise-weighted inner product that is
common in GW data analysis and helpers for computation.
"""



# ---------- PSD Handling ----------

def psd_from_file(
    fname: str,
    is_asd: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read Power spectral density (PSD) values from a file into numpy
    arrays. The file must be readable by `~numpy.loadtxt`.

    Parameters
    ----------
    fname : str
        File with two columns, left one representing frequencies and
        right one representing the corresponding PSD values.
    is_asd : bool, optional, default = False
        If true, values in file are taken to be ASD values rather than
        PSD values and thus a squared version of them is returned.

    Returns
    -------
    freqs, psd : tuple[~numpy.array]
        Frequencies and PSD values as numpy arrays.

    See also
    --------
    numpy.loadtxt : Used to read the file.
    """

    file_vals = np.loadtxt(fname)
    freqs, psd = file_vals[:, 0], file_vals[:, 1]

    if is_asd:
        psd = psd**2

    return freqs, psd


def psd_from_file_to_FreqSeries(
    fname: str,
    is_asd: bool = False,
    **kwargs
) -> FrequencySeries:
    """
    Read Power spectral density (PSD) values from file into a GWpy
    ``FrequencySeries``.

    Parameters
    ----------
    fname : str
        File with two columns, left one representing frequencies and
        right one representing the corresponding PSD values.
    is_asd : bool, optional, default = False
        If true, values in file are taken to be ASD values rather than
        PSD values and thus a squared version of them is returned.
    **kwargs
        Other keyword arguments that are passed to ``FrequencySeries``
        constructor. Can be used to assign name to series and more.

    Returns
    -------
    ~gwpy.frequencyseries.FrequencySeries
        PSD as a ``FrequencySeries``.
    """

    file_vals = np.loadtxt(fname)
    freqs, psd = file_vals[:, 0], file_vals[:, 1]

    if is_asd:
        psd = psd**2

    freqs, psd = psd_from_file(fname, is_asd=is_asd)

    return FrequencySeries(
        psd,
        frequencies=freqs,
        unit=1 / u.Hz,
        **kwargs
    )


def get_FreqSeries_from_dict(
    psd: dict,
    psd_vals_key: str,
    is_asd: bool = False,
    **kwargs
) -> FrequencySeries:
    """
    Converts dictionary with Power spectral density (PSD) values into a
    GWpy ``FrequencySeries``. Frequencies are expected to be accessible
    using the key 'frequencies'.

    Parameters
    ----------
    psd : dict
        Dictionary with PSD values and corresponding frequencies.
    psd_vals_key : str
        Key that holds PSD values.
    is_asd : bool, optional, default = False
        If true, values in file are taken to be ASD values rather than
        PSD values and thus a squared version of them is returned.
    **kwargs
        Other keyword arguments that are passed to ``FrequencySeries``
        constructor. Can be used to assign name to series and more.

    Returns
    -------
    ~gwpy.frequencyseries.FrequencySeries
        Data from input dict in a ``FrequencySeries``.
    """

    return FrequencySeries(
        psd[psd_vals_key]**2 if is_asd else psd[psd_vals_key],
        frequencies=psd['frequencies'],
        **kwargs
    )

# TODO (potentially): move to psd folder, e.g. given into __init__ file?



# ---------- Inner Product Implementation ----------

def inner_product(
    signal1: TimeSeries | FrequencySeries,
    signal2: TimeSeries | FrequencySeries,
    psd: Optional[FrequencySeries] = None,
    f_range: Optional[list[float] | list[u.Quantity]] = None,
    df: Optional[float | u.Quantity] = None,
    optimize_time_and_phase: bool = False  # Call it 'compute_match'?
) -> float | tuple[TimeSeries, float, float]:
    """
    Calculates the noise-weighted inner product of two signals.

    Parameters
    ----------
    signal1 : ~gwpy.timeseries.TimeSeries or
    ~gwpy.frequencyseries.FrequencySeries
        First signal
    signal2 : ~gwpy.timeseries.TimeSeries or
    ~gwpy.frequencyseries.FrequencySeries
        Second signal
    psd : ~gwpy.frequencyseries.FrequencySeries, optional,
    default = None
        Power spectral density to use in inner product. If None, it is taken
        to be 1 1/Hz at all frequencies.
    f_range : list[float] or list[~astropy.units.Quantity], optional,
    default = None
        Frequency range to compute inner product over. Is potentially
        cropped if bounds are greater than frequency range of one of the
        input signals.

        The whole argument can be None, otherwise it must have length 2.
        However, one of the bounds (or both) can be None, which
        indicates that no boundary shall be set.
    df : float or ~astropy.units.Quantity, optional, default = None
        Distance df between samples in frequency domain to use in
        integration.
        If None, it is set to 0.0625 Hz (or whatever frequency unit is
        used in the signals), which is the default df of frequency
        domain waveforms generated by `~lalsimulation.gwsignal`.
    optimize_time_and_phase : bool, optional, default = False
        Determines if a match is computed or just a "regular" inner
        product. The match will be optimized over relative time and
        phase shifts between `signal1` and `signal2`.

    Returns
    -------
    float or tuple[~gwpy.timeseries.TimeSeries, float,
    ~astropy.units.Quantity]
        Inner product value with `signal1`, `signal2` inserted.

    Raises
    ------
    TypeError
        In case either one of `signal1`, `signal2`, `psd` has wrong type.
    ValueError
        In case format of `f_range` parameter is not correct.

    See also
    --------
    gw_signal_tools.waveform_utils.td_to_fd_waveform : 
        Used to convert ``TimeSeries`` input to a ``FrequencySeries``.
    gwpy.frequencyseries.frequencyseries.interpolate :
        Function used to get signals to same sampling rate.
    
    Notes
    -----
    Some tips regarding the `df` parameter:
    (i) It should not be too large, otherwise results of the inner
    product may be erroneous.
    (ii) If chosen very small (e.g. in range of 0.001Hz), you should
    consider selecting only powers of two like 2**-10 Hz because these
    work best with certain function calls that utilize the Fourier
    transform of the involved signals. An indicator this might be
    necessary is a shape mismatch error.
    """
    # ----- Handling of units -----
    if isinstance(signal1, FrequencySeries):
        frequ_unit = signal1.frequencies.unit
        signal_unit = signal1.unit
    elif isinstance(signal1, TimeSeries):
        frequ_unit = signal1.times.unit * u.Hz / u.s
        signal_unit = signal1.unit * u.s
    else:
        raise TypeError('`signal1` has to be a GWpy TimeSeries or FrequencySeries')

    if isinstance(signal2, FrequencySeries):
        assert frequ_unit == signal2.frequencies.unit,\
        'Need consistent frequency/time units for `signal1` and `signal2`.'
        
        assert signal_unit == signal2.unit,\
        'Need consistent units of `signal1` and `signal2`.'
    elif isinstance(signal2, TimeSeries):
        assert frequ_unit == signal2.times.unit * u.Hz / u.s,\
        'Need consistent frequency/time units times for `signal1` and `signal2`.'
        
        assert signal_unit == signal2.unit * u.s,\
        'Need consistent signal units for `signal1` and `signal2`.'
    else:
        raise TypeError('`signal2` has to be a GWpy TimeSeries or FrequencySeries')
    

    if psd is None:
        from .PSDs import psd_no_noise

        psd = psd_no_noise

        # TODO: create psd_no_noise_geom and use this one in case frequ_unit == u.dimensionless_unscaled
        # -> update on that: no good idea, in this case one should be given.
        # Because Hz remains Hz in geometrical units, no difference.
        # Dimensionless would be rescaled with masses, but we should not
        # account for this
    
    if isinstance(psd, FrequencySeries):
        assert frequ_unit == psd.frequencies.unit,\
        'Need consistent frequency/time units for `psd` and `signal1`, `signal2`.'
        
        # assert signal_unit / u.Hz == psd.unit,\
        # ('Need consistent signal units for psd and `signal1`, `signal2`,'
        #  'i.e. psd unit has to be signal unit per Hz.')

        # assert signal_unit**2 / psd.unit * frequ_unit == u.dimensionless_unscaled,\
        # ('Need consistent signal units for psd and `signal1`, `signal2`, '
        #  'i.e. the product `signal1`.unit * `signal2`.unit / `psd`.unit * '
        #  '`df`.unit has to be dimensionless (note that `df`.unit is set to '
        #  'the match of the one of signal1.frequencies etc. if not given).')
        # TODO: rethink this. Partial derivatives, for example, are not dimensionless
    else:
        raise TypeError('`psd` has to be a GWpy FrequencySeries or None.')

    # Copying does not seem to be necessary. So we avoid the operations



    # ----- Handling of df argument -----
    if df is None:
        df = 0.0625 * frequ_unit  # Default value of output of FDWaveform
    else:
        try:
            df = u.Quantity(df, unit=frequ_unit)
        except u.UnitConversionError:
        # Conversion only fails if df is already Quantity and has
        # non-matching unit, so we can assume that df.unit works
            raise ValueError(
                f'Need consistent frequency units for `df` ({df.unit}) and'
                f' signals ({frequ_unit}).'
            )


    # If necessary, do fft. We apply padding to ensure sufficient
    # resolution in frequency domain.
    if isinstance(signal1, TimeSeries):
        signal1 = pad_to_get_target_df(signal1, df)
        # TODO: do if part from padding function here? Would avoid function call
        signal1 = td_to_fd_waveform(signal1)
    # elif type(signal1) != FrequencySeries:
    #     raise TypeError('`signal1` has to be a GWpy TimeSeries or FrequencySeries')
    # Should be obsolete now, we check this in units handling

    if isinstance(signal2, TimeSeries):
        signal2 = pad_to_get_target_df(signal2, df)
        signal2 = td_to_fd_waveform(signal2)
    # elif type(signal2) != FrequencySeries:
    #     raise TypeError('`signal2` has to be a GWpy TimeSeries or FrequencySeries')
        
    logging.debug(f'{signal_unit}, {signal1.unit}, {signal2.unit}, {psd.unit}')
    

    # ----- Set default values -----
    f_lower, f_upper =[
        max([signal1.frequencies[0], signal2.frequencies[0], psd.frequencies[0]]),
        min([signal1.frequencies[-1], signal2.frequencies[-1], psd.frequencies[-1]])
    ]

    # ----- If bounds are given, check that they fit the input data -----
    if f_range is not None:
        if len(f_range) != 2:
            raise ValueError(
                '`f_range` must contain lower and upper frequency bounds for'
                'integration. One of them or both can be `None`, but both'
                'have to be specified if `f_range` is given.'
            )
        
        # Check if both lower and upper are given or one of them is None
        if f_range[0] is not None:
            try:
                f_lower_new = u.Quantity(f_range[0], unit=frequ_unit)
            except u.UnitConversionError:
        # Conversion only fails if df is already Quantity and has
        # non-matching unit, so we can assume that df.unit works
                raise ValueError(
                    'Need consistent frequency units for `f_range` members'
                    f' ({f_range[0].unit}) and signals ({frequ_unit}).'
                )
            # TODO: change unit to signal1.unit? Because sometimes, we may want to use geometric units -> perhaps along with check that all input signals have same frequency unit
        else:
            f_lower_new = f_lower
        
        if f_range[1] is not None:
            try:
                f_upper_new = u.Quantity(f_range[1], unit=frequ_unit)
            except u.UnitConversionError:
        # Conversion only fails if df is already Quantity and has
        # non-matching unit, so we can assume that df.unit works
                raise ValueError(
                    'Need consistent frequency units for `f_range` members'
                    f' ({f_range[1].unit}) and signals ({frequ_unit}).'
                )

            # TODO: implement check of f_max with Nyquist of signals
        else:
            f_upper_new = f_upper


        # New lower bound must be greater than current lower bound,
        # otherwise no values for the range are available in signals
        if f_lower_new >= f_lower:
            f_lower = f_lower_new
        else:
            # Leave lower bound at f_lower, no update
            logging.info(
                f'Given lower bound of {f_lower_new} is smaller than '
                'values available from given signals. Taking a lower '
                f'bound of {f_lower} instead.'
            )

        # New upper bound must be smaller than current upper bound,
        # otherwise no values for the range are available in signals
        if f_upper_new <= f_upper:
            f_upper = f_upper_new
        else:
            # Leave upper bound at f_upper, no update
            logging.info(
                f'Given upper bound of {f_upper_new} is larger than '
                'values available from given signals. Taking an upper '
                f'bound of {f_upper} instead.'
            )


    # ----- Get signals to same frequencies, i.e. make df equal -----
    # ----- (if necessary) and then restrict range -----
    df_val = df.value if isinstance(df, u.Quantity) else df  # interpolate wants dimensionless df

    psd_unit = psd.unit  # Has to be kept as long as GWPy bug present
    
    signal1 = signal1.interpolate(df_val) if not np.isclose(signal1.df, df, rtol=0.01) else signal1
    signal2 = signal2.interpolate(df_val) if not np.isclose(signal2.df, df, rtol=0.01) else signal2
    logging.debug(df_val)
    logging.debug(psd.df.value)
    psd = psd.interpolate(df_val) if not np.isclose(psd.df, df, rtol=0.01) else psd

    # NOTE: these lines are temporary and shall fix bug in GWPy interpolate
    if signal1.unit != signal_unit:
        signal1 *= (signal_unit / signal1.unit)
    if signal2.unit != signal_unit:
        signal2 *= (signal_unit / signal2.unit)
    if psd.unit != psd_unit:
        psd *= (psd_unit / psd.unit)

    # signal1 = signal1.crop(start=f_lower, end=f_upper)
    # signal2 = signal2.crop(start=f_lower, end=f_upper)
    # psd = psd.crop(start=f_lower, end=f_upper)

    # TODO: check if rounding needed due to use of `floor` in crop?
    signal1 = signal1.crop(start=f_lower + 0.9 * df, end=f_upper)
    signal2 = signal2.crop(start=f_lower + 0.9 * df, end=f_upper)
    psd = psd.crop(start=f_lower + 0.9 * df, end=f_upper)

    logging.debug(signal1.frequencies)
    logging.debug(signal2.frequencies)
    logging.debug(psd.frequencies)

    logging.debug(f'{signal_unit}, {signal1.unit}, {signal2.unit}, {psd.unit}')

    if optimize_time_and_phase:
        # ifft wants start at f=0, so we may have to pad signals with zeros

        number_to_append = int((f_lower.value - 0.0) / df_val)  # Symbolic -0.0 to make it clear what happens
        # number_to_append = int(np.floor((f_lower.value - 0.0) / df_val))  # Symbolic -0.0 to make it clear what happens

        if number_to_append < 0:
            raise ValueError(
                'Cannot pad signal to f=0. Signal frequencies might be negative.'
            )

        logging.debug(number_to_append)
        
        logging.debug(f'{signal_unit}, {signal1.unit}, {signal2.unit}, {psd.unit}')
        
        f_series_to_pad = FrequencySeries(
            np.zeros(number_to_append),
            # unit=signal1.unit,  # TODO: use signal_unit?
            unit=signal_unit,  # TODO: use signal_unit?
            f0=0.0,
            df=df,
            dtype=complex  # TODO: use signal1.dtype?
        )

        # TODO: check if this has effect on FT, is not smooth right?
        # -> Maybe use 'linear_ramp' option of np.pad?
        # See https://numpy.org/doc/stable/reference/generated/numpy.pad.html#numpy.pad

        try:
            # Note: `pad` argument of append does not help, does what we do
            signal1 = f_series_to_pad.append(signal1, inplace=False)

            # f_series_to_pad *= signal2.unit / f_series_to_pad.unit  # Right now, we assure signal1 and signal2 have same unit
            signal2 = f_series_to_pad.append(signal2, inplace=False)

            # # psd = f_series_to_pad.fill(1.0).append(psd, inplace=False)  # Otherwise division by zero. Contribution is zero anyway because signals are zero there
            # # f_series_to_pad.fill(1.0)  # No return here, thus has to be done separately
            # f_series_to_pad.fill(1.0 * f_series_to_pad.unit)  # No return here, thus has to be done separately
            # # f_series_to_pad *= psd.unit   # Conver unit, may be Hz
            # f_series_to_pad *= psd.unit / f_series_to_pad.unit   # Conver unit, may be Hz
            # TODO: look for more efficient way than multiplication, can we avoid operations? On the other hand, padding is perhaps not too long

            
            f_series_to_pad.override_unit(psd.unit)
            # Should be much more efficient, orders of magnitude in test with
            # waveforms. Ok to work inplace here since f_series_pad is local
            f_series_to_pad.fill(1.0 * f_series_to_pad.unit)

            psd = f_series_to_pad.append(psd, inplace=False)  # Otherwise division by zero. Contribution is zero anyway because signals are zero there
        except ValueError as err:
            err_msg = str(err)

            if 'Cannot append discontiguous FrequencySeries' in err_msg:
                raise ValueError(
                    'Lower frequency bound and frequency spacing `df` do not '
                    'match, cannot smoothly continue signals to f=0 (required '
                    'for Fourier transform). This could be fixed by specifying '
                    'a lower bound that is some integer multiple of the given '
                    '`df` (if none was given, it is equal to 0.0625, where the'
                    'unit is derived from the signal frequencies) or by '
                    'adjusting the given `df`.'#\n'
                    # 'This message is printed in addition to the following one, '
                    # 'which is raised by GWPy:\n' +
                    # err_msg
                    # Interesting, this is printed anyway?  
                )
            else:
                # Raise error that would have been raised without exception 
                raise ValueError(err_msg)


        # TODO: implementation using built-in function -> done; but for some reason this seems to be less efficient...
        # signal1 = signal1.pad(number_to_append, mode='constant', constant_values=0.0)
        # signal2 = signal2.pad(number_to_append, mode='constant', constant_values=0.0)
        # psd = psd.pad(number_to_append, mode='constant', constant_values=1.0)

        return optimized_inner_product(signal1, signal2, psd)
        # return signal1, signal2, psd
        # TODO: decide if we divide by norm here? Or in overlap? Maybe do in
        # separate match function that always sets optimize_time_and_phase=True?
        # -> use overlap function for that?
    else:        
        return inner_product_computation(signal1, signal2, psd)


def inner_product_computation(
    signal1: FrequencySeries,
    signal2: FrequencySeries,
    psd: FrequencySeries
) -> float:
    """
    Lower level function for inner product calculation. Assumes that
    signal conditioning has been done so that they contain values at the
    same frequencies and then carries out the actual computation.

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
    float
        Inner product value with `signal1`, `signal2` inserted.
    
    See also
    --------
    scipy.integrate.simpson : Used for evaluation of inner product.
    """

    # ----- Assure same distance of samples -----
    assert np.isclose(signal1.df, psd.df, rtol=0.01) and np.isclose(signal2.df, psd.df, rtol=0.01),\
           'Signals must have equal frequency spacing.'
    
    # Second step: assure frequencies are sufficiently equal.
    # Maximum deviation allowed between the is given df, which
    # determines accuracy the signals have been sampled with.
    custom_error_msg = (
        'Frequency samples of input signals are not equal. This might be '
        'due to `df` being too large. If `df` is already small, consider '
        'choosing a (negative) power of two as these seem to work best.'
    )
    # Is perhaps because interpolate uses fft, which works best with powers of two
    try:
        assert (np.all(np.isclose(signal1.frequencies, signal2.frequencies, rtol=0.01))
                and np.all(np.isclose(signal1.frequencies, psd.frequencies, rtol=0.01))),\
                custom_error_msg
    except ValueError:
        # Due to unequal sample size. Since this is automatically checked by
        # numpy, we can be sure that signal1.size = signal2.size = psd.size
        raise ValueError(custom_error_msg)
    

    return 4.0 * np.real(
        simpson(y=signal1 * signal2.conjugate() / psd, x=signal1.frequencies)
    )


def optimized_inner_product(
    signal1: FrequencySeries,
    signal2: FrequencySeries,
    psd: FrequencySeries,
    # use_irfft: bool = False  # Results for False are usually a bit better
) -> tuple[TimeSeries, float, u.Quantity]:
    """
    Lower level function for inner product calculation. Assumes that
    signal conditioning has been done so that they contain values at the
    same frequencies and then carries out the actual computation.

    In contrast to `inner_product_computation`, this function optimizes
    the inner product over time and phase shifts.

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
    tuple[~gwpy.timeseries.TimeSeries, float, ~astropy.units.Quantity]
        A tuple with three values:
        (i) a ``TimeSeries`` where values of the inner product for
        different relative time shifts between `signal1`, `signal2`
        are stored (optimization over phase is already carried out)
        (ii) maximum value of (i)
        (iii) time at which maximum value (ii) occurs in (i)
    """

    # ----- First step: assure same distance of samples -----
    assert np.isclose(signal1.df, psd.df, rtol=0.01) and np.isclose(signal2.df, psd.df, rtol=0.01),\
           'Signals must have equal frequency spacing.'
    
    # ----- Second step: make sure all signals start at f=0 -----
    assert (np.isclose(signal1.f0, 0.0, rtol=0.01)
            and np.isclose(signal2.f0, 0.0, rtol=0.01)
            and np.isclose(psd.f0, 0.0, rtol=0.01)),\
            'All signals must start at f=0.'

    # ----- Third step: assure frequencies are sufficiently equal. -----
    # ----- Maximum deviation allowed between the is given df, which -----
    # ----- determines accuracy the signals have been sampled with. -----
    custom_error_msg = (
        'Frequency samples of input signals are not equal. This might be '
        'due to `df` being too large. If `df` is already small, consider '
        'choosing a (negative) power of two as these seem to work best.'
    )
    # Is perhaps because interpolate uses fft, which works best with powers of two
    try:
        assert (np.all(np.isclose(signal1.frequencies, signal2.frequencies, rtol=0.01))
                and np.all(np.isclose(signal1.frequencies, psd.frequencies, rtol=0.01))),\
                custom_error_msg
    except ValueError:
        # Due to unequal sample size. Since this is automatically checked by
        # numpy, we can be sure that signal1.size = signal2.size = psd.size
        raise ValueError(custom_error_msg)
    
    # ----- Fourth step: computations -----
    dft_series = signal1 * signal2.conjugate() / psd

    # Append zeros so that ifft can be used
    full_dft_series = np.append(dft_series.value, np.zeros(dft_series.size - 1))


    dt = 1.0 / (full_dft_series.size * signal1.df)

    match_series = 4.0 * TimeSeries(
        np.fft.ifft(full_dft_series / dt.value),  # Discrete -> continuous
        unit=u.dimensionless_unscaled,
        t0=0.0 * dt.unit,
        dt=dt
    )#.abs()
    # NOTE: it has been asserted that this series is dimensionless in the
    # inner_product function that should be used prior to this function

    # match_result = match_series.max()
    match_result = match_series.abs().max().value


    number_to_roll = match_series.size // 2  # Value chosen, no deep meaning
    match_series = np.roll(match_series, shift=number_to_roll)

    match_series.shift(-match_series.times[number_to_roll])  # Shouldn't it be sufficient to substract from t0? -> perhaps only with __array_finalize__ afterwards

    # TODO: check if t0=0 and then shifting is correct...

    # Compute peak time
    peak_index = np.argmax(match_series)
    peak_time = match_series.times[peak_index]
    
    return match_series, match_result, peak_time


def norm(
    signal: TimeSeries | FrequencySeries,
    *args,
    **kwargs
) -> float | tuple[TimeSeries, float, u.Quantity]:
    """
    Wrapper function for calculation of the norm of the given `signal`
    (i.e. square root of inner product between `signal` and `signal`,
    its SNR) as measured by the noise-weighted inner product
    implemented in `inner_product`.

    Parameters
    ----------
    signal : ~gwpy.timeseries.TimeSeries or
    ~gwpy.frequencyseries.FrequencySeries
        Signal to compute norm for.
    *args, **kwargs
        Additional arguments, will be passed to `inner_product` function.

    Returns
    -------
    float or tuple[~gwpy.timeseries.TimeSeries, float,
    ~astropy.units.Quantity]
        Norm of `signal`, i.e. square root of inner product of `signal`
        with itself.

        If `optimize_time_and_phase = True`, a tuple is returned that
        contains a ``TimeSeries``, the aforementioned norm and a time.
        See `optimized_inner_product` for details on the return.
    
    See also
    --------
    gw_signal_tools.inner_product.inner_product :
        Arguments are passed to this function for calculations.
    """

    out = inner_product(signal, signal, *args, **kwargs)

    if isinstance(out, float):
        return np.sqrt(out)
    else:
        return np.sqrt(out[0]), np.sqrt(out[1]), out[2]


def overlap(
    signal1: TimeSeries | FrequencySeries,
    signal2: TimeSeries | FrequencySeries,
    *args,
    **kwargs
) -> float | tuple[TimeSeries, float, u.Quantity]:
    """
    Wrapper for calculation of the overlap of two given signals as
    measured by the noise-weighted inner product implemented in
    `inner_product`. This means the signals are normalized to have unit
    norm (with respect to this inner product) and then inserted into
    `inner_product`.

    Parameters
    ----------
    signal1 : ~gwpy.timeseries.TimeSeries or
    ~gwpy.frequencyseries.FrequencySeries
        First signal.
    signal2 : ~gwpy.timeseries.TimeSeries or
    ~gwpy.frequencyseries.FrequencySeries
        Second signal.
    *args, **kwargs
        Additional arguments, will be passed to `inner_product` function.

    Returns
    -------
    float or tuple[~gwpy.timeseries.TimeSeries, float,
    ~astropy.units.Quantity]
        Overlap of `signal1` and `signal2`.

        If `optimize_time_and_phase = True`, a tuple is returned that
        contains a ``TimeSeries``, the aforementioned norm and a time.
        See `optimized_inner_product` for details on the return.
    
    See also
    --------
    gw_signal_tools.inner_product.inner_product :
        Arguments are passed to this function for calculations.
    """

    out = inner_product(signal1, signal2, *args, **kwargs)

    normalization = 1.0  # Default value

    if isinstance(norm1 := norm(signal1, *args, **kwargs), float):
        normalization *= norm1
    else:
        normalization *= norm1[1]

    if isinstance(norm2 := norm(signal2, *args, **kwargs), float):
        normalization *= norm2
    else:
        normalization *= norm2[1]


    if isinstance(out, float):
        return out / normalization
    else:
        return out[0] / normalization, out[1] / normalization, out[2]
