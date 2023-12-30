import numpy as np
from scipy.integrate import simpson

from gwpy.timeseries.timeseries import TimeSeries
from gwpy.frequencyseries.frequencyseries import FrequencySeries
import astropy.units as u


__doc__ = ('Implementation of noise-weighted inner product that is '
           'common in GW data analysis and helpers for computation.')



# ---------- Waveform Helpers ----------

def td_to_fd_waveform(signal: TimeSeries) -> FrequencySeries:
    """
    Transform given `signal` to Fourier domain. Note that the output is
    normalized to represent the continuous frequency components, not the
    discrete ones. Furthermore, a phase shift is applied to account for
    the starting of `signal`.

    Parameters
    ----------
    signal : TimeSeries
        Signal to be transformed.

    Returns
    -------
    out : FrequencySeries
        Transformed signal.

    See also
    --------
    `numpy.fft.rfft`
        Fourier transformation used here.
    """

    # Get discrete Fourier coefficients and corresponding frequencies
    out = FrequencySeries(
        np.fft.rfft(signal),
        frequencies=np.fft.rfftfreq(signal.size, d=signal.dx),
        unit=u.dimensionless_unscaled
    )

    # Convert discrete Fourier components to continuous
    out *= signal.dx / u.s
    
    # Account for non-zero starting time
    out *= np.exp(-1.j * 2 * np.pi * out.frequencies * signal.t0)
    
    return out


def restrict_f_range(
    signal: FrequencySeries,
    f_range: list[float] | list[u.quantity.Quantity],
    fill_val: float = 0.0,
    pad_to_f_zero: bool = False,
    cut_upper: bool = False,
    copy: bool = True
) -> FrequencySeries:
    """
    Set frequency components of signal outside of f_range to fill_val.

    Parameters
    ----------
    signal : FrequencySeries
        Signal to be restricted.
    f_range : list[float] | list[u.quantity.Quantity]
        Two-tuple specifying lower and upper frequency bounds that will
        be used as cutoffs.
    fill_val : float, optional, default = 0.0
        Value that will be used to fill `signal` outside of `f_range`.
    pad_to_f_zero : boolean, optional, default = False
        If true, signal is padded with `fill_val` to start at f=0.

        Convenient option if signal shall be prepared for inverse
        Fourier transformation, where start at f=0 is usually expected.
    cut_upper : bool, optional, default = False
        If true, signal will be cut off at upper frequency limit.

        Convenient in preparation for computations with multiple signals
        where equal frequency ranges might be needed.

    Returns
    -------
    FrequencySeries
        Copy of signal where values outside of `f_range` have been
        changed. If the interval defined by `f_range` is larger than
        the one spanned by signal.frequencies, no entry will be changed.

    Raises
    ------
    ValueError
        If `f_range` does not contain exactly two elements.
    """

    if len(f_range) != 2:
        raise ValueError('f_range must contain lower and upper frequency bounds.')

    f_lower = f_range[0] if f_range[0] is not None else 0.0
    f_upper = f_range[1] if f_range[1] is not None else signal.frequencies[-1]

    f_lower = u.Quantity(f_lower, unit=u.Hz)
    f_upper = u.Quantity(f_upper, unit=u.Hz)


    if pad_to_f_zero:# and (signal.f0 > signal.df):
        # Padding to zero frequency component shall be done and is needed
        number_to_append = int(signal.f0.value / signal.df.value)

        signal = FrequencySeries(
            np.zeros(number_to_append),
            unit=signal.unit,
            f0=0.0,
            df=signal.df,
            dtype=signal.dtype
        ).append(signal)
    else:
        # Otherwise filling is inplace
        signal = signal.copy()

    lower_filter = signal.frequencies < f_lower
    lower_number_to_discard = lower_filter[lower_filter == True].size
    # lower_number_to_discard = int((f_lower.value - signal.f0.value) / signal.df.value)  # Trying to avoid array stuff for efficiency reasons -> helps a bit, but not sure if number 1ßß% correct at all times... Does not look like that
    signal[:lower_number_to_discard].fill(fill_val)
    # signal[:lower_number_to_discard] = np.full(lower_number_to_discard, fill_val)  # Still inplace
    # signal = np.append(np.full(lower_number_to_discard, fill_val), (signal[lower_number_to_discard:]))  # Does not modify frequencies, problem

    # upper_filter = signal.frequencies > f_upper
    if cut_upper:
        # signal = signal[np.logical_not(upper_filter)]

        upper_filter = signal.frequencies <= f_upper
        signal = signal[upper_filter]
    else:
        upper_filter = signal.frequencies > f_upper
        upper_number_to_discard = upper_filter[upper_filter == True].size
        # upper_number_to_discard = int((f_upper.value - signal.frequencies[-1].value) / signal.df.value)  # Trying to avoid array stuff for efficiency reasons
        signal[signal.size - upper_number_to_discard:].fill(fill_val)
        # signal[signal.size - upper_number_to_discard:] = np.full(upper_number_to_discard, fill_val)  # Still inplace
        # signal = np.append(signal[upper_number_to_discard:], np.full(upper_number_to_discard, fill_val))  # Does not modify frequencies, problem

    return signal


def fd_to_td_waveform(
    signal: FrequencySeries,
    f_range: list[float] | list[u.quantity.Quantity] = None
) -> TimeSeries:
    """
    Transform given `signal` to time domain. Note that the input is
    expected to be normalized according to `td_to_fd_waveform`, i.e. so
    that the components in `signal` are continuous frequency components.

    Parameters
    ----------
    signal : FrequencySeries
        Signal to be transformed
    f_range : list[float] | list[u.quantity.Quantity], optional, default = None
        Range of frequency components to take into account. Is used as
        input for `restrict_f_range` function.

    Returns
    -------
    out : TimeSeries
        Transformed signal.

    Raises
    ------
    ValueError
        If `f_range` does not contain exactly two elements.

    See also
    --------
    `numpy.fft.irfft`
        Inverse Fourier transformation used here.
    """

    if f_range is not None:
        if len(f_range) != 2:
            raise ValueError('f_range must contain lower and upper frequency bounds.')

        signal = restrict_f_range(signal, f_range)
        

    # dt = 1 / (2 * signal.size * signal.df)  # Two because rfft has only half size
    dt = 1 / (2 * (signal.size - 1) * signal.df)
    # Note: 2*(n-1) follows normalization that happens according to the docs:
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html
    out = TimeSeries(
        np.fft.irfft(signal / dt.value),
        unit=u.dimensionless_unscaled,
        t0=0.0 * u.s,
        dt=dt
    )
    # Note: dividing by dt is necessary because irfft uses discrete
    # Fourier coefficients, but the input is expected to be continuous
    # (as this is true for the output of waveform generators in lal).
    # Equivalently, one could say that we first revert the numpy
    # normalization and then make the transition from continuous to
    # discrete Fourier coefficients by approximating the corresponding
    # integral (df comes in)

    # Handle wrap-around of signal
    number_to_roll = out.size * 7 // 8  # Value chosen, no deep meaning
    out = np.roll(out, shift=number_to_roll)

    out.times -= out.times[number_to_roll]  # Use .shift(7 / 8 * out.duration) or so?

    # TODO: make optional argument taper? TimeSeries has built-in function
    # .taper() to handle this

    return out


def pad_to_get_target_df(
    signal: TimeSeries,
    df: float | u.quantity.Quantity
) -> TimeSeries:
    """
    Pads `signal` with zeros after its end until a fft of it has desired
    resolution of `df`.

    Parameters
    ----------
    signal : TimeSeries
        Signal that will be padded.
    df : float | u.quantity.Quantity
        Desired resolution in frequency domain.

    Returns
    -------
    padded_signal : TimeSeries
        Padded signal, still in time domain.
    """

    df = u.quantity.Quantity(df, unit=u.Hz)

    # Compute what would be current df
    df_current = 1.0 / (signal.size * signal.dt)

    if df_current > df:
        target_sample_number = int(1.0 / (signal.dt * df))

        number_to_append = target_sample_number - signal.size

        padding_series = TimeSeries(
            np.zeros(number_to_append),
            unit=u.dimensionless_unscaled,
            t0=signal.times[-1] + signal.dt,
            dt=signal.dt
        )

        padded_signal = signal.append(padding_series, inplace=False)
    else:
        # Nothing has to be done
        padded_signal = signal

    return padded_signal



# ---------- PSD Handling ----------

def psd_from_file(fname: str, is_asd: bool = False) -> tuple[np.array]:
    """
    Read Power spectral density (PSD) values from a file into numpy
    arrays. The file must be readable by `numpy.loadtxt`.

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
    freqs, psd : tuple[np.array]
        Frequencies and PSD values as numpy arrays.

    See also
    --------
    `numpy.loadtxt`
        Routine used to read the file.
    """

    file_vals = np.loadtxt(fname)
    freqs, psd = file_vals[:, 0], file_vals[:, 1]

    if is_asd:
        psd = psd**2

    return freqs, psd


def psd_from_file_to_FreqSeries(
    fname: str,
    is_asd: bool = False,
    **kw_args
) -> FrequencySeries:
    """
    Read Power spectral density (PSD) values from file into a GWpy
    FrequencySeries.

    Parameters
    ----------
    fname : str
        File with two columns, left one representing frequencies and
        right one representing the corresponding PSD values.
    is_asd : bool, optional, default = False
        If true, values in file are taken to be ASD values rather than
        PSD values and thus a squared version of them is returned.
    **kw_args
        Other keyword arguments that will be passed to FrequencySeries
        constructor. Can be used to assign name to series and more.

    Returns
    -------
    FrequencySeries
        PSD as a FrequencySeries.
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
        **kw_args
    )


def get_FreqSeries_from_dict(
    psd: dict,
    psd_vals_key: str,
    is_asd: bool = False,
    **kw_args
) -> FrequencySeries:
    """
    Converts dictionary with Power spectral density (PSD) values into a
    GWpy FrequencySeries. Frequencies are expected to be accessible
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
    **kw_args
        Other keyword arguments that will be passed to FrequencySeries
        constructor. Can be used to assign name to series and more.

    Returns
    -------
    FrequencySeries
        Data from input dict in a GWpy FrequencySeries.
    """

    return FrequencySeries(
        psd[psd_vals_key]**2 if is_asd else psd[psd_vals_key],
        frequencies=psd['frequencies'],
        **kw_args
    )

# TODO (potentially): move to psd folder, e.g. given into __init__ file?



# ---------- Inner Product Implementation ----------

def inner_product(
    signal1: TimeSeries | FrequencySeries,
    signal2: TimeSeries | FrequencySeries,
    psd: FrequencySeries,
    f_range: list[float] | list[u.quantity.Quantity] = None,
    df: float | u.quantity.Quantity = None,
    optimize_time_and_phase: bool = False  # Call it 'compute_match'?
) -> float:
    """
    Calculates the noise-weighted inner product of two signals.

    Parameters
    ----------
    signal1 : TimeSeries | FrequencySeries
        First signal
    signal2 : TimeSeries | FrequencySeries
        Second signal
    psd : FrequencySeries
        Power spectral density to use in inner product.
    f_range : list[float] | list[u.quantity.Quantity], optional, default = None
        Frequency range
        Mention that f_range is potentially cropped, depending on
        frequency range of input
    df : float | u.quantity.Quantity, optional, default = None
        Distance df between samples in frequency domain to use in
        integration. Must not be too large, otherwise results of inner
        product may be erroneous. If chosen very small (e.g. in range
        of 0.001Hz), you should consider selecting only powers of two
        like 2**-10 because these work best with certain function calls
        that utilize the Fourier transform of the involved signals. An
        indicator this might be necessary is a shape mismatch error.
        If None, it is set to 0.0625 Hz, which is the default df of
        frequency domain waveforms generated by lal.
    optimize_time_and_phase : bool, optional, default = False
        Determines if a match is computed or just a "regular" inner
        product. The match will be optimized over relative time and
        phase shifts between `signal1` and `signal2`.

    Returns
    -------
    float
        Inner product value with `signal1`, `signal2` inserted.

    Raises
    ------
    TypeError
        In case either one of `signal1`, `signal2`, `psd` has wrong type.
    ValueError
        In case format of `f_range` parameter is not correct.
    """

    # Copying does not seem to be necessary. So we avoid the operations

    if type(psd) != FrequencySeries:
        raise TypeError('`psd` has to be a GWpy FrequencySeries.')
    # TODO: allow psd to be None? In this case, set it to 1? -> maybe handle that
    # via special FreqSeries, after all we use psd.frequencies etc


    # Handling of df argument
    if df is None:
        df = 0.0625 * u.Hz  # Default value of output of FDWaveform
    else:
        df = u.Quantity(df, unit=u.Hz)  # Default value of output of FDWaveform


    # If necessary, do fft. We apply padding to ensure sufficient
    # resolution in frequency domain.
    if type(signal1) == TimeSeries:
        signal1 = pad_to_get_target_df(signal1, df)
        # Do if part from padding function here? Would avoid function call
        signal1 = td_to_fd_waveform(signal1)
    elif type(signal1) != FrequencySeries:
        raise TypeError('`signal1` has to be a GWpy TimeSeries or FrequencySeries')

    if type(signal2) == TimeSeries:
        signal2 = pad_to_get_target_df(signal2, df)
        signal2 = td_to_fd_waveform(signal2)
    elif type(signal2) != FrequencySeries:
        raise TypeError('`signal2` has to be a GWpy TimeSeries or FrequencySeries')
    

    # Set default values
    f_lower, f_upper =[
            max([signal1.frequencies[0], signal2.frequencies[0], psd.frequencies[0]]),
                min([signal1.frequencies[-1], signal2.frequencies[-1], psd.frequencies[-1]])
    ]

    # If bounds are given, check that they fit the input data
    if f_range is not None:
        if len(f_range) != 2:
            raise ValueError(
                ('`f_range` must contain lower and upper frequency bounds for'
                 'integration. One of them or both can be `None`, but both'
                 'have to be specified if `f_range` is given.')
            )
        
        # Check if both lower and upper are given or one of them is None
        if f_range[0] is not None:
            f_lower_new = u.Quantity(f_range[0], unit='Hz')
        else:
            f_lower_new = f_lower
        
        if f_range[1] is not None:
            f_upper_new = u.Quantity(f_range[1], unit='Hz')

            # TODO: implement check of f_max with Nyquist of signals
        else:
            f_upper_new = f_upper


        # New lower bound must be greater than current lower bound,
        # otherwise no values for the range are available in signals
        if f_lower_new >= f_lower:
            f_lower = f_lower_new
        else:
            # Leave lower bound at f_lower, no update
            print((f'Given lower bound of {f_lower_new} is smaller than '
                   f'values available from given signals. Taking a lower '
                   f'bound of {f_lower} instead.'))

        # New upper bound must be smaller than current upper bound,
        # otherwise no values for the range are available in signals
        if f_upper_new <= f_upper:
            f_upper = f_upper_new
        else:
            # Leave upper bound at f_upper, no update
            print((f'Given upper bound of {f_upper_new} is larger than '
                   f'values available from given signals. Taking an upper '
                   f'bound of {f_upper} instead.'))


    # Get signals to same frequencies, i.e. make df equal and then restrict range
    df_float = df.value if type(df) == u.Quantity else df  # interpolate wants dimensionless df
    # TODO: check if interpolate does something if df is the one of signal
    # -> if yes, replace lines with something like signal = signal.interpolate() if signal.df != df else signal
    signal1 = signal1.interpolate(df_float)
    signal2 = signal2.interpolate(df_float)
    psd = psd.interpolate(df_float)


    # signal1 = restrict_f_range(signal1, [f_lower, f_upper], fill_val=0.0, pad_to_f_zero=True, cut_upper=True)
    # # signal1 = signal1[signal1.frequencies <= f_upper]

    # signal2 = restrict_f_range(signal2, [f_lower, f_upper], fill_val=0.0, pad_to_f_zero=True, cut_upper=True)
    # # signal2 = signal2[signal2.frequencies <= f_upper]

    # psd = restrict_f_range(psd, [f_lower, f_upper], fill_val=1.0, pad_to_f_zero=True, cut_upper=True)
    # # psd = psd[psd.frequencies <= f_upper]

    # # print(signal1.frequencies)
    # # print(signal2.frequencies)
    # # print(psd.frequencies)

    # if optimize_time_and_phase:
    #     return optimized_inner_product(signal1, signal2, psd)
    # else:
    #     return inner_product_computation(signal1, signal2, psd)
    
    # Wow, this is actually less efficient... Roughly 1ms slower...


    # Older versions, new one should be more efficient
    signal1 = signal1[(signal1.frequencies >= f_lower) & (signal1.frequencies <= f_upper)]
    signal2 = signal2[(signal2.frequencies >= f_lower) & (signal2.frequencies <= f_upper)]
    psd = psd[(psd.frequencies >= f_lower) & (psd.frequencies <= f_upper)]
    # Note: frequencies may not be changed by that, but is not needed

    # signal1 = restrict_f_range(signal1, f_range=[f_lower, f_upper])
    # signal2 = restrict_f_range(signal2, f_range=[f_lower, f_upper])
    # psd = restrict_f_range(psd, f_range=[f_lower, f_upper], fill_val=1.0)
    # Note: we fill with ones for psd to avoid division by zero

    # -> ah this does not work because we have unequal length in that case
    # So maybe use restrict_f_range for lower limit and cut off upper?

    # -> haha, next problem: they may not start at same frequency...
    # So initial solution might be best. Or we pad to f=0 here already...
    # But this would mean more operations in norm... Maybe do handling
    # separately for optimize and not
    

    if optimize_time_and_phase:
        # Shit, problem: we cut f_range... But need start at zero for ifft -> maybe do correcction in optimized_inner_product function?
        # -> should be solved now by use of restrict_f_range beforehand
        # -> nope, is not; doing that in optimized seems to be good idea and
        # also there is not other way I think since we absolutely need padding
        # to f=0 for ifft

        # signal1 = signal1[signal1.frequencies <= f_upper]
        # signal1 = restrict_f_range(signal1, f_range=[f_lower, None])

        # signal2 = signal2[signal2.frequencies <= f_upper]
        # signal2 = restrict_f_range(signal2, f_range=[f_lower, None])

        # psd = psd[psd.frequencies <= f_upper]
        # psd = restrict_f_range(psd, f_range=[f_lower, None], fill_val=1.0)

        number_to_append = int((f_lower.value - 0.0) / df_float)  # Symbolic -0.0 to make it clear what happens
        
        f_series_to_pad = FrequencySeries(
            np.zeros(number_to_append),
            unit=u.dimensionless_unscaled,
            f0=0.0,
            df=df,
            dtype=complex  # Use signal1.dtype?
        )

        signal1 = f_series_to_pad.append(signal1, inplace=False)
        signal2 = f_series_to_pad.append(signal2, inplace=False)

        # psd = f_series_to_pad.fill(1.0).append(psd, inplace=False)  # Otherwise division by zero. Contribution is zero anyway because signals are zero there
        f_series_to_pad.fill(1.0)  # No return here, thus has to be done separately
        psd = f_series_to_pad.append(psd, inplace=False)  # Otherwise division by zero. Contribution is zero anyway because signals are zero there

        return optimized_inner_product(signal1, signal2, psd)
        # return signal1, signal2, psd
        # TODO: decide if we divide by norm here? Or in overlap? Maybe do in
        # separate match function that always sets optimize_time_and_phase=True?
        # -> use overlap function for that?
    else:
        # signal1 = signal1[(signal1.frequencies >= f_lower) & (signal1.frequencies <= f_upper)]
        # signal2 = signal2[(signal2.frequencies >= f_lower) & (signal2.frequencies <= f_upper)]
        # psd = psd[(psd.frequencies >= f_lower) & (psd.frequencies <= f_upper)]
        
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
    signal1 : FrequencySeries
        First signal.
    signal2 : FrequencySeries
        Second signal.
    psd : FrequencySeries
        Power spectral density to use in inner product.

    Returns
    -------
    float
        Inner product value with `signal1`, `signal2` inserted.
    """

    # First step: assure same distance of samples
    assert np.isclose(signal1.df, psd.df, rtol=0.01) and np.isclose(signal2.df, psd.df),\
           'Signals must have equal frequency spacing.'
    
    # Second step: assure frequencies are sufficiently equal.
    # Maximum deviation allowed between the is given df, which
    # determines accuracy the signals have been sampled with.
    custom_error_msg = (
        'Frequency samples of input signals are not equal. This might be '
        'due to `df` being too large. If `df` is very small, consider '
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
    

    return 4.0 * np.real(simpson(y=np.multiply(signal1, signal2.conjugate()) / psd,
                                 x=signal1.frequencies))


def optimized_inner_product(
    signal1: FrequencySeries,
    signal2: FrequencySeries,
    psd: FrequencySeries,
    # use_irfft: bool = False  # Results for False are usually a bit better
) -> float:

    # First step: assure same distance of samples
    assert np.isclose(signal1.df, psd.df, rtol=0.01) and np.isclose(signal2.df, psd.df),\
           'Signals must have equal frequency spacing.'

    # Second step: assure frequencies are sufficiently equal.
    # Maximum deviation allowed between the is given df, which
    # determines accuracy the signals have been sampled with.
    custom_error_msg = (
        'Frequency samples of input signals are not equal. This might be '
        'due to `df` being too large. If `df` is very small, consider '
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
    
    
    dft_series = signal1 * signal2.conjugate() / psd
    

    negative_freq_terms = FrequencySeries(
        np.zeros(dft_series.size - 1),
        unit=u.dimensionless_unscaled,
        f0=-dft_series.frequencies[-1],
        df=dft_series.df,
        dtype=complex
    )

    full_dft_series = np.fft.ifftshift(negative_freq_terms.append(dft_series))
    # Get correct ordering for numpy ifft (f=0 component first, then positive
    # terms, then negative terms)


    dt = 1.0 / (full_dft_series.size * signal1.df)

    match_series = TimeSeries(
        np.fft.ifft(full_dft_series / dt.value),  # Discrete -> continuous
        unit=u.dimensionless_unscaled,
        t0=0.0 * u.s,
        dt=dt
    ).abs()

    match_series *= 4.0

    match_result = match_series.max()


    number_to_roll = match_series.size // 2  # Value chosen, no deep meaning
    match_series = np.roll(match_series, shift=number_to_roll)

    match_series.times -= match_series.times[number_to_roll]  # Use .shift()?

    # Compute peak time
    peak_index = np.argmax(match_series)
    # t_zero_index = match_series[match_series.times <= 0].size
    # peak_time = (peak_index - t_zero_index) * match_series.dt.value
    peak_time = match_series.times[peak_index]
    
    # return match_result
    return match_series, match_result, peak_time


def norm(
    signal: TimeSeries | FrequencySeries,
    psd: FrequencySeries,
    **kw_args
) -> float:
    """
    Calculates the norm (i.e. SNR) of a given `signal` as measured by
    the noise-weighted inner product implemented in `inner_product`.

    Parameters
    ----------
    signal : TimeSeries | FrequencySeries
        Signal to compute norm for.
    psd : FrequencySeries
        Power spectral density to use in inner product.
    **kw_args
        Additional arguments, will be passed to `inner_product` function.

    Returns
    -------
    float
        Norm of `signal`.
    """

    # return np.sqrt(inner_product(signal, signal, psd, **kw_args))
    if 'optimize_time_and_phase' in kw_args and kw_args['optimize_time_and_phase'] == True:
        return np.sqrt(inner_product(signal, signal, psd, **kw_args)[1])
    else:
        return np.sqrt(inner_product(signal, signal, psd, **kw_args))
    
    # TODO: handle case where much stuff is returned by inner_product better


def overlap(
    signal1: TimeSeries | FrequencySeries,
    signal2: TimeSeries | FrequencySeries,
    psd: FrequencySeries,
    # optimize_time_and_phase: bool = True,
    **kw_args
) -> float:
    """
    Calculates the overlap of two given signals as measured by the
    noise-weighted inner product of the corresponding signals normalized
    to have unit norm (with respect to this inner product).

    Parameters
    ----------
    signal1 : TimeSeries | FrequencySeries
        First signal.
    signal2 : TimeSeries | FrequencySeries
        Second signal.
    psd : FrequencySeries
        Power spectral density to use in inner product.
    **kw_args
        Additional arguments, will be passed to `inner_product` function.

    Returns
    -------
    float
        Overlap of `signal1` and `signal2`.
    """

    # TODO: use keyword is_normalized to indicate that both signals
    # already have unit norm? Would save some operations sometimes
    
    # out = inner_product(signal1, signal2, psd, optimize_time_and_phase=optimize_time_and_phase, **kwargs)
    out = inner_product(signal1, signal2, psd, **kw_args)
    
    normalization = norm(signal1, psd, **kw_args) * norm(signal2, psd, **kw_args)
    # No need to give optimization to norm function, right? Because
    # Simpson rule is more accurate

    # if optimize_time_and_phase:
    # if 'optimize_time_and_phase' in kw_args:# and optimize_time_and_phase:
    if 'optimize_time_and_phase' in kw_args and kw_args['optimize_time_and_phase'] == True:
        return out[0] / normalization, out[1] / normalization, out[2]
    else:
        return out / normalization
    
    # TODO: handle case where much stuff is returned by inner_product better
