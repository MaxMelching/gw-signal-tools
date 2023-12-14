import numpy as np
from scipy.integrate import simpson
import os
# from scipy.signal import resample

from gwpy.timeseries.timeseries import TimeSeries
from gwpy.frequencyseries.frequencyseries import FrequencySeries
import astropy.units as u



# ---------- Waveform Helpers ----------

def td_to_fd_waveform(signal: TimeSeries) -> FrequencySeries:
    # Get discrete Fourier coefficients and corresponding frequencies
    out = FrequencySeries(
        np.fft.rfft(signal),
        frequencies=np.fft.rfftfreq(signal.size, d=signal.dx),
        unit=u.dimensionless_unscaled
    )

    # Convert to continuous Fourier components
    out *= signal.dx / u.s
    
    # Account for non-zero starting time
    out *= np.exp(-1.j * 2 * np.pi * out.frequencies * signal.t0)
    
    return out


def restrict_f_range(signal, f_range):
    f_lower = f_range[0] if f_range[0] is not None else 0.0
    f_upper = f_range[1] if f_range[1] is not None else signal.frequencies[-1]

    f_lower = u.Quantity(f_lower, unit=u.Hz)
    f_upper = u.Quantity(f_upper, unit=u.Hz)

    lower_filter = signal.frequencies < f_lower
    lower_number_to_discard = lower_filter[lower_filter == True].size
    upper_filter = signal.frequencies > f_upper
    upper_number_to_discard = upper_filter[upper_filter == True].size

    # upper_number_to_discard = np.array((signal.frequencies >= f_upper) == True).size

    # signal[:lower_number_to_discard] = np.zeros(lower_number_to_discard)
    # signal[upper_number_to_discard:] = np.zeros(upper_number_to_discard)
    signal[:lower_number_to_discard].fill(0.0)
    signal[signal.size - upper_number_to_discard:].fill(0.0)

    return signal


def fd_to_td_waveform(signal: FrequencySeries, f_range: list[float] = None) -> TimeSeries:
    if f_range is not None:
        if len(f_range) != 2:
            raise ValueError('f_range must contain lower and upper frequency bounds.')

        signal = restrict_f_range(signal, f_range)
        

    out = TimeSeries(
        np.fft.irfft(signal * (2 * signal.size * signal.df / u.Hz)),  # Two because rfft has only half size
        unit=u.dimensionless_unscaled,
        t0=0.0 * u.s,
        dt=1 / (2 * signal.size * signal.df)  # Two because rfft has only half size
    )

    # Handle wrap-around of signal
    number_to_roll = out.size * 7 // 8  # Value chosen, no deep meaning
    out = np.roll(out, shift=number_to_roll)

    out.times -= out.times[number_to_roll]

    # TODO: check if/how padding influences this

    # TODO: set option f_range

    return out


# def get_to_target_df(signal: FrequencySeries) -> FrequencySeries:
#     # Fi

def pad_to_get_target_df(signal: TimeSeries, df: float) -> TimeSeries:
    # Compute what would be current df
    df_current = 1.0 / (signal.size * signal.dt)
    # print(df_current)

    if df_current > df:
        target_sample_number = int(1.0 / (signal.dt * df)) - 1
        # +1 to account for rounding, make sure resolution is indeed achieved
        # print(target_sample_number, signal.size)
        # -> interesting, is not needed (in fact, makes resolution worse...
        # Need -1 instead to make sure it is reached -> should also use
        # that, right? In case int there has some effect)

        number_to_append = target_sample_number - signal.size

        padding_series = TimeSeries(
            np.zeros(number_to_append),
            t0=signal.times[-1] + signal.dt,
            dt=signal.dt
        )

        padded_signal = signal.append(padding_series, inplace=False)
    else:
        # Nothing has to be done
        padded_signal = signal

    return padded_signal



# ---------- PSD Handling ----------

def psd_from_file_to_FreqSeries(fname, is_asd=False):
    file_vals = np.loadtxt(fname)
    freqs, psd = file_vals[:, 0], file_vals[:, 1]

    if is_asd:
        psd = psd**2

    out = FrequencySeries(psd, frequencies=freqs, unit=1 / u.Hz)

    return out


# psd_path = 'PSDs/'
# psd_path = os.path.join(gw_signal_tools.__path__, 'PSDs/')
psd_path = os.path.join(os.path.dirname(__file__), 'PSDs/')


# psd_gw150914 = psd_from_file_to_FreqSeries(os.path.join('PSDs/GW150914_psd.txt'))
# psd_gw150914 = psd_from_file_to_FreqSeries(os.path.join(psd_path, 'GW150914_psd.txt'))
# psd_o3_h1 = psd_from_file_to_FreqSeries('PSDs/O3_H1_asd.txt', is_asd=True)
# psd_o3_l1 = psd_from_file_to_FreqSeries('PSDs/O3_L1_asd.txt', is_asd=True)
# psd_o3_v1 = psd_from_file_to_FreqSeries('PSDs/O3_V1_asd.txt', is_asd=True)
# psd_sim = psd_from_file_to_FreqSeries('PSDs/sim_psd.txt')



# ---------- Inner Product Implementation ----------

def inner_product(
    signal1: TimeSeries | FrequencySeries,
    signal2: TimeSeries | FrequencySeries,
    psd: FrequencySeries | dict,
    detector = 'hanford',
    f_range: list[float] | list[u.quantity.Quantity] = None,
    df: float | u.quantity.Quantity = None
) -> float:
    """
    Mention that f_range is potentially cropped, depending on frequency
    range of input
    """

    # signal1 = signal1.copy()
    # signal2 = signal2.copy()
    if type(psd) == dict:
        psd = FrequencySeries(psd[detector], frequencies=psd['frequencies'])
    elif type(psd) == FrequencySeries:
        pass
    #     psd = psd.copy()
    else:
        raise TypeError('psd has to be a FrequencySeries or a dict.')
    # Copying does not seem to be necessary. So we avoid the operations


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

    if type(signal2) == TimeSeries:
        signal2 = pad_to_get_target_df(signal2, df)
        signal2 = td_to_fd_waveform(signal2)


    # IDEA: interpolation should be here, so set df first? Restriction
    # of frequency range can happen at arbitrary point later
    # -> use signal1.is_compatible(signal2)?
    
    # TODO: implement check of f_max with Nyquist
    

    # Set default values
    f_lower, f_upper =[
            max([signal1.frequencies[0], signal2.frequencies[0], psd.frequencies[0]]),
                min([signal1.frequencies[-1], signal2.frequencies[-1], psd.frequencies[-1]])
    ]

    # If bounds are given, check that they fit the input data
    if f_range is not None:
        if len(f_range) != 2:
            raise ValueError('f_range must contain lower and upper frequency bounds for integration.')
        
        f_lower_new = u.Quantity(f_range[0], unit='Hz')
        f_upper_new = u.Quantity(f_range[1], unit='Hz')

        # TODO: allow one bound to be None

        # New lower bound must be greater than biggest lower bound
        if f_lower_new > f_lower:
            f_lower = f_lower_new

        # New upper bound must be smaller than smallest upper bound
        if f_upper_new < f_upper:
            f_upper = f_upper_new


    # Get signals to same frequencies, i.e. make df equal and then restrict range
    df_float = float(df / df.unit) if type(df) == u.Quantity else df  # interpolate wants dimensionless df
    signal1 = signal1.interpolate(df_float)
    signal2 = signal2.interpolate(df_float)
    psd = psd.interpolate(df_float)
    # frequencies = np.arange(float(f_lower / u.Hz), float(f_upper / u.Hz) + df, step=df)
    # frequencies really needed? On the other hand, we would have to choose other array otherwise...


    signal1 = signal1[(signal1.frequencies >= f_lower) & (signal1.frequencies <= f_upper)]
    signal2 = signal2[(signal2.frequencies >= f_lower) & (signal2.frequencies <= f_upper)]
    psd = psd[(psd.frequencies >= f_lower) & (psd.frequencies <= f_upper)]
    # Note: frequencies may not be changed by that, but is not needed


    return inner_product_computation(signal1, signal2, psd)


def inner_product_computation(
    signal1: FrequencySeries,
    signal2: FrequencySeries,
    psd: FrequencySeries) -> float:
    # assert signal1.frequencies == signal2.frequencies == psd.frequencies
    assert signal1.df == signal2.df == psd.df

    # assert that frequencies are equal to within some precision? If not
    # possible for arrays, do for start and end
    # -> the following should be perfect

    # Maximum deviation allowed between the is given df, which
    # determines accuracy the signals have been sampled with
    assert (np.all(np.isclose(signal1.frequencies, signal2.frequencies, atol=signal1.df))
            and np.all(np.isclose(signal1.frequencies, psd.frequencies, atol=signal1.df))),\
            'Frequency samples of input signals are not equal. This might be due to `df` being too large.'

    return 4 * np.real(simpson(y=np.multiply(signal1, signal2.conjugate()) / psd,
                               x=signal1.frequencies))



# Good source for functions: https://gwsignal.docs.ligo.org/gwsignal-docs/gwsignal.core.html#module-gwsignal.core.gw

# Useful functions: gwsignal.core.conditioning_subroutines.resize_gwpy_timeseries(hp, start_id, new_length)


def norm(
    signal: TimeSeries | FrequencySeries,
    psd: FrequencySeries | dict,
    **kwargs
) -> float:
    return np.sqrt(inner_product(signal, signal, psd, **kwargs))



def overlap(
    signal1: TimeSeries | FrequencySeries,
    signal2: TimeSeries | FrequencySeries,
    psd: FrequencySeries | dict,
    **kwargs
) -> float:
    return inner_product(signal1, signal2, psd, **kwargs)\
           / norm(signal1, psd, **kwargs)\
           / norm(signal2, psd, **kwargs)



# PSD Stuff

def psd_from_file(fname: str) -> FrequencySeries:
    file_vals = np.loadtxt(fname)
    freqs, psd = file_vals[:, 0], file_vals[:, 1]

    return freqs, psd



def psd_from_file_to_FreqSeries(
    fname: str,
    is_asd: bool = False
) -> FrequencySeries:
    file_vals = np.loadtxt(fname)
    freqs, psd = file_vals[:, 0], file_vals[:, 1]

    if is_asd:
        psd = psd**2

    out = FrequencySeries(psd, frequencies=freqs, unit=1 / u.Hz)

    return out