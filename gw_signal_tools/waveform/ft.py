# -- Third Party Imports
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import numpy as np

# -- Local Package Imports
from ._error_helpers import _q_convert


__all__ = ('td_to_fd_waveform', 'fd_to_td_waveform')


__doc__ = """

"""


# TODO: once gwsignal with FT is on main branch, maybe just point to
# these functions?

try:
    # from lalsimulation.gwsignal.core.utils import (
    #     td_to_fd, fd_to_td, zero_pad, shift_signal, roll_signal
    # )

    import astropy.units as u

    from typing import Literal
    from gwpy.frequencyseries import FrequencySeries
    from gwpy.timeseries import TimeSeries

    FT_CONVENTION_DEFAULT: str = 'wrap'


    def td_to_fd(
        signal: TimeSeries,
        convention: Literal['wrap', 'unwrap'] = FT_CONVENTION_DEFAULT
    ) -> FrequencySeries:
        """
        Transform given :code:`signal` into Fourier domain. Note that the
        output is normalized to represent the continuous frequency
        components, not the discrete ones. Furthermore, depending on the
        chosen `convention`, a phase shift is applied to account for the
        epoch (= starting time) of the :code:`signal`.

        Parameters
        ----------
        signal : ~gwpy.timeseries.TimeSeries
            Signal to be transformed.
        convention : Literal['wrap', 'unwrap'], optional, default = 'wrap'
            Determines how the output is to be interpreted. In case of
            `'unwrap'`, information about the starting time is stored in
            the epoch, but also imprinted onto `signal` itself via a
            corresponding phase factor. For `'wrap'`, on the other hand,
            this information is stored in the epoch only.

        Returns
        -------
        out : ~gwpy.frequencyseries.FrequencySeries
            Transformed :code:`signal`.

        See Also
        --------
        numpy.fft.rfft, numpy.fft.fft : Fourier transformations used.

        Notes
        -----
        Conditioning routines of LAL waveform generators operate under the
        `'wrap'` convention. Thus, the interplay between waveforms and
        (inverse) Fourier transforms using the `'unwrap'` convention is
        likely to produce inconsistent results!
        """
        # -- Check if rfft can be performed or full fft needed
        if np.iscomplexobj(signal):
            out = FrequencySeries(
                np.fft.fftshift(np.fft.fft(signal)) * signal.dx,  # Discrete -> continuous
                frequencies=np.fft.fftshift(np.fft.fftfreq(signal.size, d=signal.dx.value)) << 1 / signal.dx.unit,
                unit=signal.unit * signal.dx.unit,
                name=('Fourier transform of '
                    + signal.name if signal.name is not None else None),
                channel=signal.channel,
                epoch=signal.epoch.value
            )
        else:
            out = FrequencySeries(
                np.fft.rfft(signal) * signal.dx,  # Discrete -> continuous
                frequencies=np.fft.rfftfreq(signal.size, d=signal.dx.value) << 1 / signal.dx.unit,
                unit=signal.unit * signal.dx.unit,
                name=('Fourier transform of '
                    + signal.name if signal.name is not None else None),
                channel=signal.channel,
                epoch=signal.epoch.value
            )

        if convention == 'unwrap':
            # -- Account for non-zero starting time via phase factor
            return out*np.exp(-2.j*np.pi*out.frequencies*signal.t0)
            # -- Note: for TimeSeries t0=epoch
        elif convention == 'wrap':
            return out
        else:
            raise ValueError(f"Invalid convention '{convention}'.")


    def fd_to_td(
        signal: FrequencySeries,
        convention: Literal['wrap', 'unwrap'] = FT_CONVENTION_DEFAULT
    ) -> TimeSeries:
        """
        Transform given :code:`signal` into time domain. Note that the
        output is normalized to represent the continuous frequency
        components, not the discrete ones.

        Parameters
        ----------
        signal : ~gwpy.timeseries.TimeSeries
            Signal to be transformed.
        convention : Literal['wrap', 'unwrap'], optional, default = 'wrap'
            Determines how the input is interpreted. In case of `'unwrap'`,
            information about the starting time is assumed to be stored in
            the epoch, but also imprinted onto `signal` itself via a
            corresponding phase factor. For `'wrap'`, on the other hand,
            this information is assumed to be stored in the epoch only.

        Returns
        -------
        out : ~gwpy.frequencyseries.FrequencySeries
            Transformed :code:`signal`.

        See Also
        --------
        numpy.fft.rfft, numpy.fft.fft : Fourier transformations used.

        Notes
        -----
        Conditioning routines of LAL waveform generators operate under the
        `'wrap'` convention. Thus, the interplay between waveforms and
        (inverse) Fourier transforms using the `'unwrap'` convention is
        likely to produce inconsistent results!
        """
        start_time = signal.epoch.value*u.s  # Convert Time to Quantity

        if convention == 'unwrap':
            # -- Avoid wrap-around of signal by rolling in frequency
            # -- domain. Is doneby shifting signal to starting time t0=0
            # -- here and then shifting IFT signal back to t0=start_time
            # -- (this order is very desirable because otherwise, we
            # -- might be off by a single sample in time when performing
            # -- repeated FFTs, IFFTs and then compare signals)
            _signal = signal*np.exp(2.j*np.pi*signal.frequencies*start_time)
        elif convention == 'wrap':
            _signal = signal
        else:
            raise ValueError(f"Invalid convention '{convention}'.")

        # -- Check if irfft can be performed or full ifft needed
        if _signal.f0 == 0.0:
            dt = 1 / (2 * (_signal.size - 1) * _signal.df)
            # 2*(n-1) follows normalization that happens according to the docs:
            # https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html

            out = TimeSeries(
                np.fft.irfft(_signal / dt),
                unit=_signal.unit / dt.unit,
                t0=start_time,  # t0=epoch for TimeSeries
                dt=dt,
                name=('Inverse Fourier transform of '
                    + _signal.name if _signal.name is not None else None),
                channel=_signal.channel
            )
        elif _signal.f0 < 0.0:
            if np.fft.ifftshift(_signal.frequencies)[0] != 0.0:
                raise ValueError(
                    '`signal` does not have correct format for ifft. Please check '
                    'https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html#numpy.fft.ifft'
                    'for the requirements regarding frequency range.'
                )

            dt = 1 / (_signal.size * _signal.df)
            # Follows normalization that happens according to the docs:
            # https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html

            out = TimeSeries(
                np.fft.ifft(np.fft.ifftshift(_signal) / dt),
                unit=_signal.unit / dt.unit,
                t0=start_time,  # t0=epoch for TimeSeries
                dt=dt,
                name=('Inverse Fourier transform of '
                    + _signal.name if _signal.name is not None else None),
                channel=_signal.channel
            )
        else:
            raise ValueError(
                'Signal starts at positive frequency. Need either f0=0 (for irfft)'
                ' or negative f0 (for ifft).'
            )

        return out


    def correct_for_conditioning(signal: FrequencySeries) -> FrequencySeries:
        """
        Change end time of signal to zero by rolling back the portion at
        positive times. In particular, this corrects for an effect
        introduced when :code:`signal` has been conditioned.

        Notes
        -----
        This function is analogous to what the bilby package does in its
        :code:`gw/source` functions :code:`gw_signal_binary_black_hole` and
        :code:`lal_binary_black_hole` (cf. the code blocks in l.246ff and
        l.665ff, respectively; lines are quoted for v2.3.0). It is NOT meant
        to convert between the different Fourier conventions.
        """
        time_shift = 1./signal.df + signal.epoch.value*u.s
        # -- Note: 1/df is duration of IFT signal (T=N*dt=N/(N*df)=1/df)

        if time_shift.value == 0.:
            return signal
        else:
            return shift_signal_cyclic(signal, -time_shift)


    def shift_signal_cyclic(
        signal: TimeSeries | FrequencySeries,
        time_shift: u.Quantity,
        convention: Literal['wrap', 'unwrap'] = FT_CONVENTION_DEFAULT
    ) -> TimeSeries | FrequencySeries:
        """
        Roll `signal` by `amount` to the right (i.e. take part of this
        length from beginning of `signal` and append it to the end; an
        equivalent description is "cyclic time shift").

        Notes
        -----
        For TimeSeries, it is basically a wrapper around `numpy.roll`, which
        means the behaviour its similar to this function (but we roll by
        minus the amount). For FrequencySeries, things are more complicated.

        To shift a ``TimeSeries`` by certain number of indices, pass
        :math:`time_shift=index_shift*signal.dt`. For a ``FrequencySeries``,
        the same shift can be achieved by using the relation
        :math:`dt = 1/(N*df)` where :code:`N = len(signal)`.
        """
        if isinstance(signal, TimeSeries):
            index_shift = int(time_shift / signal.dt)
            _signal = np.roll(signal, -index_shift)
            _signal.t0 = _signal.t0 + index_shift*_signal.dt  # NOT the same as +=
            # -- Note: we do not add time_shift because of potential
            # -- round-off errors (index_shift*dt is more accurate)
            return _signal
        elif isinstance(signal, FrequencySeries):
            if convention == 'unwrap':
                _signal = signal.copy()
                _signal.epoch = _signal.epoch.value*u.s + time_shift
            elif convention == 'wrap':
                _signal = signal * np.exp(2.j*np.pi*time_shift*signal.frequencies)
                _signal.epoch = _signal.epoch.value*u.s + time_shift  # NOT the same as +=
            else:
                raise ValueError(f"Invalid convention '{convention}'.")

            return _signal
        else:
            raise TypeError('`signal` must be a GWpy `FrequencySeries` or '
                            '`TimeSeries`.')


    def shift_signal(
        signal: TimeSeries | FrequencySeries,
        time_shift: u.Quantity,
        convention: Literal['wrap', 'unwrap'] = FT_CONVENTION_DEFAULT
    ) -> TimeSeries | FrequencySeries:
        """
        Shift `signal` by `time_shift` in time.
        """
        # -- Note: for TimeSeries.t0, the operation += does NOT have the
        # -- desired effect of automatically adjusting TimeSeries.times.
        # -- Thus we use setter instead. Similar concerns apply to the
        # -- FrequencySeries.epoch property.
        if isinstance(signal, TimeSeries):
            index_shift = int(time_shift / signal.dt)
            _signal = signal.copy()
            _signal.t0 = _signal.t0 + index_shift*_signal.dt  # NOT the same as +=
            return _signal
        elif isinstance(signal, FrequencySeries):
            if convention == 'unwrap':
                _signal = signal * np.exp(-2.j*np.pi*time_shift*signal.frequencies)
                _signal.epoch = _signal.epoch.value*u.s + time_shift  # NOT the same as +=
            elif convention == 'wrap':
                _signal = signal.copy()
                _signal.epoch = _signal.epoch.value*u.s + time_shift
            else:
                raise ValueError(f"Invalid convention '{convention}'.")

            return _signal
        else:
            raise TypeError('`signal` must be a GWpy `FrequencySeries` or '
                            '`TimeSeries`.')


    def zero_pad(
        signal: TimeSeries,
        df: float | u.Quantity
        # TODO: option to pad beginning and end? Or just beginning because
        # at end we are not consistent with Fourier convention anymore
    ) -> TimeSeries:
        """
        Pads :code:`signal` with zeros so that a FFT of it has the desired
        resolution of :code:`df`. If the resolution is already at the
        required level, it does not nothing.

        Parameters
        ----------
        signal : ~gwpy.timeseries.TimeSeries
            Signal that will be padded.
        df : float or ~astropy.units.Quantity
            Desired resolution in frequency domain.

        Returns
        -------
        padded_signal : ~gwpy.timeseries.TimeSeries
            Padded signal, still in time domain.
        """
        frequ_unit = 1/signal.times.unit

        try:
            df = u.Quantity(df, unit=frequ_unit)
        except u.UnitConversionError:
            # -- Conversion only fails if df is already Quantity and has
            # -- non-matching unit, so we can assume that df.unit works
            raise ValueError(
                f'Need consistent units for `df` ({df.unit}) and '
                f'`signals.frequencies` ({frequ_unit}).'
            )

        # -- Compute what would be current df of FT
        df_current = 1.0 / (signal.size * signal.dt)

        if df_current > df:
            target_sample_number = int(1.0 / (signal.dt * df))
            number_to_append = target_sample_number - signal.size

            return TimeSeries(
                np.zeros(number_to_append),
                unit=signal.unit,
                t0=signal.times[0] - number_to_append*signal.dt,
                dt=signal.dt
            ).append(signal)
        else:
            # -- No padding required
            return signal



    td_to_fd_waveform = td_to_fd
    fd_to_td_waveform = fd_to_td
except ImportError:
    ...
    # define them manually

    # Point to functions or make quick wrapper (in case additional
    # rolling for example is wanted)


    def td_to_fd_waveform(signal: TimeSeries) -> FrequencySeries:
        """
        Transform given :code:`signal` to Fourier domain. Note that the
        output is normalized to represent the continuous frequency
        components, not the discrete ones. Furthermore, a phase shift is
        applied to account for the starting of :code:`signal`.

        Parameters
        ----------
        signal : ~gwpy.timeseries.TimeSeries
            Signal to be transformed.

        Returns
        -------
        out : ~gwpy.frequencyseries.FrequencySeries
            Transformed :code:`signal`.

        See Also
        --------
        numpy.fft.rfft, numpy.fft.fft : Fourier transformations used.
        """
        # Check if rfft can be performed or full fft needed
        if np.iscomplexobj(signal):
            out = FrequencySeries(
                np.fft.fftshift(np.fft.fft(signal)) * signal.dx,  # Discrete -> continuous
                frequencies=np.fft.fftshift(np.fft.fftfreq(signal.size, d=signal.dx.value)) << 1 / signal.dx.unit,
                unit=signal.unit * signal.dx.unit,  # Make sure numpy functions carry unit correctly
                name=('Fourier transform of '
                    + signal.name if signal.name is not None else None),
                channel=signal.channel,
                epoch=signal.epoch.value + (-1 if signal.epoch.value < 0 else 0)
            )
        else:
            out = FrequencySeries(
                np.fft.rfft(signal) * signal.dx,  # Discrete -> continuous
                frequencies=np.fft.rfftfreq(signal.size, d=signal.dx.value) << 1 / signal.dx.unit,
                unit=signal.unit * signal.dx.unit,  # Make sure numpy functions carry unit correctly
                name=('Fourier transform of '
                    + signal.name if signal.name is not None else None),
                channel=signal.channel,
                epoch=signal.epoch.value + (-1 if signal.epoch.value < 0 else 0)
            )
        
        # Account for non-zero starting time
        # out *= np.exp(-1.j * 2 * np.pi * out.frequencies * signal.t0)
        # Equivalent to multiplication with epoch because for TimeSeries, t0=epoch
        return out


    def fd_to_td_waveform(signal: FrequencySeries) -> TimeSeries:
        """
        Transform given :code:`signal` to time domain. Note that the input
        is expected to be normalized according to :code:`td_to_fd_waveform`,
        i.e. so that the components in :code:`signal` are continuous
        frequency components.

        Parameters
        ----------
        signal : ~gwpy.frequencyseries.FrequencySeries
            Signal to be transformed.

        Returns
        -------
        out : ~gwpy.timeseries.TimeSeries
            Transformed :code:`signal`.

        Notes
        -----
        In case you experience a wrap-around after applying this function,
        there are two options: either interpolating :code:`signal` to a
        higher resolution in frequency space or, if :code:`signal` itself is
        obtained from a Fourier transform, pad the time domain signal with
        zeros at its end (if the signal goes to zero at its end), e.g. using
        the function :code:`~gw_signal_tools.waveform.utils.
        pad_to_target_df`. The time span to be padded should be roughly the
        same as the epoch (i.e. the starting time) of the signal, so that
        a shift by it does not cause a wrap-around.

        See Also
        --------
        numpy.fft.irfft, numpy.fft.ifft :
            Inverse Fourier transformations used.
        """
        # Avoid wrap-around of signal by manually setting starting time and making
        # sure signal starts at zero using time shift with negative epoch
        # signal = signal * np.exp(1.j*2*np.pi*signal.frequencies.value*signal.epoch.value)
        # TODO: call _signal?
        # NOTE: taking value of epoch here is important, otherwise no conversion
        # to number is performed from Time class

        # Check if irfft can be performed or full ifft needed
        if signal.f0 == 0.0:
            dt = 1 / (2 * (signal.size - 1) * signal.df)
            # NOTE: 2*(n-1) follows normalization that happens according to the docs:
            # https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html

            out = TimeSeries(
                np.fft.irfft(signal / dt),
                unit=signal.unit / dt.unit,  # Make sure numpy functions carry unit correctly
                t0=signal.epoch.value * dt.unit,  # t0=epoch for TimeSeries
                dt=dt,  # Units might not be s, thus no use of .to()
                name=('Inverse Fourier transform of '
                    + signal.name if signal.name is not None else None),
                channel=signal.channel
            )
        elif signal.f0 < 0.0:
            if np.fft.ifftshift(signal.frequencies)[0] != 0.0:
                raise ValueError(
                    '`signal` does not have correct format for ifft. Please check '
                    'https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html#numpy.fft.ifft'
                    'for the requirements regarding frequency range.'
                )

            dt = 1 / (signal.size * signal.df)
            # Note: follows normalization that happens according to the docs:
            # https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html

            out = TimeSeries(
                np.fft.ifft(np.fft.ifftshift(signal) / dt),
                unit=signal.unit / dt.unit,  # Make sure numpy functions carry unit correctly
                t0=signal.epoch.value * dt.unit,  # t0=epoch for TimeSeries
                dt=dt,  # Units might not be s, thus no use of .to()
                name=('Inverse Fourier transform of '
                    + signal.name if signal.name is not None else None),
                channel=signal.channel
            )
        else:
            raise ValueError(
                'Signal starts at positive frequency. Need either f0=0 (for irfft)'
                ' or negative f0 (for ifft).'
            )

        return out
