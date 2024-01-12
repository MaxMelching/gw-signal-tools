import numpy as np
from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries
import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm

import logging
from typing import Optional, Any

from .inner_product import inner_product, norm


# def num_diff(signal: Series | np.ndarray) -> Series | np.ndarray:
#     """
#     Implementation of five-point stencil method for numerical
#     differentiation for instances of GWpy Series class (which includes
#     instances of TimeSeries, FrequencySeries).

#     Parameters
#     ----------
#     signal : gwpy.types.series.Series
#         Input to compute derivative for.

#     Returns
#     -------
#     gwpy.types.series.Series
#         Derivative of `signal`.
#     """


#     signal_deriv = signal.copy()
#     signal_deriv.fill(0.0)

#     signal_deriv += np.roll(signal, 2)
#     # signal_deriv[0] -= signal[-2]
#     # signal_deriv[1] -= signal[-1]

#     signal_deriv -= 8.0 * np.roll(signal, 1)
#     # signal_deriv[0] += 8.0 * signal[-1]

#     signal_deriv += 8.0 * np.roll(signal, -1)
#     # signal_deriv[-1] -= 8.0 * signal[0]

#     signal_deriv -= np.roll(signal, -2)
#     # signal_deriv[-1] += signal[1]
#     # signal_deriv[-2] += signal[0]

#     signal_deriv[0] = signal_deriv[1] = signal_deriv[2]
#     signal_deriv[-1] = signal_deriv[-2] = signal_deriv[-3]

#     return signal_deriv / (12.0 * signal.dx)


def num_diff(
    signal: Series | np.ndarray,
    h: Optional[float | u.Quantity] = None
) -> Series | np.ndarray:
    """
    Implementation of five-point stencil method for numerical
    differentiation for numpy arrays and instances of GWpy Series class
    (which includes instances of TimeSeries, FrequencySeries).

    Parameters
    ----------
    signal : gwpy.types.series.Series or numpy.ndarray
        Input to compute derivative for.
    h : float or astropy.units.Quantity, optional, default = None
        Distance between elements of signal. Is computed automatically
        in case signal is a GWpy Series. If None, is assumed to be 1.

    Returns
    -------
    gwpy.types.series.Series or numpy.ndarray
        Derivative of `signal`.
    """
    
    if isinstance(signal, Series):
        if h is not None:
            logging.info(
                '`signal` is instance of `gwpy.Series` class, `h` is'
                'taken from there and input is ignored.'
            )  # overridden would be nice word here
        
        h = signal.dx
    else:
        # Make sure signal is array, we utilize numpy operations
        signal = np.asarray(signal)

        # Check if h is set
        if h is None:
            h = 1.0


    signal_deriv = signal.copy()
    signal_deriv.fill(0.0)

    signal_deriv += np.roll(signal, 2)
    # signal_deriv[0] -= signal[-2]
    # signal_deriv[1] -= signal[-1]

    signal_deriv -= 8.0 * np.roll(signal, 1)
    # signal_deriv[0] += 8.0 * signal[-1]

    signal_deriv += 8.0 * np.roll(signal, -1)
    # signal_deriv[-1] -= 8.0 * signal[0]

    signal_deriv -= np.roll(signal, -2)
    # signal_deriv[-1] += signal[1]
    # signal_deriv[-2] += signal[0]

    signal_deriv[0] = signal_deriv[1] = signal_deriv[2]
    signal_deriv[-1] = signal_deriv[-2] = signal_deriv[-3]

    return signal_deriv / (12.0 * h)


# TODO: implement fitting derivative method


# NOTE: this version does not work
# def fisher_val_at_point(
#     wf_params_at_point: dict[str, u.Quantity],
#     param_to_vary: str,
#     wf_generator: Any,
#     psd: FrequencySeries,
#     tolerance: float = 0.01
# ) -> float | u.Quantity:  # TODO: decide here
#     # Right now only for derivatives of Fisher value
    
#     param_center_val = wf_params_at_point[param_to_vary]

#     derivative_vals = [np.inf]  # Make sure first difference is too large

#     for step_size in [0.5, 0.1, 0.01, 0.001, 0.0001]:
#         param_vals = param_center_val + u.Quantity(np.arange(-2.0, 2.1, step=1.0) * step_size, unit=param_center_val.unit)

#         waveforms = [
#             wfm.GenerateFDWaveform(
#                 wf_params_at_point | {param_to_vary: param_val},
#                 wf_generator
#             )[0] for param_val in param_vals
#         ]

#         temp_deriv_series = FrequencySeries(
#             num_diff(waveforms, h=step_size),
#             frequencies=waveforms[2].frequencies
#         )

#         derivative_at_point = norm(temp_deriv_series, psd)


#         # Check for convergence
#         if np.abs(derivative_at_point - derivative_vals[-1]) / derivative_at_point <= tolerance:
#             return derivative_at_point
#         else:
#             derivative_vals += [derivative_at_point]
    
#     logging.info(
#         f'Desired relative accuracy of {tolerance:.3f} could not be reached.'
#         'Best estimate at step size 0.0001 had relative difference of '
#         f'{np.abs(derivative_at_point - derivative_vals[-1]) / derivative_at_point:.4f} '
#         'to estimate with previous step size at 0.001.'
#     )

#     return derivative_at_point


def fisher_val_at_point(
    wf_params_at_point: dict[str, u.Quantity],
    param_to_vary: str,
    wf_generator: Any,
    psd: FrequencySeries,
    tolerance: float = 0.01
) -> float | u.Quantity:  # TODO: decide here
    # Right now only for derivatives of Fisher value
    
    param_center_val = wf_params_at_point[param_to_vary]

    derivative_vals = [np.inf]  # Make sure first difference is too large

    for step_size in [0.1, 0.01, 0.001, 0.0001]:
        # param_vals = param_center_val + u.Quantity(np.arange(-2.0, 2.1, step=1.0) * step_size, unit=param_center_val.unit)
        # param_vals = param_center_val + u.Quantity(np.array([-2.0, -1.0, 1.0, 2.0]) * step_size, unit=param_center_val.unit)
        param_vals = param_center_val + np.array([-2.0, -1.0, 1.0, 2.0]) * step_size * param_center_val

        waveforms = [
            wfm.GenerateFDWaveform(
                wf_params_at_point | {param_to_vary: param_val},
                wf_generator
            )[0] for param_val in param_vals
        ]

        # print(np.asanyarray(waveforms, dtype=FrequencySeries))
            
        # print(num_diff(waveforms, h=step_size))

        # temp_deriv_series = FrequencySeries(
        #     np.zeros(waveforms[2].size),
        #     frequencies=waveforms[2].frequencies
        # )


        deriv_series = waveforms[0].copy()

        deriv_series -= 8.0 * waveforms[1]

        deriv_series += 8.0 * waveforms[2]

        deriv_series -= waveforms[3]

        deriv_series /= 12.0 * step_size * param_center_val

        # deriv_series += num_diff(waveforms, h=step_size)

        derivative_at_point = norm(deriv_series, psd)

        # logging.info(np.abs((derivative_at_point - derivative_vals[-1]) / derivative_at_point) <= tolerance)
        # Check for convergence
        if np.abs((derivative_at_point - derivative_vals[-1]) / derivative_at_point) <= tolerance:
            return derivative_at_point
        else:
            derivative_vals += [derivative_at_point]

    # logging.info(derivative_vals)
    
    logging.info(
        f'Desired relative accuracy of {tolerance} could not be reached. '
        'Best estimate at step size 0.0001 had relative difference of '
        f'{np.abs((derivative_vals[-2] - derivative_vals[-1]) / derivative_at_point):.4f} '
        'to estimate with previous step size at 0.001.'
    )

    return derivative_at_point