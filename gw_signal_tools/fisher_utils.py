import logging
import warnings
from typing import Optional, Any

import numpy as np

from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries
import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm

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

    At the boundary points, less accurate methods like the central,
    forward and backward difference are used.

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
        # if h is not None:
        #     logging.info(
        #         '`signal` is instance of `gwpy.Series` class, `h` is '
        #         'taken from there and input is ignored.'
        #     )  # overridden would be nice word here
        
        # h = signal.dx
        if h is None:
            h = signal.dx
        else:
            h = u.Quantity(h, signal.unit)

            if h != signal.dx:
                warnings.warn(
                    'Given `h` does not coincide with `signal.dx`.'
                )
    else:
        # Make sure signal is array, we utilize numpy operations
        signal = np.asarray(signal)

        # Check if h is set
        if h is None:
            h = 1.0
        # elif isinstance(h, u.Quantity):
        #     logging.info(
        #         '`h` has a unit, but no information about the unit of '
        #         '`signal` is available.'
        #     )
        # TODO: decide if necessary


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

    # signal_deriv[0] = signal_deriv[1] = signal_deriv[2]
    # signal_deriv[-1] = signal_deriv[-2] = signal_deriv[-3]

    signal_deriv /= 12.0 * h

    signal_deriv[0] = (signal[1] - signal[0]) / h  # Forward difference
    signal_deriv[1] = (signal[2] - signal[0]) / (2.0 * h)  # Central difference

    signal_deriv[-2] = (signal[-1] - signal[-3]) / (2.0 * h)  # Central difference
    signal_deriv[-1] = (signal[-1] - signal[-2]) / h  # Backward difference

    return signal_deriv


# TODO: implement fitting derivative method -> hm, this might be hard


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



# The following is supposed to be a more general version of function below
# def fisher_val_v2(
#     wf_params_at_point: dict[str, u.Quantity],
#     params_to_vary: str | list[str],
#     wf_generator: Any,
#     psd: FrequencySeries,
#     tolerance: float = 0.01,
#     **inner_prod_kw_args
# ) -> float | u.Quantity:  # TODO: decide here
#     # Right now only for derivatives of Fisher value

#     if type(params_to_vary) == str:
#         params_to_vary = [params_to_vary]
#     # TODO: check if parameters like phase in there? Where computation is not just applying inner_product

#     param_numb = len(params_to_vary)

#     fisher_matrix = np.zeros((param_numb, param_numb))

#     for i, param_i in enumerate(params_to_vary):
#         for j, param_j in enumerate(params_to_vary):   

#             derivative_vals = [np.inf]  # Make sure first difference is too large

#             for step_size in [0.01, 0.001, 0.0001]:  # Removed 0.1, only adds unnecessary operations
                
#                 deriv_param_i = get_waveform_derivative_1D(
#                     wf_params_at_point,
#                     param_i,
#                     wf_generator,
#                     step_size
#                 )

#                 deriv_param_j = get_waveform_derivative_1D(
#                     wf_params_at_point,
#                     param_j,
#                     wf_generator,
#                     step_size
#                 )

#                 derivative_at_point = inner_product(deriv_param_i, deriv_param_j, psd, **inner_prod_kw_args)
#                 # TODO: add optimize=True as default? Do we need that?

#                 # logging.info(np.abs((derivative_at_point - derivative_vals[-1]) / derivative_at_point) <= tolerance)
#                 # Check for convergence
#                 if np.abs((derivative_at_point - derivative_vals[-1]) / derivative_at_point) <= tolerance:
#                     break
#                 else:
#                     derivative_vals += [derivative_at_point]

#             logging.info(
#                 f'Desired relative accuracy of {tolerance} could not be reached '
#                 f'for parameter combination {param_i}, {param_j}'
#                 'Best estimate at step size 0.0001 had relative difference of '
#                 f'{np.abs((derivative_vals[-2] - derivative_vals[-1]) / derivative_at_point):.4f} '
#                 'to estimate with previous step size at 0.001.'
#             )
#             # Ah shit, problem here: we print this even if convergence is good -> use definition 'deviation:=' in condition?

#             fisher_matrix[i, j] = fisher_matrix[j, i] = derivative_vals[-1]


#     return fisher_matrix


# v2 is wayyy to redundant
def fisher_val_v3(
    wf_params_at_point: dict[str, u.Quantity],
    params_to_vary: str | list[str],
    wf_generator: Any,
    psd: FrequencySeries,
    tolerance: float = 0.01,
    **inner_prod_kw_args
) -> np.ndarray:
    # Right now only for derivatives of Fisher value

    if type(params_to_vary) == str:
        params_to_vary = [params_to_vary]
    # TODO: check if parameters like phase in there? Where computation is not just applying inner_product

    param_numb = len(params_to_vary)

    fisher_matrix = np.zeros((param_numb, param_numb))
    # fisher_matrix = u.Quantity(np.zeros((param_numb, param_numb)))
    # TODO: check if it has unit


    # Compute relevant derivatives in frequency domain
    deriv_series_storage = {}

    for param in params_to_vary:
        derivative_vals = [np.inf]  # Make sure first difference is too large
        deriv_norms = [np.inf]

        # for relative_step_size in [0.01, 0.001, 0.0001]:  # Removed 0.1, only adds unnecessary operations
        for relative_step_size in [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:  # Maybe better than jumping to very small value quickly (numerical errors)
            deriv_param = get_waveform_derivative_1D(
                wf_params_at_point,
                param,
                wf_generator,
                relative_step_size
            )

            # Check for convergence -> maybe bad idea? Due to additional operations
            derivative_norm = norm(deriv_param, psd, **inner_prod_kw_args)
            if 'optimize_time_and_phase' in inner_prod_kw_args.keys():
                derivative_norm = derivative_norm[1]

            # if np.abs((derivative_norm - derivative_vals[-1]) / derivative_norm) <= tolerance:
            #     break
            # else:
            #     derivative_vals += [deriv_param]

            # logging.info(relative_step_size)
                
            
            derivative_vals += [deriv_param]
            deriv_norms += [derivative_norm]

            if (relative_deviation:=np.abs((derivative_norm - deriv_norms[-2]) / derivative_norm)) <= tolerance:
                # -2 because derivativenorm is deriv_norm[-1]
                break
            
            # logging.info(relative_deviation)
            # logging.info(np.abs((deriv_norms[-1] - deriv_norms[-2]) / derivative_norm))
            # Equal

        if relative_deviation > tolerance:
        # TODO: check if this works. Otherwise just compute difference again, i.e. do
        # if (np.abs((derivative_vals[-1] - derivative_vals[-2]) / derivative_norm)) <= tolerance:
        # -> Update: does work, checked via logging.info calls above
            logging.info(
                f'Desired relative accuracy of {tolerance} could not be reached '
                f'for parameter combination {param_i}, {param_j}'
                'Best estimate at step size 0.0001 had relative difference of '
                f'{np.abs((derivative_vals[-2] - derivative_vals[-1]) / derivative_norm):.4f} '
                'to estimate with previous step size at 0.001.'
            )
        
        # logging.debug(np.array(deriv_norms)**2)

        deriv_series_storage[param] = derivative_vals[-1]
    

    # Populate Fisher matrix
    for i, param_i in enumerate(params_to_vary):
        for j, param_j in enumerate(params_to_vary):
            fisher_val = inner_product(deriv_series_storage[param_i], deriv_series_storage[param_j], psd, **inner_prod_kw_args)
            # TODO: add optimize=True as default? Do we need that?

            if 'optimize_time_and_phase' in inner_prod_kw_args.keys():
                fisher_val = fisher_val[1]

            # fisher_matrix[i, j] = fisher_matrix[j, i] = fisher_val
            fisher_matrix[i, j] = fisher_val
            fisher_matrix[j, i] = fisher_val


    return fisher_matrix


# Apply five-point-stencil for certain step size
def get_waveform_derivative_1D(
    wf_params_at_point: dict[str, u.Quantity],
    param_to_vary: str,
    wf_generator: Any,
    relative_step_size: float,
) -> FrequencySeries:
    param_center_val = wf_params_at_point[param_to_vary]
    # TODO cover case where param_center_val = 0 -> set to 1 maybe? Otherwise step_size always zero

    step_size = relative_step_size * param_center_val
    
    param_vals = param_center_val + np.array([-2.0, -1.0, 1.0, 2.0]) * step_size

    waveforms = [
        wfm.GenerateFDWaveform(
            wf_params_at_point | {param_to_vary: param_val},
            wf_generator
        )[0] for param_val in param_vals
    ]

    # deriv_series = waveforms[0].copy()
    # deriv_series -= 8.0 * waveforms[1]
    # deriv_series += 8.0 * waveforms[2]
    # deriv_series -= waveforms[3]

    deriv_series = waveforms[0] - 8.0 * waveforms[1] + 8.0 * waveforms[2] - waveforms[3]
    # Python interpreter probably does this anyway. But maybe no copy is somehow more efficient
    # -> ok, wow; this is about 30 times faster!!!

    deriv_series /= 12.0 * step_size

    return deriv_series
    




def fisher_val_at_point(  # TODO: rename to fisher_val?
    wf_params_at_point: dict[str, u.Quantity],
    param_to_vary: str,
    wf_generator: Any,
    psd: FrequencySeries,
    tolerance: float = 0.01
) -> float | u.Quantity:  # TODO: decide here
    # Right now only for derivatives of Fisher value
    
    param_center_val = wf_params_at_point[param_to_vary]

    derivative_vals = [np.inf]  # Make sure first difference is too large

    # for relative_step_size in [0.1, 0.01, 0.001, 0.0001]:
    for relative_step_size in [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:  # Maybe better than jumping to very small value quickly (numerical errors)
        # param_vals = param_center_val + u.Quantity(np.arange(-2.0, 2.1, step=1.0) * relative_step_size, unit=param_center_val.unit)
        # param_vals = param_center_val + u.Quantity(np.array([-2.0, -1.0, 1.0, 2.0]) * relative_step_size, unit=param_center_val.unit)
        # step_size = relative_step_size * param_center_val

        # param_vals = param_center_val + np.array([-2.0, -1.0, 1.0, 2.0]) * step_size

        # waveforms = [
        #     wfm.GenerateFDWaveform(
        #         wf_params_at_point | {param_to_vary: param_val},
        #         wf_generator
        #     )[0] for param_val in param_vals
        # ]

        # # print(np.asanyarray(waveforms, dtype=FrequencySeries))
            
        # # print(num_diff(waveforms, h=step_size))

        # # temp_deriv_series = FrequencySeries(
        # #     np.zeros(waveforms[2].size),
        # #     frequencies=waveforms[2].frequencies
        # # )


        # # deriv_series = waveforms[0].copy()
        # # deriv_series -= 8.0 * waveforms[1]
        # # deriv_series += 8.0 * waveforms[2]
        # # deriv_series -= waveforms[3]
    
        # deriv_series = waveforms[0] - 8.0 * waveforms[1] + 8.0 * waveforms[2] - waveforms[3]

        # deriv_series /= 12.0 * step_size

        # deriv_series += num_diff(waveforms, h=step_size)


        deriv_series = get_waveform_derivative_1D(
            wf_params_at_point,
            param_to_vary,
            wf_generator,
            relative_step_size
        )

        # fisher_val = norm(deriv_series, psd)**2
        fisher_val = inner_product(deriv_series, deriv_series, psd)  # Changes nothing, as it should be

        # logging.info(relative_step_size)
        
        # logging.info(np.abs((fisher_val - derivative_vals[-1]) / fisher_val) <= tolerance)
        # Check for convergence
        # if np.abs((fisher_val - derivative_vals[-1]) / fisher_val) <= tolerance:
        #     # logging.info(derivative_vals)
        #     return fisher_val
        # else:
        #     derivative_vals += [fisher_val]


        derivative_vals += [fisher_val]
        if np.abs((fisher_val - derivative_vals[-2]) / fisher_val) <= tolerance:
            # -2 because derivativenorm is deriv_norm[-1]
            break

    # logging.debug(derivative_vals)
    
    logging.info(
        f'Desired relative accuracy of {tolerance} could not be reached. '
        'Best estimate at step size 0.0001 had relative difference of '
        f'{np.abs((derivative_vals[-2] - derivative_vals[-1]) / derivative_vals[-1]):.4f} '
        'to estimate with previous step size at 0.001.'
    )

    return fisher_val