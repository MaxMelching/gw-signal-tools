import logging
import warnings
from typing import Optional, Any, Literal

import numpy as np
import matplotlib.pyplot as plt

from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries
import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm

from .inner_product import inner_product, inner_product_computation, norm
from .waveform_utils import restrict_f_range


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
        elif not isinstance(h, u.Quantity):
            h = u.Quantity(h, signal.xindex.unit)

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


    # signal_deriv = signal.copy()
    # signal_deriv.fill(0.0)

    # signal_deriv += np.roll(signal, 2)
    # # signal_deriv[0] -= signal[-2]
    # # signal_deriv[1] -= signal[-1]

    # signal_deriv -= 8.0 * np.roll(signal, 1)
    # # signal_deriv[0] += 8.0 * signal[-1]

    # signal_deriv += 8.0 * np.roll(signal, -1)
    # # signal_deriv[-1] -= 8.0 * signal[0]

    # signal_deriv -= np.roll(signal, -2)
    # # signal_deriv[-1] += signal[1]
    # # signal_deriv[-2] += signal[0]

    # # signal_deriv[0] = signal_deriv[1] = signal_deriv[2]
    # # signal_deriv[-1] = signal_deriv[-2] = signal_deriv[-3]

    # Having learned from other derivative, here is more efficient implementation
    signal_deriv = np.roll(signal, 2) - 8.0 * np.roll(signal, 1) + 8.0 * np.roll(signal, -1) - np.roll(signal, -2)

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
    psd: Optional[FrequencySeries] = None,
    # # tolerance: float = 0.01,
    # tolerance: float = 1.0,  # Role is now taken by convergence_threshold
    step_sizes: Optional[list[float]] = None,
    convergence_check: Optional[Literal['rel_deviation', 'stem', 'mismatch', 'diff_norm']] = None,
    convergence_threshold: float = None,  # Rename to threshold?
    convergence_plot: bool = False,
    **inner_prod_kw_args
) -> np.ndarray:
    """
    _summary_

    Parameters
    ----------
    wf_params_at_point : dict[str, u.Quantity]
        _description_
    params_to_vary : str | list[str]
        _description_
    wf_generator : Any
        _description_
    psd : Optional[FrequencySeries], optional
        _description_, by default None
    step_sizes : Optional[list[float]], optional
        _description_, by default None

        If None, values in the range 5e-2 to 1e-5 are taken
    convergence_check : Optional[Literal['rel_deviation', 'stem', 'mismatch', 'diff_norm']], optional
        _description_, by default None

        stem is adapted from Python class implemented by Niko Sacevic, whose
        work was based on following paper: https://arxiv.org/abs/1606.03451 (see appendix)

        diff_norm means we do not loose more than threshold percent of SNR
        between consecutive derivative calculations
    convergence_threshold : float, optional
        _description_, by default None

    Returns
    -------
    np.ndarray
        _description_
    """

    if isinstance(params_to_vary, str):
        params_to_vary = [params_to_vary]
    # TODO: check if parameters like phase in there? Where computation is not just applying inner_product

    param_numb = len(params_to_vary)


    if psd is None:
        from .PSDs import psd_no_noise

        psd = psd_no_noise


    if step_sizes is None:
        # step_sizes = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        # step_sizes = np.reshape(np.outer([1e-2, 1e-4, 1e-6], [5, 2, 1]), -1)
        # step_sizes = np.reshape(np.outer([1e-2, 1e-3, 1e-5, 1e-6], [5, 1]), -1)
        step_sizes = np.reshape(np.outer([1e-2, 1e-3, 1e-4, 1e-5], [5, 1]), -1)  # Seems most reasonable choice at the moment -> testing showed that 1e-6 might be too small for good results :OO
        # step_sizes = np.reshape(np.outer([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], [5, 1]), -1)  # For more detailed testing of convergence

    if convergence_check is None:
        convergence_check = 'rel_deviation'
    else:
        if convergence_check not in ['rel_deviation', 'stem', 'mismatch', 'diff_norm']:
            raise ValueError(
                    'Invalid value for `convergence_check`.'
                )

    if convergence_threshold is None:
        match convergence_check:
            case 'rel_deviation':
                convergence_threshold = 0.01
                # convergence_threshold = 0.005
            case 'stem':
                # convergence_threshold = 0.005  # For linear fit method
                # convergence_threshold = 0.01  # For mean method
                convergence_threshold = 0.001  # For mean method -> seems very small, maybe only suitable for chirp mass
            case 'mismatch':
                convergence_threshold = 0.01  # Maybe choose 0.03? Or 0.001?
                # convergence_threshold = 0.001
            case 'diff_norm':
                convergence_threshold = 1.0  # TODO: check this one
            # case _:
            #     raise ValueError(
            #         'Invalid value for `convergence_check`.'
            #     )
    


    fisher_matrix = np.zeros((param_numb, param_numb))
    # fisher_matrix = u.Quantity(np.zeros((param_numb, param_numb)))
    # TODO: check if it has unit -> would have to add one in inner_product output


    # Compute relevant derivatives in frequency domain
    deriv_series_storage = {}

    for i, param in enumerate(params_to_vary):
        derivative_vals = [np.array(np.inf)]  # Make sure first difference is too large
        deriv_norms = [np.inf]
        convergence_val = np.inf
        is_converged = False
        # TODO: better names for parameters in this for loop

        # TODO: store convergence values and then, in case we do not reach threshold,
        # take the one where convergence value was minimal -> idea: handle this via
        # argument break_upon_convergence? Which determines whether or not to
        # break once convergence value is smaller than threshold for the first
        # time (this would be faster, but we might miss out on slightly better
        # result that is encountered later on)

        for relative_step_size in step_sizes:  # Maybe better than jumping to very small value quickly (numerical errors)
            deriv_param = get_waveform_derivative_1D(
                wf_params_at_point,
                param,
                wf_generator,
                relative_step_size
            )

            derivative_norm = norm(deriv_param, psd, **inner_prod_kw_args)
            if 'optimize_time_and_phase' in inner_prod_kw_args.keys():
                derivative_norm = derivative_norm[1]**2
            else:
                derivative_norm **= 2

            derivative_vals += [deriv_param]
            deriv_norms += [derivative_norm]
            logging.debug(derivative_vals)
            logging.debug(deriv_norms)

            # logging.debug(deriv_difference.frequencies)
            # logging.debug(psd.frequencies)


            match convergence_check:
                case 'rel_deviation':
                    # convergence_val = np.abs(derivative_norm - deriv_norms[-2])
                    # convergence_val = np.abs((derivative_norm - deriv_norms[-2]) / derivative_norm)
                    convergence_val = np.abs((derivative_norm - deriv_norms[-2]) / derivative_norm)
                # case 'stem':
                #     ...   # TODO: implement -> ah, nothing to do in loop
                case 'mismatch':
                    # Compute mismatch, using that we already know norms
                    if len(derivative_vals) > 2:
                        convergence_val = 1.0 - inner_product(deriv_param, derivative_vals[-2], psd, **inner_prod_kw_args) / np.sqrt(derivative_norm * deriv_norms[-2])  # Index -1 is deriv_param
                    else:
                        convergence_val = np.inf
                case 'diff_norm':
                    # convergence_val = norm(deriv_param - derivative_vals[-2], psd, **inner_prod_kw_args)
                    # convergence_val = norm(deriv_param - derivative_vals[-2], psd, **inner_prod_kw_args) / np.sqrt(derivative_norm * deriv_norms[-2])  # Index -1 is deriv_param
                    convergence_val = norm(deriv_param - derivative_vals[-2], psd, **inner_prod_kw_args) / np.sqrt(derivative_norm)  # Index -1 is deriv_param


            logging.info(convergence_val)                
            

            if convergence_val <= convergence_threshold:
                is_converged = True
                break
        
        # Remove np.inf from beginning
        deriv_norms.pop(0)
        derivative_vals.pop(0)
        
        # if convergence_check == 'stem':
        #     # fit_vals_x = step_sizes.copy()
        #     fit_vals_x = list(step_sizes.copy())
        #     fit_vals_y = deriv_norms.copy()  # Do we even need to copy? Do not need deriv_norms afterwards anymore -> need for convergence_plots
        #     # fit_vals_y = deriv_norms

        #     while len(fit_vals_y) > 1:
        #         # slope, intercept, residuals, _, _, _ = np.polyfit(np.log10(fit_vals_x), fit_vals, 1, full=True)
        #         slope, intercept = np.polyfit(np.log10(fit_vals_x), fit_vals_y, 1, full=False)
        #         # intercept = Fisher value, slope = change in Fisher value

        #         # deviations = (intercept - fit_vals)**2 / fit_vals
        #         deviations = np.abs(intercept - fit_vals_y) / fit_vals_y

        #         max_dev_index, min_dev_index = np.argmax(deviations), np.argmin(deviations)


        #         fit_vals_x.pop(max_dev_index)
        #         fit_vals_y.pop(max_dev_index)
        #         # convergence_val = deviations[min_dev_index]  # Hmmm, or slope?
        #         convergence_val = slope

        #         logging.info(fit_vals_y)
        #         logging.info(deviations)

        #         # if slope < convergence_threshold:
        #         if convergence_val < convergence_threshold:
        #             is_converged = True
        #             break
            
        #     logging.info(convergence_val)

        #     # Last values are not necessarily best fit here, so they have to
        #     # be set manually
        #     derivative_vals = [derivative_vals[min_dev_index]]
        #     derivative_norm = deriv_norms[min_dev_index]

        # PROBLEM: does not work so well because values close to x=0 are
        # more similar to intercept value, not matter how well it fits
        # together with all other values
        
        # if convergence_check == 'stem':
        #     fit_vals = deriv_norms.copy()  # Do we even need to copy? Do not need deriv_norms afterwards anymore -> need for convergence_plots
        #     # fit_vals = deriv_norms

        #     while len(fit_vals) > 1:
        #         mean_val = np.mean(fit_vals)

        #         deviations = np.abs(mean_val - fit_vals) / fit_vals

        #         max_dev_index, min_dev_index = np.argmax(deviations), np.argmin(deviations)

        #         fit_vals.pop(max_dev_index)
        #         fit_vals.pop(max_dev_index)
        #         convergence_val = deviations[min_dev_index]

        #         logging.info(fit_vals)
        #         logging.info(mean_val)
        #         logging.info(deviations)

        #         # if slope < convergence_threshold:
        #         if convergence_val < convergence_threshold:
        #             is_converged = True
        #             break
            
        #     logging.info(convergence_val)

        #     # Last values are not necessarily best fit here, so they have to
        #     # be set manually
        #     derivative_vals = [derivative_vals[min_dev_index]]
        #     derivative_norm = deriv_norms[min_dev_index]
        #     # -> ah shit... Problem is that min_dev_index is with respect to the popped deriv_norms...
        
        if convergence_check == 'stem':
            use_mask = len(deriv_norms) * [True]
            deriv_norms = np.asarray(deriv_norms)
            fit_vals = deriv_norms[use_mask]

            while len(fit_vals) > 1:
                mean_val = np.mean(fit_vals)

                # deviations = np.abs(mean_val - fit_vals) / fit_vals
                # deviations = np.abs(mean_val - deriv_norms) / deriv_norms
                deviations = (mean_val - deriv_norms)**2 / deriv_norms

            
                # deviations[np.logical_not(use_mask)] = 0.0  # Avoid index chaos

                # # max_dev_index, min_dev_index = np.argmax(deviations), np.argmin(deviations)

                # max_dev_index = np.argmax(deviations)

                # deviations[np.logical_not(use_mask)] = np.inf  # Avoid index chaos
                # min_dev_index = np.argmin(deviations)


                deviations[np.logical_not(use_mask)] = np.nan  # Avoid index chaos while ignoring excluded points
                max_dev_index, min_dev_index = np.nanargmax(deviations), np.nanargmin(deviations)


                use_mask[max_dev_index] = False
                # use_mask[use_mask == True][max_dev_index] = False
                fit_vals = deriv_norms[use_mask]  # Correct here?
                convergence_val = deviations[min_dev_index]

                logging.info(fit_vals)
                logging.info(mean_val)
                logging.info(deviations)
                logging.info(min_dev_index)

                # # if slope < convergence_threshold:
                # if convergence_val < convergence_threshold:
                #     is_converged = True
                #     break
            
            if convergence_val < convergence_threshold:
                is_converged = True
            
            logging.info(convergence_val)

            # Last values are not necessarily best fit here, so they have to
            # be set manually
            derivative_vals = [derivative_vals[min_dev_index]]
            derivative_norm = deriv_norms[min_dev_index]



        if not is_converged:
            # logging.info(
            #     f'Desired relative accuracy of {convergence_threshold} could not be reached '
            #     f'for parameter combination {param}, {param}'
            #     'Best estimate at step size 0.0001 had relative difference of '
            #     f'{difference_norm:.4f} '
            #     'to estimate with previous step size at 0.001.'
            # )

            logging.info(
                'Calculations using the selected relative step sizes '
                f'did not converge for parameter `{param}` using convergence '
                f'check method `{convergence_check}`. Last value of criterion '
                f'was {convergence_val}, which is above threshold of '
                f'{convergence_threshold}.'
            )
        
        # logging.debug(np.array(deriv_norms)**2)

        deriv_series_storage[param] = derivative_vals[-1]
        fisher_matrix[i, i] = derivative_norm


        if convergence_plot:
            plt.figure(figsize=(13,12))

            plt.plot(step_sizes[:len(deriv_norms)], deriv_norms, 'x-')#, label='Evolution of\nFisher Values')

            plt.plot([step_sizes[0], step_sizes[-1]], 2 * [derivative_norm], 'r--', label='Result')

            plt.legend()
            plt.xlim([1.1 * max(step_sizes), 0.9 * min(step_sizes)])
            plt.xscale('log')

            # Enable log scale on y-axis in case more than one order of magnitude difference
            if abs(np.max(deriv_norms) / min(deriv_norms)) > 10:
                plt.yscale('log')

            plt.xlabel('Relative Step Size')
            plt.ylabel('Diagonal Fisher Matrix Entry')
            plt.title(f'Parameter: {param}')

            plt.show()
    

    # Populate Fisher matrix
    for i, param_i in enumerate(params_to_vary):
        for j, param_j in enumerate(params_to_vary):

            if i == j:
                # fisher_matrix[i, i] = deriv_norms[i]
                continue
            else:
                fisher_val = inner_product(deriv_series_storage[param_i], deriv_series_storage[param_j], psd, **inner_prod_kw_args)
                # TODO: add optimize=True as default? Do we need that? -> nope, should not be used (ideally), has physical implications

                if 'optimize_time_and_phase' in inner_prod_kw_args.keys():
                    fisher_val = fisher_val[1]
                fisher_matrix[i, j] = fisher_matrix[j, i] = fisher_val


    # TODO: decide if this is good idea, would potentially mess up functions that acces indices
    # if len(params_to_vary) == 1:
    #     return fisher_matrix[0, 0]
    # else:
    #     return fisher_matrix
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

    step_size = relative_step_size * param_center_val if param_center_val != 0.0 else relative_step_size
    
    param_vals = param_center_val + np.array([-2.0, -1.0, 1.0, 2.0]) * step_size

    # if 'f22_start' in wf_params_at_point:
    #     f_min = 0.9 * wf_params_at_point['f22_start']
    # else:
    #     f_min = 0.9 * 20.*u.Hz  # Default value

    waveforms = [
        wfm.GenerateFDWaveform(
            wf_params_at_point | {param_to_vary: param_val},
            # wf_params_at_point | {param_to_vary: param_val, 'f22_start': f_min},
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

    # deriv_series = restrict_f_range(deriv_series, f_range=[f_min / 0.9, None])

    return deriv_series


def fisher_matrix_optimize_over_time_and_phase():
    raise NotImplementedError
    




def fisher_val_at_point(  # TODO: rename to fisher_val?
    wf_params_at_point: dict[str, u.Quantity],
    param_to_vary: str,
    wf_generator: Any,
    psd: FrequencySeries,
    tolerance: float = 0.01
) -> float | u.Quantity:  # TODO: decide here
    # Right now only for derivatives of Fisher value
    
    # param_center_val = wf_params_at_point[param_to_vary]

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

        # logging.debug(relative_step_size)
        
        # logging.debug(np.abs((fisher_val - derivative_vals[-1]) / fisher_val) <= tolerance)
        # Check for convergence
        # if np.abs((fisher_val - derivative_vals[-1]) / fisher_val) <= tolerance:
        #     # logging.info(derivative_vals)
        #     return fisher_val
        # else:
        #     derivative_vals += [fisher_val]


        derivative_vals += [fisher_val]
        if (relative_deviation:=np.abs((fisher_val - derivative_vals[-2]) / fisher_val)) <= tolerance:
            # -2 because derivativenorm is deriv_norm[-1]
            break

    # logging.debug(derivative_vals)
    if relative_deviation > tolerance:
        logging.info(
            f'Desired relative accuracy of {tolerance} could not be reached. '
            'Best estimate at step size 0.0001 had relative difference of '
            f'{np.abs((derivative_vals[-2] - derivative_vals[-1]) / derivative_vals[-1]):.4f} '
            'to estimate with previous step size at 0.001.'
        )

    return fisher_val


class FisherMatrix():
    def __init__(self):
        ...
    """
    Idea: we could store Fisher matrix and inverse in here and also
    diagonal version, from which we would get condition number.
    
    -> maybe just compute Fisher and diagonal; then inform about
    condition number when inverse is called

    -> ah, condition number is norm(A)*norm(A^-1); so computing inverse
    would also be just fine
    """


def fisher_inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)

def condition_number(matrix: np.ndarray, inverse: np.ndarray) -> float:
    return np.linalg.norm(matrix) * np.linalg.norm(inverse)