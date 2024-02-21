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
    signal : ~gwpy.types.series.Series or numpy.ndarray
        Input to compute derivative for.
    h : float or ~astropy.units.Quantity, optional, default = None
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


    signal_deriv = np.roll(signal, 2) - 8.0 * np.roll(signal, 1) + 8.0 * np.roll(signal, -1) - np.roll(signal, -2)

    signal_deriv /= 12.0 * h

    signal_deriv[0] = (signal[1] - signal[0]) / h  # Forward difference
    signal_deriv[1] = (signal[2] - signal[0]) / (2.0 * h)  # Central difference

    signal_deriv[-2] = (signal[-1] - signal[-3]) / (2.0 * h)  # Central difference
    signal_deriv[-1] = (signal[-1] - signal[-2]) / h  # Backward difference

    return signal_deriv


def fisher_matrix(
    wf_params_at_point: dict[str, u.Quantity],
    params_to_vary: str | list[str],
    wf_generator: Any,
    psd: Optional[FrequencySeries] = None,
    # # tolerance: float = 0.01,
    # tolerance: float = 1.0,  # Role is now taken by convergence_threshold
    step_sizes: Optional[list[float]] = None,
    convergence_check: Optional[Literal['diff_norm', 'mismatch']] = None,
    break_upon_convergence: bool = False,
    convergence_threshold: float = None,  # Rename to threshold?
    return_info: bool = False,
    **inner_prod_kwargs
) -> np.ndarray:
    """
    Compute Fisher matrix at a fixed point.

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

        stem is adapted from Python class implemented by Niko Sarcevic, whose
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
        step_sizes = np.reshape(np.outer([1e-2, 1e-3, 1e-4, 1e-5], [5, 1]), -1)  # Seems most reasonable choice at the moment -> testing showed that 1e-6 might be too small for good results :OO
        # step_sizes = np.reshape(np.outer([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], [5, 1]), -1)  # For more detailed testing of convergence

    if convergence_check is None:
        convergence_check = 'diff_norm'
    else:
        if convergence_check not in ['mismatch', 'diff_norm']:
            raise ValueError(
                    'Invalid value for `convergence_check`.'
                )


    # fisher_matrix = np.zeros((param_numb, param_numb))
    # fisher_matrix = u.Quantity(np.zeros((param_numb, param_numb)))
    # fisher_matrix = u.Quantity(np.zeros((param_numb, param_numb)), dtype=Any)
    # fisher_matrix = u.Quantity(np.zeros((param_numb, param_numb)), u.StructuredUnit())
    # TODO: check if it has unit -> would have to add one in inner_product output
    # fisher_matrix = [[0.0 for _ in range(param_numb)] for _ in range(param_numb)]
    fisher_matrix = np.zeros((param_numb, param_numb), dtype=object)



    # Compute relevant derivatives in frequency domain
    deriv_series_storage = {}

    for i, param in enumerate(params_to_vary):
        deriv_series_storage[param], info = get_waveform_derivative_1D_with_convergence(
            wf_params_at_point=wf_params_at_point,
            params_to_vary=params_to_vary,
            wf_generator=wf_generator,
            psd=psd,
            step_sizes=step_sizes,
            convergence_check=convergence_check,
            break_upon_convergence=break_upon_convergence,
            convergence_threshold=convergence_threshold,
            return_info=True,
            **inner_prod_kwargs
        )
        fisher_matrix[i, i] = info['norm']**2
        # fisher_matrix[i][i] = info['norm']**2

        # TODO: decide if norm also returned or newly computed here -> e.g. from info_dict
    

    # Populate Fisher matrix
    for i, param_i in enumerate(params_to_vary):
        for j, param_j in enumerate(params_to_vary):

            if i == j:
                continue
            else:
                fisher_val = inner_product(deriv_series_storage[param_i], deriv_series_storage[param_j], psd, **inner_prod_kwargs)
                # TODO: add optimize=True as default? Do we need that? -> nope, should not be used (ideally), has physical implications

                if 'optimize_time_and_phase' in inner_prod_kwargs.keys():
                    fisher_val = fisher_val[1]
                fisher_matrix[i, j] = fisher_matrix[j, i] = fisher_val
                # fisher_matrix[i][j] = fisher_matrix[j][i] = fisher_val

    return fisher_matrix


def get_waveform_derivative_1D_with_convergence(
    wf_params_at_point: dict[str, u.Quantity],
    param_to_vary: str,
    wf_generator: Any,
    psd: Optional[FrequencySeries] = None,
    step_sizes: Optional[list[float]] = None,
    convergence_check: Optional[Literal['diff_norm', 'mismatch']] = None,
    break_upon_convergence: bool = True,
    convergence_threshold: float = None,  # Rename to threshold?
    return_info: bool = False,
    **inner_prod_kwargs
) -> FrequencySeries | tuple[FrequencySeries, u.Quantity]:
    
    if psd is None:
        from .PSDs import psd_no_noise

        psd = psd_no_noise


    if convergence_check is None:
        convergence_check = 'diff_norm'
    else:
        if convergence_check not in ['mismatch', 'diff_norm']:
            raise ValueError(
                    'Invalid value for `convergence_check`.'
                )

    if convergence_threshold is None:
        match convergence_check:
            case 'diff_norm':
                convergence_threshold = 0.01
            case 'mismatch':
                convergence_threshold = 0.01  # Maybe choose 0.03? Or 0.001?
                # convergence_threshold = 0.001
                

    is_converged = False

    refine_numb = 0
    for _ in range(2):  # Number of refinements of step size
        derivative_vals = []
        deriv_norms = []
        convergence_vals = []

        for i, relative_step_size in enumerate(step_sizes):  # Maybe better than jumping to very small value quickly (numerical errors)
            deriv_param = get_waveform_derivative_1D(
                wf_params_at_point,
                param_to_vary,
                wf_generator,
                relative_step_size
            )

            derivative_norm = norm(deriv_param, psd, **inner_prod_kwargs)
            if 'optimize_time_and_phase' in inner_prod_kwargs.keys():
                derivative_norm = derivative_norm[1]**2
            else:
                derivative_norm **= 2

            derivative_vals += [deriv_param]
            deriv_norms += [derivative_norm]

            logging.debug(derivative_vals)
            logging.debug(deriv_norms)


            match convergence_check:
                case 'diff_norm':
                    if len(derivative_vals) >= 2:
                        # convergence_vals += [norm(deriv_param - derivative_vals[-2], psd, **inner_prod_kw_args)]
                        # convergence_vals += [norm(deriv_param - derivative_vals[-2], psd, **inner_prod_kw_args) / np.sqrt(derivative_norm * deriv_norms[-2])]  # Index -1 is deriv_param
                        convergence_vals += [norm(deriv_param - derivative_vals[-2], psd, **inner_prod_kwargs) / np.sqrt(derivative_norm)]  # Index -1 is deriv_param
                    else:
                        continue
                case 'mismatch':
                    # Compute mismatch, using that we already know norms
                    if len(derivative_vals) >= 2:
                        convergence_vals += [1.0 - inner_product(deriv_param, derivative_vals[-2], psd, **inner_prod_kwargs) / np.sqrt(derivative_norm * deriv_norms[-2])]  # Index -1 is deriv_param
                    else:
                        continue


            logging.info(convergence_vals)
            logging.info([len(derivative_vals), i, len(convergence_vals)])


            if (len(convergence_vals) >= 2
                and (convergence_vals[-2] <= convergence_threshold)
                and (convergence_vals[-1] <= convergence_threshold)):
                is_converged = True  # Remains true, is never set to False again

                if break_upon_convergence:
                    # min_dev_index = -1
                    # min_dev_index = len(convergence_vals) - 1  # Then it can also be used to access step_sizes
                    # -> uhhh, problem with this one: convergence_vals is not as long as other lists!!!
                    # -> so either take i, as done below, or len(derivative_vals)
                    # -> other idea: make first element nan?
                    min_dev_index = i  # Then it can also be used to access step_sizes
                    break
        

        # Check if convergence was reached using these step sizes, refine them if not
        if (not break_upon_convergence or not is_converged) and (convergence_check != 'stem'):
            min_dev_index = np.nanargmin(convergence_vals)  # Should not have nan, but who knows
            # TODO: rename min_dev_index
            # step_sizes = (1.0 + np.array([-0.05, -0.01, 0.01, 0.05])) * step_sizes[min_dev_index]

            # Cut steps made around step size with best criterion value in half
            # compared to current steps (we take average step size in case
            # difference to left and right is unequl)
            # new_step = (step_sizes[min_dev_index + 1] - step_sizes[min_dev_index - 1]) / 8.0  # Due to factor of two below
            # step_sizes = step_sizes[min_dev_index] + np.array([-2.0, -1.0, 1.0, 2.0]) * new_step

            left_step = (step_sizes[min_dev_index - 1] - step_sizes[min_dev_index]) / 4.0
            right_step = (step_sizes[min_dev_index + 1] - step_sizes[min_dev_index]) / 4.0
            # 4.0 due to factor of two below
            step_sizes = step_sizes[min_dev_index] + np.array(
                [2.0 * left_step, 1.0 * left_step, 1.0 * right_step, 2.0 * right_step]
            )

            refine_numb += 1
        else:
            # TODO: check if this one makes sense
            break

    # logging.info(min_dev_index)

    if not is_converged:
        # TODO: definitely make remark that despite trying very hard, no convergence was possible. Consider changing starting step sizes
        logging.info(
            'Calculations using the selected relative step sizes '
            f'did not converge for parameter `{param_to_vary}` using convergence '
            f'check method `{convergence_check}`. Last value of criterion '  # Best instead of last?
            f'was {convergence_vals[-1]}, which is above '  # convergence_vals[min_dev_index]?
            f'threshold of {convergence_threshold}.'
        )
    

    if return_info:
        # from . import PLOT_STYLE_SHEET
        # plt.style.use(PLOT_STYLE_SHEET)
        # Should rather be set by user

        fig, ax = plt.subplots()

        for i in range(len(derivative_vals)):
            ax.plot(derivative_vals[i], '--', label=f'{step_sizes[i]}')

        # plt.legend(title='Step Sizes', ncols=2 if len(derivative_vals) > 3 else 1)
        ax.legend(title='Step Sizes', ncols=max(1, len(derivative_vals) % 3))  # Maybe even 4?

        ax.set_xlabel('$f$')
        ax.set_ylabel('Derivative')
        ax.set_title(f'Parameter: {param_to_vary}')

        # plt.close()

        return derivative_vals[min_dev_index], {
            'norm': deriv_norms[min_dev_index],
            'final_step_size': step_sizes[min_dev_index],
            'final_convergence_val': convergence_vals[min_dev_index - 1 if min_dev_index != -1 else -1],
            'number_of_refinements': refine_numb,
            'final_set_of_rel_step_sizes': step_sizes,
            'convergence_plot': ax
        }
    else:
        return derivative_vals[min_dev_index]


def get_waveform_derivative_1D(
    wf_params_at_point: dict[str, u.Quantity],
    param_to_vary: str,
    wf_generator: Any,
    step_size: float
) -> FrequencySeries:
    """
    Use five-point stencil method to calculate numerical derivatives
    with respect to a waveform parameter.

    Parameters
    ----------
    wf_params_at_point : dict[str, u.Quantity]
        Dictionary with parameters that determine point at which
        derivative is calculated.
    param_to_vary : str
        Derivative is taken with respect to this parameter (has to be
        key passed to waveform generators).
    wf_generator : Any
        Instance of `~lalsimulation.gwsignal.core.waveform.
        LALCompactBinaryCoalescenceGenerator` class that is used for
        waveform generation.
    step_size : float
        Step size used in numerical differentiation.

    Returns
    -------
    FrequencySeries
        Derivative in frequency domain.
    """

    param_center_val = wf_params_at_point[param_to_vary]
    
    # TODO: ensure that f_min and f_max are given, otherwise potentially inconsistent waveform lengths, right?

    step_size = step_size * param_center_val.unit    
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

    # # deriv_series = restrict_f_range(deriv_series, f_range=[f_min / 0.9, None])


    # Just for fun central difference
    # -> WOW, results are pretty good, actually; I had thought that maybe taking
    # all these difference may cause issue with floating point error, but
    # five point stencil is still more accurate (although more calculations involved)
    # -> numdifftools also uses central method, so it is natural that they have
    # comparable accuracy, but five point stencil is not significantly more
    # accurate! So we could save operations and thus e.g. go to lower
    # thresholds when using central difference
    # param_vals = param_center_val + np.array([-1.0, 1.0]) * step_size

    # waveforms = [
    #     wfm.GenerateFDWaveform(
    #         wf_params_at_point | {param_to_vary: param_val},
    #         # wf_params_at_point | {param_to_vary: param_val, 'f22_start': f_min},
    #         wf_generator
    #     )[0] for param_val in param_vals
    # ]
    # # TODO: catch ValueError: Input domain error? This is caused if we
    # # cross bounds of parameter. In that case we could automatically
    # # decrease step size... On the other hand, this is again something
    # # that happens under the hood, not necessarily good idea...

    # deriv_series = waveforms[1] - waveforms[0]

    # deriv_series /= 2.0 * step_size

    return deriv_series


def fisher_val_at_point(  # TODO: rename to fisher_val?
    wf_params_at_point: dict[str, u.Quantity],
    param_to_vary: str,
    wf_generator: Any,
    **deriv_kwargs
) -> float | u.Quantity:  # TODO: decide here
    
    _, info = get_waveform_derivative_1D_with_convergence(
        wf_params_at_point,
        param_to_vary,
        wf_generator,
        return_info=True,
        **deriv_kwargs
    )

    fisher_val = info['norm']

    return fisher_val
