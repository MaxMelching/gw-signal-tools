# ----- Standard Lib Imports -----
import logging
import warnings
from typing import Optional, Any, Literal, Callable

# ----- Third Party Imports -----
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries
import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm

# ----- Local Package Imports -----
from .inner_product import inner_product, norm
from .matrix_with_units import MatrixWithUnits


def num_diff(
    signal: Series | np.ndarray,
    h: Optional[float | u.Quantity] = None
) -> Series | np.ndarray:
    """
    Implementation of five-point stencil method for numerical
    differentiation for numpy arrays and instances of GWpy Series class
    (which includes instances of TimeSeries, FrequencySeries). The
    differentiation is carried out with respect to the respective
    quantity that spans `signal.xindex`.

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


    signal_deriv = np.roll(signal, 2) - 8.0 * np.roll(signal, 1) + 8.0 * np.roll(signal, -1) - np.roll(signal, -2)

    signal_deriv /= 12.0 * h

    signal_deriv[0] = (signal[1] - signal[0]) / h  # Forward difference
    signal_deriv[1] = (signal[2] - signal[0]) / (2.0 * h)  # Central difference

    signal_deriv[-2] = (signal[-1] - signal[-3]) / (2.0 * h)  # Central difference
    signal_deriv[-1] = (signal[-1] - signal[-2]) / h  # Backward difference

    return signal_deriv


# NOTE: removing some of the arguments to pass all kwargs to derivative does
# not work because we want to be able to pass kwargs to inner_product function
# in fisher_matrix itself
def fisher_matrix(
    wf_params_at_point: dict[str, u.Quantity],
    params_to_vary: str | list[str],
    wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries | ArrayLike],
    step_sizes: Optional[list[float]] = None,
    convergence_check: Optional[Literal['diff_norm', 'mismatch']] = None,
    convergence_threshold: float = None,
    break_upon_convergence: bool = True,
    return_info: bool = False,
    **inner_prod_kwargs
) -> MatrixWithUnits:
    """
    Compute Fisher matrix at a fixed point. To assess the stability of
    the result, this function calculates the involved derivatives for
    several step sizes and compares the results using what we call a
    convergence checker.

    Parameters
    ----------
    wf_params_at_point : dict[str, u.Quantity]
        Point in parameter space at which the Fisher matrix is
        evaluated, encoded as key-value-pairs. Given as input to
        `wf_generator`.
    params_to_vary : str or list[str]
        Parameter(s) with respect to which the derivatives will be
        computed, the norms of which constitute the Fisher matrix.
        Must be keys in `wf_params_at_point`.
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]],
    FrequencySeries or ArrayLike]
        Arbitrary function that is used for waveform generation. The
        required signature means that it has one non-optional argument,
        which is expected to accept the input provided in
        `wf_params_at_point`, while the output is either a ``~gwpy.
        frequencyseries.FrequencySeries`` or of type ``ArrayLike``, so
        that its subtraction is carried out element-wise. The preferred
        type is ``FrequencySeries`` because it supports astropy units
        (and it is the standard output of LAL gwsignal generators).

        A convenient option is to use the method `FisherMatrix.
        get_wf_generator`, which generates a suitable function from
        a few arguments.
    step_sizes : list[float], optional, default = None
        Step sizes that are used in the numerical differention.
    convergence_check : Optional[Literal['diff_norm', 'mismatch']],
    optional, default = None
        Criterion used to asses stability of the result. Currently, two
        are available:
        - diff_norm: calculates the norm of the difference of two
          consecutive derivatives (using the function `~gw_signal_tools.
          inner_product.norm`). This is compared to the norm of the most
          recent derivative and if their fraction is smaller than some
          threshold (specified in `convergence_threshold`), the result
          is taken to be converged because the differences become
          negligible on the relevant scales (provided by the norm of
          the derivative).
        - mismatch: calculates the mismatch between consecutive
          derivatives (also using the function `~gw_signal_tools.
          inner_product.norm`), which is defined as :math:`1 - overlap`.
          Again, the result is taken to be converged if this mismatch
          falls under a certain threshold, provided by
          `convergence_threshold`.
    convergence_threshold : float, optional, default = None
        Threshold that is used to decide if result is converged. This
        will be the case once the value of the criterion specified in
        `convergence_check` is smaller than `convergence_threshold` two
        iterations in a row.
    break_upon_convergence : bool, optional, default = True
        Whether to break upon the convergence described previously
        (difference smaller than given threshold two times in a row) or
        not. If not, results for all step sizes are calculated and the
        one with minimal convergence criterion value is selected.

    Returns
    -------
    gw_signal_tools.matrix_with_units.MatrixWithUnits
        Type of the returned matrix. Entries are Fisher values, where
        index :math:`i j` corresponds to the inner product of
        derivatives with respect to the parameters `params_to_vary[i]`,
        `params_to_vary[j]`.
    
    Notes
    -----
    The main reason behind choosing ``MatrixWithUnits`` as the data
    type was that information about units is available from our
    calculations, so simply discarding it would not make sense.
    Moreover, "regular" calculations using e.g. numpy arrays can also
    be carried out fairly easily using this type, namely by extracting
    this value using by applying `.value` to the class instance.

    See also
    --------
    gw_signal_tools.fisher_utils.
    get_waveform_derivative_1D_with_convergence : 
        Method used for numerical differentiation. Almost all arguments
        are passed straight to this function.
    """
    # ----- Handle some default values -----
    if isinstance(params_to_vary, str):
        params_to_vary = [params_to_vary]

    param_numb = len(params_to_vary)


    # ----- Initialize Fisher Matrix as MatrixWithUnits instance -----
    fisher_matrix = MatrixWithUnits(
        np.zeros((param_numb, param_numb), dtype=float),
        np.full((param_numb, param_numb), u.dimensionless_unscaled, dtype=object)
    )


    # ----- Compute relevant derivatives in frequency domain -----
    deriv_series_storage = {}
    deriv_info = {}

    for i, param in enumerate(params_to_vary):
        deriv_series_storage[param], info = get_waveform_derivative_1D_with_convergence(
            wf_params_at_point=wf_params_at_point,
            param_to_vary=param,
            wf_generator=wf_generator,
            step_sizes=step_sizes,
            convergence_check=convergence_check,
            break_upon_convergence=break_upon_convergence,
            convergence_threshold=convergence_threshold,
            return_info=True,
            **inner_prod_kwargs
        )

        fisher_matrix[i, i] = info['norm_squared']

        if return_info:
            # TODO: maybe copy selected stuff only?
            deriv_info[param] = info


    # ----- Populate Fisher matrix -----
    for i, param_i in enumerate(params_to_vary):
        for j, param_j in enumerate(params_to_vary):

            if i == j:
                continue
            else:
                unit_i = deriv_series_storage[param_i].unit
                unit_j = deriv_series_storage[param_j].unit

                if unit_i == unit_j:
                    fisher_val = inner_product(
                        deriv_series_storage[param_i],
                        deriv_series_storage[param_j],
                        **inner_prod_kwargs
                    )
                else:
                    deriv_series_storage[param_i].override_unit(u.dimensionless_unscaled)
                    deriv_series_storage[param_j].override_unit(u.dimensionless_unscaled)

                    fisher_val = inner_product(
                        deriv_series_storage[param_i],
                        deriv_series_storage[param_j],
                        **inner_prod_kwargs
                    )

                    deriv_series_storage[param_i].override_unit(unit_i)
                    deriv_series_storage[param_j].override_unit(unit_j)

                    # fisher_val *= unit_i * unit_j
                    fisher_val *= (unit_i * unit_j).si  # Also transform to SI for consistency with results from norm

                if 'optimize_time_and_phase' in inner_prod_kwargs.keys():
                    fisher_val = fisher_val[1]

                fisher_matrix[i, j] = fisher_matrix[j, i] = fisher_val

    if return_info:
        return fisher_matrix, deriv_info
    else:
        return fisher_matrix


def get_waveform_derivative_1D_with_convergence(
    wf_params_at_point: dict[str, u.Quantity],
    param_to_vary: str,
    wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries | ArrayLike],
    step_sizes: Optional[list[float]] = None,
    convergence_check: Optional[Literal['diff_norm', 'mismatch']] = None,
    convergence_threshold: float = None,
    break_upon_convergence: bool = True,
    return_info: bool = False,
    **inner_prod_kwargs
) -> FrequencySeries | tuple[FrequencySeries, u.Quantity]:
    """
    Calculate numerical derivative with respect to a waveform parameter,
    using the five-point-stencil method for different step sizes and a
    variable criterion check the "quality" of approximation.

    Parameters
    ----------
    wf_params_at_point : dict[str, u.Quantity]
        can in principle also be any, but param_to_vary has to be
        accessible as a key and value has to be value of point that we
        want to compute derivative around. Given as input to
        `wf_generator`.
    param_to_vary : str
        Parameter with respect to which the derivative is taken. Must be
        a key in `wf_params_at_point`.
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]],
    FrequencySeries or ArrayLike]
        Arbitrary function that is used for waveform generation. The
        required signature means that it has one non-optional argument,
        which is expected to accept the input provided in
        `wf_params_at_point`, while the output is either a ``~gwpy.
        frequencyseries.FrequencySeries`` or of type ``ArrayLike``, so
        that its subtraction is carried out element-wise. The preferred
        type is ``FrequencySeries`` because it supports astropy units
        (and it is the standard output of LAL gwsignal generators).

        A convenient option is to use the method `FisherMatrix.
        get_wf_generator`, which generates a suitable function from
        a few arguments.
    step_sizes : list[float], optional, default = None
        Step sizes that are used in the numerical differention.
    convergence_check : Optional[Literal['diff_norm', 'mismatch']],
    optional, default = None
        Criterion used to asses stability of the result. Currently, two
        are available:
        - diff_norm: calculates the norm of the difference of two
          consecutive derivatives (using the function `~gw_signal_tools.
          inner_product.norm`). This is compared to the norm of the most
          recent derivative and if their fraction is smaller than some
          threshold (specified in `convergence_threshold`), the result
          is taken to be converged because the differences become
          negligible on the relevant scales (provided by the norm of
          the derivative).
        - mismatch: calculates the mismatch between consecutive
          derivatives (also using the function `~gw_signal_tools.
          inner_product.norm`), which is defined as :math:`1 - overlap`.
          Again, the result is taken to be converged if this mismatch
          falls under a certain threshold, provided by
          `convergence_threshold`.
    convergence_threshold : float, optional, default = None
        Threshold that is used to decide if result is converged. This
        will be the case once the value of the criterion specified in
        `convergence_check` is smaller than `convergence_threshold` two
        iterations in a row.
    break_upon_convergence : bool, optional, default = True
        Whether to break upon the convergence described previously
        (difference smaller than given threshold two times in a row) or
        not. If not, results for all step sizes are calculated and the
        one with minimal convergence criterion value is selected.

    Returns
    -------
    ~gwpy.frequencyseries.FrequencySeries or tuple[~gwpy.
    frequencyseries.FrequencySeries, dict[str, Any]]
        Derivative in frequency space with respect to `param_to_vary`.
    `~gw_signal_tools.inner_product.norm` :
        Function used to create the involved inner products.

    Raises
    ------
    ValueError
        If an invalid value for convergence_check is provided.
    """
    
    if step_sizes is None:
        step_sizes = np.reshape(np.outer([1e-2, 1e-3, 1e-4, 1e-5], [5, 1]), -1)  # Seems most reasonable choice at the moment -> testing showed that 1e-6 might be too small for good results :OO
        # step_sizes = np.reshape(np.outer([1e-3, 1e-4, 1e-5], [5, 1]), -1)  # 1e-2 might be too large for some parameters -> on the other hand, 1e-5 too small for others. So there is no perfect choise
        # step_sizes = np.reshape(np.outer([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], [5, 1]), -1)  # For more detailed testing of convergence


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
    for _ in range(3):  # Maximum number of refinements of step size
        derivative_vals = []
        deriv_norms = []
        convergence_vals = []

        for i, step_size in enumerate(step_sizes):  # Maybe better than jumping to very small value quickly (numerical errors)
            # deriv_param = get_waveform_derivative_1D(
            #     wf_params_at_point,
            #     param_to_vary,
            #     wf_generator,
            #     step_size
            # )

            try:
                deriv_param = get_waveform_derivative_1D(
                    wf_params_at_point,
                    param_to_vary,
                    wf_generator,
                    step_size
                )
            except ValueError as err:
                err_msg = str(err)

                if 'Input domain error' in err_msg:
                    logging.info(
                        f'{step_size} is not a valid step size for a parameter'
                        f'value of {wf_params_at_point[param_to_vary]}. '
                        'Skipping this step size.'
                    )

                    # Still have to append something to lists, otherwise
                    # indices become inconsistent with step_sizes
                    derivative_vals += [0.0]
                    deriv_norms += [np.inf]
                    convergence_vals += [np.inf]

                    continue
                else:
                    raise ValueError(err_msg)


            derivative_norm = norm(deriv_param, **inner_prod_kwargs)
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
                        convergence_vals += [
                            norm(deriv_param - derivative_vals[-2],
                                 **inner_prod_kwargs) \
                            / np.sqrt(derivative_norm)
                        ]  # Index -1 is deriv_param
                    else:
                        continue
                case 'mismatch':
                    # Compute mismatch, using that we already know norms
                    if len(derivative_vals) >= 2:
                        convergence_vals += [
                            1.0 - inner_product(deriv_param, derivative_vals[-2],
                                                **inner_prod_kwargs) \
                            / np.sqrt(derivative_norm * deriv_norms[-2])
                        ]  # Index -1 is deriv_param
                    else:
                        continue


            logging.debug(convergence_vals)
            logging.debug([len(derivative_vals), i, len(convergence_vals)])


            if (len(convergence_vals) >= 2
                and (convergence_vals[-2] <= convergence_threshold)
                and (convergence_vals[-1] <= convergence_threshold)):
                is_converged = True  # Remains true, is never set to False again

                if break_upon_convergence:
                    min_dev_index = i  # Then it can also be used to access step_sizes
                    break
        

        # Check (i) if we shall break upon convergence and in case yes (ii) if
        # convergence was reached using these step sizes (refine them if not)
        if not break_upon_convergence or not is_converged:
            min_dev_index = np.nanargmin(convergence_vals)  # Should not have nan, but who knows
            # TODO: rename min_dev_index

            # Cut steps made around step size with best criterion value in half
            # compared to current steps (we take average step size in case
            # difference to left and right is unequl)
            left_step = (step_sizes[min_dev_index - 1] - step_sizes[min_dev_index]) / 4.0
            right_step = (step_sizes[min_dev_index + 1] - step_sizes[min_dev_index]) / 4.0
            # 4.0 due to factor of two in step_sizes below
            step_sizes = step_sizes[min_dev_index] + np.array(
                [2.0 * left_step, 1.0 * left_step, 1.0 * right_step, 2.0 * right_step]
            )

            refine_numb += 1
        else:
            break

    logging.debug(min_dev_index)

    if not is_converged:
        logging.info(
            'Calculations using the selected step sizes did not converge '
            f'for parameter `{param_to_vary}` using convergence check method '
            f'`{convergence_check}`, even after {refine_numb} refinements of '
            'step sizes. The minimal value of the criterion was '
            f'{convergence_vals[min_dev_index]}, which is above the selected '
            f'threshold of {convergence_threshold}.'
            'If you are not satisfied with the result (for an eye test, you '
            'can plot the `convergence_plot` value returned in case '
            '`return_info=True`), consider changing the initial step sizes.'
        )
    

    if return_info:
        fig, ax = plt.subplots()

        for i in range(len(derivative_vals)):
            ax.plot(derivative_vals[i].real, '--', label=f'{step_sizes[i]} (Re)')
            ax.plot(derivative_vals[i].imag, ':', label=f'{step_sizes[i]} (Im)')

        ax.legend(title='Step Sizes', ncols=max(1, len(derivative_vals) % 3))

        ax.set_xlabel('$f$')
        ax.set_ylabel('Derivative')
        ax.set_title(f'Parameter: {param_to_vary}')

        # plt.close()

        # TODO: check if this has unwanted effect, like having an open axis flying around after function call

        return derivative_vals[min_dev_index], {
            'norm_squared': deriv_norms[min_dev_index],
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
    wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries | ArrayLike],
    step_size: float
) -> FrequencySeries:
    """
    Use five-point stencil method to calculate numerical derivatives
    with respect to a waveform parameter.

    Parameters
    ----------
    wf_params_at_point : dict[str, ~astropy.units.Quantity]
        Dictionary with parameters that determine point at which
        derivative is calculated.
    param_to_vary : str
        Derivative is taken with respect to this parameter (has to be
        key passed to waveform generators).
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]],
    FrequencySeries or ArrayLike]
        Arbitrary function that is used for waveform generation. The
        required signature means that it has one non-optional argument,
        which is expected to accept the input provided in
        `wf_params_at_point`, while the output is either a ``~gwpy.
        frequencyseries.FrequencySeries`` or of type ``ArrayLike``, so
        that its subtraction is carried out element-wise. The preferred
        type is ``FrequencySeries`` because it supports astropy units
        (and it is the standard output of LAL gwsignal generators).
    step_size : float
        Step size used in numerical differentiation.

    Returns
    -------
    ~gwpy.frequencyseries.FrequencySeries
        Derivative in frequency domain.
    
    Notes
    -----
    In case the function throws an error because shape mismatches or so,
    consider providing fixed starting frequency, end frequency and
    frequency spacing so that the waveforms are guaranteed to have the
    same frequency range.
    """

    param_center_val = wf_params_at_point[param_to_vary]
    step_size = u.Quantity(step_size, unit=param_center_val.unit)
    param_vals = param_center_val + np.array([-2., -1., 1., 2.]) * step_size


    waveforms = [
        wf_generator(wf_params_at_point | {param_to_vary: param_val}
                     ) for param_val in param_vals
    ]

    
    # TODO: change override once lalsuite gets it right!!!
    for wf in waveforms:
        try:
            wf.override_unit(u.s)
        except AttributeError:
            # Could also turn into u.Quantity here... But usecase not there perhaps
            pass


    deriv_series = waveforms[0] - 8.0 * waveforms[1] + 8.0 * waveforms[2] - waveforms[3]    
    deriv_series /= 12.0 * step_size

    # Central Difference -> make this option in function?
    # deriv_series = waveforms[1] - waveforms[0]
    # deriv_series /= 2.0 * step_size

    return deriv_series
