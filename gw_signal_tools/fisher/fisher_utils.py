# ----- Standard Lib Imports -----
import warnings
from typing import Optional, Any, Literal, Callable

# ----- Third Party Imports -----
import numpy as np
import matplotlib.pyplot as plt
from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries
import astropy.units as u
import numdifftools as nd

# ----- Local Package Imports -----
from ..logging import logger
from ..waveform.inner_product import inner_product, norm, _INNER_PROD_ARGS
from ..types import MatrixWithUnits
from ..test_utils import allclose_quantity


__doc__ = """
Module that contains functions to calculate numerical derivatives of
gravitational waveforms and also a wrapper to calculate a Fisher matrix.
"""


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
    ~gwpy.types.series.Series or ~numpy.ndarray
        Derivative of `signal`.
    """    
    if isinstance(signal, Series):
        if h is None:
            h = signal.dx
        elif not isinstance(h, u.Quantity):
            h = u.Quantity(h, signal.xindex.unit)

        if not allclose_quantity(h.value, signal.dx.value,
                                 atol=0., rtol=1e-3):  # pragma: no cover
            warnings.warn(
                'Given `h` does not coincide with `signal.dx`.'
            )
    else:
        # Make sure signal is array, we utilize numpy operations
        signal = np.asarray(signal)

        # Check if h is set
        if h is None:
            h = 1.
        else:
            if isinstance(h, u.Quantity):
                signal = u.Quantity(signal, u.dimensionless_unscaled)

    signal_deriv = (np.roll(signal, 2) - 8.*np.roll(signal, 1)
                    + 8.*np.roll(signal, -1) - np.roll(signal, -2))
    signal_deriv /= 12.*h

    signal_deriv[0] = (signal[1] - signal[0]) / h  # Forward difference
    signal_deriv[1] = (signal[2] - signal[0]) / (2.*h)  # Central difference

    signal_deriv[-2] = (signal[-1] - signal[-3]) / (2.*h)  # Central difference
    signal_deriv[-1] = (signal[-1] - signal[-2]) / h  # Backward difference

    return signal_deriv


def fisher_matrix(
    deriv_routine: Literal['gw_signal_tools', 'numdifftools'],
    **deriv_and_inner_prod_kwargs
) -> MatrixWithUnits | tuple[MatrixWithUnits, dict[str, dict[str, str]]]:
    """
    Wrapper that allows to select between the two available routines for
    Fisher matrix calculation by passing the argument `deriv_routine`.
    All other arguments are passed to the selected routine and described
    in more detail in the corresponding functions
    `fisher_matrix_gw_signal_tools`, `fisher_matrix_numdifftools`.
    
    Notes
    -----
    The two routines differ in the way they calculate derivatives, as
    the argument name already indicates. The `'gw_signal_tools'` option
    uses a custom routine that applies the five-point stencil method to
    the output of `wf_generator` and assesses convergence based on a
    criteria whose origin lies in GW data analysis (utilizing the
    noise-weighted inner product defined in this context).

    The `'numdifftools'` option, on the other hand, calculates the
    derivative of amplitude and phase of the waveform separately, which
    are recombined using the product rule to give the waveform
    derivative. These derivatives are calculated the `numdifftools`
    package.

    Both routines have advantages and disadvantages: since two
    derivatives have to be estimated for the latter case,
    `'gw_signal_tools'` is typically faster than `'numdifftools'`.
    However, `'numdifftools'` typically yields more stable results for
    highly oscillatory derivatives, where `'gw_signal_tools'` can
    struggle to find results with comparable accuracy (which means that
    it does find results that do look similar, but the convergence when
    decreasing step sizes is sometimes not as stable; reason is use of
    direct waveform differences, combined with five-point stencil that
    can encounter numerical issues in some cases).
    """
    match deriv_routine:
        case 'gw_signal_tools':
            return fisher_matrix_gw_signal_tools(
                **deriv_and_inner_prod_kwargs
            )
        case 'numdifftools':
            return fisher_matrix_numdifftools(
                **deriv_and_inner_prod_kwargs
            )
        case _:  # pragma: no cover
            raise ValueError('Invalid `deriv_routine`.')


# NOTE: removing some of the arguments to pass all kwargs to derivative
# does not work because we want to be able to pass kwargs to
# inner_product function in fisher_matrix itself
def fisher_matrix_gw_signal_tools(
    wf_params_at_point: dict[str, u.Quantity],
    params_to_vary: str | list[str],
    wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
    return_info: bool = False,
    **deriv_and_inner_prod_kwargs
) -> MatrixWithUnits | tuple[MatrixWithUnits, dict[str, dict[str, Any]]]:
    r"""
    Compute Fisher matrix at a fixed point, with the derivative
    calculation being carried out by the custom derivative calculator
    implemented in :code:`gw_signal_tools`.

    Parameters
    ----------
    wf_params_at_point : dict[str, ~astropy.units.Quantity]
        Point in parameter space at which the Fisher matrix is
        evaluated, encoded as key-value pairs representing
        parameter-value pairs. Given as input to :code:`wf_generator`.
    params_to_vary : str or list[str]
        Parameter(s) with respect to which the derivatives will be
        computed, the norms of which constitute the Fisher matrix.
        Must be compatible with :code:`param_to_vary` input to the
        function :code:`~gw_signal_tools.fisher.fisher_utils.
        get_waveform_derivative_1D_with_convergence`, i.e. either
        :code:`'tc'` (equivalent: :code:`'time'`), :code:`'psi'`
        (equivalent up to a factor: :code:`'phase' = 2*'psi'`) or a key
        in :code:`wf_params_at_point`.
        
        For time and phase shifts, analytical derivatives are applied.
        This is possible because they contribute only to a factor
        :math:`\exp(i \cdot 2 \psi - i \cdot 2 \pi \cdot f \cdot t_c)`
        in the waveform generated by :code:`wf_generator`. They
        correspond to the parameters typically called coalescence time
        :math:`t_c` and polarization angle :math:`\psi`.

        The last analytical derivative is the one for the luminosity
        distance :math:`D_L`, which enters in waveforms only as an
        amplitude factor :math:`1/D_L`. Note that can only be done
        if the parameter recognized, i.e. if it is called `'distance'`.
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]], ~gwpy.frequencyseries.FrequencySeries]
        Arbitrary function that is used for waveform generation. The
        required signature means that it has one non-optional argument,
        which is expected to accept the input provided in
        :code:`wf_params_at_point`, while the output must be a ``~gwpy.
        frequencyseries.FrequencySeries`` (the standard output of
        LAL gwsignal generators) because it carries information about
        value, frequencies and units, which are all required for the
        calculations that are carried out.

        A convenient option is to use the method
        :code:`~gw_signal_tools.waveform_utils.get_wf_generator`, which
        generates a suitable function from a few arguments.
    return_info : boolean, optional, default = True
        Whether to return information collected during the derivative
        calculations. Can be used as a sort of custom cache to also
        return derivatives.
    deriv_and_inner_prod_kwargs :
        All other keyword arguments are passed to the derivative
        and inner product routines involved in the Fisher matrix
        calculations.

    Returns
    -------
    ~gw_signal_tools.matrix_with_units.MatrixWithUnits
        A ``MatrixWithUnits`` instance. Entries are Fisher values, where
        index :math:`(i, j)` corresponds to the inner product of
        derivatives with respect to the parameters
        :code:`params_to_vary[i]`, :code:`params_to_vary[j]`.
    
    Notes
    -----
    The main reason behind choosing ``MatrixWithUnits`` as the data
    type was that information about units is available from our
    calculations, so simply discarding it would not make sense.
    Moreover, "regular" calculations using e.g. numpy arrays can also
    be carried out fairly easily using this type, namely by extracting
    this value using by applying `.value` to the class instance.

    See Also
    --------
    gw_signal_tools.fisher.get_waveform_derivative_1D_with_convergence :
        Method used for numerical differentiation. Almost all arguments
        are passed straight to this function.
    """
    # ----- Separate deriv and inner_prod kwargs, check defaults -----
    _deriv_kw_args = {}
    _inner_prod_kwargs = {}
    for key, value in deriv_and_inner_prod_kwargs.items():
        if key in _INNER_PROD_ARGS:
            _inner_prod_kwargs[key] = value
        else:
            _deriv_kw_args[key] = value
    _inner_prod_kwargs['return_opt_info'] = False
    # Ensure float output of inner_product, even if optimization on

    if isinstance(params_to_vary, str):
        params_to_vary = [params_to_vary]

    param_numb = len(params_to_vary)

    # ----- Initialize Fisher Matrix as MatrixWithUnits instance -----
    fisher_matrix = MatrixWithUnits(
        np.zeros(2*(param_numb, ), dtype=float),
        np.full(2*(param_numb, ), u.dimensionless_unscaled, dtype=object)
    )

    # ----- Compute relevant derivatives in frequency domain -----
    deriv_series_storage = {}
    deriv_info = {}

    for i, param in enumerate(params_to_vary):
        deriv, info = get_waveform_derivative_1D_with_convergence(
            wf_params_at_point=wf_params_at_point,
            param_to_vary=param,
            wf_generator=wf_generator,
            return_info=True,
            **deriv_and_inner_prod_kwargs
        )

        deriv_series_storage[param] = deriv
        info['deriv'] = deriv
        fisher_matrix[i, i] = info['norm_squared']

        if return_info:
            # TODO: maybe copy selected stuff only?
            deriv_info[param] = info
        else:
            plt.close('all')  # Otherwise axes remain open and eventually get displayed

    # ----- Populate Fisher matrix -----
    for i, param_i in enumerate(params_to_vary):
        for j, param_j in enumerate(params_to_vary):

            if i == j:
                # Was already set in previous loop
                continue
            else:
                fisher_matrix[i, j] = fisher_matrix[j, i] = inner_product(
                    deriv_series_storage[param_i],
                    deriv_series_storage[param_j],
                    **_inner_prod_kwargs
                )

    if return_info:
        return fisher_matrix, deriv_info
    else:
        return fisher_matrix


def get_waveform_derivative_1D_with_convergence(
    wf_params_at_point: dict[str, u.Quantity],
    param_to_vary: str,
    wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
    step_sizes: Optional[list[float] | np.ndarray] = None,
    start_step_size: Optional[float] = 1e-2,
    convergence_check: Optional[Literal['diff_norm', 'mismatch']] = None,
    convergence_threshold: Optional[float] = None,
    break_upon_convergence: bool = True,
    return_info: bool = False,
    **inner_prod_kwargs
) -> FrequencySeries | tuple[FrequencySeries, dict[str, Any]]:
    r"""
    Calculate numerical derivative of gravitational wave (GW) waveforms
    with respect to a waveform parameter in frequency domain, using the
    five-point-stencil method for different step sizes and a variable
    criterion check the "quality" of approximation.

    Parameters
    ----------
    wf_params_at_point : dict[str, ~astropy.units.Quantity]
        Point in parameter space at which the derivative is evaluated,
        encoded as key-value pairs representing parameter-value pairs.

        In principle, the keys can be arbitrary, only two requirements
        have to be fulfilled: (i) the dictionary must be accepted by
        :code:`wf_generator` since it is given as input to this functoin
        and (ii) :code:`param_to_vary` has to be accessible as a key
        (except one of the special cases mentioned in the description of
        :code:`params_to_vary` is true).
    param_to_vary : str
        Parameter with respect to which the derivative is taken. Must be
        :code:`'tc'` (equivalent: :code:`'time'`), :code:`'psi'`
        (equivalent up to a factor: :code:`'phase' = 2*'psi'`) or a key
        in :code:`wf_params_at_point`.
        
        For time and phase shifts, analytical derivatives are applied.
        This is possible because they contribute only to a factor
        :math:`\exp(i \cdot 2 \psi - i \cdot 2 \pi \cdot f \cdot t_c)`
        in the waveform generated by `wf_generator`. They correspond
        to the parameters that are typically called coalescence
        time :math:`t_c` and polarization angle :math:`\psi`.

        The last analytical derivative is the one for the luminosity
        distance :math:`D_L`, which enters in waveforms only as an
        amplitude factor :math:`1/D_L`. Note that can only be done
        if the parameter recognized, i.e. if it is called `'distance'`.
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]], ~gwpy.frequencyseries.FrequencySeries]
        Arbitrary function that is used for waveform generation. The
        required signature means that it has one non-optional argument,
        which is expected to accept the input provided in
        :code:`wf_params_at_point`, while the output must be a ``~gwpy.
        frequencyseries.FrequencySeries`` (the standard output of
        LAL gwsignal generators) because it carries information about
        value, frequencies and units, which are all required for the
        calculations that are carried out.

        A convenient option is to use the method
        :code:`~gw_signal_tools.waveform_utils.get_wf_generator`, which
        generates a suitable function from a few arguments.
    step_sizes : list[float], optional, default = None
        Step sizes used in the numerical differention. Based on the
        evaluation point, these are used as relative or absolute steps.
    start_step_size: float, optional, default = 1e-2
        Alternative way to control the relative step sizes. Determines
        the largest relative step size that is tried.
    convergence_check : Literal['diff_norm', 'mismatch'], optional, default = None
        Criterion used to asses stability of the result. Currently, two
        are available:

            * diff_norm: calculates the norm of the difference of two
              consecutive derivatives (using the function
              :code:`~gw_signal_tools.inner_product.norm`). This is
              compared to the norm of the most recent derivative and if
              their fraction is smaller than some threshold (specified
              in :code:`convergence_threshold`), the result is taken to
              be converged because the differences become negligible on
              the relevant scales (provided by the norm of the
              derivative).
            * mismatch: calculates the mismatch between consecutive
              derivatives (also using the function
              :code:`~gw_signal_tools.inner_product.norm`), which is
              defined as :math:`1-overlap`. Again, the result is taken
              to be converged if this mismatch falls under a certain
              threshold, provided by :code:`convergence_threshold`.

        For larger differences, they might produce different results,
        but their behaviour for small distances should be very similar
        because they coincide in the infinitesimal limit (they induce
        the same metric).
    convergence_threshold : float, optional, default = None
        Threshold that is used to decide if result is converged. This
        will be the case once the value of the criterion specified in
        :code:`convergence_check` is smaller than
        :code:`convergence_threshold` two iterations in a row.
    break_upon_convergence : bool, optional, default = True
        Whether to break upon the convergence described previously
        (difference smaller than given threshold two times in a row) or
        not. If not, results for all step sizes are calculated and the
        one with minimal convergence criterion value is selected.
    inner_prod_kwargs :
        All additional keyword arguments are passed to the inner product
        function during the corresponding calculations.

    Returns
    -------
    ~gwpy.frequencyseries.FrequencySeries or tuple[~gwpy.frequencyseries.FrequencySeries, dict[str, Any]]
        Derivative in frequency space with respect to
        :code:`param_to_vary`. If :code:`return_info = True`, also a
        dictionary with information about the result.
    
    See Also
    --------
    gw_signal_tools.inner_product.norm :
        Function used to create the involved inner products.

    Raises
    ------
    ValueError
        If an invalid value for convergence_check is provided.
    AssertionError
        If an invalid :code:`params_to_vary` is provided.
    """
    # ----- Check defaults -----
    inner_prod_kwargs['return_opt_info'] = False
    # Ensure float output of inner_product, even if optimization on

    if step_sizes is None:
        step_sizes = np.reshape(np.outer([start_step_size/10**i for i in range(5)], [5, 1]), -1)[1:]  # Indexing makes sure we do not start at 5*start_step_size
        # NOTE: keeping 1e-2 as default start_step_size is most likely too
        # high for relative ones. But depending on point that derivative is
        # evaluated in, absolute step sizes are used sometimes, too, and in
        # that case, 1e-2 seems to be a valid starting point.
        # Also, five-point-stencil does not need values as small as the ones
        # commonly used with e.g. central difference.

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
                convergence_threshold = 0.001
            case 'mismatch':
                convergence_threshold = 0.001

    # ----- Calculation -----
    if (param_to_vary == 'time' or param_to_vary == 'tc'):
        wf = wf_generator(wf_params_at_point)
        deriv = wf * (-1.j * 2. * np.pi * wf.frequencies)

        derivative_norm = norm(deriv, **inner_prod_kwargs)**2

        if return_info:
            return deriv, {
                'norm_squared': derivative_norm,
                'description': 'This derivative is exact.'
            }
        else:
            return deriv
    elif (param_to_vary == 'phase' or param_to_vary == 'psi'):
        wf = wf_generator(wf_params_at_point)

        if param_to_vary == 'phase':
            deriv = wf * 1.j / u.rad
        else:
            deriv = wf * 2.j / u.rad

        derivative_norm = norm(deriv, **inner_prod_kwargs)**2

        if return_info:
            return deriv, {
                'norm_squared': derivative_norm,
                'description': 'This derivative is exact.'
            }
        else:
            return deriv
    elif param_to_vary == 'distance':
        wf = wf_generator(wf_params_at_point)

        deriv = (-1./wf_params_at_point['distance']) * wf

        derivative_norm = norm(deriv, **inner_prod_kwargs)**2

        if return_info:
            return deriv, {
                'norm_squared': derivative_norm,
                'description': 'This derivative is exact.'
            }
        else:
            return deriv
    else:
        assert param_to_vary in wf_params_at_point, \
            ('`param_to_vary` must be `\'tc\'`/`\'time\'`, `\'psi\'`/'
             '`\'phase`\' or a key in `wf_params_at_point`.')

    is_converged = False
    refine_numb = 0
    for _ in range(3):  # Maximum number of refinements of step size
        derivative_vals = []
        deriv_norms = []
        convergence_vals = []

        for i, step_size in enumerate(step_sizes):
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
                    logger.info(
                        f'{step_size} is not a valid step size for a parameter'
                        f' value of {wf_params_at_point[param_to_vary]}. '
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


            derivative_norm = norm(deriv_param, **inner_prod_kwargs)**2

            derivative_vals += [deriv_param]
            deriv_norms += [derivative_norm]


            match convergence_check:
                case 'diff_norm':
                    if len(derivative_vals) >= 2:
                        convergence_vals += [
                            norm(deriv_param - derivative_vals[-2],
                                 **inner_prod_kwargs)/np.sqrt(derivative_norm)
                        ]
                    else:
                        convergence_vals += [np.inf]
                        continue
                case 'mismatch':
                    # Compute mismatch, using that we already know norms
                    if len(derivative_vals) >= 2:
                        convergence_vals += [
                            1. - inner_product(
                            deriv_param,
                            derivative_vals[-2],
                            **inner_prod_kwargs
                        ) / np.sqrt(derivative_norm * deriv_norms[-2])
                        ]  # Index -1 is deriv_param
                    else:
                        convergence_vals += [np.inf]
                        continue

            # if (convergence_vals[-1] <= convergence_threshold):
                # We use five-point stencil, which converges fast, so
                # that it is justified to interpret two consecutive
                # results being very similar as convergence
                # -> testing revealed that criterion below leads to more
                #    consistent results, thus we leave for now
            if (len(convergence_vals) >= 2
                and (convergence_vals[-1] <= convergence_threshold)
                and (convergence_vals[-2] <= convergence_threshold)):
                # Double checking is more robust
                is_converged = True  # Remains true, is never set to False again

                if break_upon_convergence:
                    min_dev_index = i  # Then it can also be used to access step_sizes
                    break
        
        last_used_step_sizes = step_sizes  # Save for plots
        
        # Check if step sizes shall be refined. This is be done if no breaking
        # upon convergence is wanted or if no convergence was reached yet
        if not break_upon_convergence or not is_converged:
            # TODO: remove break_upon_convergence and just handle that via convergence_threshold?
            # I.e. set to 0.0 if no breaking wanted
            
            if np.all(np.equal(convergence_vals, np.inf)):
                # Only invalid step sizes for this parameter, we have to
                # decrease further
                min_dev_index = len(step_sizes) - 1
            else:
                min_dev_index = np.nanargmin(convergence_vals)  # type: ignore
            # Explanation of ignore: it seems like a signedinteger is returned
            # by nanargmin, violates static checking for int. Note that we do
            # use nan-version here just in case something goes wrong in norm or
            # so, making it zero (should not happen, though)

            # Cut steps made around step size with best criterion value in half
            # compared to current steps (we take average step size in case
            # difference to left and right is unequl)
            if min_dev_index < (len(step_sizes) - 1):
                left_step = (step_sizes[min_dev_index - 1] - step_sizes[min_dev_index]) / 4.0
                right_step = (step_sizes[min_dev_index + 1] - step_sizes[min_dev_index]) / 4.0
                # 4.0 due to factor of two in step_sizes below

                step_sizes = step_sizes[min_dev_index] + np.array(
                    [2.*left_step, 1.*left_step, 1.*right_step, 2.*right_step]
                )
                # TODO: also include 0.0 here? I.e. the optimal one, as of now?
            else:
                # Smallest convergence value at smallest step size, so
                # min_dev_index + 1 is invalid index. Instead of zooming in,
                # smaller step sizes are explored

                # Refine in same way that we do with start_step_size
                step_sizes = np.reshape(np.outer([step_sizes[min_dev_index]/10**i for i in range(4)], [5, 1]), -1)[1:]  # Indexing makes sure we do not start at 5*start_step_size


            refine_numb += 1
        else:
            break

    # ----- Verification of result and information collection -----
    if not is_converged:
        logger.info(
            'Calculations using the selected step sizes did not converge '
            f'for parameter `{param_to_vary}` using convergence check method '
            f'`{convergence_check}`, even after {refine_numb} refinements of '
            'step sizes. The minimal value of the criterion was '
            f'{convergence_vals[min_dev_index]}, ' + ((f'which is above the '
            f'selected threshold of {convergence_threshold}. ')
            if convergence_vals[min_dev_index] > convergence_threshold else (
            f'which is below the selected threshold of {convergence_threshold}'
            ', but the previous and following value were not.')) +
            'If you are not satisfied with the result (for an eye test, you '
            'can plot the `convergence_plot` value returned in case '
            '`return_info=True`), consider changing the initial step sizes.'
        )
    
    if return_info:
        fig = plt.figure()
        ax = fig.subplots(nrows=2, sharex=True)

        for i in range(len(derivative_vals)):
            ax[0].plot(derivative_vals[i].real, '--',
                       label=f'{last_used_step_sizes[i]:.3e}')
            ax[1].plot(derivative_vals[i].imag, '--')
            # No label for second because otherwise, everything shows up twice
            # in figure legend

        fig.legend(
            title='Step Sizes',
            bbox_to_anchor=(0.96, 0.5),
            loc='center left'
        )
        
        fig.suptitle(f'Parameter: {param_to_vary}')  # TODO: use latexparams here?
        ax[1].set_xlabel('$f$')
        ax[0].set_ylabel('Derivative Re')
        ax[1].set_ylabel('Derivative Im')

        return derivative_vals[min_dev_index], {
            'norm_squared': deriv_norms[min_dev_index],
            'final_step_size': step_sizes[min_dev_index],
            'final_convergence_val': convergence_vals[min_dev_index],
            'number_of_refinements': refine_numb,
            'final_set_of_step_sizes': step_sizes,
            'convergence_plot': ax
        }
    else:
        return derivative_vals[min_dev_index]


def get_waveform_derivative_1D(
    wf_params_at_point: dict[str, u.Quantity],
    param_to_vary: str,
    wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
    step_size: float
) -> FrequencySeries:
    """
    Use five-point stencil method to calculate numerical derivatives
    with respect to a waveform parameter (in very special cases,
    other finite difference methods might be used, for example in case
    of extreme mass ratio values).

    Parameters
    ----------
    wf_params_at_point : dict[str, ~astropy.units.Quantity]
        Dictionary with parameters that determine point at which
        derivative is calculated.

        Can in principle also be any, but param_to_vary has to be
        accessible as a key and value has to be value of point that we
        want to compute derivative around. Given as input to
        :code:`wf_generator`.
    param_to_vary : str
        Derivative is taken with respect to this parameter (has to be a
        key from :code:`wf_params_at_point`).
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]], ~gwpy.frequencyseries.FrequencySeries]
        Arbitrary function that is used for waveform generation. The
        required signature means that it has one non-optional argument,
        which is expected to accept the input provided in
        :code:`wf_params_at_point`, while the output must be a ``~gwpy.
        frequencyseries.FrequencySeries`` (the standard output of
        LAL gwsignal generators) because it carries information about
        value, frequencies and units, which are all required for the
        calculations that are carried out.

        A convenient option is to use the method
        :code:`~gw_signal_tools.waveform_utils.get_wf_generator`, which
        generates a suitable function from a few arguments.
    step_size : float
        Step size used in the numerical differention. Based on the
        evaluation point, this is used as relative or absolute step.

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
    _wf_params_at_point = wf_params_at_point
    # if 'f_max' in wf_params_at_point and 'deltaF' in wf_params_at_point:
    #     old_f_max = wf_params_at_point['f_max']
    #     _df = wf_params_at_point['deltaF']
    # else:
    #     _wf = wf_generator(wf_params_at_point)
    #     old_f_max = _wf.frequencies[-1]
    #     _df = _wf.df
    
    # _wf_params_at_point['f_max'] = old_f_max + 3.*_df
    # print(old_f_max, _df, _wf_params_at_point['f_max'])
    # 3.1 instead of 3. to prevent numerical errors from entering

    param_center_val = _wf_params_at_point[param_to_vary]
    
    # Choose relative or absolute step size, based on param value
    if np.log10(param_center_val.value) < 1:
        step_size = np.abs(u.Quantity(step_size, unit=param_center_val.unit))
    else:
        step_size = np.abs(u.Quantity(step_size * param_center_val, unit=param_center_val.unit))
    
    # Note: we need to pay attention to the mass ratio q because there
    # are two conventions, which are both accepted by LAL. This becomes
    # a problem for values close to 1, where this convention switches.
    if (param_to_vary == 'mass_ratio'
        and (param_center_val + 2.*step_size) > 1.
        and (param_center_val - 2.*step_size) < 1.):
        if param_center_val <= 1.:
            param_vals = param_center_val + np.array([0., -1.])*step_size
        else:
            param_vals = param_center_val + np.array([1., 0.])*step_size

        waveforms = [
            wf_generator(_wf_params_at_point | {param_to_vary: param_val}
                        ) for param_val in param_vals
        ]

        deriv_series = (waveforms[0] - waveforms[1]) / step_size
    elif (param_to_vary == 'mass_ratio'
          and (param_center_val - 2.*step_size) < 0.):
        # mass ratio is very close to 0, use forward difference
        param_vals = param_center_val + np.array([1., 0.])*step_size

        waveforms = [
            wf_generator(_wf_params_at_point | {param_to_vary: param_val}
                        ) for param_val in param_vals
        ]

        deriv_series = (waveforms[0] - waveforms[1]) / step_size
    else:
        # Five-point stencil method can be used
        param_vals = param_center_val + np.array([-2., -1., 1., 2.])*step_size

        waveforms = [
            wf_generator(_wf_params_at_point | {param_to_vary: param_val}
                        ) for param_val in param_vals
        ]

        deriv_series = (waveforms[0] - 8.*waveforms[1]
                        + 8.*waveforms[2] - waveforms[3])
        deriv_series /= 12.*step_size

    # deriv_series.value[-3:] = 0.  # Solution for now
    # Following perhaps more robust, in principle value has no setter
    # deriv_series[-3:] *= 0.  # Solution for now

    return deriv_series


def fisher_matrix_numdifftools(
    wf_params_at_point: dict[str, u.Quantity],
    params_to_vary: str | list[str],
    wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
    return_info: bool = False,
    **deriv_and_inner_prod_kwargs
) -> MatrixWithUnits | tuple[MatrixWithUnits, dict[str, dict[str, Any]]]:
    r"""
    Compute Fisher matrix at a fixed point, with the derivative
    calculation being carried out by routines implemented in the
    external package :code:`numdifftools`.

    Parameters
    ----------
    wf_params_at_point : dict[str, ~astropy.units.Quantity]
        Point in parameter space at which the Fisher matrix is
        evaluated, encoded as key-value pairs representing
        parameter-value pairs. Given as input to :code:`wf_generator`.
    params_to_vary : str or list[str]
        Parameter(s) with respect to which the derivatives will be
        computed, the norms of which constitute the Fisher matrix.
        Must be compatible with :code:`param_to_vary` input to the
        function :code:`~gw_signal_tools.fisher.fisher_utils.
        get_waveform_derivative_1D_with_convergence`, i.e. either
        :code:`'tc'` (equivalent: :code:`'time'`), :code:`'psi'`
        (equivalent up to a factor: :code:`'phase' = 2*'psi'`) or a key
        in :code:`wf_params_at_point`.
        
        For time and phase shifts, analytical derivatives are applied.
        This is possible because they contribute only to a factor
        :math:`\exp(i \cdot 2 \psi - i \cdot 2 \pi \cdot f \cdot t_c)`
        in the waveform generated by :code:`wf_generator`. They
        correspond to the parameters typically called coalescence time
        :math:`t_c` and polarization angle :math:`\psi`.

        The last analytical derivative is the one for the luminosity
        distance :math:`D_L`, which enters in waveforms only as an
        amplitude factor :math:`1/D_L`. Note that can only be done
        if the parameter recognized, i.e. if it is called `'distance'`.
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]], ~gwpy.frequencyseries.FrequencySeries]
        Arbitrary function that is used for waveform generation. The
        required signature means that it has one non-optional argument,
        which is expected to accept the input provided in
        :code:`wf_params_at_point`, while the output must be a ``~gwpy.
        frequencyseries.FrequencySeries`` (the standard output of
        LAL gwsignal generators) because it carries information about
        value, frequencies and units, which are all required for the
        calculations that are carried out.

        A convenient option is to use the method
        :code:`~gw_signal_tools.waveform_utils.get_wf_generator`, which
        generates a suitable function from a few arguments.
    deriv_and_inner_prod_kwargs :
        All other keyword arguments are passed to the derivative
        and inner product routines involved in the Fisher matrix
        calculations.

    Returns
    -------
    ~gw_signal_tools.matrix_with_units.MatrixWithUnits
        A ``MatrixWithUnits`` instance. Entries are Fisher values, where
        index :math:`(i, j)` corresponds to the inner product of
        derivatives with respect to the parameters
        :code:`params_to_vary[i]`, :code:`params_to_vary[j]`.
    
    Notes
    -----
    The main reason behind choosing ``MatrixWithUnits`` as the data
    type was that information about units is available from our
    calculations, so simply discarding it would not make sense.
    Moreover, "regular" calculations using e.g. numpy arrays can also
    be carried out fairly easily using this type, namely by extracting
    this value using by applying `.value` to the class instance.

    See Also
    --------
    gw_signal_tools.fisher.get_waveform_derivative_1D_numdifftools :
        Method used for numerical differentiation. Almost all arguments
        are passed straight to this function.
    """
    # ----- Separate deriv and inner_prod kwargs, check defaults -----
    _deriv_kw_args = {}
    _inner_prod_kwargs = {}
    for key, value in deriv_and_inner_prod_kwargs.items():
        if key in _INNER_PROD_ARGS:
            _inner_prod_kwargs[key] = value
        else:
            _deriv_kw_args[key] = value
    _inner_prod_kwargs['return_opt_info'] = False
    # Ensure float output of inner_product, even if optimization on

    if isinstance(params_to_vary, str):
        params_to_vary = [params_to_vary]

    param_numb = len(params_to_vary)

    # ----- Initialize Fisher Matrix as MatrixWithUnits instance -----
    fisher_matrix = MatrixWithUnits(
        np.zeros(2*(param_numb, ), dtype=float),
        np.full(2*(param_numb, ), u.dimensionless_unscaled, dtype=object)
    )

    # ----- Compute relevant derivatives in frequency domain -----
    deriv_series_storage = {}
    deriv_info = {}

    for i, param in enumerate(params_to_vary):
        deriv = get_waveform_derivative_1D_numdifftools(
            wf_params_at_point=wf_params_at_point,
            param_to_vary=param,
            wf_generator=wf_generator,
            **_deriv_kw_args
        )

        deriv_series_storage[param] = deriv
        fisher_matrix[i, i] = norm(deriv, **_inner_prod_kwargs)**2

        if return_info:
            # Useful as storage for derivatives
            deriv_info[param] = {'deriv': deriv}

    # ----- Populate Fisher matrix -----
    for i, param_i in enumerate(params_to_vary):
        for j, param_j in enumerate(params_to_vary):

            if i == j:
                # Was already set in previous loop
                continue
            else:
                fisher_matrix[i, j] = fisher_matrix[j, i] = inner_product(
                    deriv_series_storage[param_i],
                    deriv_series_storage[param_j],
                    **_inner_prod_kwargs
                )

    if return_info:
        return fisher_matrix, deriv_info
    else:
        return fisher_matrix


def get_waveform_derivative_1D_numdifftools(
    wf_params_at_point: dict[str, u.Quantity],
    param_to_vary: str,
    wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
    **deriv_kwargs
) -> FrequencySeries | tuple[FrequencySeries, dict[str, Any]]:
    r"""
    Calculate numerical derivative of gravitational wave (GW) waveforms
    with respect to a waveform parameter in frequency domain, using the
    `numdifftools` package.

    Parameters
    ----------
    wf_params_at_point : dict[str, ~astropy.units.Quantity]
        Point in parameter space at which the derivative is evaluated,
        encoded as key-value pairs representing parameter-value pairs.

        In principle, the keys can be arbitrary, only two requirements
        have to be fulfilled: (i) the dictionary must be accepted by
        :code:`wf_generator` since it is given as input to this functoin
        and (ii) :code:`param_to_vary` has to be accessible as a key
        (except one of the special cases mentioned in the description of
        :code:`params_to_vary` is true).
    param_to_vary : str
        Parameter with respect to which the derivative is taken. Must be
        :code:`'tc'` (equivalent: :code:`'time'`), :code:`'psi'`
        (equivalent up to a factor: :code:`'phase' = 2*'psi'`) or a key
        in :code:`wf_params_at_point`.
        
        For time and phase shifts, analytical derivatives are applied.
        This is possible because they contribute only to a factor
        :math:`\exp(i \cdot 2 \psi - i \cdot 2 \pi \cdot f \cdot t_c)`
        in the waveform generated by `wf_generator`. They correspond
        to the parameters that are typically called coalescence
        time :math:`t_c` and polarization angle :math:`\psi`.

        The last analytical derivative is the one for the luminosity
        distance :math:`D_L`, which enters in waveforms only as an
        amplitude factor :math:`1/D_L`. Note that can only be done
        if the parameter recognized, i.e. if it is called `'distance'`.
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]], ~gwpy.frequencyseries.FrequencySeries]
        Arbitrary function that is used for waveform generation. The
        required signature means that it has one non-optional argument,
        which is expected to accept the input provided in
        :code:`wf_params_at_point`, while the output must be a ``~gwpy.
        frequencyseries.FrequencySeries`` (the standard output of
        LAL gwsignal generators) because it carries information about
        value, frequencies and units, which are all required for the
        calculations that are carried out.

        A convenient option is to use the method
        :code:`~gw_signal_tools.waveform_utils.get_wf_generator`, which
        generates a suitable function from a few arguments.

    Returns
    -------
    ~gwpy.frequencyseries.FrequencySeries | tuple[~gwpy.frequencyseries.FrequencySeries, dict[str, Any]]
        Derivative in frequency space with respect to
        :code:`param_to_vary`.

    Raises
    ------
    AssertionError
        If an invalid :code:`params_to_vary` is provided.
    """
    _wf_at_point = wf_generator(wf_params_at_point)

    # ----- Check if analytical derivative can be used -----
    if (param_to_vary == 'time' or param_to_vary == 'tc'):
        return _wf_at_point * (-1.j * 2. * np.pi * _wf_at_point.frequencies)
    elif param_to_vary == 'phase':
        return _wf_at_point * 1.j / u.rad
    elif param_to_vary == 'psi':
        return _wf_at_point * 2.j / u.rad
    elif param_to_vary == 'distance':
        return (-1./wf_params_at_point['distance']) * _wf_at_point
    else:
        assert param_to_vary in wf_params_at_point, \
            ('`param_to_vary` must be `\'tc\'`/`\'time\'`, `\'psi\'`/'
             '`\'phase`\' or a key in `wf_params_at_point`.')

    # ----- Need numerical derivative -----
    param_center_val = wf_params_at_point[param_to_vary].value
    param_center_unit = wf_params_at_point[param_to_vary].unit

    # Set automatic step size, if given
    _deriv_kwargs = deriv_kwargs.copy()
    if 'base_step' not in deriv_kwargs:
        if 'start_step_size' in deriv_kwargs:
            # Allowed as alias
            _deriv_kwargs['base_step'] = _deriv_kwargs.pop('start_step_size')
        else:
            _deriv_kwargs['base_step'] = 1e-2*param_center_val

    def abs_wrapper(param_val):
        _wf_params_at_point = wf_params_at_point |{
            param_to_vary: param_val * param_center_unit
        }
        return np.abs(wf_generator(_wf_params_at_point).value)

    def phase_wrapper(param_val):
        _wf_params_at_point = wf_params_at_point |{
            param_to_vary: param_val * param_center_unit
        }
        return np.unwrap(np.angle(wf_generator(_wf_params_at_point).value))
    
    deriv_abs = nd.Derivative(abs_wrapper, **_deriv_kwargs)
    deriv_phase = nd.Derivative(phase_wrapper, **_deriv_kwargs)

    amp = np.abs(_wf_at_point).value
    pha = np.unwrap(np.angle(_wf_at_point)).value

    return FrequencySeries(
        (deriv_abs(param_center_val)
         + 1.j*amp*deriv_phase(param_center_val)) * np.exp(1j*pha),
        frequencies=_wf_at_point.frequencies,
        unit=_wf_at_point.unit/param_center_unit  # TODO: compose this?
    )
