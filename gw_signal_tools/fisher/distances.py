# ---------- Standard Lib Imports ----------
from typing import Any, Optional, Literal, Callable

# ---------- Third Party Imports ----------
import numpy as np
import astropy.units as u

from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries

# ---------- Local Package Imports ----------
from ..fisher import phenomx_generator, FisherMatrix
from ..inner_product import inner_product, norm, overlap
from ..matrix_with_units import MatrixWithUnits


# TODO: make interval handling more sophisticated? Example idea:
# param_range can be both limits of interval or whole interval,
# depending on the value of step_size. If None, it is assumed that
# param_range contains all points to evaluate at, and if step_size
# has certain value, an interval will be constructed

def distance(
    param_to_vary: str,
    param_range: list[u.Quantity, u.Quantity],
    wf_params: dict[str, u.Quantity],
    step_size: u.Quantity | float,
    distance_kind: Literal['diff_norm', 'mismatch_norm'] = 'diff_norm',
    wf_generator: Optional[Callable[[dict[str, u.Quantity]],
                                    FrequencySeries]] = None,
    **inner_prod_kwargs
) -> Series:
    r"""
    Calculate the distance between waveforms upon variation of a single
    parameter over a given range.

    Parameters
    ----------
    param_to_vary : str
        Parameter that is varied arount the respective value given in
        `wf_params`.
    param_range : list[u.Quantity, u.Quantity]
        Range to over which distances for different values of
        `param_to_vary` are calculated.
    wf_params : dict[str, u.Quantity]
        Waveform parameters specifying the point with respect to which
        the distances are calculated. Will be given as input to
        `wf_generator` and must contain the `param_to_vary` as key.
    step_size : u.Quantity | float
        Step size of points in the discretized interval `param_range`,
        at which the distance values will be calculated.
    distance_kind : Literal['diff_norm', 'mismatch_norm'], optional,
    default = 'diff_norm'
        Distance notion to use. At the moment, two possibilities can be
        selected, 'diff_norm' and 'mismatch_norm'. For details on them,
        refer to the **Notes** section.
    wf_generator : Callable[[dict[str, u.Quantity]], FrequencySeries],
    optional, default = None
        Function that takes dicionary of waveform parameters as input
        and produces a waveform (stored in a GWpy ``FrequencySeries``).

    Returns
    -------
    ~gwpy.types.Series
        A ``Series`` that contains the distances calculated, along with
        the parameter values that the distance has been calculated at.

    Raises
    ------
    ValueError
        For an invalid `distance_kind`.
    
    Notes
    -----
    Here we want to provide a more detailed explanation of the possible
    choices for `distance_kind`.

    'diff_norm' represents

    .. math:: ||h_1 - h_2|| = ||h(\theta_1) - h(\theta_2)||

    and thus the norm of the difference of waveforms generated at
    different points :math:`\theta_1, \theta_2` in the parameter space.
    The norm referred to here is the familiar noise-weighted inner
    product from gravitational wave data analysis and the
    implementation from `~gw_signal_tools.inner_product.inner_product`
    is used here.

    'mismatch_norm' represents the mismatch

    .. math:: 1 - \langle h_1, h_2 \rangle

    in the same language as before (i.e. same inner product etc.).
    """
    if wf_generator is None:
        wf_generator = phenomx_generator

    center_val = wf_params[param_to_vary]

    assert len(param_range) == 2, \
        '`param_range` must contain exactly two elements'

    assert (param_range[0] <= center_val) and (center_val <= param_range[1]), \
        ('The value of `param_to_vary` provided in `wf_params` has to be in '
         'the interval given by `param_range`.')
    
    step_size = u.Quantity(step_size, unit=center_val.unit)
    
    param_vals = np.arange(
        param_range[0].value,
        (param_range[1] + 0.9*step_size).value,
        step=step_size.value
    )*center_val.unit

    distances = []
    
    center_wf = wf_generator(wf_params)

    norm_center_wf = norm(center_wf, **inner_prod_kwargs)

    for param_val in param_vals:
        wf = wf_generator(wf_params | {param_to_vary: param_val})

        if distance_kind == 'diff_norm':  
            if isinstance(norm_center_wf, u.Quantity):
                distance_val = norm(wf - center_wf, **inner_prod_kwargs)
            else:
                # Optimization is wanted, have to rewrite for this to take action
                norm_wf = norm(wf, **inner_prod_kwargs)
                overlap = inner_product(wf, center_wf, **inner_prod_kwargs)
                # distance_val = (norm_wf[1]**2 + norm_center_wf[1]**2 - 2*overlap[1])**(1/2)
                distance_val = np.sqrt(norm_wf[1]**2 + norm_center_wf[1]**2 - 2*overlap[1])
        elif distance_kind == 'mismatch_norm':
            overlap = inner_product(wf, center_wf, **inner_prod_kwargs)

            if isinstance(overlap, u.Quantity):
                distance_val = 1 - overlap
            else:
                distance_val = 1 - overlap[1]
            
            # distance_val = 1 - overlap(wf, center_wf, **inner_prod_kwargs)
            # distance_val = norm(center_wf, **inner_prod_kwargs)**2 - inner_product(wf, center_wf, **inner_prod_kwargs)
        else:
            raise ValueError('Invalid `distance_kind` is given.')

        distances += [
            distance_val
        ]

    return Series(
        distances,
        xindex=param_vals
    )

def linearized_distance(
    param_to_vary: str | list[str],
    param_range: list[u.Quantity, u.Quantity],
    wf_params: dict[str, u.Quantity],
    step_size: u.Quantity | float,
    params_to_project: Optional[list[str]] = None,
    wf_generator: Optional[Callable[[dict[str, u.Quantity]],
                                    FrequencySeries]] = None,
    **inner_prod_kwargs
) -> Series:
    r"""
    Calculate the approximate distance between waveforms upon variation
    of a single parameter over a given range. The approximation used
    here is a simple linearization around the point that distances are
    calculated with respect to (:math:`\theta_1`),

    .. math:: ||h_1 - h_2|| = ||h(\theta_1) - h(\theta_2)||
    \simeq \sqrt{\Gamma_{\mu \nu} (\theta_1 - \theta_2)^2}

    where :math:`\Gamma_{\mu \nu}` is the Fisher matrix (evaluated at
    :math:`\theta_1`).

    Parameters
    ----------
    param_to_vary : str | list[str]
        Parameters that the Fisher matrix is calculated for. In
        principle, this should be just one parameter. However, giving
        more than one is permitted as long as they are also in
        `params_to_project`.
    param_range : list[u.Quantity, u.Quantity]
        Range to over which distances for different values of
        `param_to_vary` are calculated.
    wf_params : dict[str, u.Quantity]
        Waveform parameters specifying the point with respect to which
        the distances are calculated. Will be given as input to
        `wf_generator` and must contain the `param_to_vary` as key.
    step_size : u.Quantity | float
        Step size of points in the discretized interval `param_range`,
        at which the distance values will be calculated.
    params_to_project : str | list[str], optional, default = None
        One or multiple parameters that the linearized distance will
        be optimized over (by projecting the Fisher matrix on the
        subspace that is orthogonal to this parameter).

        Need not be in `param_to_vary`, but can as long as just a single
        element of `param_to_vary` is not in `params_to_project`.
    wf_generator : Callable[[dict[str, u.Quantity]], FrequencySeries],
    optional, default = None
        Function that takes dicionary of waveform parameters as input
        and produces a waveform (stored in a GWpy ``FrequencySeries``).

    Returns
    -------
    ~gwpy.types.Series
        A ``Series`` that contains the distances calculated, along with
        the parameter values that the distance has been calculated at.

    Raises
    ------
    ValueError
        For invalid combinations of `param_to_vary` and
        `params_to_project`.
    """
    if wf_generator is None:
        wf_generator = phenomx_generator
    
    if not isinstance(param_to_vary, str):
        if len(param_to_vary) == 1:
            # Only one element given, fine
            param_to_vary = param_to_vary[0]
        elif params_to_project is not None:
            # A projection is carried out
            if isinstance(params_to_project, str):
                params_to_project = [params_to_project]
            
            non_proj_params = [val for val in param_to_vary if val not in params_to_project]

            if len(non_proj_params) != 1:
                raise ValueError(
                    'Invalid `param_to_vary` input is given, more than one '
                    'parameter remains after projection.'
                )
            
            param_to_vary = non_proj_params[0]
        else:
            raise ValueError(
                'Invalid `param_to_vary` input is given, contains more than '
                'one parameter (and no parameters to project on).'
            )

    center_val = wf_params[param_to_vary]

    assert len(param_range) == 2, \
        '`param_range` must contain exactly two elements'

    assert (param_range[0] <= center_val) and (center_val <= param_range[1]), \
        ('The value of `param_to_vary` provided in `wf_params` has to be in '
         'the interval given by `param_range`.')
    
    step_size = u.Quantity(step_size, unit=center_val.unit)
    
    param_vals = np.arange(
        param_range[0].value,
        (param_range[1] + 0.9*step_size).value,
        step=step_size.value
    )*center_val.unit
    
    if params_to_project is None:
        full_fisher = FisherMatrix(wf_params, param_to_vary, wf_generator,
                                   return_info=False, **inner_prod_kwargs)
        fisher = full_fisher.fisher
    else:
        param_to_vary = [param_to_vary] + params_to_project
        full_fisher = FisherMatrix(wf_params, param_to_vary, wf_generator,
                                   return_info=False, **inner_prod_kwargs)
        fisher = full_fisher.project_fisher(params_to_project)

    if params_to_project is not None:
        non_proj_indices = [i for i, val in enumerate(param_to_vary) \
                            if val not in params_to_project]
        index = non_proj_indices[0]
    else:
        index = 0

    return Series(
        # MatrixWithUnits.sqrt(abs(fisher))[index, index]*np.abs(param_vals - center_val),
        # Hmmm we should not need abs right?
        # -> some entries might be negative, rather take sqrt only of
        #    relevant entry (which is non-negative) to avoid error
        np.sqrt(fisher[index, index])*np.abs(param_vals - center_val),
        xindex=param_vals
    )
