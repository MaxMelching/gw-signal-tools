# ---------- Standard Lib Imports ----------
from typing import Optional, Literal, Callable

# ---------- Third Party Imports ----------
import numpy as np
import astropy.units as u
from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries

# ---------- Local Package Imports ----------
from ..fisher import FisherMatrix
from ..inner_product import inner_product, norm, overlap


__doc__ = """
Module that allows convenient calculation of various distances that can
be defined on signal manifolds in gravitational wave data analysis.
"""


def distance(
    param_to_vary: str,
    param_vals: u.Quantity | float | list[u.Quantity | float],
    wf_params: dict[str, u.Quantity],
    wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
    param_step_size: Optional[u.Quantity | float] = None,
    distance_kind: Literal['diff_norm', 'mismatch_norm'] = 'diff_norm',
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
    param_vals : ~astropy.units.Quantity | float | list[~astropy.units.
    Quantity] | list[float]
        Interval over which distances for different values of
        `param_to_vary` are calculated. The way this is input is
        interpreted depends on the value of `param_step_size`: if this
        is None, the distances are evaluated at the points given in
        `param_vals`; if it is not None, the interval is resampled in
        the bounds given by first and last element of `param_vals`.

        Note that it will be converted to an astropy Quantity with the
        same unit that `param_to_vary` has in `wf_params`. Thus, if no
        units are specified for `param_vals` (i.e. a list of floats is
        passed), make sure the values are given in the correct units. To
        avoid potential inconsistencies, giving a Quantity as input here
        is recommended.
    wf_params : dict[str, ~astropy.units.Quantity]
        Waveform parameters specifying the point with respect to which
        the distances are calculated. Will be given as input to
        `wf_generator` and must contain the `param_to_vary` as key.
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]],
    FrequencySeries]
        Function that takes dicionary of waveform parameters as input
        and produces a waveform (stored in a GWpy ``FrequencySeries``).
    param_step_size : ~astropy.units.Quantity | float, default = None
        Step size of points in the discretized interval `param_vals`, at
        which the distance values will be calculated. Can be a ``float``
        or an astropy Quantity. Its effect is described in the
        description of `param_vals`.

        Note that it will be converted to an astropy Quantity with the
        same unit that `param_to_vary` has in `wf_params`. Thus, if no
        units are specified for `param_step_size` (i.e. a list of floats
        is passed), make sure the values are given in the correct units.
        To avoid potential inconsistencies, giving a Quantity as input
        here is recommended.
    distance_kind : Literal['diff_norm', 'mismatch_norm'], optional,
    default = 'diff_norm'
        Distance notion to use. At the moment, two possibilities can be
        selected, 'diff_norm' and 'mismatch_norm'. For details on them,
        refer to the **Notes** section.

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

    .. math:: 1 - \frac{\langle h_1, h_2 \rangle}{||h_1|| \cdot ||h_2||}

    in the same language as before (i.e. same inner product etc.).
    """
    center_val = wf_params[param_to_vary]

    # ----- Parameter range handling -----
    param_vals = u.Quantity(param_vals, unit=center_val.unit)

    assert (param_vals[0] <= center_val) and (center_val <= param_vals[-1]), \
        ('The value of `param_to_vary` provided in `wf_params` has to be in '
         'the interval given by `param_vals`.')
    
    # Check if new interval shall be created, otherwise use param_vals
    if param_step_size is not None:
        param_step_size = u.Quantity(param_step_size, unit=center_val.unit)
        
        param_vals = np.arange(
            param_vals[0].value,
            (param_vals[-1] + 0.9*param_step_size).value,
            step=param_step_size.value
        )*center_val.unit

    # ----- Initialization of variables and calculation -----
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
                wf_overlap = inner_product(wf, center_wf, **inner_prod_kwargs)
                # distance_val = (norm_wf[1]**2 + norm_center_wf[1]**2 - 2*wf_overlap[1])**(1/2)
                distance_val = np.sqrt(norm_wf[1]**2 + norm_center_wf[1]**2 - 2*wf_overlap[1])
        elif distance_kind == 'mismatch_norm':
            # wf_overlap = inner_product(wf, center_wf, **inner_prod_kwargs)
            wf_overlap = overlap(wf, center_wf, **inner_prod_kwargs)

            if isinstance(wf_overlap, u.Quantity):
                distance_val = 1 - wf_overlap
            else:
                distance_val = 1 - wf_overlap[1]
            
            # distance_val = 1 - wf_overlap(wf, center_wf, **inner_prod_kwargs)
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
    param_vals: u.Quantity | float | list[u.Quantity | float],
    wf_params: dict[str, u.Quantity],
    wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
    params_to_project: Optional[list[str]] = None,
    param_step_size: Optional[u.Quantity] = None,
    **inner_prod_kwargs
) -> Series:
    r"""
    Calculate the approximate distance between waveforms upon variation
    of a single parameter over a given range. The approximation used
    here is a simple linearization around the point that distances are
    calculated with respect to (:math:`\theta_1`),

    .. math::
        ||h_1 - h_2|| = ||h(\theta_1) - h(\theta_2)||
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
    param_vals : ~astropy.units.Quantity | float | list[~astropy.units.Quantity] | list[float]
        Interval over which distances for different values of
        :code:`param_to_vary` are calculated. The way this is input is
        interpreted depends on the value of :code:`param_step_size`:
        if this is None, the distances are evaluated at the points given
        in :code:`param_vals`; if it is not None, the interval is
        resampled in the bounds given by first and last element of
        :code:`param_vals`.

        Note that it will be converted to an astropy Quantity with the
        same unit that :code:`param_to_vary` has in :code:`wf_params`.
        Thus, if no units are specified for :code:`param_vals` (i.e. a
        list of floats is passed), make sure the values are given in the
        correct units. To avoid potential inconsistencies, giving an
        astropy ``Quantity`` as input here is recommended.
    wf_params : dict[str, ~astropy.units.Quantity]
        Waveform parameters specifying the point with respect to which
        the distances are calculated. Will be given as input to
        :code:`wf_generator` and must contain :code:`param_to_vary` as
        a key.
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]],FrequencySeries]
        Function that takes dicionary of waveform parameters as input
        and produces a waveform (stored in a GWpy ``FrequencySeries``).
    params_to_project : str | list[str], optional, default = None
        One or multiple parameters that the linearized distance will
        be optimized over (by projecting the Fisher matrix on the
        subspace that is orthogonal to this parameter).

        Need not be in :code:`param_to_vary`, but can as long as just a
        single element of :code:`param_to_vary` is not in
        :code:`params_to_project`.
    param_step_size : ~astropy.units.Quantity | float, default = None
        Step size of points in the discretized interval
        :code:`param_vals`, at which the distance values will be
        calculated. Can be a ``float`` or an astropy ``Quantity``. Its
        effect is described in the description of :code:`param_vals`.

        Note that it will be converted to an astropy Quantity with the
        same unit that :code:`param_to_vary` has in :code:`wf_params`.
        Thus, if no units are specified for :code:`param_step_size`
        (i.e. a list of floats is passed), make sure the values are
        given in the correct units. To avoid potential inconsistencies,
        giving an astropy ``Quantity`` as input here is recommended.

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
    # ----- Parameter input handling -----
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
    
    if params_to_project is None:
        full_fisher = FisherMatrix(wf_params, param_to_vary, wf_generator,
                                   return_info=False, **inner_prod_kwargs)
        fisher = full_fisher.fisher
    else:
        param_to_vary = [param_to_vary] + params_to_project
        full_fisher = FisherMatrix(wf_params, param_to_vary, wf_generator,
                                   return_info=False, **inner_prod_kwargs)
        fisher = full_fisher.project_fisher(params_to_project).fisher

    if params_to_project is not None:
        non_proj_indices = [i for i, val in enumerate(param_to_vary) \
                            if val not in params_to_project]
        index = non_proj_indices[0]
    else:
        index = 0

    # ----- Parameter range handling -----
    param_vals = u.Quantity(param_vals, unit=center_val.unit)

    assert (param_vals[0] <= center_val) and (center_val <= param_vals[-1]), \
        ('The value of `param_to_vary` provided in `wf_params` has to be in '
         'the interval given by `param_vals`.')
    
    # Check if new interval shall be created, otherwise use param_vals
    if param_step_size is not None:
        param_step_size = u.Quantity(param_step_size, unit=center_val.unit)
        
        param_vals = np.arange(
            param_vals[0].value,
            (param_vals[-1] + 0.9*param_step_size).value,
            step=param_step_size.value
        )*center_val.unit

    return Series(
        # MatrixWithUnits.sqrt(abs(fisher))[index, index]*np.abs(param_vals - center_val),
        # Hmmm we should not need abs right?
        # -> some entries might be negative, rather take sqrt only of
        #    relevant entry (which is non-negative) to avoid error
        np.sqrt(fisher[index, index])*np.abs(param_vals - center_val),
        xindex=param_vals
    )
