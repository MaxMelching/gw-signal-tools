from typing import Any, Optional, Literal

import numpy as np
import astropy.units as u
from gwpy.types import Series

from gw_signal_tools.fisher import phenomx_generator, FisherMatrix
from ..inner_product import inner_product, norm, overlap
from ..matrix_with_units import MatrixWithUnits


def distance(
    param_to_vary: str,
    param_range: list[u.Quantity, u.Quantity],
    wf_params: dict[str, u.Quantity],
    step_size: u.Quantity | float,
    distance_kind: Literal['diff_norm', 'mismatch_norm'],
    wf_generator: Optional[Any] = None,
    **inner_prod_kwargs
) -> u.Quantity:
    if wf_generator is None:
        wf_generator = phenomx_generator

    center_val = wf_params[param_to_vary]

    assert len(param_range) == 2, \
        '`param_range` must contain exactly two elements'

    # TODO: decide if float param_ranges should be permitted. Yes for convenience,
    # but no for error source (mightbe given - unintended - in different units)
    assert (param_range[0] <= center_val) and (center_val <= param_range[1]), \
        '`center_val` has to be in `param_range`.'
    
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
                distance_val = (norm_wf[1]**2 + norm_center_wf[1]**2 - 2*overlap[1])**(1/2)
        elif distance_kind == 'mismatch_norm':
            overlap = inner_product(wf, center_wf, **inner_prod_kwargs)

            if isinstance(overlap, u.Quantity):
                distance_val = 1 - overlap
            else:
                distance_val = 1 - overlap[1]
            
            # distance_val = 1 - overlap(wf, center_wf, **inner_prod_kwargs)
            # distance_val = norm(center_wf, **inner_prod_kwargs)**2 - inner_product(wf, center_wf, **inner_prod_kwargs)

        if not isinstance(distance_val, u.Quantity):
            distance_val = distance_val[1]

        distances += [
            distance_val
        ]

    # distances = u.Quantity(distances)  # Series should take care of that
    return Series(
        distances,
        # unit=distances.unit,
        xindex=param_vals
    )

def linearized_distance(
    param_to_vary: str | list[str],
    param_range: list[u.Quantity, u.Quantity],
    wf_params: dict[str, u.Quantity],
    step_size: u.Quantity | float,
    params_to_project: Optional[list[str]] = None,
    wf_generator: Optional[Any] = None,
    **inner_prod_kwargs
) -> u.Quantity:
    if wf_generator is None:
        wf_generator = phenomx_generator
    
    if not isinstance(param_to_vary, str):
        if len(param_to_vary) == 1:
            # Only one element given, fine
            param_to_vary = param_to_vary[0]
        elif params_to_project is not None:
            # A projection is carried out
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

    # TODO: decide if float param_ranges should be permitted. Yes for convenience,
    # but no for error source (mightbe given - unintended - in different units)
    assert (param_range[0] <= center_val) and (center_val <= param_range[1]), \
        '`center_val` has to be in `param_range`.'
    
    step_size = u.Quantity(step_size, unit=center_val.unit)
    
    param_vals = np.arange(
        param_range[0].value,
        (param_range[1] + 0.9*step_size).value,
        step=step_size.value
    )*center_val.unit
    
    if params_to_project is None:
        fisher = FisherMatrix(wf_params, param_to_vary, wf_generator, return_info=False, **inner_prod_kwargs)
    else:
        param_to_vary = [param_to_vary] + params_to_project
        full_fisher = FisherMatrix(wf_params, param_to_vary, wf_generator, return_info=False, **inner_prod_kwargs)

        fisher = full_fisher.project_fisher(params_to_project)

    if params_to_project is not None:
        non_proj_indices = [i for i, val in enumerate(param_to_vary) if val not in params_to_project]
        index = non_proj_indices[0]
    else:
        index = 0

    return Series(
        # fisher[index, index]**(1/2)*np.abs(param_vals - center_val),
        MatrixWithUnits.sqrt(fisher)[index, index]*np.abs(param_vals - center_val),
        xindex=param_vals
    )
