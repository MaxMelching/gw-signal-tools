from typing import Any, Optional, Literal

import numpy as np
import astropy.units as u
from gwpy.types import Series

from gw_signal_tools.fisher import phenomx_generator
from ..inner_product import inner_product, norm, overlap


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
    # distances = u.Quantity(np.zeros(len(param_vals)))
    
    center_wf = wf_generator(wf_params)

    norm_center_wf = norm(center_wf, **inner_prod_kwargs)

    # for i, param_val in enumerate(param_vals):
    #     wf = phenomx_generator(wf_params | {param_to_vary: param_val})

    #     if distance_kind == 'diff_norm':
    #         distances[i] = norm(wf - center_wf, **inner_prod_kwargs)
    #     elif distance_kind == 'mismatch_norm':
    #         distances[i] = overlap(wf, center_wf, **inner_prod_kwargs)

    for param_val in param_vals:
        wf = wf_generator(wf_params | {param_to_vary: param_val})

        if distance_kind == 'diff_norm':
            # distance_val = norm(wf - center_wf, **inner_prod_kwargs)
            
            # norm_wf = norm(wf, **inner_prod_kwargs)
            # overlap = inner_product(wf, center_wf, **inner_prod_kwargs)
  
            # if isinstance(norm_center_wf, u.Quantity):
            #     distance_val = (norm_wf**2 + norm_center_wf**2 - 2*overlap)**(1/2)
            # else:
            #     distance_val = (norm_wf[1]**2 + norm_center_wf[1]**2 - 2*overlap[1])**(1/2)

  
            if isinstance(norm_center_wf, u.Quantity):
                distance_val = norm(wf - center_wf, **inner_prod_kwargs)
            else:
                # Optimization is wanted, have to rewrite for this to take action
                norm_wf = norm(wf, **inner_prod_kwargs)
                overlap = inner_product(wf, center_wf, **inner_prod_kwargs)
                distance_val = (norm_wf[1]**2 + norm_center_wf[1]**2 - 2*overlap[1])**(1/2)
        elif distance_kind == 'mismatch_norm':
            # distance_val = 1 - overlap(wf, center_wf, **inner_prod_kwargs)
            # inner_prod_kwargs['optimize_time_and_phase'] = True
            # distance_val = 1 - inner_product(wf, center_wf, **inner_prod_kwargs)[1]
            distance_val = 1 - inner_product(wf, center_wf, **inner_prod_kwargs)
            # TODO: support optimization
            # distance_val = norm(center_wf, **inner_prod_kwargs)**2 - inner_product(wf, center_wf, **inner_prod_kwargs)

        if not isinstance(distance_val, u.Quantity):
            distance_val = distance_val[1]

        distances += [
            distance_val
        ]

    # return distances

    distances = u.Quantity(distances)
    return Series(
        distances,
        unit=distances.unit,
        xindex=param_vals
    )
