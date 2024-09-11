# -- Standard Lib Imports
from typing import Callable, Any

# -- Third Party Imports
import numdifftools as nd
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries


__doc__ = """
Module for `WaveformDerivativeNumdifftools` and
`WaveformDerivativeAmplitudePhase` classes.
"""


class WaveformDerivativeNumdifftools(nd.Derivative):
    """
    Wrapper around numdifftools `Derivative` class specifically for
    waveform callers from new LAL interface

    Note: for a time domain model, you have to make sure that output
    always has the same size (and is defined on same times)!!!
    Otherwise the required operations do not work
    """
    def __init__(
        self,
        wf_params_at_point: dict[str, u.Quantity],
        param_to_vary: str,
        wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries | TimeSeries],
        *args,
        **kwds
    ) -> None:
        # -- Check if parameter has analytical derivative
        if (param_to_vary == 'time' or param_to_vary == 'tc'):
            wf = wf_generator(wf_params_at_point)
            deriv = wf * (-1.j * 2. * np.pi * wf.frequencies)

            self.deriv_info = {
                'description': 'This derivative is exact.'
            }
            self._ana_deriv = deriv
            return None
        elif (param_to_vary == 'phase' or param_to_vary == 'psi'):
            wf = wf_generator(wf_params_at_point)
            
            if param_to_vary == 'phase':
                deriv = wf * 1.j / u.rad
            else:
                deriv = wf * 2.j / u.rad

            self.deriv_info = {
                'description': 'This derivative is exact.'
            }
            self._ana_deriv = deriv
            return None
        
        self._param_center_val = wf_params_at_point[param_to_vary]
        param_unit = self._param_center_val.unit
        self._wf_generator = wf_generator
        self._wf_params_at_point = wf_params_at_point
        self._param_to_vary = param_to_vary

        if 'base_step' not in kwds:
                kwds['base_step'] = 1e-2*self._param_center_val.value

        def fun(x):
            return wf_generator(wf_params_at_point | {param_to_vary: x*param_unit})
        
        
        # super().__init__(fun, step, method, order, n, **options)
        super().__init__(fun, *args, **kwds)
    
    def __call__(self, x=None, *args, **kwds) -> Any:
        """
        Get derivative at parameter value x.

        Note that derivative options cannot be passed here anymore! All
        args and kwds are passed to the function, i.e. the waveform
        generator!

        Return has same type as return of wf_generator. Should, in
        principle, be either FrequencySeries or TimeSeries, but we only
        rely on Series properties being defined and so it could also be
        just a regular GWpy Series

        information gathered during calculation is stored in self.deriv_info
        """
        # -- Check if analytical derivative has already been calculated
        if hasattr(self, '_ana_deriv'):
            return self._ana_deriv
        
        # -- Check selected arguments
        if x is None:
            x = self._param_center_val.value
        

        # -- Check if parameter has analytical derivative (cannot be in
        # -- previous check because dependent on point)
        if self._param_to_vary == 'distance':
            dist_val = x*self._param_center_val.unit
            wf = self._wf_generator(self._wf_params_at_point | {'distance': dist_val})
            deriv = (-1./dist_val) * wf

            # derivative_norm = norm(deriv, **self.inner_prod_kwargs)**2

            self.deriv_info = {
                # 'norm_squared': derivative_norm,
                'description': 'This derivative is exact.'
            }
            return deriv
        
        self.full_output = True
        deriv, info = super().__call__(x, *args, **kwds)
        self.deriv_info = info._asdict()

        # TODO: use test_point function in case of Input domain error
        # -> could maybe adjust base_step and also the deriv routine
        # -> works like this: self.fd_rule.method = 'central'


        param_unit = self._param_center_val.unit

        wf = self._wf_generator(self._wf_params_at_point)
        # Idea: use type that wf_generator returns to have flexibility
        # with respect to whether TimeSeries/FrequencySeries is passed
        out = type(wf)(
        # out = NDFrequencySeries(
            data=deriv,
            xindex=wf.frequencies,
            unit=wf.unit / param_unit
        )

        return out
    
    @property
    def deriv(self) -> Any:
        """Alias for calling with no arguments."""
        return self.__call__()


# -- Now: fix bug in nd.Derivative, complex input will throw error. This
# -- is due to some numpy changes that were not accounted for by nd
from numdifftools.limits import _Limit
import numpy as np
import warnings

def _add_error_to_outliers_fixed(der, trim_fact=10):
    """
    discard any estimate that differs wildly from the
    median of all estimates. A factor of 10 to 1 in either
    direction is probably wild enough here. The actual
    trimming factor is defined as a parameter.
    """
    if np.iscomplexobj(der):    
        return np.sqrt(
            _add_error_to_outliers_fixed(np.real(der), trim_fact)**2
            + _add_error_to_outliers_fixed(np.imag(der), trim_fact)**2
        )
    
    try:
        if np.any(np.isnan(der)):
            p25, median, p75 = np.nanpercentile(der, [25,50, 75], axis=0) 
        else:
            p25, median, p75 = np.percentile(der, [25,50, 75], axis=0)

        iqr = np.abs(p75 - p25)
    except ValueError as msg:
        warnings.warn(str(msg))
        return 0 * der

    a_median = np.abs(median)
    outliers = (((abs(der) < (a_median / trim_fact)) +
                (abs(der) > (a_median * trim_fact))) * (a_median > 1e-8) +
                ((der < p25 - 1.5 * iqr) + (p75 + 1.5 * iqr < der)))
    errors = outliers * np.abs(der - median)
    return errors

_Limit._add_error_to_outliers = staticmethod(_add_error_to_outliers_fixed)


# class AmplitudePhaseDerivative():
class WaveformDerivativeAmplitudePhase():
    """
    Calculate numerical derivative using chain rule. Behaves potentially
    better mathematically, but also from code perspective we have
    difference to other derivatives: no straightforward way to get
    something like overall final_step_size from the ones for amplitude
    and phase, so class structure and attributes are very different
    """
    # def abs_wrapper(param_val):
    #     _wf_params_at_point = wf_params_at_point |{
    #         param_to_vary: param_val * param_center_unit
    #     }
    #     return np.abs(wf_generator(_wf_params_at_point).value)

    # def phase_wrapper(param_val):
    #     _wf_params_at_point = wf_params_at_point |{
    #         param_to_vary: param_val * param_center_unit
    #     }
    #     return np.unwrap(np.angle(wf_generator(_wf_params_at_point).value))
    
    # deriv_abs = nd.Derivative(abs_wrapper, **_deriv_kwargs)
    # deriv_phase = nd.Derivative(phase_wrapper, **_deriv_kwargs)

    # amp = np.abs(_wf_at_point).value
    # pha = np.unwrap(np.angle(_wf_at_point)).value

    # return FrequencySeries(
    #     (deriv_abs(param_center_val)
    #      + 1.j*amp*deriv_phase(param_center_val)) * np.exp(1j*pha),
    #     frequencies=_wf_at_point.frequencies,
    #     unit=_wf_at_point.unit/param_center_unit  # TODO: compose this?
    # )
