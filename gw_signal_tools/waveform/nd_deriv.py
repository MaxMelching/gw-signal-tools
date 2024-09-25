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

__all__ = ('WaveformDerivativeNumdifftools', 'WaveformDerivativeAmplitudePhase')


class WaveformDerivativeNumdifftools(nd.Derivative):
    """
    Wrapper around numdifftools `Derivative` class specifically for
    waveform callers from new LAL interface

    Note: for a time domain model, you have to make sure that output
    always has the same size (and is defined on same times)!!!
    Otherwise the required operations do not work

    Note that most attributes here are immutable!


    Arbitrary function that is used for waveform generation. The
    required signature means that it has one non-optional argument,
    which is expected to accept the input provided in
    :code:`self.wf_params_at_point`.
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
        param_unit = self.param_center_val.unit
        self._wf_generator = wf_generator
        self._wf_params_at_point = wf_params_at_point
        self._param_to_vary = param_to_vary

        if 'base_step' not in kwds:
                kwds['base_step'] = 1e-2*self.param_center_val.value

        def fun(x):
            return self.wf_generator(wf_params_at_point | {param_to_vary: x*param_unit})
        # -- Next line stores this function in self.fun
        
        super().__init__(fun, *args, **kwds)
    
    def __call__(self, x=None) -> Any:
        """
        Get derivative at parameter value x.

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
            x = self.param_center_val.value
        elif isinstance(x, u.Quantity):
            x = x.to_value(self.param_center_val.unit)

        # -- Check if parameter has analytical derivative (cannot be in
        # -- previous check because dependent on point)
        if self.param_to_vary == 'distance':
            dist_val = x*self.param_center_val.unit
            wf = self.wf_generator(self.wf_params_at_point | {'distance': dist_val})
            deriv = (-1./dist_val) * wf

            self.deriv_info = {
                'description': 'This derivative is exact.'
            }
            return deriv
        
        self.full_output = True
        deriv, info = super().__call__(x)
        self.deriv_info = info._asdict()

        # TODO: use test_point function in case of Input domain error
        # -> could maybe adjust base_step and also the deriv routine
        # -> works like this: self.fd_rule.method = 'central'


        param_unit = self.param_center_val.unit

        wf = self.fun(x)
        # Idea: use type that wf_generator returns to have flexibility
        # with respect to whether TimeSeries/FrequencySeries is passed
        out = type(wf)(
            data=deriv,
            xindex=wf.xindex,
            unit=wf.unit / param_unit
        )

        self.error_estimate = type(wf)(
            data=info.error_estimate,
            xindex=wf.xindex,
            unit=wf.unit / param_unit
        )
        self.deriv_info['error_estimate'] = self.error_estimate

        return out
    
    # -- In case calling seems unintiutive, create attribute
    @property
    def deriv(self) -> Any:
        """Alias for calling with no arguments."""
        return self.__call__()
    
    # -- Define certain properties. These have NO setters, on purpose!
    @property
    def param_to_vary(self) -> str:
        """
        Parameter that derivative is taken with respect to.

        :type: `str`
        """
        return self._param_to_vary
    
    @property
    def param_center_val(self) -> u.Quantity:
        """
        Value of `self.param_to_vary` at which derivative is taken by
        default.

        :type: `~astropy.units.Quantity`
        """
        return self._param_center_val
    
    @property
    def wf_generator(self) -> Callable[[dict[str, u.Quantity]], FrequencySeries | TimeSeries]:
        """
        Generator for waveform model that is differentiated.

        :type: `Callable[[dict[str, ~astropy.units.Quantity]], ~gwpy.frequencyseries.FrequencySeries | ~gwpy.timeseries.TimeSeries]`
        """
        return self._wf_generator
    
    @property
    def wf_params_at_point(self) -> dict[str, u.Quantity]:
        """
        Point in parameter space at which waveform is differentiated,
        encoded as key-value pairs representing parameter-value pairs.

        :type: `dict[str, ~astropy.units.Quantity]`
        """
        return self._wf_params_at_point


# -- Now: fix bug in nd.Derivative, complex input throws error. This is
# -- due to numpy changes that were not (yet) addressed by numdifftools
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


class WaveformDerivativeAmplitudePhase():
    """
    Calculate numerical derivative using chain rule. Behaves potentially
    better mathematically, but also from code perspective we have
    difference to other derivatives: no straightforward way to get
    something like overall final_step_size from the ones for amplitude
    and phase, so class structure and attributes are very different

    disadvantage: has double the calls to waveforms. But in case other
    routines fail, it might be worth a try. Moreover, if we have
    waveform caching, this issue should disappear
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
        param_unit = self.param_center_val.unit
        self._wf_generator = wf_generator
        self._wf_params_at_point = wf_params_at_point
        self._param_to_vary = param_to_vary

        if 'base_step' not in kwds:
                kwds['base_step'] = 1e-2*self.param_center_val.value
        
        # def abs_wrapper(x):
        #     _wf_params_at_point = wf_params_at_point |{
        #         param_to_vary: x * param_unit
        #     }
        #     return np.abs(wf_generator(_wf_params_at_point).value)

        # def phase_wrapper(x):
        #     _wf_params_at_point = wf_params_at_point |{
        #         param_to_vary: x * param_unit
        #     }
        #     return np.unwrap(np.angle(wf_generator(_wf_params_at_point).value))

        # -- Defining fun turns out to be useful later on
        # TODO: decide if calling it in abs_wrapper, phase_wrapper makes sense
        # -> it does make nice code. And one additional call with single argument should not be too bad
        def fun(x):
            return self.wf_generator(wf_params_at_point | {param_to_vary: x*param_unit})
        self.fun = fun

        def abs_wrapper(x):
            return np.abs(self.fun(x).value)

        def phase_wrapper(x):
            return np.unwrap(np.angle(self.fun(x).value))
        
        self._abs_deriv = nd.Derivative(abs_wrapper, *args, **kwds)
        self._phase_deriv = nd.Derivative(phase_wrapper, *args, **kwds)
    
    def __call__(self, x=None) -> Any:
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
            x = self.param_center_val.value
        elif isinstance(x, u.Quantity):
            x = x.to_value(self.param_center_val.unit)        

        # -- Check if parameter has analytical derivative (cannot be in
        # -- previous check because dependent on point)
        if self.param_to_vary == 'distance':
            dist_val = x*self.param_center_val.unit
            wf = self.wf_generator(self.wf_params_at_point | {'distance': dist_val})
            deriv = (-1./dist_val) * wf

            self.deriv_info = {
                'description': 'This derivative is exact.'
            }
            return deriv
        

        self.abs_deriv.full_output = True
        abs_deriv, abs_info = self.abs_deriv(x)

        self.phase_deriv.full_output = True
        phase_deriv, phase_info = self.phase_deriv(x)

        self.deriv_info = {'abs': abs_info._asdict(),
                           'phase': phase_info._asdict()}

        # TODO: use test_point function in case of Input domain error
        # -> could maybe adjust base_step and also the deriv routine
        # -> works like this: self.fd_rule.method = 'central'


        param_unit = self.param_center_val.unit

        wf = self.fun(x)
        ampl = np.abs(wf).value
        phase = np.unwrap(np.angle(wf)).value
        # -- Following would be more future proof I think... But involves
        # -- more calls to function... So should we do it?
        # ampl = self.abs_fun(x)
        # phase = self.phase_fun(x)

        deriv = (abs_deriv + 1.j*ampl*phase_deriv) * np.exp(1j*phase)

        # Idea: use type that wf_generator returns to have flexibility
        # with respect to whether TimeSeries/FrequencySeries is passed
        out = type(wf)(
            data=deriv,
            xindex=wf.frequencies,
            unit=wf.unit / param_unit
        )

        return out
    
    @property
    def deriv(self) -> Any:
        """Alias for calling with no arguments."""
        return self.__call__()
    
    # -- Define certain properties. These have NO setters, on purpose!
    param_to_vary = WaveformDerivativeNumdifftools.param_to_vary
    param_center_val = WaveformDerivativeNumdifftools.param_center_val
    wf_generator = WaveformDerivativeNumdifftools.wf_generator
    wf_params_at_point = WaveformDerivativeNumdifftools.wf_params_at_point

    @property
    def abs_deriv(self) -> nd.Derivative:
        """
        Wrapper that calculates derivative of waveform amplitude.

        :type: `~numdifftools.core.Derivative`
        """
        return self._abs_deriv
    
    @property
    def abs_fun(self) -> Callable[[float], np.ndarray]:
        """
        Function that calculates the waveform amplitude.

        :type: `~Callable[[float], ~numpy.ndarray]`
        """
        return self.abs_deriv.fun
    
    @property
    def phase_deriv(self) -> nd.Derivative:
        """
        Wrapper that calculates derivative of waveform phase.

        :type: `~numdifftools.core.Derivative`
        """
        return self._phase_deriv
    
    @property
    def phase_fun(self) -> Callable[[float], np.ndarray]:
        """
        Function that calculates the waveform phase.

        :type: `~Callable[[float], ~numpy.ndarray]`
        """
        return self.phase_deriv.fun
