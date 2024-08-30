import numdifftools as nd

# from gwpy.frequencyseries import FrequencySeries
# class NDFrequencySeries(FrequencySeries):
#     _ndim=2
# -- From testing with n-dim output. Did not use in the end


class WaveformDerivative(nd.Derivative):
    """
    Note: for a time domain model, you have to make sure that output
    always has the same size!!! Otherwise operations do not work
    """
    def __init__(self, wf_generator, wf_params_at_point, param_to_vary,
                #  step=None, method='central', order=2, n=1, **options):
                 *args, **kwds):
        self._param_center_val = wf_params_at_point[param_to_vary]
        param_unit = self._param_center_val.unit
        self._wf_generator = wf_generator
        self._wf_params_at_point = wf_params_at_point
        self._param_to_vary = param_to_vary

        def fun(x):
            return wf_generator(wf_params_at_point | {param_to_vary: x*param_unit})#.value
        
            # -- Testing n-dim output
            # wf = wf_generator(wf_params_at_point | {param_to_vary: x*param_unit})

            # return np.stack([wf, wf])

            # WORKS!!! This is great, means that we can easily pass a
            # NDWaveform as well, right?
            # -> but definitely check if each row is handled separately
            #    or if things are handled for each column (don't think
            #    so, but we should make sure; otherwise just make
            #    separate calls to the derivative)
        
        
        # super().__init__(fun, step, method, order, n, **options)
        super().__init__(fun, *args, **kwds)
    
    def __call__(self, x=None, *args, **kwds):
        if x is None:
            x = self._param_center_val.value
        
        deriv = super().__call__(x, *args, **kwds)

        # TODO: use test_point function in case of Input domain error
        # -> could maybe adjust base_step and also the deriv routine

        param_unit = self._param_center_val.unit
        # out = self._wf_generator(self._wf_params_at_point)
        # # out = self._wf_generator(self._wf_params_at_point
        # #                          | {self._param_to_vary: x*param_unit})
        # out._value = deriv  # DOES NOT WORK!!!
        # out.override_unit(out.unit / param_unit)

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
        # return deriv  # Testing n-dim output
    
    @property
    def deriv(self):
        return self.__call__()


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
        # return (_add_error_to_outliers_fixed(np.real(der), trim_fact)
        #         + _add_error_to_outliers_fixed(np.imag(der), trim_fact))
    
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


# -- Multi-function Testing
func1_counter = 0
func2_counter = 0

def func1(x):
    global func1_counter
    func1_counter += 1
    # return x**2
    return np.sin(x)

def func2(x):
    global func2_counter
    func2_counter += 1
    return np.exp(x)

def func(x):
    return np.stack([func1(x), func2(x)])

func_deriv = nd.Derivative(
    func,
    base_step=1,  # To provoke slower convergence, test if there is difference
    full_output=True
)

# point = 3
point = np.linspace(0, 2, num=5)
num_deriv, info = func_deriv(point)
print(np.vstack([np.cos(point), np.exp(point)]))
print(num_deriv)

print(func1_counter, func2_counter)
print(info.final_step)
# Ok, so both counters are equal, which of course makes sense because
# they are called simultaneously. final_step is more important and it
# indeed shows that every entry is handled separately (i.e. each row is,
# just like each column is)
# -> that means from a calling perspective, NDWaveformGenerator would
#    work with this Derivative class here. Would also be convenient
#    because return of attributes would be handled well. On the other
#    hand, it could be that we accumulate waveform calls although the
#    corresponding derivative is already converged... But I would think
#    this discrepancy should not be too large, so perhaps code clarity
#    is king here


# -- Testing
# from gw_signal_tools.waveform_utils import get_wf_generator
# import astropy.units as u
# import matplotlib.pyplot as plt

# f_min = 20.*u.Hz
# f_max = 1024.*u.Hz

# wf_params = {
#     'total_mass': 100.*u.solMass,
#     'mass_ratio': 0.42*u.dimensionless_unscaled,
#     'deltaT': 1./2048.*u.s,
#     'f22_start': f_min,
#     'f_max': f_max,
#     'deltaF': 2**-5*u.Hz,
#     'f22_ref': 20.*u.Hz,
#     'phi_ref': 0.*u.rad,
#     'distance': 1.*u.Mpc,
#     'inclination': 0.0*u.rad,
#     'eccentricity': 0.*u.dimensionless_unscaled,
#     'longAscNodes': 0.*u.rad,
#     'meanPerAno': 0.*u.rad,
#     'condition': 0
# }

# test_params = ['total_mass', 'mass_ratio']


# approximant = 'IMRPhenomXPHM'
# wf_generator = get_wf_generator(approximant)#, mode='mixed')

# # Make sure mass1 and mass2 are not in default_dict (makes messy behaviour)
# import lalsimulation.gwsignal.core.parameter_conventions as pc
# pc.default_dict.pop('mass1', None);
# pc.default_dict.pop('mass2', None);

# # test_param = 'total_mass'
# test_param = 'mass_ratio'
# # test_param = 'distance'
# test = WaveformDerivative(
#     wf_generator,
#     wf_params,
#     test_param,
#     # base_step=1e-2
#     base_step=1e-2*wf_params[test_param].value,
#     # method='forward'
#     # method='complex'  # Does not work for complex input
# )


# from gw_signal_tools.types.deriv import Derivative
# # print(Derivative.__dict__)
# # print('five_point' in Derivative.__dict__)


# test_deriv_object = Derivative(
#     wf_params_at_point=wf_params,
#     param_to_vary=test_param,
#     wf_generator=wf_generator
# )

# test_deriv = test_deriv_object.deriv

# from gw_signal_tools.fisher.fisher_utils import get_waveform_derivative_1D_numdifftools
# test_deriv_3 = get_waveform_derivative_1D_numdifftools(
#     wf_params_at_point=wf_params,
#     param_to_vary=test_param,
#     wf_generator=wf_generator,
# )

# # plt.plot(test())
# plt.plot(test.deriv)
# plt.plot(test_deriv, '--')
# plt.plot(test_deriv_3, ':')


# plt.show()


# # -- Testing n-dim output
# # deriv = test.deriv
# # print(deriv)
# # print(np.allclose(deriv[0], deriv[1], atol=0., rtol=0.))
