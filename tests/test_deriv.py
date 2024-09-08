# ----- Standard Lib Imports -----
import unittest

# ----- Third Party Imports -----
import astropy.units as u
import matplotlib.pyplot as plt

# ----- Local Package Imports -----
from gw_signal_tools.waveform.utils import get_wf_generator
from gw_signal_tools.fisher import get_waveform_derivative_1D_with_convergence


#%% ----- Initializing commonly used variables for Fisher tests -----
f_min = 20.*u.Hz
f_max = 1024.*u.Hz

wf_params = {
    'total_mass': 100.*u.solMass,
    'mass_ratio': 0.42*u.dimensionless_unscaled,
    'deltaT': 1./2048.*u.s,
    'f22_start': f_min,
    'f_max': f_max,
    'deltaF': 2**-5*u.Hz,
    'f22_ref': 20.*u.Hz,
    'phi_ref': 0.*u.rad,
    'distance': 1.*u.Mpc,
    'inclination': 0.0*u.rad,
    'eccentricity': 0.*u.dimensionless_unscaled,
    'longAscNodes': 0.*u.rad,
    'meanPerAno': 0.*u.rad,
    'condition': 0
}


approximant = 'IMRPhenomXPHM'
wf_generator = get_wf_generator(approximant)

# Make sure mass1 and mass2 are not in default_dict (makes messy behaviour)
import lalsimulation.gwsignal.core.parameter_conventions as pc
pc.default_dict.pop('mass1', None);
pc.default_dict.pop('mass2', None);


from gw_signal_tools.waveform.deriv import Derivative

param_to_vary = 'total_mass'
# param_to_vary = 'mass_ratio'
# param_to_vary = 'distance'


# -- Test consistency with old function

test_deriv_object = Derivative(
    wf_params_at_point=wf_params,
    param_to_vary=param_to_vary,
    wf_generator=wf_generator
)


test_deriv = test_deriv_object.deriv
test_deriv_object._deriv
test_deriv_object.deriv


test_deriv_2, info_2 = get_waveform_derivative_1D_with_convergence(
    wf_params_at_point=wf_params,
    param_to_vary=param_to_vary,
    wf_generator=wf_generator,
    return_info=True
)
plt.close()
# from gw_signal_tools.fisher.fisher_utils import get_waveform_derivative_1D_numdifftools
# test_deriv_3 = get_waveform_derivative_1D_numdifftools(
#     wf_params_at_point=wf_params,
#     param_to_vary=param_to_vary,
#     wf_generator=wf_generator,
# )


print(info_2)
print(test_deriv_object.deriv_info)
print(test_deriv_object.step_sizes, test_deriv_object._convergence_vals,
      test_deriv_object.min_dev_index, test_deriv_object.refine_numb,
      test_deriv_object.final_step_size, test_deriv_object.final_convergence_val)

# plt.plot(test_deriv)
# plt.plot(test_deriv_2, '--')
# plt.plot(test_deriv_3, ':')
plt.plot(test_deriv - test_deriv_2)
plt.show()