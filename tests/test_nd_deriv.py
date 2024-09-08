# ----- Third Party Imports -----
import astropy.units as u
import matplotlib.pyplot as plt

# ----- Local Package Imports -----
from gw_signal_tools.waveform.utils import get_wf_generator
from gw_signal_tools.waveform.nd_deriv import WaveformDerivative
from gw_signal_tools.waveform.deriv import Derivative


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

test_params = ['total_mass', 'mass_ratio']


approximant = 'IMRPhenomXPHM'
wf_generator = get_wf_generator(approximant)#, mode='mixed')

# Make sure mass1 and mass2 are not in default_dict (makes messy behaviour)
import lalsimulation.gwsignal.core.parameter_conventions as pc
pc.default_dict.pop('mass1', None);
pc.default_dict.pop('mass2', None);

# test_param = 'total_mass'
test_param = 'mass_ratio'
# test_param = 'distance'
test = WaveformDerivative(
    wf_generator,
    wf_params,
    test_param,
    # base_step=1e-2
    base_step=1e-2*wf_params[test_param].value,
    # method='forward'
    # method='complex'  # Does not work for complex input
)


# -- Comparison with custom Derivative class
# print(Derivative.__dict__)
# print('five_point' in Derivative.__dict__)


test_deriv_object = Derivative(
    wf_params_at_point=wf_params,
    param_to_vary=test_param,
    wf_generator=wf_generator
)

test_deriv = test_deriv_object.deriv

from gw_signal_tools.fisher.fisher_utils import get_waveform_derivative_1D_numdifftools
test_deriv_3 = get_waveform_derivative_1D_numdifftools(
    wf_params_at_point=wf_params,
    param_to_vary=test_param,
    wf_generator=wf_generator,
)

# plt.plot(test())
plt.plot(test.deriv)
plt.plot(test_deriv, '--')
plt.plot(test_deriv_3, ':')


plt.show()


# -- Testing n-dim output
# deriv = test.deriv
# print(deriv)
# print(np.allclose(deriv[0], deriv[1], atol=0., rtol=0.))
