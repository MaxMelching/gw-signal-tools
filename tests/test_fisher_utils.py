# ----- Standard Lib Imports -----
import unittest

# ----- Third Party Imports -----
import numpy as np
import astropy.units as u
from gw_signal_tools.test_utils import (
    assert_allclose_quantity, assert_allclose_frequseries,
    assert_allclose_timeseries
)
from gwpy.testing.utils import assert_quantity_equal

# ----- Local Package Imports -----
from gw_signal_tools.fisher_utils import (
    num_diff, get_waveform_derivative_1D,
    get_waveform_derivative_1D_with_convergence,
    fisher_matrix, fisher_element
)
from gw_signal_tools.fisher_matrix import FisherMatrix


#%% ---------- Testing Derivative Methods ----------
def test_num_diff():
    step_size = 0.01
    x_vals = np.arange(0.0, 2.0, step=step_size)

    derivative_vals = num_diff(x_vals, h=step_size)

    assert np.all(np.isclose(derivative_vals, np.ones(derivative_vals.size), atol=0.0, rtol=0.01))

    func_vals = 0.5 * x_vals**2
    derivative_vals = num_diff(func_vals, h=step_size)

    assert np.all(np.isclose(derivative_vals[2 : -2], x_vals[2 : -2], atol=0.0, rtol=0.01))
    assert np.all(np.isclose(derivative_vals[1:2], x_vals[1:2], atol=0.0, rtol=0.01))  # First correct value is zero, thus relative deviation is always 1
    assert np.all(np.isclose(derivative_vals[-2:], x_vals[-2:], atol=0.0, rtol=0.01))
    # NOTE: for values at border of interval, rule is not applicable.
    # Thus we make separate checks, methods could be less accurate there

    func_vals = np.sin(x_vals)

    derivative_vals = num_diff(func_vals, h=step_size)

    assert np.all(np.isclose(derivative_vals[2 : -2], np.cos(x_vals)[2 : -2], atol=0.0, rtol=0.01))
    assert np.all(np.isclose(derivative_vals[:2], np.cos(x_vals)[:2], atol=0.0, rtol=0.01))
    assert np.all(np.isclose(derivative_vals[-2:], np.cos(x_vals)[-2:], atol=0.0, rtol=0.02))


def test_wf_deriv():
    # TODO: compare with numdifftools
    ...


#%% Initializing commonly used variables
f_min = 20.*u.Hz
f_max = 1024.*u.Hz

wf_params = {
    'total_mass': 100.*u.solMass,
    'mass_ratio': 0.42*u.dimensionless_unscaled,
    'deltaT': 1./2048.*u.s,
    'f22_start': f_min,
    'f_max': f_max,
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

phenomx_generator = FisherMatrix.get_wf_generator(approximant, 'frequency')

deriv = get_waveform_derivative_1D_with_convergence(
    wf_params,
    'total_mass',
    phenomx_generator,
    return_info=False
)

print(deriv)

fisher_val, info = fisher_matrix(
    wf_params,
    # 'total_mass',
    ['total_mass', 'distance'],
    phenomx_generator,
    return_info=True
)

print(fisher_val)
print(fisher_val.value)
print(fisher_val.unit)
print(fisher_val[0, 0].to(1/(u.Msun)**2))


# Do some tests