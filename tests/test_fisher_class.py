# ----- Standard Lib Imports -----
import unittest

# ----- Third Party Imports -----
import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u

import matplotlib.pyplot as plt

import pytest

# ----- Local Package Imports -----
from gw_signal_tools.inner_product import norm
from gw_signal_tools.matrix_with_units import MatrixWithUnits
from gw_signal_tools.fisher import (
    fisher_matrix, FisherMatrix
)
from gw_signal_tools import PLOT_STYLE_SHEET
plt.style.use(PLOT_STYLE_SHEET)


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
phenomx_generator = FisherMatrix.get_wf_generator(approximant)
phenomx_cross_generator = FisherMatrix.get_wf_generator(approximant, mode='cross')

# Make sure mass1 and mass2 are not in default_dict (makes messy behaviour)
import lalsimulation.gwsignal.core.parameter_conventions as pc

try:
    pc.default_dict.pop('mass1')
except KeyError:
    pass

try:
    pc.default_dict.pop('mass2')
except KeyError:
    pass


fisher_tot_mass = FisherMatrix(
    wf_params,
    'total_mass',
    wf_generator=phenomx_generator,
    return_info=True
)


#%% Simple consistency tests
def test_unit():
    # Both ways of accessing must work
    assert fisher_tot_mass.fisher[0, 0].unit == 1/u.solMass**2
    assert fisher_tot_mass.fisher.unit[0, 0] == 1/u.solMass**2

def test_inverse():
    assert np.all(np.equal(np.linalg.inv(fisher_tot_mass.fisher.value),
                           fisher_tot_mass.fisher_inverse.value))
    
    assert np.all(np.equal(fisher_tot_mass.fisher.unit**-1,
                           fisher_tot_mass.fisher_inverse.unit))
    
def test_fisher_calc():
    fisher_tot_mass_2 = fisher_matrix(wf_params, 'total_mass',
                                      phenomx_generator)

    assert fisher_tot_mass.fisher == fisher_tot_mass_2

def test_criterion_consistency():
    fisher_tot_mass_2 = fisher_matrix(wf_params, 'total_mass',
                                      phenomx_generator, convergence_check='mismatch')
    
    assert fisher_tot_mass.fisher == fisher_tot_mass_2
    # Interesting, even equal. Roughly equal would also be sufficient, might
    # converge at different step sizes and thus have slightly different results


#%% Feature tests
def test_project():
    test_params = ['total_mass', 'mass_ratio', 'time', 'phase', 'distance']

    fisher = FisherMatrix(
        wf_params,
        test_params,
        wf_generator=phenomx_generator,
    )

    fisher_fully_projected = fisher.project_fisher(test_params)

    assert_allclose(
        fisher_fully_projected.value,
        0.0,
        atol=1e-10 * np.max(np.abs(fisher.value)),
        rtol=0
    )
    # Large values are decreased by 10 orders of magnitude. For smaller
    # values, this test will pass almost automatically, but they have
    # been close to zero in some cases anyway, so testing with
    # rtol=1e-10 or testing that smaller than 1e-10*fisher.value would
    # also not make sense (values that are already essentially zero
    # will not change by 10 orders of magnitude)

    fisher_partially_projected = fisher.project_fisher('time')  # Test if works

def test_plot():
    fisher = FisherMatrix(
        wf_params,
        ['total_mass', 'mass_ratio', 'distance'],
        wf_generator=phenomx_generator,
        return_info=False
    )
    plt.close()

    print(fisher)  # For verification

    MatrixWithUnits.plot(fisher.fisher)
    # plt.show()
    plt.close()

    fisher.plot()
    # plt.show()
    plt.close()

    fisher.plot(only_fisher=True)
    # plt.show()
    plt.close()

    fisher.plot(only_fisher_inverse=True)
    # plt.show()
    plt.close()

def test_cond():
    assert fisher_tot_mass.cond() == fisher_tot_mass.fisher.cond()

def test_array():
    assert np.all(np.array(fisher_tot_mass) == np.array(fisher_tot_mass.fisher))

@pytest.mark.parametrize('new_wf_params_at_point', [None, wf_params | {'total_mass': 42.*u.solMass}])
@pytest.mark.parametrize('new_params_to_vary', [None, ['mass_ratio', 'distance']])
@pytest.mark.parametrize('new_wf_generator', [None, phenomx_cross_generator])
@pytest.mark.parametrize('new_metadata', [None, {'return_info': False, 'convergence_check': 'mismatch'}])
def test_update_attrs(new_wf_params_at_point, new_params_to_vary,
                      new_wf_generator, new_metadata):
    if new_metadata is None:
        new_metadata = {}

    fisher_tot_mass_v2 = fisher_tot_mass.update_attrs(
        new_wf_params_at_point,
        new_params_to_vary,
        new_wf_generator,
        **new_metadata
    )

def test_copy():
    ...

#%% Confirm that certain errors are raised
class ErrorRaising(unittest.TestCase):
    def test_immutable(self):
        # Setting Fisher matrix related attributes should throw error
        with self.assertRaises(AttributeError):
            fisher_tot_mass.fisher = 42
    
        with self.assertRaises(AttributeError):
            fisher_tot_mass.fisher_inverse = 42
