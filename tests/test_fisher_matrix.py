# ----- Standard Lib Imports -----
import unittest

# ----- Third Party Imports -----
import numpy as np

import astropy.units as u

# ----- Local Package Imports -----
from gw_signal_tools.inner_product import norm
from gw_signal_tools.matrix_with_units import MatrixWithUnits
from gw_signal_tools.fisher_matrix import FisherMatrix
from gw_signal_tools.fisher_utils import fisher_matrix


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


#%% Confirm that certain errors are raised
class ErrorRaising(unittest.TestCase):
    def test_immutable(self):
        # Setting Fisher matrix related attributes should throw error
        with self.assertRaises(AttributeError):
            fisher_tot_mass.fisher = 42
    
        with self.assertRaises(AttributeError):
            fisher_tot_mass.fisher_inverse = 42
