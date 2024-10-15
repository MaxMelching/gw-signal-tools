# -- Standard Lib Imports
import unittest

# -- Third Party Imports
import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import astropy.units as u
from gwpy.types import Series
import pytest

# -- Local Package Imports
from gw_signal_tools.test_utils import (
    assert_allclose_MatrixWithUnits
)
from gw_signal_tools.waveform import get_wf_generator
from gw_signal_tools.fisher import num_diff, fisher_matrix
from gw_signal_tools.types import HashableDict


#%% -- Testing Derivative Methods ---------------------------------------------
def test_num_diff():
    step_size = 0.01
    x_vals = np.arange(0.0, 2.0, step=step_size)

    derivative_vals = num_diff(x_vals, h=step_size)

    assert_allclose(derivative_vals, np.ones(derivative_vals.size), atol=0.0, rtol=0.01)

    func_vals = 0.5 * x_vals**2
    derivative_vals = num_diff(func_vals, h=step_size)

    assert_allclose(derivative_vals[2 : -2], x_vals[2 : -2], atol=0.0, rtol=0.01)
    assert_allclose(derivative_vals[1:2], x_vals[1:2], atol=0.0, rtol=0.01)
    assert_allclose(derivative_vals[-2:], x_vals[-2:], atol=0.0, rtol=0.01)
    # -- Note: for values at border of interval, rule is not applicable.
    # -- Thus we make separate checks, methods could be less accurate
    # -- there. For example, the first correct value is zero, thus the
    # -- relative deviation is always 1

    func_vals = np.sin(x_vals)

    derivative_vals = num_diff(func_vals, h=step_size)

    assert_allclose(derivative_vals[2 : -2], np.cos(x_vals)[2 : -2], atol=0.0, rtol=0.01)
    assert_allclose(derivative_vals[:2], np.cos(x_vals)[:2], atol=0.0, rtol=0.01)
    assert_allclose(derivative_vals[-2:], np.cos(x_vals)[-2:], atol=0.0, rtol=0.02)


@pytest.mark.parametrize('h', [None, 1e-2, 1e-2*u.s])
def test_num_diff_input(h):
    step_size = 1e-2
    x_vals = np.arange(0.0, 2.0, step=step_size)
    func_vals = 0.5 * x_vals**2
    num_diff(func_vals, h)

    func_vals = Series(func_vals, xindex=x_vals*u.s)
    num_diff(func_vals, h)
    # No need to compare something, is just to test that h is accepted


#%% -- Initializing commonly used variables for Fisher tests ------------------
f_min = 20.*u.Hz
f_max = 1024.*u.Hz

wf_params = HashableDict({
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
})

test_params = ['total_mass', 'mass_ratio']

from gw_signal_tools import enable_caching, disable_caching
# enable_caching()
disable_caching()

approximant = 'IMRPhenomXPHM'
wf_generator = get_wf_generator(approximant)

# -- Make sure mass1 and mass2 are not in default_dict
import lalsimulation.gwsignal.core.parameter_conventions as pc
pc.default_dict.pop('mass1', None);
pc.default_dict.pop('mass2', None);


#%% -- Fisher consistency checks ----------------------------------------------
@pytest.mark.parametrize('break_conv', [True, False])
def test_convergence_check(break_conv):
    fisher_diff_norm = fisher_matrix(
        wf_params,
        test_params,
        wf_generator,
        convergence_check='diff_norm',
        break_upon_convergence=break_conv,
        deriv_routine='gw_signal_tools'
    )
    
    fisher_mismatch = fisher_matrix(
        wf_params,
        test_params,
        wf_generator,
        convergence_check='mismatch',
        break_upon_convergence=break_conv,
        deriv_routine='gw_signal_tools'
    )

    if break_conv:
        assert_allclose_MatrixWithUnits(fisher_diff_norm, fisher_mismatch,
                                        atol=0.0, rtol=4e-4)
    else:
        assert_allclose_MatrixWithUnits(fisher_diff_norm, fisher_mismatch,
                                        atol=0.0, rtol=3e-6)


@pytest.mark.parametrize('crit', ['diff_norm', 'mismatch'])
def test_break_upon_convergence(crit):
    fisher_without_convergence = fisher_matrix(
        wf_params,
        test_params,
        wf_generator,
        convergence_check=crit,
        break_upon_convergence=True,
        deriv_routine='gw_signal_tools'
    )

    fisher_with_convergence = fisher_matrix(
        wf_params,
        test_params,
        wf_generator,
        convergence_check=crit,
        break_upon_convergence=False,
        deriv_routine='gw_signal_tools'
    )

    plt.close()

    assert_allclose_MatrixWithUnits(fisher_without_convergence, fisher_with_convergence, atol=0.0, rtol=2e-3)
    # --  Small deviations are expected, different final step sizes
    # -- might be selected -> total mass has largest deviations,
    # -- otherwise rtol=1e-3 would work


@pytest.mark.parametrize('conv_crit', ['diff_norm', 'mismatch'])
def test_optimize(conv_crit):
    params_to_vary = ['total_mass', 'time', 'phase']

    # -- For diagonal values, optimization must yield same result (up to
    # -- differences in the routines)
    fisher_non_opt = fisher_matrix(
        wf_params,
        params_to_vary,
        wf_generator,
        convergence_check=conv_crit,
        deriv_routine='gw_signal_tools'
    )
    fisher_opt = fisher_matrix(
        wf_params,
        params_to_vary,
        wf_generator,
        optimize_time_and_phase=True,
        convergence_check=conv_crit,
        deriv_routine='gw_signal_tools'
    )
    
    assert_allclose_MatrixWithUnits(
        fisher_non_opt.diagonal(),
        fisher_opt.diagonal(),
        atol=0.0, rtol=0.005
    )


def test_start_step_size():
    fisher_1 = fisher_matrix(
        wf_params,
        test_params,
        wf_generator,
        start_step_size=1e-1,
        deriv_routine='gw_signal_tools'
    )
    
    fisher_2 = fisher_matrix(
        wf_params,
        test_params,
        wf_generator,
        start_step_size=1e-2,
        deriv_routine='gw_signal_tools'
    )

    assert_allclose_MatrixWithUnits(fisher_1, fisher_2, atol=0.0, rtol=1e-7)
    # -- Idea: they should converge at similar step size because 1e-1 is
    # -- very large, no good results will be produced there
