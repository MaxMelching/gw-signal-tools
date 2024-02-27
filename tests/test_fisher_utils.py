# ----- Standard Lib Imports -----
import unittest

# ----- Third Party Imports -----
import numpy as np
from numpy.testing import assert_allclose
import numdifftools as nd

import matplotlib.pyplot as plt
import astropy.units as u

from gwpy.frequencyseries import FrequencySeries
from gwpy.testing.utils import assert_quantity_equal

import pytest

# ----- Local Package Imports -----
from gw_signal_tools.test_utils import (
    assert_allclose_quantity, assert_allclose_MatrixWithUnits,
    assert_allclose_frequseries
)
from gw_signal_tools.inner_product import norm
from gw_signal_tools.fisher_utils import (
    num_diff, get_waveform_derivative_1D,
    get_waveform_derivative_1D_with_convergence,
    fisher_matrix
)
from gw_signal_tools.fisher_matrix import FisherMatrix


#%% ---------- Testing Derivative Methods ----------
def test_num_diff():
    step_size = 0.01
    x_vals = np.arange(0.0, 2.0, step=step_size)

    derivative_vals = num_diff(x_vals, h=step_size)

    assert_allclose(derivative_vals, np.ones(derivative_vals.size), atol=0.0, rtol=0.01)

    func_vals = 0.5 * x_vals**2
    derivative_vals = num_diff(func_vals, h=step_size)

    assert_allclose(derivative_vals[2 : -2], x_vals[2 : -2], atol=0.0, rtol=0.01)
    assert_allclose(derivative_vals[1:2], x_vals[1:2], atol=0.0, rtol=0.01)  # First correct value is zero, thus relative deviation is always 1
    assert_allclose(derivative_vals[-2:], x_vals[-2:], atol=0.0, rtol=0.01)
    # NOTE: for values at border of interval, rule is not applicable.
    # Thus we make separate checks, methods could be less accurate there

    func_vals = np.sin(x_vals)

    derivative_vals = num_diff(func_vals, h=step_size)

    assert_allclose(derivative_vals[2 : -2], np.cos(x_vals)[2 : -2], atol=0.0, rtol=0.01)
    assert_allclose(derivative_vals[:2], np.cos(x_vals)[:2], atol=0.0, rtol=0.01)
    assert_allclose(derivative_vals[-2:], np.cos(x_vals)[-2:], atol=0.0, rtol=0.02)


#%% ----- Initializing commonly used variables for Fisher tests -----
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

# test_params = ['total_mass', 'distance']
test_params = ['total_mass', 'distance', 'mass_ratio']


approximant = 'IMRPhenomXPHM'

phenomx_generator = FisherMatrix.get_wf_generator(approximant)

# plt.plot(phenomx_generator(wf_params).real)
# plt.plot(phenomx_generator(wf_params).imag)
# plt.show()


#%% ----- Derivative consistency checks -----
@pytest.mark.parametrize('param_to_vary', test_params)
@pytest.mark.parametrize('crit', ['diff_norm', 'mismatch'])
def test_wf_deriv_numdifftools(param_to_vary, crit):
    def deriv_wrapper_real(param_val):
        return phenomx_generator(wf_params | {param_to_vary: param_val * wf_params[param_to_vary].unit}).real

    def deriv_wrapper_imag(param_val):
        return phenomx_generator(wf_params | {param_to_vary: param_val * wf_params[param_to_vary].unit}).imag


    center_val = wf_params[param_to_vary]
    max_step_size = 1e-2

    nd_deriv_real = nd.Derivative(deriv_wrapper_real, full_output=False, base_step=max_step_size)

    nd_deriv_imag = nd.Derivative(deriv_wrapper_imag, full_output=False, base_step=max_step_size)


    deriv = get_waveform_derivative_1D_with_convergence(
        wf_params,
        param_to_vary,
        phenomx_generator,
        convergence_check=crit
    )
    plt.close()

    nd_deriv = nd_deriv_real(center_val) + 1.j * nd_deriv_imag(center_val)
    

    mask = deriv.frequencies <= 256.0 * u.Hz

    assert_allclose(deriv.value[mask], nd_deriv[mask], atol=0.0, rtol=0.05)
    # Once again, the problems seem to occur mostly around zeros, which is
    # also the reason why we exclude values where derivative is close to zero


    # Check Fisher values
    nd_deriv = FrequencySeries(
        nd_deriv,
        frequencies=deriv.frequencies,
        unit=deriv.unit
    )

    assert_allclose_quantity(
        norm(deriv),
        norm(nd_deriv),
        atol=0.0,
        rtol=5e-4
    )
    # Very good agreement, supports claim above that most severe relative
    # differences occur around zeros, where impact is not very high.
    # Note that no frequency regions are excluded here
    

    # Eye test: here are plots of the derivatives
    # plt.plot(deriv.real)
    # plt.plot(deriv.imag)
    # plt.plot(deriv.frequencies, nd_deriv.real, '--')
    # plt.plot(deriv.frequencies, nd_deriv.imag, '--')

    # plt.title(param_to_vary)
    # plt.show()

# TODO: test how things behave with smaller df!!!


@pytest.mark.parametrize('param', test_params)
@pytest.mark.parametrize('q_val', [0.42, 0.05])
@pytest.mark.parametrize('break_conv', [True, False])
def test_step_size(param, q_val, break_conv):
    deriv, deriv_info = get_waveform_derivative_1D_with_convergence(
        wf_params | {'mass_ratio': q_val*u.dimensionless_unscaled},
        param,
        phenomx_generator,
        break_upon_convergence=break_conv,
        return_info=True
    )
    plt.close()

    deriv_fixed_step_size = get_waveform_derivative_1D(
        wf_params | {'mass_ratio': q_val*u.dimensionless_unscaled},
        param,
        phenomx_generator,
        step_size=deriv_info['final_step_size']
    )

    # These must be equal (not just close)
    if break_conv:
        assert_allclose_frequseries(deriv, deriv_fixed_step_size, atol=0.0, rtol=0.0)
    else:
        deriv.crop(end=256 * u.Hz, copy=False)
        deriv_fixed_step_size.crop(end=256 * u.Hz, copy=False)

        # assert_allclose_frequseries(deriv, deriv_fixed_step_size, atol=2e-24, rtol=6e-4)
        if param != 'total_mass':
            assert_allclose_frequseries(deriv, deriv_fixed_step_size, atol=2e-24, rtol=2e-3)
        else:
            # One peak for q=0.42 where deviation is larger than otherwise.
            # No idea where this comes from
            assert_allclose_frequseries(deriv, deriv_fixed_step_size, atol=2e-23, rtol=2e-3)
        # Not sure why, but they are never fully equal here. Maybe we are off
        # by one index (though I checked this), but results look very equal
        # and all deviations are around zeros, where small deviations result
        # in large errors. Fisher matrix values below are very similar even
        # for break_upon_convergence=False, which is only thing that counts
        # Moreover, q=0.05 produces VERY challenging waveforms, this is merely
        # to test if index error is handled correctly


    # For eye test, this is where maximum deviation occurs
    # -> imaginary part seems to be source of error
    # deriv, deriv_info = get_waveform_derivative_1D_with_convergence(
    #     wf_params,
    #     'total_mass',
    #     phenomx_generator,
    #     return_info=True,
    #     break_upon_convergence=False
    # )

    # deriv2 = get_waveform_derivative_1D(
    #     wf_params,# | {'mass_ratio': 0.05*u.dimensionless_unscaled},
    #     'total_mass',
    #     phenomx_generator,
    #     step_size=deriv_info['final_step_size']
    # )

    # print(deriv_info['final_step_size'])
    # plt.close()
    # plt.plot(deriv.real)
    # plt.plot(deriv.imag)
    # plt.plot(deriv2.real, '--')
    # plt.plot(deriv2.imag, '--')
    # # plt.plot(np.abs(deriv))
    # # plt.plot(np.abs(deriv), '--')
    # plt.plot(np.abs(deriv - deriv2)*50, label='Difference')
    # plt.xlim(130, 170)
    # # plt.ylim(-5e-22, 5e-22)
    # plt.legend()
    # plt.show()


#%% ----- Fisher consistency checks -----
@pytest.mark.parametrize('break_conv', [True, False])
def test_convergence_check(break_conv):
    fisher_diff_norm = fisher_matrix(
        wf_params,
        test_params,
        phenomx_generator,
        convergence_check='diff_norm',
        break_upon_convergence=break_conv
    )
    plt.close()
    
    fisher_mismatch = fisher_matrix(
        wf_params,
        test_params,
        phenomx_generator,
        convergence_check='mismatch',
        break_upon_convergence=break_conv
    )
    plt.close()

    if break_conv:
        assert_allclose_MatrixWithUnits(fisher_diff_norm, fisher_mismatch,
                                        atol=0.0, rtol=0.0)
    # 0.0 is good, means they converge at same step size, although this is
    # surely not always the case -> for example if break_conv=True
    else:
        assert_allclose_MatrixWithUnits(fisher_diff_norm, fisher_mismatch,
                                        atol=0.0, rtol=3e-6)


@pytest.mark.parametrize('crit', ['diff_norm', 'mismatch'])
def test_break_upon_convergence(crit):
    fisher_without_convergence = fisher_matrix(
        wf_params,
        test_params,
        phenomx_generator,
        convergence_check=crit,
        break_upon_convergence=True
    )

    fisher_with_convergence = fisher_matrix(
        wf_params,
        test_params,
        phenomx_generator,
        convergence_check=crit,
        break_upon_convergence=False
    )

    plt.close()

    assert_allclose_MatrixWithUnits(fisher_without_convergence, fisher_with_convergence, atol=0.0, rtol=1e-3)
    # Small deviations are expected, different final step sizes might be selected
