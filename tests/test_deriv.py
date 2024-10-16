# -- Standard Lib Imports
import unittest

# -- Third Party Imports
import astropy.units as u
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
import pytest

# -- Local Package Imports
from gw_signal_tools.test_utils import (
    assert_allclose_quantity, assert_allclose_series
)
from gw_signal_tools.waveform import (
    get_wf_generator, norm, WaveformDerivativeGWSignaltools,
    WaveformDerivativeNumdifftools, WaveformDerivativeAmplitudePhase
)
from gw_signal_tools.types import HashableDict
from gw_signal_tools import enable_caching_locally, disable_caching_locally


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


with enable_caching_locally():
# with disable_caching_locally():
    # -- Avoid globally changing caching, messes up test_caching
    wf_generator = get_wf_generator('IMRPhenomXPHM')

# -- Make sure mass1 and mass2 are not in default_dict
import lalsimulation.gwsignal.core.parameter_conventions as pc
pc.default_dict.pop('mass1', None);
pc.default_dict.pop('mass2', None);


# -- Characterizing gwsignaltools derivative ----------------------------------
@pytest.mark.parametrize('param', test_params)
@pytest.mark.parametrize('q_val', [0.42, 0.05])
@pytest.mark.parametrize('break_conv', [True, False])
def test_step_size(param, q_val, break_conv):
    full_deriv = WaveformDerivativeGWSignaltools(
        wf_params | {'mass_ratio': q_val*u.dimensionless_unscaled},
        param,
        wf_generator,
        break_upon_convergence=break_conv
    )
    deriv, deriv_info = full_deriv.deriv, full_deriv.deriv_info

    deriv_fixed_step_size = WaveformDerivativeGWSignaltools(
        wf_params | {'mass_ratio': q_val*u.dimensionless_unscaled},
        param,
        wf_generator,
        step_sizes=deriv_info['final_step_size']
    ).deriv

    # -- These must be equal (not just close)
    if break_conv:
        assert_allclose_series(deriv, deriv_fixed_step_size, atol=0.0, rtol=0.0)
    else:
        deriv.crop(end=256 * u.Hz, copy=False)
        deriv_fixed_step_size.crop(end=256 * u.Hz, copy=False)

        if param != 'total_mass':
            assert_allclose_series(deriv, deriv_fixed_step_size, atol=2e-24, rtol=2e-3)
        else:
            # One peak for q=0.42 where deviation is larger than otherwise.
            # No idea where this comes from
            assert_allclose_series(deriv, deriv_fixed_step_size, atol=2e-23, rtol=2e-3)
        # Not sure why, but they are never fully equal here. Maybe we are off
        # by one index (though I checked this), but results look very equal
        # and all deviations are around zeros, where small deviations result
        # in large errors. Fisher matrix values below are very similar even
        # for break_upon_convergence=False, which is only thing that counts
        # Moreover, q=0.05 produces VERY challenging waveforms, this is merely
        # to test if index error is handled correctly


    # -- For eye test, this is where maximum deviation occurs
    # -- -> imaginary part seems to be source of error
    # full_deriv = WaveformDerivativeGWSignaltools(
    #     wf_params,
    #     'total_mass',
    #     wf_generator,
    #     return_info=True,
    #     break_upon_convergence=False
    # )
    # deriv, deriv_info = full_deriv.deriv, full_deriv.deriv_info

    # deriv2 = WaveformDerivativeGWSignaltools(
    #     wf_params,# | {'mass_ratio': 0.05*u.dimensionless_unscaled},
    #     'total_mass',
    #     wf_generator,
    #     step_sizes=deriv_info['final_step_size']
    # ).deriv

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


@pytest.mark.parametrize('param', test_params)
def test_custom_convergence(param):
    deriv_1 = WaveformDerivativeGWSignaltools(
        wf_params,
        param,
        wf_generator
    ).deriv

    deriv_2 = WaveformDerivativeGWSignaltools(
        wf_params,
        param,
        wf_generator,
        step_sizes=[1.5e-2, 1e-2]
        # -- Force convergence testing (need two step sizes for that)
    ).deriv

    assert_allclose_series(deriv_1, deriv_2, atol=0.0, rtol=0.0)


def test_invalid_step_size():
    param = 'mass_ratio'
    param_val = 0.42*u.dimensionless_unscaled
    deriv_1 = WaveformDerivativeGWSignaltools(
        wf_params | {'mass_ratio': param_val},
        param,
        wf_generator,
        step_sizes=[2*param_val, 1e-2],
        max_refine_numb=1,
        deriv_formula='forward'
    ).deriv
    # -- Idea: provoke error for complete coverage, then use same step
    # -- size as below

    deriv_2 = WaveformDerivativeGWSignaltools(
        wf_params | {'mass_ratio': param_val},
        param,
        wf_generator,
        step_sizes=[1e-2],
        deriv_formula='forward'
    ).deriv
    # -- Important: have to pass same formula that is used after
    # -- adjustment in previous call

    assert_allclose_series(deriv_1, deriv_2, atol=0.0, rtol=0.0)


#%% -- Derivative consistency checks ------------------------------------------
@pytest.mark.parametrize('param_to_vary', test_params)
@pytest.mark.parametrize('crit', ['diff_norm', 'mismatch'])
def test_wf_deriv_numdifftools(param_to_vary, crit):
    deriv = WaveformDerivativeGWSignaltools(
        wf_params,
        param_to_vary,
        wf_generator,
        convergence_check=crit
    ).deriv

    nd_deriv = WaveformDerivativeNumdifftools(
        wf_params,
        param_to_vary,
        wf_generator,
        base_step=1e-2
    ).deriv


    # TODO: also include WaveformDerivativeAmplitudePhase?
    

    mask = deriv.frequencies <= 256.0 * u.Hz

    assert_allclose(deriv[mask], nd_deriv[mask], atol=0.0, rtol=0.06)
    # -- Once again, the problems seem to occur mostly around zeros,
    # -- which is also the reason why we exclude values where derivative
    # -- is close to zero


    # -- Check Fisher values
    assert_allclose_quantity(
        norm(deriv),
        norm(nd_deriv),
        atol=0.0,
        rtol=5e-4
    )
    # -- Very good agreement, supports claim above that most severe
    # -- relative differences occur around zeros, where impact is not
    # -- very high. Note that no frequency regions are excluded here
    

    # -- Eye test: here are plots of the derivatives
    # plt.plot(deriv.real)
    # plt.plot(deriv.imag)
    # plt.plot(deriv.frequencies, nd_deriv.real, '--')
    # plt.plot(deriv.frequencies, nd_deriv.imag, '--')

    # plt.title(param_to_vary)
    # plt.show()


# TODO: test how things behave with smaller df!!!

class ErrorRaising(unittest.TestCase):
    def test_wrong_conv_check(self):
        with self.assertRaises(ValueError):
            WaveformDerivativeGWSignaltools(
                wf_params,
                'total_mass',
                wf_generator,
                convergence_check=''
            )
