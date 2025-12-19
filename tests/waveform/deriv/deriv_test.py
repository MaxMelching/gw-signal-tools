# -- Standard Lib Imports
import unittest

# -- Third Party Imports
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pytest

# -- Local Package Imports
from gw_signal_tools.test_utils import assert_allclose_quantity, assert_allclose_series
from gw_signal_tools.waveform import (
    get_wf_generator,
    norm,
    WaveformDerivativeGWSignaltools,
    WaveformDerivativeNumdifftools,
    WaveformDerivativeAmplitudePhase,
    WaveformDerivative,
    WaveformDerivativeBase,
)
from gw_signal_tools.types import HashableDict
from gw_signal_tools import enable_caching_locally, disable_caching_locally  # noqa: F401


# %% -- Initializing commonly used variables for Fisher tests -----------------
f_min = 20.0 * u.Hz
f_max = 1024.0 * u.Hz

wf_params = HashableDict(
    {
        'total_mass': 100.0 * u.solMass,
        'mass_ratio': 0.42 * u.dimensionless_unscaled,
        'deltaT': 1.0 / 2048.0 * u.s,
        'f22_start': f_min,
        'f_max': f_max,
        'deltaF': 2**-5 * u.Hz,
        'f22_ref': 20.0 * u.Hz,
        'phi_ref': 0.0 * u.rad,
        'distance': 1.0 * u.Mpc,
        'inclination': 0.0 * u.rad,
        'eccentricity': 0.0 * u.dimensionless_unscaled,
        'longAscNodes': 0.0 * u.rad,
        'meanPerAno': 0.0 * u.rad,
        'condition': 0,
    }
)

test_params = ['total_mass', 'mass_ratio']


with enable_caching_locally():
    # with disable_caching_locally():
    # -- Avoid globally changing caching, messes up test_caching
    wf_generator = get_wf_generator('IMRPhenomXPHM')

# -- Make sure mass1 and mass2 are not in default_dict
import lalsimulation.gwsignal.core.parameter_conventions as pc  # noqa: E402

pc.default_dict.pop('mass1', None)
pc.default_dict.pop('mass2', None)


# %% -- Testing WaveformDerivative --------------------------------------------
def test_deriv_routines():
    # -- Test default first, then all other options
    full_deriv = WaveformDerivative(wf_params, 'total_mass', wf_generator)
    assert isinstance(full_deriv, WaveformDerivativeNumdifftools)

    full_deriv = WaveformDerivative(
        wf_params,
        'total_mass',
        wf_generator,
        deriv_routine='gw_signal_tools',
        pass_inn_prod_kwargs_to_deriv=True,
    )
    assert isinstance(full_deriv, WaveformDerivativeGWSignaltools)

    full_deriv = WaveformDerivative(
        wf_params, 'total_mass', wf_generator, deriv_routine='numdifftools'
    )
    assert isinstance(full_deriv, WaveformDerivativeNumdifftools)

    full_deriv = WaveformDerivative(
        wf_params, 'total_mass', wf_generator, deriv_routine='amplitude_phase'
    )
    assert isinstance(full_deriv, WaveformDerivativeAmplitudePhase)

    # -- Test invalid inputs
    with pytest.raises(ValueError, match='Invalid deriv_routine'):
        full_deriv = WaveformDerivative(
            wf_params, 'total_mass', wf_generator, deriv_routine=''
        )

    with pytest.raises(
        RuntimeError, match='`deriv_routine` you provided is not callable '
    ):
        full_deriv = WaveformDerivative(
            wf_params, 'total_mass', wf_generator, deriv_routine=None
        )


def test_class_passing():
    full_deriv = WaveformDerivative(
        wf_params,
        'total_mass',
        wf_generator,
        deriv_routine=WaveformDerivativeNumdifftools,
    )
    assert isinstance(full_deriv, WaveformDerivativeNumdifftools)


def test_str_mapping():
    class CustomDeriv(WaveformDerivativeBase):
        pass

    WaveformDerivative.deriv_routine_class_map['custom_routine'] = CustomDeriv
    full_deriv = WaveformDerivative(
        wf_params, 'total_mass', wf_generator, deriv_routine='custom_routine'
    )

    assert isinstance(full_deriv, CustomDeriv)

    WaveformDerivative.deriv_routine_class_map.pop('custom_routine')

    with pytest.raises(ValueError, match='Invalid deriv_routine'):
        full_deriv = WaveformDerivative(
            wf_params, 'total_mass', wf_generator, deriv_routine='custom_routine'
        )


@pytest.mark.parametrize(
    'deriv_routine', ['numdifftools', 'gw_signal_tools', 'amplitude_phase']
)
def test_info_setting(deriv_routine):
    # -- Test default first, then all other options
    full_deriv = WaveformDerivative(
        wf_params, 'total_mass', wf_generator, deriv_routine=deriv_routine
    )
    _, info = full_deriv.deriv, full_deriv.info

    info.is_exact_deriv
    full_deriv.is_exact_deriv

    with pytest.raises(
        AttributeError,
        match=f"'{full_deriv.__class__.__name__}' object has no attribute",
    ):
        full_deriv.non_existent_attribute

    with pytest.raises(TypeError, match='`info` must be a dict or namedtuple.'):
        full_deriv.info = 'Not a dict or namedtuple'


# %% -- Characterizing gwsignaltools derivative -------------------------------
@pytest.mark.parametrize('param', test_params)
@pytest.mark.parametrize('q_val', [0.42, 0.05])
@pytest.mark.parametrize('break_conv', [True, False])
def test_step_size(param, q_val, break_conv):
    full_deriv = WaveformDerivativeGWSignaltools(
        wf_params | {'mass_ratio': q_val * u.dimensionless_unscaled},
        param,
        wf_generator,
        break_upon_convergence=break_conv,
    )
    deriv, info = full_deriv.deriv, full_deriv.info

    deriv_fixed_step_size = WaveformDerivativeGWSignaltools(
        wf_params | {'mass_ratio': q_val * u.dimensionless_unscaled},
        param,
        wf_generator,
        step_sizes=info.final_step_size,
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
    # deriv, info = full_deriv.deriv, full_deriv.info

    # deriv2 = WaveformDerivativeGWSignaltools(
    #     wf_params,# | {'mass_ratio': 0.05*u.dimensionless_unscaled},
    #     'total_mass',
    #     wf_generator,
    #     step_sizes=info.final_step_size
    # ).deriv

    # print(info.final_step_size)
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
    deriv_1 = WaveformDerivativeGWSignaltools(wf_params, param, wf_generator).deriv

    deriv_2 = WaveformDerivativeGWSignaltools(
        wf_params,
        param,
        wf_generator,
        step_sizes=[1.5e-2, 1e-2],
        # -- Force convergence testing (need two step sizes for that)
    ).deriv

    assert_allclose_series(deriv_1, deriv_2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    'param, param_val, invalid_step, expected_formula',
    [
        ['total_mass', 10 * u.Msun, 15.0, 'forward'],
        ['mass_ratio', 0.1 * u.dimensionless_unscaled, 0.2, 'forward'],
        ['mass_ratio', 0.8 * u.dimensionless_unscaled, 0.2, 'backward'],
        [
            'mass_ratio',
            0.9 * u.dimensionless_unscaled,
            1.0,
            'five_point',
        ],  # Not backward because lower bound also violated
        [
            'mass_ratio',
            1.1 * u.dimensionless_unscaled,
            0.2 * u.dimensionless_unscaled,
            'forward',
        ],
    ],
)
def test_invalid_step_size(param, param_val, invalid_step, expected_formula):
    deriv_1 = WaveformDerivativeGWSignaltools(
        wf_params | {param: param_val},
        param,
        wf_generator,
        step_sizes=[invalid_step, 1e-2],
        max_refine_numb=1,
        deriv_formula='five_point',
    )
    # -- Idea: provoke error for complete coverage, then use same step
    # -- size as below

    deriv_2 = WaveformDerivativeGWSignaltools(
        wf_params | {param: param_val},
        param,
        wf_generator,
        step_sizes=[1e-2],
        deriv_formula=expected_formula,
    )
    # -- Important: have to pass same formula that is used after
    # -- adjustment in previous call

    assert_allclose_series(deriv_1.deriv, deriv_2.deriv, atol=0.0, rtol=0.0)
    assert deriv_1.info.deriv_formula == deriv_2.info.deriv_formula


@pytest.mark.parametrize('param', test_params)
def test_convergence_plot(param):
    deriv = WaveformDerivativeGWSignaltools(wf_params, param, wf_generator)
    deriv()

    deriv.convergence_plot()
    plt.close()


def test_calling():
    full_deriv = WaveformDerivativeGWSignaltools(wf_params, 'total_mass', wf_generator)
    deriv = full_deriv()

    deriv_2 = full_deriv(50 * u.Msun)

    assert not np.all(deriv == deriv_2)


# %% -- Derivative consistency checks -----------------------------------------
@pytest.mark.parametrize('param_to_vary', test_params)
@pytest.mark.parametrize('crit', ['diff_norm', 'mismatch'])
def test_wf_deriv_numdifftools(param_to_vary, crit):
    deriv = WaveformDerivativeGWSignaltools(
        wf_params, param_to_vary, wf_generator, convergence_check=crit
    ).deriv

    nd_deriv = WaveformDerivativeNumdifftools(
        wf_params, param_to_vary, wf_generator, base_step=1e-2
    ).deriv

    # TODO: also include WaveformDerivativeAmplitudePhase?

    mask = deriv.frequencies <= 256.0 * u.Hz

    assert_allclose(deriv[mask], nd_deriv[mask], atol=0.0, rtol=0.06)
    # -- Once again, the problems seem to occur mostly around zeros,
    # -- which is also the reason why we exclude values where derivative
    # -- is close to zero

    # -- Check Fisher values
    assert_allclose_quantity(norm(deriv), norm(nd_deriv), atol=0.0, rtol=5e-4)
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
                wf_params, 'total_mass', wf_generator, convergence_check=''
            )
