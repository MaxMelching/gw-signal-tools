# -- Third Party Imports
import astropy.units as u
import pytest
# import matplotlib.pyplot as plt

# -- Local Package Imports
from gw_signal_tools.waveform import WaveformDerivative, get_wf_generator
from gw_signal_tools.types import HashableDict
from gw_signal_tools import enable_caching_locally, disable_caching_locally  # noqa: F401
from gw_signal_tools.test_utils import assert_allclose_series


# %% -- Initializing commonly used variables ----------------------------------
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


# %% -- Class Tests -----------------------------------------------------------
@pytest.mark.parametrize('param', ['total_mass', 'distance'])
@pytest.mark.parametrize('routine', ['numdifftools', 'amplitude_phase'])
def test_point_calls(param, routine):
    nd_deriv = WaveformDerivative(wf_params, param, wf_generator, deriv_routine=routine)

    point = wf_params[param]
    deriv_scalar = nd_deriv(point.value)
    deriv_quantity = nd_deriv(point.decompose(bases=u.si.bases))

    avg_peak_height = (deriv_scalar.max() + deriv_quantity.max()).value / 2.0

    assert_allclose_series(
        deriv_scalar, deriv_quantity, atol=2.75e-4 * avg_peak_height, rtol=1.1e-15
    )
    # -- atol for total_mass. Not sure where it comes from, maybe from
    # -- little error in conversions. Sub-percent maximal relative
    # -- deviation (measuring on scale of peak) is still fine, though.
    # -- rtol for distance, just numerical errors. Presumably from
    # -- conversions that translate into derivatives.


@pytest.mark.parametrize(
    'param, param_val, invalid_step',
    [
        ['total_mass', 10 * u.Msun, 15.0],
        ['mass_ratio', 0.1 * u.dimensionless_unscaled, 0.2],
        ['mass_ratio', 0.8 * u.dimensionless_unscaled, 0.2],
        ['mass_ratio', 1.0 * u.dimensionless_unscaled, 2.0],
        # -- Trigger repeated violation, for coverage
        ['mass_ratio', 0.9 * u.dimensionless_unscaled, 1.0],
        # -- Not backward because lower bound also violated
        ['mass_ratio', 1.1 * u.dimensionless_unscaled, 0.2 * u.dimensionless_unscaled],
    ],
)
@pytest.mark.parametrize('routine', ['numdifftools', 'amplitude_phase'])
def test_invalid_step_size_correctable(param, param_val, invalid_step, routine):
    deriv_1 = WaveformDerivative(
        wf_params | {param: param_val},
        param,
        wf_generator,
        base_step=invalid_step,
        deriv_routine=routine,
    )
    # -- Idea: provoke error for complete coverage, then use same step
    # -- size as below

    deriv_2 = WaveformDerivative(
        wf_params | {param: param_val},
        param,
        wf_generator,
        deriv_routine=routine,
    )

    deriv_1_eval = deriv_1()
    deriv_2_eval = deriv_2()

    assert_allclose_series(deriv_1_eval, deriv_2_eval, atol=0, rtol=0)
    # -- Demand equality here because step size is corrected internally
    # -- to default one before any calculation is done.


@pytest.mark.parametrize(
    'param, param_val, invalid_step',
    [
        # -- Important that invalid_step / 2 is smaller than default base step
        ['inverse_mass_ratio', (1.0 + 1.8e-2) * u.dimensionless_unscaled, 1.8e-2],
        ['mass_ratio', (1.0 - 1.8e-2) * u.dimensionless_unscaled, 1.8e-2],
    ],
)
@pytest.mark.parametrize(
    'routine',
    [
        'numdifftools',
        'amplitude_phase',
    ],
)
def test_invalid_step_size_same_method(param, param_val, invalid_step, routine):
    def wf_gen(wf_params_inner):
        # -- Wrapper to enable inverse_mass_ratio usage
        if 'inverse_mass_ratio' in wf_params_inner:
            q_inv = wf_params_inner['inverse_mass_ratio'].value
            q = 1.0 / q_inv
            wf_params_inner = wf_params_inner.copy()
            wf_params_inner.pop('inverse_mass_ratio')
            wf_params_inner['mass_ratio'] = q * u.dimensionless_unscaled

        return wf_generator(wf_params_inner)

    deriv_1 = WaveformDerivative(
        wf_params | {param: param_val},
        param,
        wf_gen,
        base_step=invalid_step,
        deriv_routine=routine,
    )
    # -- Idea: provoke error for complete coverage, then use same step
    # -- size as below

    deriv_2 = WaveformDerivative(
        wf_params | {param: param_val},
        param,
        wf_gen,
        deriv_routine=routine,
    )

    deriv_1_eval = deriv_1()
    deriv_2_eval = deriv_2()

    if routine == 'numdifftools':
        assert deriv_1.method == deriv_2.method
        assert deriv_2.method == 'central'
    elif routine == 'amplitude_phase':
        assert deriv_1.abs_deriv.method == deriv_2.abs_deriv.method
        assert deriv_2.abs_deriv.method == 'central'

        assert deriv_1.phase_deriv.method == deriv_2.phase_deriv.method
        assert deriv_2.phase_deriv.method == 'central'

    # plt.figure(figsize=(12, 6))
    # plt.plot(deriv_1_eval.real, label='deriv 1')
    # plt.plot(deriv_2_eval.real, '--', label='deriv 2')
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.plot(deriv_1_eval.imag, label='deriv 1')
    # plt.plot(deriv_2_eval.imag, '--', label='deriv 2')
    # plt.legend()
    # plt.show()

    avg_peak_height = (deriv_1_eval.abs().max() + deriv_2_eval.abs().max()).value / 2.0

    assert_allclose_series(
        deriv_1_eval, deriv_2_eval, atol=10e-2 * avg_peak_height, rtol=0
    )
    # -- Idea: different step sizes will be used, but still same method.
    # -- So we expect certain deviations, but they should be small.
    # -- Hmm, 10% is a little too much if you ask me... Needed for second
    # -- one, first one would be fine with 1% (still a little too much).
    # -- But we have to live with it, both methods have same behavior.


@pytest.mark.parametrize(
    'param, param_val, invalid_step',
    [
        # -- Important that invalid_step / 2 is still smaller than default
        # -- base step, but large enough that it still produces invalid
        # -- param value. And the point must be chosen so that default
        # -- base step / 2 still has valid param value.
        ['inverse_mass_ratio', (1.0 + 0.8e-2) * u.dimensionless_unscaled, 2e-2],
        ['mass_ratio', (1.0 - 0.8e-2) * u.dimensionless_unscaled, 2e-2],
    ],
)
@pytest.mark.parametrize(
    'routine',
    [
        'numdifftools',
        # 'amplitude_phase',
    ],
)
def test_invalid_step_size(param, param_val, invalid_step, routine):
    def wf_gen(wf_params_inner):
        # -- Wrapper to enable inverse_mass_ratio usage
        if 'inverse_mass_ratio' in wf_params_inner:
            q_inv = wf_params_inner['inverse_mass_ratio'].value
            q = 1.0 / q_inv
            wf_params_inner = wf_params_inner.copy()
            wf_params_inner.pop('inverse_mass_ratio')
            wf_params_inner['mass_ratio'] = q * u.dimensionless_unscaled

        return wf_generator(wf_params_inner)

    deriv_1 = WaveformDerivative(
        wf_params | {param: param_val},
        param,
        wf_gen,
        base_step=invalid_step,
        num_steps=30,  # Does not help...
        deriv_routine=routine,
    )
    # -- Idea: provoke error for complete coverage, then use same step
    # -- size as below

    deriv_2 = WaveformDerivative(
        wf_params | {param: param_val},
        param,
        wf_gen,
        deriv_routine=routine,
    )

    deriv_1_eval = deriv_1()  # noqa: F841 - important to change routine
    deriv_2_eval = deriv_2()  # noqa: F841 - important to change routine

    if routine == 'numdifftools':
        assert deriv_1.method != deriv_2.method
        assert deriv_2.method == 'central'
    elif routine == 'amplitude_phase':
        assert deriv_1.abs_deriv.method != deriv_2.abs_deriv.method
        assert deriv_2.abs_deriv.method == 'central'

        assert deriv_1.phase_deriv.method != deriv_2.phase_deriv.method
        assert deriv_2.phase_deriv.method == 'central'

    # plt.figure(figsize=(12, 6))
    # plt.plot(deriv_1_eval.real, label='deriv 1')
    # plt.plot(deriv_2_eval.real, label='deriv 2')
    # plt.legend()
    # # plt.savefig(f'test_invalid_step_size_{param}_{routine}_real_python311.png')
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.plot(deriv_1_eval.imag, label='deriv 1')
    # plt.plot(deriv_2_eval.imag, label='deriv 2')
    # plt.legend()
    # # plt.savefig(f'test_invalid_step_size_{param}_{routine}_imag_python311.png')
    # plt.show()

    f_comp_max = 50.0 * u.Hz

    deriv_1_eval = deriv_1_eval[deriv_1_eval.frequencies <= f_comp_max]
    deriv_2_eval = deriv_2_eval[deriv_2_eval.frequencies <= f_comp_max]

    avg_peak_height = (deriv_1_eval.abs().max() + deriv_2_eval.abs().max()).value / 2.0

    assert_allclose_series(
        deriv_1_eval,
        deriv_2_eval,
        atol=4e-2 * avg_peak_height,
        rtol=0,
    )
    # -- For rest, our initial chosen step size seems way too large...
    # -- -> ah, no. Increasing num_steps does not help. So perhaps it is
    # -- the forward, backward methods that are not accurate enough.
