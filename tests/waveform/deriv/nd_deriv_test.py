# -- Third Party Imports
import astropy.units as u
import pytest
import matplotlib.pyplot as plt

# -- Local Package Imports
from gw_signal_tools.waveform import (
    WaveformDerivativeGWSignaltools,
    WaveformDerivativeNumdifftools,
    WaveformDerivativeAmplitudePhase,
    WaveformDerivative,
    get_wf_generator,
)
from gw_signal_tools.types import HashableDict
from gw_signal_tools import enable_caching_locally, disable_caching_locally  # noqa: F401
from gw_signal_tools.test_utils import assert_allclose_series


# %% -- Initializing commonly used variables ----------------------------------
f_min = 20.0 * u.Hz
f_max = 1024.0 * u.Hz

wf_params = HashableDict(
    {
        "total_mass": 100.0 * u.solMass,
        "mass_ratio": 0.42 * u.dimensionless_unscaled,
        "deltaT": 1.0 / 2048.0 * u.s,
        "f22_start": f_min,
        "f_max": f_max,
        "deltaF": 2**-5 * u.Hz,
        "f22_ref": 20.0 * u.Hz,
        "phi_ref": 0.0 * u.rad,
        "distance": 1.0 * u.Mpc,
        "inclination": 0.0 * u.rad,
        "eccentricity": 0.0 * u.dimensionless_unscaled,
        "longAscNodes": 0.0 * u.rad,
        "meanPerAno": 0.0 * u.rad,
        "condition": 0,
    }
)

test_params = ["total_mass", "mass_ratio"]


with enable_caching_locally():
    # with disable_caching_locally():
    # -- Avoid globally changing caching, messes up test_caching
    wf_generator = get_wf_generator("IMRPhenomXPHM")

# -- Make sure mass1 and mass2 are not in default_dict
import lalsimulation.gwsignal.core.parameter_conventions as pc  # noqa: E402

pc.default_dict.pop("mass1", None)
pc.default_dict.pop("mass2", None)


# %% -- Class Tests -----------------------------------------------------------
@pytest.mark.parametrize("param", ["total_mass", "distance"])
@pytest.mark.parametrize("routine", ["numdifftools", "amplitude_phase"])
def test_point_calls(param, routine):
    nd_deriv = WaveformDerivative(wf_params, param, wf_generator, deriv_routine=routine)

    point = wf_params[param]
    deriv_scalar = nd_deriv(point.value)
    deriv_quantity = nd_deriv(point.decompose(bases=u.si.bases))

    avg_peak_height = (deriv_scalar.max() + deriv_quantity.max()).value / 2.0

    assert_allclose_series(
        deriv_scalar, deriv_quantity, atol=2e-4 * avg_peak_height, rtol=1.1e-15
    )
    # -- atol for total_mass. Not sure where it comes from, maybe from
    # -- little error in conversions. Sub-percent maximal relative
    # -- deviation (measuring on scale of peak) is still fine, though.
    # -- rtol for distance, just numerical errors. Presumably from
    # -- conversions that translate into derivatives.


@pytest.mark.parametrize(
    "param, param_val, invalid_step",
    [
        ["total_mass", 10 * u.Msun, 15.0],
        ["mass_ratio", 0.1 * u.dimensionless_unscaled, 0.2],
        ["mass_ratio", 0.8 * u.dimensionless_unscaled, 0.2],
        # -- Works in test_deriv, but not here. One segment of frequency
        # -- interval is badly approximated by numdifftools
        [
            "mass_ratio",
            0.9 * u.dimensionless_unscaled,
            1.0,
        ],  # Not backward because lower bound also violated
        ["mass_ratio", 1.1 * u.dimensionless_unscaled, 0.2 * u.dimensionless_unscaled],
        # -- Following tests do not work, sym_mass_ratio is not in wf_params
        # ['sym_mass_ratio', 0.1*u.dimensionless_unscaled, 0.2*u.dimensionless_unscaled, 'forward'],
        # ['sym_mass_ratio', 0.2*u.dimensionless_unscaled, 0.1*u.dimensionless_unscaled, 'backward'],
    ],
)
@pytest.mark.parametrize("routine", ["numdifftools", "amplitude_phase"])
def test_invalid_step_size(param, param_val, invalid_step, routine):
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
    # -- Either very low relative deviation or the deviations are
    # -- essentially zero on scale of derivative (0.1% of peak). Latter
    # -- mainly needed for issues at high frequency end


# %% -- Comparison with custom Derivative class -------------------------------
# print(WaveformDerivativeGWSignaltools.__dict__)
# print(WaveformDerivativeNumdifftools.__dict__)
# -- To check that docstring is transferred


# test_param = 'total_mass'  # Differences between routines are 1e-4 smaller than actual values. Good agreement
test_param = "mass_ratio"  # Differences between routines are 1e-4 smaller than actual values. Great agreement
# test_param = 'distance'  # Perfectly equal, as expected
nd_deriv = WaveformDerivativeNumdifftools(
    wf_params,
    test_param,
    wf_generator,
    # base_step=1e-2
    # base_step=1e-2*wf_params[test_param].value,
    # method='forward'
    # method='complex'  # Does not work for complex input
)

gwsignal_deriv = WaveformDerivativeGWSignaltools(
    point=wf_params, param_to_vary=test_param, wf_generator=wf_generator
)

amp_phase_deriv = WaveformDerivativeAmplitudePhase(
    point=wf_params,
    param_to_vary=test_param,
    wf_generator=wf_generator,
)


fig, [ax1, ax2, ax3] = plt.subplots(figsize=(18, 24), nrows=3)

# ax1.plot(test.deriv, '-', label='Numdifftools')
# ax1.plot(test_2.deriv, '--', label='GWSignaltools')
# ax1.plot(test_3.deriv, ':', label='AmplitudePhase')

# eval_point = wf_params[test_param]
eval_point = wf_params[test_param] * 0.9
# eval_point = wf_params[test_param]*1.2
# ax1.plot(nd_deriv(eval_point), '-', label='Numdifftools')
# # ax1.plot(gwsignal_deriv(wf_params | {test_param: eval_point}), '--', label='GWSignaltools')
# ax1.plot(gwsignal_deriv(eval_point), '--', label='GWSignaltools')
# ax1.plot(amp_phase_deriv(eval_point), ':', label='AmplitudePhase')

nd_deriv_eval = nd_deriv(eval_point)
gwsignal_deriv_eval = gwsignal_deriv(eval_point)
amp_phase_deriv_eval = amp_phase_deriv(eval_point)

ax1.plot(nd_deriv(eval_point).real, "-", label="Numdifftools")
# ax1.plot(gwsignal_deriv(wf_params | {test_param: eval_point}).real, '--', label='GWSignaltools')
ax1.plot(gwsignal_deriv(eval_point).real, "--", label="GWSignaltools")
ax1.plot(amp_phase_deriv(eval_point).real, ":", label="AmplitudePhase")

ax1.set_title("Real Part")
ax1.legend()

ax2.plot(nd_deriv(eval_point).imag, "-", label="Numdifftools")
# ax2.plot(gwsignal_deriv(wf_params | {test_param: eval_point}).imag, '--', label='GWSignaltools')
ax2.plot(gwsignal_deriv(eval_point).imag, "--", label="GWSignaltools")
ax2.plot(amp_phase_deriv(eval_point).imag, ":", label="AmplitudePhase")

ax2.set_title("Imaginary Part")
ax2.legend()

# ax3.plot(nd_deriv.deriv - gwsignal_deriv.deriv, '-', label='Numdifftools - GWSignaltools')
# ax3.plot(nd_deriv.deriv - amp_phase_deriv.deriv, '--', label='Numdifftools - AmplitudePhase')
# ax3.plot(gwsignal_deriv.deriv - amp_phase_deriv.deriv, ':', label='GWSignaltools - AmplitudePhase')
# ax3.plot((nd_deriv.deriv - gwsignal_deriv.deriv).abs(), '-', label='Numdifftools - GWSignaltools')
# ax3.plot((nd_deriv.deriv - amp_phase_deriv.deriv).abs(), '--', label='Numdifftools - AmplitudePhase')
# ax3.plot((gwsignal_deriv.deriv - amp_phase_deriv.deriv).abs(), ':', label='GWSignaltools - AmplitudePhase')

ax3.plot(
    (nd_deriv_eval - gwsignal_deriv_eval).abs(),
    "-",
    label="Numdifftools - GWSignaltools",
)
ax3.plot(
    (nd_deriv_eval - amp_phase_deriv_eval).abs(),
    "--",
    label="Numdifftools - AmplitudePhase",
)
ax3.plot(
    (gwsignal_deriv_eval - amp_phase_deriv_eval).abs(),
    ":",
    label="GWSignaltools - AmplitudePhase",
)

ax3.legend()

# plt.show()
plt.close()  # -- Activate when running pytest


# -- Testing n-dim output
# deriv = test.deriv
# print(deriv)
# print(np.allclose(deriv[0], deriv[1], atol=0., rtol=0.))

# -- Had following definition in the class for this test
# def fun(x):
#     # -- Testing n-dim output
#     wf = wf_generator(point | {param_to_vary: x*param_unit})

#     return np.stack([wf, wf])

# from gwpy.frequencyseries import FrequencySeries
# class NDFrequencySeries(FrequencySeries):
#     _ndim=2
# -- From testing with n-dim output. Did not use in the end


# -- Multi-function Testing
# import numpy as np
# import numdifftools as nd


# func1_counter = 0
# func2_counter = 0

# def func1(x):
#     global func1_counter
#     func1_counter += 1
#     # return x**2
#     return np.sin(x)

# def func2(x):
#     global func2_counter
#     func2_counter += 1
#     return np.exp(x)

# def func(x):
#     return np.stack([func1(x), func2(x)])

# func_deriv = nd.Derivative(
#     func,
#     base_step=1,  # To provoke slower convergence, test if there is difference
#     full_output=True
# )

# # point = 3
# point = np.linspace(0, 2, num=5)
# num_deriv, info = func_deriv(point)
# print(np.vstack([np.cos(point), np.exp(point)]))
# print(num_deriv)

# print(func1_counter, func2_counter)
# print(info.final_step)

# Ok, so both counters are equal, which of course makes sense because
# they are called simultaneously. final_step is more important and it
# indeed shows that every entry is handled separately (i.e. each row is,
# just like each column is)
# -> that means from a calling perspective, NDWaveformGenerator would
#    work with this Derivative class here. Would also be convenient
#    because return of attributes would be handled well. On the other
#    hand, it could be that we accumulate waveform calls although the
#    corresponding derivative is already converged... But I would think
#    this discrepancy should not be too large, so perhaps code clarity
#    is king here
