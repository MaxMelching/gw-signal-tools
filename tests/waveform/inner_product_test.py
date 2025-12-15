# -- Standard Lib Imports
import unittest

# -- Third Party Imports
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
from lalsimulation.gwsignal import gwsignal_get_waveform_generator
import lalsimulation.gwsignal.core.waveform as wfm
import pytest

# -- Local Package Imports
from gw_signal_tools.waveform import (
    td_to_fd,
    pad_to_dx,
    signal_at_xindex,
    inner_product,
    norm,
    overlap,
    optimize_overlap,
    optimized_inner_product,
    time_phase_wrapper,
    apply_time_phase_shift,
)
from gwpy.testing.utils import assert_quantity_equal
from gw_signal_tools.PSDs import psd_no_noise
from gw_signal_tools.test_utils import assert_allclose_quantity
from gw_signal_tools import enable_caching_locally, disable_caching_locally  # noqa: F401
from gw_signal_tools.types import HashableDict


# %% -- Initializing commonly used variables -----------------------------------
f_min = 20.0 * u.Hz
f_max = 1024.0 * u.Hz

wf_params = HashableDict(
    {
        'mass1': 36 * u.solMass,
        'mass2': 29 * u.solMass,
        'deltaT': 1.0 / 2048.0 * u.s,
        'f22_start': f_min,
        'f_max': f_max,
        'f22_ref': 20.0 * u.Hz,
        'phi_ref': 0.0 * u.rad,
        'distance': 440.0 * u.Mpc,
        'inclination': 0.0 * u.rad,
        'eccentricity': 0.0 * u.dimensionless_unscaled,
        'longAscNodes': 0.0 * u.rad,
        'meanPerAno': 0.0 * u.rad,
        'condition': 0,
    }
)

# -- Make sure mass1 and mass2 are not in default_dict
import lalsimulation.gwsignal.core.parameter_conventions as pc  # noqa: E402

pc.default_dict.pop('mass1', None)
pc.default_dict.pop('mass2', None)

approximant = 'IMRPhenomXPHM'
gen = gwsignal_get_waveform_generator(approximant)

from gw_signal_tools.waveform.utils import _CORRECT_H_UNIT_TIME, _CORRECT_H_UNIT_FREQU  # noqa: E402

with enable_caching_locally():
    # with disable_caching_locally():
    from gw_signal_tools.caching import cache_func

    @cache_func
    def td_wf_gen(wf_params):
        hp, hc = wfm.GenerateTDWaveform(wf_params, gen)
        return hp * _CORRECT_H_UNIT_TIME, hc * _CORRECT_H_UNIT_TIME

    @cache_func
    def fd_wf_gen(wf_params):
        hp, hc = wfm.GenerateFDWaveform(wf_params, gen)
        return hp * _CORRECT_H_UNIT_FREQU, hc * _CORRECT_H_UNIT_FREQU


# -- NOTE: unit conversion is needed because of inconsistent handling of
# -- units in lal, not because of error in gw_signal_tools code


hp_t, hc_t = td_wf_gen(wf_params)
hp_f_fine, hc_f_fine = fd_wf_gen(wf_params)
hp_f_coarse, hc_f_coarse = fd_wf_gen(
    wf_params | {'deltaF': 1.0 / (hp_t.size * hp_t.dx)}
)


# %% -- Technical test if signals are edited inplace ---------------------------
@pytest.mark.parametrize('optimize_time_and_phase', [False, True])
def test_no_inplace_editing_of_signals(optimize_time_and_phase):
    from gw_signal_tools.PSDs import psd_no_noise

    psd = psd_no_noise.copy()
    hp_f_fine_2 = hp_f_fine.copy()
    # TODO: check they do not share memory

    norm(
        hp_f_fine_2,
        f_range=[2 * f_min, 0.5 * f_max],
        df=hp_f_fine_2.df,
        optimize_time_and_phase=optimize_time_and_phase,
    )
    # -- Ensure some conversions take place, which could change signal
    # -- inplace. Setting df so that no interpolation takes place is
    # -- important because otherwise, interpolate would copy.

    assert_quantity_equal(hp_f_fine, hp_f_fine_2)

    # -- The following assertion is based on bug that was present for a
    # -- short time, where no copying of input PSD took place, so it was
    # -- changed inplace
    hp_f_fine_2.frequencies *= u.s
    norm(hp_f_fine_2, optimize_time_and_phase=optimize_time_and_phase)

    from gw_signal_tools.PSDs import psd_no_noise

    assert_quantity_equal(psd, psd_no_noise)


# %% -- Consistency tests with inner_product function --------------------------
def test_fd_td_match_consistency():
    norm_td_coarse = norm(hp_t, df=2**-2, f_range=[f_min, None])
    norm_fd_coarse = norm(hp_f_coarse, df=2**-2, f_range=[f_min, None])

    assert_allclose_quantity(norm_td_coarse, norm_fd_coarse, atol=0.0, rtol=0.11)

    norm_td_fine = norm(pad_to_dx(hp_t, dx=2**-4), df=2**-4, f_range=[f_min, None])
    norm_fd_fine = norm(hp_f_fine, df=2**-4, f_range=[f_min, None])

    assert_allclose_quantity(norm_td_fine, norm_fd_fine, atol=0.0, rtol=0.005)


def test_fd_td_overlap_consistency():
    norm_td = overlap(hp_t, hp_t, df=2**-4, f_range=[f_min, None])
    norm_fd_coarse = overlap(hp_f_coarse, hp_f_coarse, df=2**-2, f_range=[f_min, None])
    norm_fd_fine = overlap(hp_f_fine, hp_f_fine, df=2**-4, f_range=[f_min, None])

    # assert_allclose_quantity(u.Quantity([norm_td, norm_fd_coarse, norm_fd_fine]), u.Quantity([1.0, 1.0, 1.0]), atol=0.0, rtol=0.005)
    assert_allclose_quantity(
        norm_td, 1.0 * u.dimensionless_unscaled, atol=0.0, rtol=0.005
    )
    assert_allclose_quantity(
        norm_fd_coarse, 1.0 * u.dimensionless_unscaled, atol=0.0, rtol=0.005
    )
    assert_allclose_quantity(
        norm_fd_fine, 1.0 * u.dimensionless_unscaled, atol=0.0, rtol=0.005
    )
    assert_allclose_quantity(norm_td, norm_fd_fine, atol=0.0, rtol=0.005)


@pytest.mark.parametrize('hp_f', [hp_f_fine, hp_f_coarse])
def test_frequ_sampling_consistency(hp_f):
    # -- Sample on shifted frequencies
    delta_f = hp_f.df
    hp_f_2, _ = fd_wf_gen(
        wf_params
        | {
            'f22_start': wf_params['f22_start'] + delta_f / 3.0,
            'f_max': wf_params['f_max'] + delta_f / 3.0,
            'deltaF': delta_f,
        }
    )
    norm_mixed = overlap(hp_f, hp_f_2, df=2**-4, f_range=[f_min, None])
    assert_allclose_quantity(
        norm_mixed, 1.0 * u.dimensionless_unscaled, atol=0.0, rtol=1e-15
    )


def test_optimize_match_consistency():
    norm1_coarse = norm(hp_f_coarse)
    # norm2_coarse, info_coarse = norm(hp_f_coarse, optimize_time_and_phase=True,
    norm2_coarse, info_coarse = norm(
        hp_f_coarse, optimize_time=True, optimize_phase=True, return_opt_info=True
    )
    time_coarse = info_coarse['peak_time']
    phase_coarse = info_coarse['peak_phase']

    assert_allclose_quantity(norm1_coarse, norm2_coarse, atol=0.0, rtol=0.11)
    assert_allclose_quantity(0.0 * u.s, time_coarse, atol=1e-10, rtol=0.0)
    assert_allclose_quantity(0.0 * u.rad, phase_coarse, atol=3e-18, rtol=0.0)

    norm1_fine = norm(hp_f_fine)
    norm2_fine, info_fine = norm(
        hp_f_fine, optimize_time_and_phase=True, return_opt_info=True
    )
    time_fine = info_fine['peak_time']
    phase_fine = info_fine['peak_phase']

    assert_allclose_quantity(norm1_fine, norm2_fine, atol=0.0, rtol=5e-4)
    assert_allclose_quantity(0.0 * u.s, time_fine, atol=1e-12, rtol=0.0)
    assert_allclose_quantity(0.0 * u.rad, phase_fine, atol=1e-17, rtol=0.0)


@pytest.mark.parametrize('time_shift', [0.0 * u.s, 1e-3 * u.s, -0.2 * u.s, 0.5 * u.s])
@pytest.mark.parametrize(
    'phase_shift', [0.0 * u.rad, 0.12 * u.rad, -0.3 * np.pi * u.rad]
)
def test_optimize_match(time_shift, phase_shift):
    # norm_coarse = norm(hp_f_coarse)**2
    # hp_f_coarse_shifted = apply_time_phase_shift(hp_f_coarse, time_shift, phase_shift)

    # overlap_coarse, info_coarse = inner_product(
    #     hp_f_coarse,
    #     hp_f_coarse_shifted,
    #     optimize_time_and_phase=True,
    #     return_opt_info=True
    # )
    # time_coarse = info_coarse['peak_time']
    # phase_coarse = info_coarse['peak_phase']
    # match_series_coarse = info_coarse['match_series']

    # assert_allclose_quantity(norm_coarse, overlap_coarse, atol=0.0, rtol=4e-2)
    # assert_allclose_quantity(time_shift, time_coarse, atol=0.8*match_series_coarse.dx.value, rtol=0.0)
    # # assert_allclose_quantity(0.*u.rad, np.abs(phase_shift - phase_coarse) % (2.*np.pi*u.rad), atol=2e-17, rtol=0.01)
    # assert_allclose_quantity(phase_shift, phase_coarse, atol=0.06, rtol=0.0)

    # -- coarse performs REALLY bad, thus omitted for these tests

    norm_fine = norm(hp_f_fine) ** 2
    hp_f_fine_shifted = apply_time_phase_shift(hp_f_fine, time_shift, phase_shift)
    overlap_fine, info_fine = inner_product(
        hp_f_fine_shifted,
        hp_f_fine,
        optimize_time_and_phase=True,
        return_opt_info=True,
        # df=2**-3,  # Would decrease overlap agreement, thus comment
        min_dt_prec=1e-5 * u.s,
    )
    time_fine = info_fine['peak_time']
    phase_fine = info_fine['peak_phase']
    match_series_fine = info_fine['match_series']

    assert_allclose_quantity(norm_fine, overlap_fine, atol=0.0, rtol=9e-4)
    assert_allclose_quantity(
        time_shift, time_fine, atol=0.8 * match_series_fine.dx.value, rtol=0.0
    )
    assert_allclose_quantity(phase_shift, phase_fine, atol=1e-3, rtol=0.01)
    # -- To get accurate phase recovery, min_dt_prec has to be
    # -- sufficiently small (more accurate time shift = more accurate
    # -- match, since this is computed at this time shift)


@pytest.mark.parametrize(
    'signal',
    [
        hp_f_fine,
        FrequencySeries(
            np.flip(np.conjugate(hp_f_fine)[1:]),  # Exclude zero component
            f0=-hp_f_fine.frequencies[-1],
            df=hp_f_fine.df,
            unit=hp_f_fine.unit,
        ).append(hp_f_fine, inplace=True),
    ],
)
@pytest.mark.parametrize('min_dt_prec', [None, 1e-5])
def test_even_sample_size(signal, min_dt_prec):
    # psd = psd_no_noise.crop(start=signal.f0, end=signal.frequencies[-1])
    psd = signal_at_xindex(psd_no_noise, signal.frequencies, 1.0 * psd_no_noise.unit)
    norm_1, info_1 = optimized_inner_product(
        signal,
        signal,
        psd=psd,
        optimize_time=True,
        optimize_phase=True,
        return_opt_info=True,
        min_dt_prec=min_dt_prec,
    )

    _signal = signal[:-1]
    # _psd = psd_no_noise.crop(start=_signal.f0, end=_signal.frequencies[-1])
    _psd = signal_at_xindex(psd_no_noise, _signal.frequencies, 1.0 * psd_no_noise.unit)
    norm_2, info_2 = optimized_inner_product(
        _signal,
        _signal,
        psd=_psd,
        optimize_time=True,
        optimize_phase=True,
        return_opt_info=True,
        min_dt_prec=min_dt_prec,
    )

    assert_allclose_quantity(norm_1, norm_2, atol=0.0, rtol=0.0)
    assert_allclose_quantity(
        0.0 * u.s, [info_1['peak_time'], info_2['peak_time']], atol=0.0, rtol=0.0
    )
    assert_allclose_quantity(
        0.0 * u.rad, [info_1['peak_phase'], info_2['peak_phase']], atol=1e-18, rtol=0.0
    )


def test_different_optimizations():
    norm1 = norm(hp_f_fine, optimize_time_and_phase=False)
    norm2, info2 = norm(hp_f_fine, optimize_time_and_phase=True, return_opt_info=True)
    norm3, info3 = norm(
        hp_f_fine, optimize_time=True, optimize_phase=False, return_opt_info=True
    )
    norm4, info4 = norm(
        hp_f_fine, optimize_time=False, optimize_phase=True, return_opt_info=True
    )

    assert_allclose_quantity(norm1, [norm2, norm3, norm4], atol=0.0, rtol=4e-4)
    # -- rtol for usual deviation between simpson result and fft one.
    # -- Next test verifies that all optimized norms are actually equal.
    assert_allclose_quantity(norm2, [norm3, norm4], atol=0.0, rtol=0.0)

    time2 = info2['peak_time']
    time3 = info3['peak_time']
    time4 = info4['peak_time']
    assert_allclose_quantity(0.0 * u.s, [time2, time3, time4], atol=3e-13, rtol=0.0)

    phase2 = info2['peak_phase']
    phase3 = info3['peak_phase']
    phase4 = info4['peak_phase']
    assert_allclose_quantity(
        0.0 * u.rad, [phase2, phase3, phase4], atol=6.3e-18, rtol=0.0
    )


@pytest.mark.parametrize('f_min', [f_min, 30.0 * u.Hz])
@pytest.mark.parametrize('f_max', [50.0 * u.Hz, f_max])
def test_f_range(f_min, f_max):
    norm1 = norm(hp_f_fine, f_range=[f_min, f_max])
    norm_no_units = norm(hp_f_fine, f_range=[f_min.value, f_max.value])
    assert_quantity_equal(norm1, norm_no_units)

    hp_f_restricted, _ = fd_wf_gen(wf_params | {'f22_start': f_min, 'f_max': f_max})
    norm2 = norm(hp_f_restricted)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=1e-3)
    # -- Not fully equal due to potentially being one sample off when filling


@pytest.mark.parametrize(
    'f_range',
    [
        [-f_min, 1.1 * f_max],  # Too large, should be adjusted by function
        [None, None],
        pytest.param(
            [0.0 * u.m, None],
            marks=pytest.mark.xfail(
                raises=ValueError, strict=True, reason='Invalid unit for f_lower'
            ),
        ),
        pytest.param(
            [None, f_max.value * u.m],
            marks=pytest.mark.xfail(
                raises=ValueError, strict=True, reason='Invalid unit for f_upper'
            ),
        ),
        pytest.param(
            [1, 2, 3],
            marks=pytest.mark.xfail(
                raises=ValueError, strict=True, reason='Three values in f_range'
            ),
        ),
    ],
)
def test_f_range_handling(f_range):
    norm1 = norm(hp_f_fine)
    norm2 = norm(hp_f_fine, f_range=f_range)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize('interp', [False, True])
def test_positive_negative_f_range_consistency(interp):
    h = td_to_fd(pad_to_dx(hp_t, dx=hp_f_fine.df))
    h_symm = td_to_fd(pad_to_dx(hp_t, dx=hp_f_fine.df) + 0.0j)
    # -- h_symm has symmetric spectrum around f=0.0 and the same
    # -- spectrum as h for positive frequencies
    assert h.f0 != h_symm.f0  # Make sure they are not the same

    f_upper = f_max

    norm1 = norm(h, f_range=[0.0, f_upper])
    norm1_opt, info1 = norm(
        h,
        f_range=[0.0, f_upper],
        signal_interpolation=interp,
        optimize_time_and_phase=True,
        return_opt_info=True,
    )
    time_1 = info1['peak_time']
    assert_allclose_quantity(norm1, norm1_opt, atol=0.0, rtol=1e-12)
    assert_allclose_quantity(0.0 * u.s, time_1, atol=1e-12, rtol=0.0)

    norm2 = norm(h_symm, f_range=[-f_upper, f_upper])
    norm2_opt, info2 = norm(
        h_symm,
        f_range=[-f_upper, f_upper],
        signal_interpolation=interp,
        optimize_time_and_phase=True,
        return_opt_info=True,
    )
    time_2 = info2['peak_time']
    assert_allclose_quantity(norm2, norm2_opt, atol=0.0, rtol=1e-12)
    assert_allclose_quantity(0.0 * u.s, time_2, atol=1e-12, rtol=0.0)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=1e-15)
    assert_allclose_quantity(norm1_opt, norm2_opt, atol=0.0, rtol=1e-12)

    norm_plus = norm(h_symm, f_range=[0.0, f_upper], signal_interpolation=interp)
    norm_minus = norm(h_symm, f_range=[-f_upper, 0.0], signal_interpolation=interp)

    assert_allclose_quantity(norm_plus, norm_minus, atol=0.0, rtol=1e-15)
    assert_allclose_quantity(norm_plus, norm2, atol=0.0, rtol=1e-15)
    assert_allclose_quantity(norm_minus, norm2, atol=0.0, rtol=1e-15)


def test_df_consistency():
    # -- Same signal, decreasing df in inner_product
    norm1 = norm(hp_f_fine, df=hp_f_fine.df)
    norm2 = norm(hp_f_fine, df=hp_f_fine.df / 2)
    norm3 = norm(hp_f_fine, df=hp_f_fine.df / 4)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=2e-3)
    assert_allclose_quantity(norm1, norm3, atol=0.0, rtol=2e-3)
    assert_quantity_equal(
        norm2, norm3
    )  # Because linear interpolation the same for them

    # -- Different signals with matching df in inner_product
    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': hp_f_fine.df / 2})
    norm2 = norm(hp_f, df=hp_f_fine.df / 2)

    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': hp_f_fine.df / 4})
    norm3 = norm(hp_f, df=hp_f_fine.df / 4)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=5e-4)
    assert_allclose_quantity(norm1, norm3, atol=0.0, rtol=6e-4)
    assert_allclose_quantity(norm2, norm3, atol=0.0, rtol=2e-4)


@pytest.mark.parametrize(
    'df1,df2',
    [
        [2**-5, 2**-5 * u.Hz],
        [2**-5, 2**-2 * u.mHz],
        pytest.param(
            2**-5,
            2**-5 * u.m,
            marks=pytest.mark.xfail(
                raises=ValueError, strict=True, reason='Invalid unit for df'
            ),
        ),
    ],
)
def test_df_handling(df1, df2):
    norm1 = norm(hp_f_fine, df=df1)
    norm2 = norm(hp_f_fine, df=df2)

    assert_quantity_equal(norm1, norm2)


@pytest.mark.parametrize('f_range', [[None, None], [f_min, f_max], [2 * f_min, None]])
@pytest.mark.parametrize(
    'eval_frequs',
    [
        hp_f_fine.frequencies,
        np.logspace(
            np.log10(f_min.value), np.log10(f_max.value), num=hp_f_fine.size // 10
        )
        << u.Hz,
    ],
)  # No resampling, unequal resampling
def test_calc_modes(f_range, eval_frequs):
    h_interp = signal_at_xindex(hp_f_fine, eval_frequs, fill_val=0.0 * hp_f_fine.unit)
    psd = FrequencySeries(
        np.ones(h_interp.size),
        frequencies=eval_frequs,
        unit=u.strain**2 / hp_f_fine.xunit,
    )

    norm1 = norm(hp_f_fine, psd=psd, f_range=f_range)  # Reference result
    norm2 = norm(h_interp, psd=psd, f_range=f_range, signal_interpolation=False)
    norm3 = norm(h_interp, psd=psd, f_range=f_range, signal_interpolation=True)
    assert_allclose_quantity(norm1, [norm2, norm3], atol=0.0, rtol=0.003)
    # -- Some deviation is expected, we compare signals evaluated at different
    # -- frequencies, i.e. there are different interpolations going on.
    assert_allclose_quantity(norm2, norm3, atol=0.0, rtol=0.0012)
    # -- Between signals evaluated on same frequencies, difference is
    # -- smaller, as it should be the case.


@pytest.mark.parametrize('f_range', [[None, None], [f_min, f_max], [2 * f_min, None]])
def test_no_interpolation(f_range):
    for opt in [False, True]:
        norm1 = norm(
            hp_f_fine,
            f_range=f_range,
            signal_interpolation=True,
            optimize_time_and_phase=opt,
        )
        norm2 = norm(
            hp_f_fine,
            f_range=f_range,
            signal_interpolation=False,
            optimize_time_and_phase=opt,
        )

        assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=0.0)
        # -- No deviation at all in this case, since default df is same, so
        # -- signal interpolation to new frequencies and restriction of
        # -- current/given frequencies are perfectly equivalent.


# TODO: now test for failures!

# giving eval_frequencies that do not match signal1
# giving unequally sampled eval_frequencies with optimization on


def test_different_units():
    norm2 = norm(hp_f_fine, psd=psd_no_noise)

    rescale_unit = u.s
    hp_f_fine_rescaled = hp_f_fine.copy()
    hp_f_fine_rescaled.frequencies *= rescale_unit
    hp_f_fine_rescaled /= rescale_unit
    # -- NOTE: rescaling the amplitude this way is not strictly
    # -- necessary, one could also get a consistent result without this
    # -- step. By doing that, we simply ensure the resulting norm is
    # -- dimensionless, making the subsequent comparison easier.

    norm1 = norm(hp_f_fine)
    norm2 = norm(hp_f_fine_rescaled)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=0.001)

    psd_no_noise_rescaled = psd_no_noise.copy()  # Verify manually what happens
    psd_no_noise_rescaled.frequencies *= rescale_unit
    psd_no_noise_rescaled /= rescale_unit
    # -- Also rescale density that it represents, psd is per frequ_unit

    norm1 = norm(hp_f_fine_rescaled, psd=psd_no_noise_rescaled)
    norm2 = norm(hp_f_fine, psd=psd_no_noise)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=0.001)

    hp_f_fine_rescaled_2 = hp_f_fine.copy()
    hp_f_fine_rescaled_2 *= u.m**2

    norm3 = np.sqrt(inner_product(hp_f_fine, hp_f_fine_rescaled_2))

    assert_allclose_quantity(norm1 * u.m, norm3, atol=0.0, rtol=0.001)


# TODO (maybe): test with mass rescaled waveforms?


# %% -- Confirm that certain errors are raised ---------------------------------
class InnProdErrorRaising(unittest.TestCase):
    def test_signal_type_checking(self):
        with self.assertRaises(TypeError):
            inner_product(np.array([42]), hp_f_fine)

        with self.assertRaises(TypeError):
            inner_product(hp_f_fine, np.array([42]))

        with self.assertRaises(TypeError):
            inner_product(hp_f_fine, hp_f_fine, psd=np.array([42]))

    def test_frequ_unit_checking(self):
        with self.assertRaises(AssertionError):
            hp_f_fine_wrong = hp_f_fine.copy()
            hp_f_fine_wrong.frequencies *= u.m

            inner_product(hp_f_fine, hp_f_fine_wrong)

        with self.assertRaises(AssertionError):
            psd_no_noise_wrong = psd_no_noise.copy()
            psd_no_noise_wrong.frequencies *= u.m

            norm(hp_f_fine, psd=psd_no_noise_wrong)

    def test_df_unit_testing(self):
        with self.assertRaises(ValueError):
            norm(hp_f_fine, df=0.0625 * u.m)

    def test_min_dt_prec_unit_testing(self):
        with self.assertRaises(ValueError):
            norm(hp_f_fine, min_dt_prec=1e-3 * u.m, optimize_time_and_phase=True)


# %% -- Confirming results with PyCBC match function ---------------------------
from gwpy.frequencyseries.frequencyseries import FrequencySeries  # noqa: E402
from pycbc.waveform import get_fd_waveform  # noqa: E402
from pycbc.filter import match  # noqa: E402
from pycbc.psd import aLIGOZeroDetHighPower  # noqa: E402

f_low, f_high = 20, 350  # f_min and some essentially arbitrary cutoff
sample_rate = 4096

# -- Enter some arbitrary parameters here
wfs_to_compare = {
    'signal1': {'mass1': 10, 'mass2': 10, 'spin1': 0.6, 'spin2': 0.0},
    'signal2': {'mass1': 96, 'mass2': 20, 'spin1': 0.0, 'spin2': 0.1},
}

hp_1_pycbc, _ = get_fd_waveform(
    approximant=approximant,
    **wfs_to_compare['signal1'],
    f_lower=f_low,
    f_upper=f_high,
    delta_f=1.0 / sample_rate,
)

hp_2_pycbc, _ = get_fd_waveform(
    approximant=approximant,
    **wfs_to_compare['signal2'],
    f_lower=f_low,
    f_upper=f_high,
    delta_f=1.0 / sample_rate,
)

tlen = max(len(hp_1_pycbc), len(hp_2_pycbc))
hp_1_pycbc.resize(tlen)
hp_2_pycbc.resize(tlen)

delta_f = 1.0 / hp_2_pycbc.duration
flen = tlen // 2 + 1
psd_pycbc = aLIGOZeroDetHighPower(flen, delta_f, f_low)

hp_1_pycbc_converted = FrequencySeries.from_pycbc(hp_1_pycbc) * u.s
hp_2_pycbc_converted = FrequencySeries.from_pycbc(hp_2_pycbc) * u.s
psd_pycbc_converted = FrequencySeries.from_pycbc(psd_pycbc) / u.Hz


def test_match_pycbc():
    overlap_pycbc, time_pycbc, phase_pycbc = match(
        hp_1_pycbc,
        hp_2_pycbc,
        v1_norm=1.0,
        v2_norm=1.0,
        psd=psd_pycbc,
        low_frequency_cutoff=f_low,
        high_frequency_cutoff=f_high,
        return_phase=True,
    )
    time_pycbc *= 1 / (2 * (tlen - 1) * delta_f)

    overlap_gw_signal_tools, info = inner_product(
        hp_1_pycbc_converted,
        hp_2_pycbc_converted,
        psd_pycbc_converted,
        f_range=[f_low, f_high],
        optimize_time_and_phase=True,
        return_opt_info=True,
    )
    time_gw_signal_tools = info['peak_time'].value
    # phase_gw_signal_tools = info["peak_phase"].value

    assert_allclose(overlap_pycbc, overlap_gw_signal_tools, atol=0.0, rtol=2e-3)
    assert_allclose(
        np.abs(time_pycbc), np.abs(time_gw_signal_tools), atol=0.0, rtol=2e-2
    )
    # assert_allclose(phase_pycbc, phase_gw_signal_tools, atol=0.0, rtol=0.0)
    # -- Phase is not matching well, seems to be due to different
    # -- conventions in what the phase output is (pycbc changes phase,
    # -- setting it to zero at certain point), potentially leading to an
    # -- unequal shift in the phases of different signals and thus a
    # -- different phase needed to align them)


def test_overlap_pycbc():
    overlap_normalized_pycbc, _ = match(
        hp_1_pycbc,
        hp_2_pycbc,
        psd=psd_pycbc,
        low_frequency_cutoff=f_low,
        high_frequency_cutoff=f_high,
    )

    overlap_normalized_gw_signal_tools = overlap(
        hp_1_pycbc_converted,
        hp_2_pycbc_converted,
        psd_pycbc_converted,
        f_range=[f_low, f_high],
        optimize_time_and_phase=True,
    )

    assert_allclose(
        overlap_normalized_pycbc,
        overlap_normalized_gw_signal_tools,
        atol=0.0,
        rtol=2e-3,
    )


def test_norm_optimized():
    norm1_gw_signal_tools = overlap(
        hp_1_pycbc_converted,
        hp_1_pycbc_converted,
        psd_pycbc_converted,
        f_range=[f_low, f_high],
        optimize_time_and_phase=True,
    )
    norm2_gw_signal_tools = overlap(
        hp_2_pycbc_converted,
        hp_2_pycbc_converted,
        psd_pycbc_converted,
        f_range=[f_low, f_high],
        optimize_time_and_phase=True,
    )

    assert_allclose(
        1.0, [norm1_gw_signal_tools, norm2_gw_signal_tools], atol=0.0, rtol=1e-5
    )


# %% -- Testing Overlap Optimization -------------------------------------------
@pytest.mark.slow  # Because mass1 is involved, time and phase are fast
@pytest.mark.parametrize(
    'opt, shift',
    [
        [False, 2.0 * u.Msun],
        [True, 2.0 * u.Msun],
        [True, 5.0 * u.Msun],
        [True, 10.0 * u.Msun],
    ],
)
def test_mass_opt(opt, shift):
    def wf_gen(wf_params):
        return fd_wf_gen(wf_params)[0]

    def shifted_wf_gen(wf_params):
        return wf_gen(wf_params | {'mass1': wf_params['mass1'] + shift})

    wf1_shifted, wf2_shifted, opt_params = optimize_overlap(
        wf_params,
        wf_gen,
        shifted_wf_gen,
        opt_params=['mass1'],
        optimize_time_and_phase=opt,
    )
    # -- Works better if time and phase optimization is on because
    # -- mismatch is a smoother function in that case (without this kind
    # -- of optimization, there are many local maxima)

    assert_allclose_quantity(
        opt_params['mass1'] + shift, wf_params['mass1'], atol=0.0, rtol=1e-2
    )

    _match = overlap(wf1_shifted, wf2_shifted)
    assert_allclose(_match, 1.0, atol=1e-3, rtol=0.0)


@pytest.mark.parametrize('tc', [0.0 * u.s, 1e-3 * u.s, -0.2 * u.s, 0.5 * u.s])
@pytest.mark.parametrize('phic', [0.0 * u.rad, 0.12 * u.rad, -0.3 * np.pi * u.rad])
def test_time_phase_opt(tc, phic):
    def wf_gen(wf_params):
        return fd_wf_gen(wf_params)[0]

    def shifted_wf_gen(wf_params):
        wf = wf_gen(wf_params)
        return apply_time_phase_shift(wf, tc, phic)

    wf1_shifted, wf2_shifted, opt_params = optimize_overlap(
        wf_params,
        shifted_wf_gen,
        wf_gen,
        opt_params=['time', 'phase'],
        df=2**-3,  # Not required, but speeds up calculations
        min_dt_prec=1e-5 * u.s,
    )

    wf1_shifted_2, wf2_shifted_2, opt_params_2 = optimize_overlap(
        wf_params,
        shifted_wf_gen,
        wf_gen,
        optimize_time_and_phase=True,
        df=2**-3,  # Not required, but speeds up calculations
        min_dt_prec=1e-5 * u.s,
    )

    assert_allclose_quantity(
        tc, [opt_params['time'], opt_params_2['time']], atol=1e-5, rtol=0.0
    )
    assert_allclose_quantity(
        phic, [opt_params['phase'], opt_params_2['phase']], atol=1e-3, rtol=1e-2
    )  # atol for phase zero

    assert_allclose(
        1.0,
        [
            overlap(wf1_shifted, wf2_shifted),
            overlap(wf1_shifted, wf2_shifted, optimize_time_and_phase=True),
            overlap(wf1_shifted_2, wf2_shifted_2),
            overlap(wf1_shifted_2, wf2_shifted_2, optimize_time_and_phase=True),
        ],
        atol=1e-4,
        rtol=0.0,
    )
    # -- Repeated optimization should not change overlap results


@pytest.mark.slow  # Because mass1 is involved, time and phase are fast
@pytest.mark.parametrize('opt_time', [True, False])
@pytest.mark.parametrize('opt_phase', [True, False])
def test_time_phase_arg_interplay(opt_time, opt_phase):
    # -- We now pass time and phase arguments, along with optimization
    # -- enabled via inner product keyword arguments
    tc = -0.2 * u.s
    phic = 0.12 * u.rad

    def wf_gen(wf_params):
        return fd_wf_gen(wf_params)[0]

    def shifted_wf_gen(wf_params):
        wf = wf_gen(wf_params)
        return apply_time_phase_shift(wf, tc, phic)

    wf1_shifted, wf2_shifted, opt_params = optimize_overlap(
        wf_params,
        shifted_wf_gen,
        wf_gen,
        opt_params=['mass1', 'time', 'phase'],
        optimize_time=opt_time,
        optimize_phase=opt_phase,
        df=2**-3,  # Not required, but speeds up calculations
        min_dt_prec=1e-5,
    )

    assert_allclose_quantity(
        wf_params['mass1'], opt_params['mass1'], atol=0.0, rtol=1e-3
    )
    assert_allclose_quantity(tc, opt_params['time'], atol=1e-5, rtol=0.0)
    assert_allclose_quantity(phic, opt_params['phase'], atol=1e-3, rtol=1e-2)
    assert_allclose(1.0, overlap(wf1_shifted, wf2_shifted), atol=1e-4, rtol=0.0)


@pytest.mark.parametrize('tc', [0.0 * u.s, 1e-3 * u.s, -0.2 * u.s, 0.5 * u.s])
@pytest.mark.parametrize('phic', [0.0 * u.rad, 0.12 * u.rad, -0.3 * np.pi * u.rad])
def test_time_phase_gen_handling(tc, phic):
    # -- See what happens when time and phase are in wf_params
    common_tc, common_phic = tc / 3.0, phic / 3.0
    # tc_diff, phic_diff = tc - common_tc, phic - common_phic
    tc_diff, phic_diff = tc, phic

    def wf_gen(wf_params):
        return fd_wf_gen(wf_params)[0]

    wrapped_wf_gen = time_phase_wrapper(wf_gen)

    def shifted_wf_gen(wf_params):
        wf = wrapped_wf_gen(wf_params)
        return apply_time_phase_shift(wf, tc_diff, phic_diff)

    wf1_shifted, wf2_shifted, opt_params = optimize_overlap(
        wf_params | {'time': common_tc, 'phase': common_phic},
        shifted_wf_gen,
        wrapped_wf_gen,
        opt_params=['time', 'phase'],
        df=2**-3,  # Not required, but speeds up calculations
        min_dt_prec=1e-5,
    )

    assert_allclose_quantity(
        common_tc + tc_diff, opt_params['time'], atol=1e-5, rtol=0.0
    )
    assert_allclose_quantity(
        common_phic + phic_diff, opt_params['phase'], atol=1e-3, rtol=1e-2
    )
    assert_allclose(1.0, overlap(wf1_shifted, wf2_shifted), atol=1e-4, rtol=0.0)
    # -- Overlap threshold slightly higher than in previous functions,
    # -- but still tolerable. And this situation is much trickier, we
    # -- deliberately introduce a degeneracy, so this is ok.


@pytest.mark.slow  # Because mass1 is involved, time and phase are fast
@pytest.mark.parametrize(
    'params', ['time', ['mass1', 'time'], 'phase', ['mass1', 'phase']]
)
def test_time_phase_arg_handling(params):
    # -- Testing strange combinations for handling
    def wf_gen(wf_params):
        return fd_wf_gen(wf_params)[0]

    _, _, opt_params = optimize_overlap(
        wf_params,
        wf_gen,
        wf_gen,
        opt_params=params,
    )

    if isinstance(params, str):
        params = [params]  # For length comparison
    assert len(opt_params) == len(params)
