# -- Standard Lib Imports
import unittest

# -- Third Party Imports
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from lalsimulation.gwsignal import gwsignal_get_waveform_generator
import lalsimulation.gwsignal.core.waveform as wfm
from gwpy.testing.utils import assert_quantity_equal
from gwpy.frequencyseries import FrequencySeries
import pytest

# -- Local Package Imports
from gw_signal_tools.waveform.utils import (
    td_to_fd_waveform, fd_to_td_waveform,
    pad_to_get_target_df, restrict_f_range,
    get_signal_at_target_df, get_signal_at_target_frequs,
    get_strain, fill_f_range,
    # get_mass_scaled_wf
)
from gw_signal_tools.test_utils import (
    assert_allclose_quantity, assert_allequal_series
)


#%% -- Initializing commonly used variables -----------------------------------
f_min = 20.*u.Hz
f_max = 1024.*u.Hz

wf_params = {
    'mass1': 36*u.solMass,
    'mass2': 29*u.solMass,
    'deltaT': 1./2048.*u.s,
    'f22_start': f_min,
    'f_max': f_max,
    'f22_ref': 20.*u.Hz,
    'phi_ref': 0.*u.rad,
    'distance': 440.*u.Mpc,
    'inclination': 0.0*u.rad,
    'eccentricity': 0.*u.dimensionless_unscaled,
    'longAscNodes': 0.*u.rad,
    'meanPerAno': 0.*u.rad,
    'condition': 0
}

# -- Make sure mass1 and mass2 are not in default_dict
import lalsimulation.gwsignal.core.parameter_conventions as pc
pc.default_dict.pop('mass1', None);
pc.default_dict.pop('mass2', None);

approximant = 'IMRPhenomXPHM'
gen = gwsignal_get_waveform_generator(approximant)

def td_wf_gen(wf_params):
    return wfm.GenerateTDWaveform(wf_params, gen)

def fd_wf_gen(wf_params):
    return wfm.GenerateFDWaveform(wf_params, gen)


hp_t, hc_t = td_wf_gen(wf_params)

hp_f_fine, hc_f_fine = fd_wf_gen(wf_params)

hp_f_coarse, hc_f_coarse = fd_wf_gen(wf_params | {'deltaF': 1.0 / (hp_t.size * hp_t.dx)})


hp_f_fine.override_unit(u.s)
hc_f_fine.override_unit(u.s)
hp_f_coarse.override_unit(u.s)
hc_f_coarse.override_unit(u.s)
# NOTE: unit conversion is needed because of inconsistent handling of units in
# lal, not because of error in gw_signal_tools code


#%% -- Testing transformation into one domain and back ------------------------
def test_ifft_fft_consistency():
    f_min_comp, f_max_comp = 20.0 * u.Hz, 512.0 * u.Hz  # Restrict to interesting region, elsewhere only values close to zero and thus numerical errors might occur

    hp_f_coarse_ifft_fft = td_to_fd_waveform(fd_to_td_waveform(hp_f_coarse))

    hp_f_coarse_cropped = hp_f_coarse.crop(start=f_min_comp, end=f_max_comp)
    hp_f_coarse_ifft_fft_cropped = hp_f_coarse_ifft_fft.crop(start=f_min_comp, end=f_max_comp)

    assert_allclose_quantity(hp_f_coarse_cropped.frequencies, hp_f_coarse_ifft_fft_cropped.frequencies, atol=0.0, rtol=1e-10)
    # assert_quantity_equal(hp_f_coarse_cropped.frequencies, hp_f_coarse_ifft_fft_cropped.frequencies)
    assert_allclose_quantity(hp_f_coarse_cropped, hp_f_coarse_ifft_fft_cropped, atol=0.0, rtol=1e-10)
    # assert_quantity_equal(hp_f_coarse_cropped, hp_f_coarse_ifft_fft_cropped)


    hp_f_fine_ifft_fft = td_to_fd_waveform(fd_to_td_waveform(hp_f_fine))

    hp_f_fine_cropped = hp_f_fine.crop(start=f_min_comp, end=f_max_comp)
    hp_f_fine_ifft_fft_cropped = hp_f_fine_ifft_fft.crop(start=f_min_comp, end=f_max_comp)

    assert_allclose_quantity(hp_f_fine_cropped.frequencies, hp_f_fine_ifft_fft_cropped.frequencies, atol=0.0, rtol=1e-10)
    # assert_quantity_equal(hp_f_fine_cropped.frequencies, hp_f_fine_ifft_fft_cropped.frequencies)
    assert_allclose_quantity(hp_f_fine_cropped, hp_f_fine_ifft_fft_cropped, atol=0.0, rtol=1e-10)
    # assert_quantity_equal(hp_f_fine_cropped, hp_f_fine_ifft_fft_cropped)


def test_fft_ifft_consistency():
    hp_t_fft_ifft = fd_to_td_waveform(td_to_fd_waveform(hp_t))

    t_min_comp, t_max_comp = max(hp_t.t0, hp_t_fft_ifft.t0), min(hp_t.times[-1], hp_t_fft_ifft.times[-1])  # hp_t_fft_ifft_fine is padded to be much longer
    # t_min_comp, t_max_comp = -0.5, 0.01  # hp_t_fft_ifft_fine is padded to be much longer

    hp_t_cropped = hp_t.crop(start=t_min_comp, end=t_max_comp)#[2:]
    hp_t_fft_ifft_cropped = hp_t_fft_ifft.crop(start=t_min_comp, end=t_max_comp)[1:]

    # Have to apply different kind of threshold here because coarse sampling is indeed very coarse
    # -> slight phase shift due to that is biggest problem, causes amplitude errors in comparison of values at same time
    # assert_allclose_quantity(hp_t_cropped.times, hp_t_fft_ifft_cropped.times, atol=2*hp_t.dt.value, rtol=0.0)  # Oof, this is a lot...
    assert_allclose_quantity(hp_t_cropped.times, hp_t_fft_ifft_cropped.times, atol=5e-4, rtol=0)
    assert_allclose_quantity(hp_t_cropped, hp_t_fft_ifft_cropped, atol=2e-23, rtol=0.0)
    # assert_quantity_equal(hp_t_cropped, hp_t_fft_ifft_cropped)


    hp_t_fft_ifft_fine = fd_to_td_waveform(td_to_fd_waveform(pad_to_get_target_df(hp_t, df=0.0625 * u.Hz)))

    t_min_comp, t_max_comp = max(hp_t.t0, hp_t_fft_ifft_fine.t0), min(hp_t.times[-1], hp_t_fft_ifft_fine.times[-1])  # hp_t_fft_ifft_fine is padded to be much longer
    
    hp_t_cropped = hp_t.crop(start=t_min_comp, end=t_max_comp)[1:]
    hp_t_fft_ifft_fine_cropped = hp_t_fft_ifft_fine.crop(start=t_min_comp, end=t_max_comp)[1:]
    # NOTE: for some reason, first sample is not equal. Thus excluded here

    # assert_allclose_quantity(hp_t_cropped.times, hp_t_fft_ifft_fine_cropped.times, atol=0.0, rtol=1e-10)
    assert_allclose_quantity(hp_t_cropped.times, hp_t_fft_ifft_fine_cropped.times, atol=5e-12, rtol=0)
    assert_allclose_quantity(hp_t_cropped, hp_t_fft_ifft_fine_cropped, atol=1e-29, rtol=1e-8)
    # assert_quantity_equal(hp_t_cropped, hp_t_fft_ifft_fine_cropped)


#%% -- Testing transformations with generated signals from different domain ---
def test_fd_td_consistency():
    # NOTE: we have to apply different thresholds for certain frequency regions here.
    # For f_min_comp close to f_min from the parameter dictionary above, the threshold
    # has to be chosen a bit higher than the usual 1%. Here, it comes into play
    # that tapering is applied to TDWaveform that we do FFT of, while this is not
    # done for FDWaveform. This causes certain differences in the Fourier components

    f_min_comp, f_max_comp = 20.0 * u.Hz, 512.0 * u.Hz  # Restrict to interesting region, elsewhere only values close to zero and thus numerical errors might occur
    
    hp_t_f_coarse = td_to_fd_waveform(hp_t)

    hp_f_coarse_cropped = hp_f_coarse.crop(start=f_min_comp, end=f_max_comp)
    hp_t_f_coarse_cropped = hp_t_f_coarse.crop(start=f_min_comp, end=f_max_comp)

    assert_allclose_quantity(hp_f_coarse_cropped.frequencies, hp_t_f_coarse_cropped.frequencies, atol=0.0, rtol=1e-14)
    # assert_quantity_equal(hp_f_coarse_cropped.frequencies, hp_t_f_coarse_cropped.frequencies)
    assert_allclose_quantity(hp_f_coarse_cropped, hp_t_f_coarse_cropped, atol=0.0, rtol=0.05)
    # assert_quantity_equal(hp_f_coarse_cropped, hp_t_f_coarse_cropped)

    # For a finer resolution, we have to pad signal
    hp_t_padded = pad_to_get_target_df(hp_t, df=hp_f_fine.df)
    hp_t_f_fine = td_to_fd_waveform(hp_t_padded)

    hp_f_fine_cropped = hp_f_fine.crop(start=f_min_comp, end=f_max_comp)
    hp_t_f_fine_cropped = hp_t_f_fine.crop(start=f_min_comp, end=f_max_comp)

    # assert_allclose_quantity(hp_f_fine_cropped.frequencies, hp_t_f_fine_cropped.frequencies, atol=0.0, rtol=0.05)
    assert_quantity_equal(hp_f_fine_cropped.frequencies, hp_t_f_fine_cropped.frequencies)
    assert_allclose_quantity(hp_f_fine_cropped, hp_t_f_fine_cropped, atol=0.0, rtol=0.01)
    # assert_quantity_equal(hp_f_fine_cropped, hp_t_f_fine_cropped)


    f_min_comp, f_max_comp = 25.0 * u.Hz, 512.0 * u.Hz  # Restrict to interesting region, elsewhere only values close to zero and thus numerical errors might occur
    
    hp_t_f_coarse = td_to_fd_waveform(hp_t)

    hp_f_coarse_cropped = hp_f_coarse.crop(start=f_min_comp, end=f_max_comp)
    hp_t_f_coarse_cropped = hp_t_f_coarse.crop(start=f_min_comp, end=f_max_comp)

    assert_allclose(hp_f_coarse_cropped, hp_t_f_coarse_cropped, atol=0.0, rtol=0.01)

    # For a finer resolution, we have to pad signal
    hp_t_padded = pad_to_get_target_df(hp_t, df=hp_f_fine.df)
    hp_t_f_fine = td_to_fd_waveform(hp_t_padded)

    hp_f_fine_cropped = hp_f_fine.crop(start=f_min_comp, end=f_max_comp)
    hp_t_f_fine_cropped = hp_t_f_fine.crop(start=f_min_comp, end=f_max_comp)

    assert_allclose(hp_f_fine_cropped, hp_t_f_fine_cropped, atol=0.0, rtol=0.01)


#%% -- Verification of complex transformation ---------------------------------
def test_complex_fft_ifft_consistency():
    h_symm = td_to_fd_waveform(pad_to_get_target_df(hp_t, df=hp_f_fine.df) + 0.j)
    # Padding hp_t to make sure resolution is sufficient and to avoid
    # wrap-around due to insufficient length
    h_symm_t = fd_to_td_waveform(h_symm)

    t_min, t_max = max(hp_t.times[0], h_symm_t.times[0]), min(hp_t.times[-1], h_symm_t.times[-1])
    assert_allclose(hp_t.crop(t_min, t_max).times, h_symm_t.crop(t_min, t_max).times, atol=5e-12, rtol=0)
    assert_allclose(hp_t.crop(t_min, t_max).value, h_symm_t.crop(t_min, t_max).value, atol=5e-27, rtol=0)


def test_complex_ifft_fft_consistency():
    h_symm = FrequencySeries(
        np.flip(np.conjugate(hp_f_fine+0.j)[1:]),
        f0=-hp_f_fine.frequencies[-1],
        df=hp_f_fine.df
    ).append(hp_f_fine+0.j, inplace=True)*u.s

    h_symm_t = fd_to_td_waveform(h_symm)

    # t_min, t_max = max(hp_t.times[0], h_symm_t.times[0]), min(hp_t.times[-1], h_symm_t.times[-1])
    
    h_symm_f = td_to_fd_waveform(h_symm_t)

    # assert_allclose_series(h_symm, h_symm_f, atol=0., rtol=0.)
    assert_allclose_quantity(h_symm.frequencies, h_symm_f.frequencies, atol=3e-11, rtol=0.)
    assert_allclose_quantity(h_symm.value, h_symm_f.value, atol=4e-33, rtol=0.)
    # Some residual imaginary part is there, but of negligible amplitude
    # f_min_comp, f_max_comp = 25.0 * u.Hz, 512.0 * u.Hz  # Restrict to interesting region, elsewhere only values close to zero and thus numerical errors might occur
    # assert_allclose_series(h_symm.crop(start=f_min_comp, end=f_max_comp), h_symm_f.crop(start=f_min_comp, end=f_max_comp), atol=0., rtol=0.)


def test_complex_and_real_fft_consistency():
    h_symm = td_to_fd_waveform(pad_to_get_target_df(hp_t, df=hp_f_fine.df) + 0.j)
    # Padding hp_t to make sure resolution is sufficient and to avoid
    # wrap-around due to insufficient length
    h_neg = h_symm[h_symm.frequencies < 0.*u.Hz][1:]  # Filter somehow includes 0 as well, thus excluding first one
    h_pos = h_symm[h_symm.frequencies > 0.*u.Hz]

    assert_allclose(-h_neg.frequencies[::-1], h_pos.frequencies, atol=0, rtol=0)
    assert_allclose(h_neg[::-1].real, h_pos.real, atol=3e-38, rtol=0)
    assert_allclose(-h_neg[::-1].imag, h_pos.imag, atol=3e-28, rtol=0)
    # Note: as https://en.wikipedia.org/wiki/Fourier_transform#Conjugation
    # shows, real signals have the following property: real part of Fourier
    # spectrum is symmetric around f=0, while imaginary part is antisymmetric.


def test_complex_and_real_ifft_consistency():
    h_symm = FrequencySeries(
        np.flip(np.conjugate(hp_f_fine)[1:]),
        f0=-hp_f_fine.frequencies[-1],
        df=hp_f_fine.df
    ).append(hp_f_fine, inplace=True)

    h_symm_t = fd_to_td_waveform(h_symm)
    h_t = fd_to_td_waveform(hp_f_fine)

    assert_allclose_quantity(h_t.times, h_symm_t[:-1].times, atol=5e-4, rtol=0)  # Deviations on scale of dt are ok
    assert_allclose_quantity(h_t, h_symm_t[:-1], atol=2e-24, rtol=0)  # Context: peaks are at roughly 1e-21


#%% -- Testing helper functions for frequency region stuff --------------------
@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df])
# These input values are powers of two, have to be reproduced exactly
def test_pad_to_target_df_exact(df):
    hp_t_padded = pad_to_get_target_df(hp_t, df)
    hp_t_f = td_to_fd_waveform(hp_t_padded)

    assert_quantity_equal(df, hp_t_f.df)


@pytest.mark.parametrize('df', [0.007*u.Hz, 0.001*u.Hz])
# These input values are not exact powers of two and thus cannot be
# reproduced exactly (thus ensure sufficient accuracy)
def test_pad_to_target_df_not_exact(df):
    hp_t_padded = pad_to_get_target_df(hp_t, df)
    hp_t_f = td_to_fd_waveform(hp_t_padded)

    assert_allclose_quantity(df, hp_t_f.df, atol=0.0, rtol=1e-5)


@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df, 0.007*u.Hz, 0.001*u.Hz])
@pytest.mark.parametrize('full_metadata', [False, True])
def test_pad_to_target_df_exact(df, full_metadata):
    hp_f_interp = get_signal_at_target_df(hp_f_fine, df, full_metadata=full_metadata)

    assert_quantity_equal(df, hp_f_interp.df)


@pytest.mark.parametrize('df', [hp_f_fine.df, hp_f_fine.df / 2, hp_f_fine.df / 4, 0.007*u.Hz, 0.001*u.Hz])  # hp_f_coarse.df too coarse for comparison to make sense
@pytest.mark.parametrize('f_low', [0.9 * f_min, f_min])
@pytest.mark.parametrize('f_high', [f_max, 1.1 * f_max])
def test_get_signal_at_target_frequs_interp_and_padding(f_low, f_high, df):
    target_frequs = np.arange(f_low.value, f_high.value , step=df.value) << u.Hz
    hp_f_at_target_frequs = get_signal_at_target_frequs(hp_f_fine, target_frequs, fill_val=0.0)

    assert_quantity_equal(hp_f_at_target_frequs.frequencies, target_frequs)


    hp_f_at_df, _ = fd_wf_gen(wf_params | {'deltaF': df})
    hp_f_at_df.override_unit(u.s)

    hp_f_at_df = hp_f_at_df[
        (hp_f_at_df.frequencies >= f_min)
        & (hp_f_at_df.frequencies <= f_max)
    ]

    hp_f_at_target_frequs_restricted_1 = hp_f_at_target_frequs[
        (hp_f_at_target_frequs.frequencies >= f_min)
        & (hp_f_at_target_frequs.frequencies <= f_max)
    ]

    assert_allclose_quantity(hp_f_at_df.f0, hp_f_at_target_frequs_restricted_1.f0, atol=df.value, rtol=0.0)
    assert_allclose_quantity(hp_f_at_df.frequencies[-1], hp_f_at_target_frequs_restricted_1.frequencies[-1], atol=df.value, rtol=0.0)

    min_size = min(hp_f_at_df.size, hp_f_at_target_frequs_restricted_1.size)

    if np.abs(hp_f_at_df.f0 - hp_f_at_target_frequs_restricted_1.f0) < 0.5 * df:
        hp_f_at_df = hp_f_at_df[:min_size]
        hp_f_at_target_frequs_restricted_1 = hp_f_at_target_frequs_restricted_1[:min_size]
    else:
        hp_f_at_df = hp_f_at_df[hp_f_at_df.size - min_size:]
        hp_f_at_target_frequs_restricted_1 = hp_f_at_target_frequs_restricted_1[hp_f_at_target_frequs_restricted_1.size - min_size:]
    

    assert_allclose_quantity(hp_f_at_df.frequencies, hp_f_at_target_frequs_restricted_1.frequencies, atol=df.value, rtol=0.0)
    assert_allclose_quantity(hp_f_at_df, hp_f_at_target_frequs_restricted_1, atol=2e-24, rtol=0.0)
    # Frequencies are slightly shifted, which means we have to allow certain
    # tolerance. rtol not suited here because we might shift away from zero
    # to finite value, causing large relative deviations
    # Could choose 1e-24 for first three, but would be too much replication
    # just for stricter threshold


    hp_f_at_target_frequs_restricted_2 = hp_f_at_target_frequs[
        hp_f_at_target_frequs.frequencies < (f_min - hp_f_fine.df)
    ]
    hp_f_at_target_frequs_restricted_3 = hp_f_at_target_frequs[
        hp_f_at_target_frequs.frequencies > (f_max + hp_f_fine.df)
    ]
    # Otherwise interpolation might be linear between last zero sample and
    # first non-zero one, leading to values that are not zero

    assert_quantity_equal(hp_f_at_target_frequs_restricted_2, 0.0 * u.s)
    assert_quantity_equal(hp_f_at_target_frequs_restricted_3, 0.0 * u.s)


@pytest.mark.parametrize('df', [hp_f_fine.df, hp_f_fine.df / 2, hp_f_fine.df / 4, 0.007 * u.Hz, 0.001 * u.Hz])  # hp_f_coarse.df too coarse for comparison to make sense
@pytest.mark.parametrize('f_low', [1.1 * f_min])
@pytest.mark.parametrize('f_high', [0.9 * f_max])
def test_get_signal_at_target_frequs_interp_and_filling(f_low, f_high, df):
    target_frequs = np.arange(f_min.value, f_max.value , step=df.value) << u.Hz
    hp_f_at_target_frequs = get_signal_at_target_frequs(hp_f_fine, target_frequs, fill_val=0.0, fill_bounds=[f_low, f_high])

    assert_quantity_equal(hp_f_at_target_frequs.frequencies, target_frequs)


    hp_f_at_df, _ = fd_wf_gen(wf_params | {'deltaF': df})
    hp_f_at_df.override_unit(u.s)

    hp_f_at_df = hp_f_at_df[
        (hp_f_at_df.frequencies >= f_low)
        & (hp_f_at_df.frequencies <= f_high)
    ]

    hp_f_at_target_frequs_restricted_1 = hp_f_at_target_frequs[
        (hp_f_at_target_frequs.frequencies >= f_low)
        & (hp_f_at_target_frequs.frequencies <= f_high)
    ]

    assert_allclose_quantity(hp_f_at_df.f0, hp_f_at_target_frequs_restricted_1.f0, atol=df.value, rtol=0.0)
    assert_allclose_quantity(hp_f_at_df.frequencies[-1], hp_f_at_target_frequs_restricted_1.frequencies[-1], atol=df.value, rtol=0.0)

    min_size = min(hp_f_at_df.size, hp_f_at_target_frequs_restricted_1.size)

    if np.abs(hp_f_at_df.f0 - hp_f_at_target_frequs_restricted_1.f0) < 0.5 * df:
        hp_f_at_df = hp_f_at_df[:min_size]
        hp_f_at_target_frequs_restricted_1 = hp_f_at_target_frequs_restricted_1[:min_size]
    else:
        hp_f_at_df = hp_f_at_df[hp_f_at_df.size - min_size:]
        hp_f_at_target_frequs_restricted_1 = hp_f_at_target_frequs_restricted_1[hp_f_at_target_frequs_restricted_1.size - min_size:]
    

    assert_allclose_quantity(hp_f_at_df.frequencies, hp_f_at_target_frequs_restricted_1.frequencies, atol=df.value, rtol=0.0)
    assert_allclose_quantity(hp_f_at_df, hp_f_at_target_frequs_restricted_1, atol=1e-24, rtol=0.0)
    # Frequencies are slightly shifted, which means we have to allow certain
    # tolerance. rtol not suited here because we might shift away from zero
    # to finite value, causing large relative deviations


    hp_f_at_target_frequs_restricted_2 = hp_f_at_target_frequs[
        hp_f_at_target_frequs.frequencies < f_low
    ]
    hp_f_at_target_frequs_restricted_3 = hp_f_at_target_frequs[
        hp_f_at_target_frequs.frequencies > f_high
    ]
    # Otherwise interpolation might be linear between last zero sample and
    # first non-zero one, leading to values that are not zero

    assert_quantity_equal(hp_f_at_target_frequs_restricted_2, 0.0 * u.s)
    assert_quantity_equal(hp_f_at_target_frequs_restricted_3, 0.0 * u.s)


def test_restrict_f_range_copy():
    ...


def test_restrict_f_range_none_args():
    hp_f, _ = fd_wf_gen(wf_params)
    hp_f.override_unit(u.s)

    hp_f_filtered = hp_f[hp_f != 0.0 * hp_f.unit]

    # First sanity check, otherwise errors later on might not be our fault
    assert_quantity_equal(hp_f.f0, 0.0 * u.Hz)
    assert_quantity_equal(hp_f_filtered.f0, f_min)
    assert_quantity_equal(hp_f.frequencies[-1], f_max)


    hp_f_restricted = restrict_f_range(hp_f)

    assert_quantity_equal(hp_f_restricted.f0, hp_f.f0)
    assert_quantity_equal(hp_f_restricted.frequencies[-1], hp_f.frequencies[-1])


    hp_f_restricted_2 = restrict_f_range(hp_f, f_range=[None, None])

    assert_quantity_equal(hp_f_restricted_2.f0, hp_f.f0)
    assert_quantity_equal(hp_f_restricted_2.frequencies[-1], hp_f.frequencies[-1])


    hp_f_restricted_2_v2 = restrict_f_range(hp_f, f_range=[f_min, f_max])

    assert_quantity_equal(hp_f_restricted_2_v2.f0, hp_f_filtered.f0)
    assert_quantity_equal(hp_f_restricted_2_v2.frequencies[-1], hp_f.frequencies[-1])


    hp_f_restricted_3 = restrict_f_range(hp_f, fill_range=[None, None])

    assert_quantity_equal(hp_f_restricted_3.f0, hp_f.f0)
    assert_quantity_equal(hp_f_restricted_3.frequencies[-1], hp_f.frequencies[-1])


    hp_f_restricted_3_v2 = restrict_f_range(hp_f, fill_range=[f_min, f_max])
    hp_f_restricted_3_v2_filtered = hp_f_restricted_3_v2[hp_f_restricted_3_v2 != 0.0 * hp_f_restricted_3_v2.unit]

    assert_quantity_equal(hp_f_restricted_3_v2.f0, hp_f.f0)
    assert_quantity_equal(hp_f_restricted_3_v2_filtered.f0, hp_f_filtered.f0)
    assert_quantity_equal(hp_f_restricted_3_v2.frequencies[-1], hp_f.frequencies[-1])


@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df, hp_f_fine.df / 4])
@pytest.mark.parametrize('f_crop_low', [0.9 * f_min, 1.1 * f_min])
@pytest.mark.parametrize('f_crop_high', [0.9 * f_max, 1.1 * f_max])
# Last one is there to demonstrate that it is not about size of df, behaviour
# is about nature of its value (power of two or not)
def test_restrict_f_range_cropping_and_padding_exact(df, f_crop_low, f_crop_high):
    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})
    hp_f.override_unit(u.s)
    
    hp_f_restricted = restrict_f_range(hp_f, f_range=[f_crop_low, f_crop_high])
    
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes a method similar to what is done in restrict_f_range.
    # Instead, more straightforward array slicing is used

    f_lower = max(hp_f.f0, hp_f_restricted.f0)
    f_upper = min(hp_f.frequencies[-1], hp_f_restricted.frequencies[-1])

    hp_f_cropped = hp_f[(hp_f.frequencies >= f_lower)
                            & (hp_f.frequencies <= f_upper)]
    hp_f_restricted_cropped = hp_f_restricted[
        (hp_f_restricted.frequencies >= f_lower)
        & (hp_f_restricted.frequencies <= f_upper)
    ]

    if hp_f_cropped.size != hp_f_restricted_cropped.size:
        # Note: this only happens for VERY small df like 0.001 where our
        # estimates of the number of points to pad/cut off may be flawed
        # and deviate by a single sample
        assert abs(hp_f_cropped.size - hp_f_restricted_cropped.size) < 2
        
        size_min = min(hp_f_cropped.size, hp_f_restricted_cropped.size)
        hp_f_cropped = hp_f_cropped[:size_min]
        hp_f_restricted_cropped = hp_f_restricted_cropped[:size_min]

    assert_quantity_equal(hp_f_cropped, hp_f_restricted_cropped)

    assert_allclose_quantity(hp_f_restricted.f0, f_crop_low,
                             atol=0.9 * df.value, rtol=0.0)
    assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
                             atol=0.9 * df.value, rtol=0.0)
    # NOTE: we cannot demand exact equality for arbitrary limits because
    # the samples are still discrete. However, the deviation mus not be
    # larger than the sample spacing, this would mean error in our code


@pytest.mark.parametrize('df', [0.007*u.Hz, 0.001*u.Hz])
@pytest.mark.parametrize('f_crop_low', [0.9 * f_min, 1.1 * f_min])
@pytest.mark.parametrize('f_crop_high', [0.9 * f_max, 1.1 * f_max])
# Checking with one that is not power of two is important to ensure
# pad_to_target_df does good job (not necessarily related to restrict_f_range)
def test_restrict_f_range_cropping_and_padding_not_exact(df, f_crop_low, f_crop_high):
    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})
    hp_f.override_unit(u.s)
    
    hp_f_restricted = restrict_f_range(hp_f, f_range=[f_crop_low, f_crop_high])
    
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes a method similar to what is done in restrict_f_range.
    # Instead, more straightforward array slicing is used

    f_lower = max(hp_f.f0, hp_f_restricted.f0)
    f_upper = min(hp_f.frequencies[-1], hp_f_restricted.frequencies[-1])

    hp_f_cropped = hp_f[(hp_f.frequencies >= f_lower)
                            & (hp_f.frequencies <= f_upper)]
    hp_f_restricted_cropped = hp_f_restricted[
        (hp_f_restricted.frequencies >= f_lower)
        & (hp_f_restricted.frequencies <= f_upper)
    ]

    if hp_f_cropped.size != hp_f_restricted_cropped.size:
        # Note: this only happens for VERY small df like 0.001 where our
        # estimates of the number of points to pad/cut off may be flawed
        # and deviate by a single sample
        assert abs(hp_f_cropped.size - hp_f_restricted_cropped.size) < 2
        
        size_min = min(hp_f_cropped.size, hp_f_restricted_cropped.size)
        hp_f_cropped = hp_f_cropped[:size_min]
        hp_f_restricted_cropped = hp_f_restricted_cropped[:size_min]

    assert_quantity_equal(hp_f_cropped, hp_f_restricted_cropped)

    assert_allclose_quantity(hp_f_restricted.f0, f_crop_low,
                             atol=df.value, rtol=0.0)
    assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
                             atol=df.value, rtol=0.0)
    # NOTE: we cannot demand exact equality for arbitrary limits because
    # the samples are still discrete. However, the deviation mus not be
    # larger than the sample spacing, this would mean error in our code


@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df])#, 0.007*u.Hz, 0.001*u.Hz])
# Checking with one that is not power of two is important to ensure
# pad_to_target_df does good job (not necessarily related to restrict_f_range)
@pytest.mark.parametrize('f_fill_low', [0.8 * f_min, f_min, 1.2 * f_min])
@pytest.mark.parametrize('f_fill_high', [0.8 * f_max, f_max, 1.2 * f_max])
def test_restrict_f_range_filling(df, f_fill_low, f_fill_high):
    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})
    hp_f.override_unit(u.s)

    hp_f_restricted = restrict_f_range(hp_f, fill_range=[f_fill_low, f_fill_high])
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes a method similar to what is done in restrict_f_range.
    # Instead, more straightforward array slicing is used

    assert_allclose_quantity(hp_f_restricted.f0, hp_f.f0,
                             atol=df.value, rtol=0.0)
    assert_allclose_quantity(hp_f_restricted.frequencies[-1], hp_f.frequencies[-1],
                             atol=df.value, rtol=0.0)

    hp_f_cropped = hp_f[(hp_f.frequencies >= f_fill_low)
                            & (hp_f.frequencies <= f_fill_high)]
    hp_f_restricted_cropped = hp_f_restricted[
        (hp_f_restricted.frequencies >= f_fill_low)
        & (hp_f_restricted.frequencies <= f_fill_high)
    ]

    assert_quantity_equal(hp_f_cropped, hp_f_restricted_cropped)

    if f_fill_low > hp_f.f0:
        assert_allclose_quantity(hp_f_restricted_cropped.f0, f_fill_low,
                                atol=df.value, rtol=0.0)
    
    if f_fill_high < hp_f.frequencies[-1]:
        assert_allclose_quantity(hp_f_restricted_cropped.frequencies[-1], f_fill_high,
                                 atol=df.value, rtol=0.0)
        
    # In respective else case, nothing should happen to the frequency ranges
    # because there is nothing to do here (by design, no filling over range
    # [f_min f_max] is applied). We have checked this by ensuring
    # hp_f_restricted covers the same range as hp_f does


    # Also check that everything has been set to zero outside of f_range
    hp_f_restricted_cropped_2 = hp_f_restricted[
        hp_f_restricted.frequencies < f_fill_low
    ]
    hp_f_restricted_cropped_3 = hp_f_restricted[
        hp_f_restricted.frequencies > f_fill_high
    ]

    if f_fill_low > hp_f.f0:
        assert_quantity_equal(0.0 * u.s, hp_f_restricted_cropped_2)
    else:
        assert len(hp_f_restricted_cropped_2) == 0

    if f_fill_high < hp_f.frequencies[-1]:
        assert_quantity_equal(0.0 * u.s, hp_f_restricted_cropped_3)
    else:
        assert len(hp_f_restricted_cropped_3) == 0


# @pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df])#, 0.007*u.Hz, 0.001*u.Hz])
# # Checking with one that is not power of two is important to ensure
# # pad_to_target_df does good job (not necessarily related to restrict_f_range)
# @pytest.mark.parametrize('f_crop_low', [0.9 * f_min, f_min])
# @pytest.mark.parametrize('f_crop_high', [f_max, 1.1 * f_max])
# @pytest.mark.parametrize('f_fill_low', [1.1 * f_min, f_min])
# @pytest.mark.parametrize('f_fill_high', [0.9 * f_max, f_max])
# def test_restrict_f_range_arg_interplay(df, f_crop_low, f_crop_high, f_fill_low, f_fill_high):
#     hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})
#     hp_f.override_unit(u.s)

#     hp_f_restricted = restrict_f_range(hp_f,
#                                        f_range=[f_crop_low, f_crop_high],
#                                        fill_range=[f_fill_low, f_fill_high])

#     assert_allclose_quantity(hp_f_restricted.f0, f_crop_low,
#                              atol=df.value, rtol=0.0)
#     assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
#                              atol=df.value, rtol=0.0)
    
#     # NOTE: we will not use Series.crop to get the comparisons because it
#     # utilizes a method similar to what is done in restrict_f_range.
#     # Instead, more straightforward array slicing is used

#     hp_f_cropped = hp_f[(hp_f.frequencies >= f_fill_low)
#                             & (hp_f.frequencies <= f_fill_high)]
#     hp_f_restricted_cropped = hp_f_restricted[
#         (hp_f_restricted.frequencies >= f_fill_low)
#         & (hp_f_restricted.frequencies <= f_fill_high)
#     ]

#     assert_quantity_equal(hp_f_cropped, hp_f_restricted_cropped)

#     # assert_allclose_quantity(hp_f_restricted_cropped.f0, f_fill_low,
#     #                          atol=df.value, rtol=0.0)
#     # assert_allclose_quantity(hp_f_restricted_cropped.frequencies[-1], f_fill_high,
#     #                          atol=df.value, rtol=0.0)


#     # Also check that everything has been set to zero outside of f_range
#     hp_f_restricted_cropped_2 = hp_f_restricted[
#         hp_f_restricted.frequencies < f_fill_low
#     ]
#     hp_f_restricted_cropped_3 = hp_f_restricted[
#         hp_f_restricted.frequencies > f_fill_high
#     ]

#     assert_quantity_equal(0.0 * u.s, hp_f_restricted_cropped_2)
#     assert_quantity_equal(0.0 * u.s, hp_f_restricted_cropped_3)
    
#     # assert_allclose_quantity(hp_f_restricted_cropped_2.f0, f_crop_low,
#     #                          atol=df.value, rtol=0.0)
#     # assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
#     #                          atol=df.value, rtol=0.0)


# @pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df, hp_f_fine.df / 4])
# # Region where pad_to_get_to
# def test_restrict_f_range_with_padding_and_cropping_exact(df):
#     f_crop_low, f_crop_high = 20.0 * u.Hz, 30.0 * u.Hz
#     # For contiguous padding to be possible, f_crop_low has to be an integer
#     # multiple of df

#     # hp_t_padded = pad_to_get_target_df(hp_t, df)
#     # hp_t_f = td_to_fd_waveform(hp_t_padded)
#     hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})
#     hp_f = hp_f[hp_f.frequencies >= f_crop_low]  # Cut off so no start at f=0
#     hp_f_restricted = restrict_f_range(hp_f, f_range=[0.0, f_crop_high],
#                                        fill_range=[f_crop_low, None])
    
#     # NOTE: we will not use Series.crop to get the comparisons because it
#     # utilizes computations similar to what is done in restrict_f_range.
#     # Instead, more straightforward array slicing is used

#     assert_quantity_equal(hp_f_restricted.f0, 0.0 * u.Hz)
#     assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
#                              atol=0.42 * df.value, rtol=0.0)
#     # We expect accuracy smaller than 0.5 here, which should be distinguishable


# @pytest.mark.parametrize('df', [0.001*u.Hz, 0.007*u.Hz])
# def test_restrict_f_range_with_padding_and_cropping_not_exact(df):
#     f_crop_low, f_crop_high = 20.0 * u.Hz, 30.0 * u.Hz

#     hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})
#     hp_f = hp_f[hp_f.frequencies >= f_crop_low]  # Cut off so no start at f=0
#     hp_f_restricted = restrict_f_range(hp_f, f_range=[0.0, f_crop_high],
#                                        fill_range=[f_crop_low, None])
    
#     # NOTE: we will not use Series.crop to get the comparisons because it
#     # utilizes computations similar to what is done in restrict_f_range.
#     # Instead, more straightforward array slicing is used

#     assert_quantity_equal(hp_f_restricted.f0, 0.0 * u.Hz)
#     assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
#                              atol=df.value, rtol=0.0)
#     # More tolerance needed here since using the more accurate slicing
#     # method used here is too expensive for use in restrict_f_range. This
#     # comes at the price of certain smaller deviations for some df


# TODO: test for various cases if copy works; e.g. if copy=False, but nothing
# is to be filled, should not change original array


@pytest.mark.parametrize('fill_val', [0., 2.])
def test_fill_f_range(fill_val):
    hf = hp_f_fine.copy()

    fill_f_range(hf, fill_val, [-f_min, 1.1*f_max])

    assert_allequal_series(hp_f_fine, hf)

    f_lower, f_upper = 30.*u.Hz, 50.*u.Hz
    fill_f_range(hf, fill_val, [f_lower, f_upper])

    filter1 = hp_f_fine.frequencies < f_lower
    filter2 = hp_f_fine.frequencies > f_upper
    filter3 = np.logical_and(np.logical_not(filter1), np.logical_not(filter2))

    fill_val = u.Quantity(fill_val, hf.unit)  # For assertions
    assert_quantity_equal(hf[filter1], fill_val)
    assert_quantity_equal(hf[filter2], fill_val)
    assert_quantity_equal(hf[filter3], hp_f_fine[filter3])



 #%% -- Testing get_strain function -------------------------------------------
# Goal is essentially just to make sure code works
def test_get_strain_no_extrinsic():
    # Not sure we can capture this in parametrize, problem is how to
    # automatically get array we want to compare with
    hp_t_test = get_strain(wf_params, 'time', generator=gen, mode='plus')
    hc_t_test = get_strain(wf_params, 'time', generator=gen, mode='cross')
    h_t_test = get_strain(wf_params, 'time', generator=gen, mode='mixed')

    assert_quantity_equal(hp_t, hp_t_test)
    assert_quantity_equal(hc_t, hc_t_test)
    assert_quantity_equal(hp_t + 1.j * hc_t, h_t_test)

    hp_f_test = get_strain(wf_params, 'frequency', generator=gen, mode='plus')
    hc_f_test = get_strain(wf_params, 'frequency', generator=gen, mode='cross')
    h_f_test = get_strain(wf_params, 'frequency', generator=gen, mode='mixed')
    
    assert_quantity_equal(hp_f_fine, hp_f_test)
    assert_quantity_equal(hc_f_fine, hc_f_test)

    h_f_fine = np.flip((np.conjugate(hp_f_fine) + 1.j * np.conjugate(hc_f_fine))[1:])
    h_f_fine.df = hp_f_fine.df
    h_f_fine.frequencies -= h_f_fine.frequencies[-1] + h_f_fine.df
    h_f_fine = h_f_fine.append(hp_f_fine + 1.j * hc_f_fine, inplace=False)

    assert_quantity_equal(h_f_fine, h_f_test)


def test_get_strain_extrinsic():
    ext_params = {'det': 'H1', 'ra': 0.2*u.rad, 'dec': 0.2*u.rad,
                  'psi': 0.5*u.rad, 'tgps': 1126259462}
    
    from lalsimulation.gwsignal.core.gw import GravitationalWavePolarizations

    params = wf_params | ext_params
    ht_test = get_strain(params, 'time', generator=gen)
    ht_test = get_strain(params, 'time', generator=gen)
    assert_quantity_equal(
        GravitationalWavePolarizations(hp_t, hc_t).strain(**ext_params),
        ht_test
    )

    hf_test = get_strain(wf_params | ext_params, 'frequency', generator=gen)
    assert_quantity_equal(
        GravitationalWavePolarizations(hp_f_fine, hc_f_fine).strain(**ext_params),
        hf_test
    )
test_get_strain_extrinsic()
class ErrorRaising(unittest.TestCase):
    def test_domain_checking(self):
        with self.assertRaises(ValueError):
            get_strain(wf_params, 'domain', generator=gen)

    def test_mode_checking(self):
        with self.assertRaises(ValueError):
            get_strain(wf_params, 'time', generator=gen, mode='mode')
    
    def test_extr_params_checking(self):
        with self.assertRaises(ValueError):
            get_strain(wf_params | {'psi': 0.5*u.rad}, 'time', generator=gen)


#%% -- Testing mass rescaling -------------------------------------------------

# TODO: get this to work

# wf_params_with_total_mass = wf_params.copy()
# wf_params_with_total_mass.pop('mass1')
# wf_params_with_total_mass.pop('mass2')

# total_mass = 100.*u.solMass
# wf_params_with_total_mass['total_mass'] = total_mass
# wf_params_with_total_mass['mass_ratio'] = 0.5 * u.dimensionless_unscaled


# @pytest.mark.parametrize('target_unit_sys', ['SI', 'cosmo', 'geom'])
# def test_scaling_fd(target_unit_sys):
#     mass1 = total_mass
#     mass2 = 0.5 * total_mass
#     mass3 = 0.25 * total_mass

#     # hp_f_M1, _ = fd_wf_gen(wf_params_with_total_mass | {'total_mass': mass1})
#     # hp_f_M2, _ = fd_wf_gen(wf_params_with_total_mass | {'total_mass': mass2})
#     # hp_f_M3, _ = fd_wf_gen(wf_params_with_total_mass | {'total_mass': mass3})

#     # hp_f_M1 = rescale_with_Mtotal(hp_f_M1, mass1, target_unit_sys)
#     # hp_f_M2 = rescale_with_Mtotal(hp_f_M2, mass2, target_unit_sys)
#     # hp_f_M3 = rescale_with_Mtotal(hp_f_M3, mass3, target_unit_sys)

#     hp_f_M1 = get_mass_scaled_wf(wf_params_with_total_mass | {'total_mass': mass1, 'deltaF': 2**-8 * u.Hz}, 'FD', gen, target_unit_sys)
#     hp_f_M2 = get_mass_scaled_wf(wf_params_with_total_mass | {'total_mass': mass2, 'deltaF': 2**-8 * u.Hz}, 'FD', gen, target_unit_sys)
#     hp_f_M3 = get_mass_scaled_wf(wf_params_with_total_mass | {'total_mass': mass3, 'deltaF': 2**-8 * u.Hz}, 'FD', gen, target_unit_sys)

#     # df_interpolate = 2**-6
#     # hp_f_M1 = hp_f_M1.interpolate(df_interpolate)
#     # hp_f_M2 = hp_f_M2.interpolate(df_interpolate)
#     # hp_f_M3 = hp_f_M3.interpolate(df_interpolate)

#     f_min = max(hp_f_M1.frequencies[0], hp_f_M2.frequencies[0], hp_f_M3.frequencies[0])
#     f_max = min(hp_f_M1.frequencies[-1], hp_f_M2.frequencies[-1], hp_f_M3.frequencies[-1])

#     hp_f_M1 = hp_f_M1.crop(start=f_min, end=f_max)
#     hp_f_M2 = hp_f_M2.crop(start=f_min, end=f_max)
#     hp_f_M3 = hp_f_M3.crop(start=f_min, end=f_max)
#     # hp_f_M1 = restrict_f_range(hp_f_M1, f_range=[f_min, f_max])
#     # hp_f_M2 = restrict_f_range(hp_f_M2, f_range=[f_min, f_max])
#     # hp_f_M3 = restrict_f_range(hp_f_M3, f_range=[f_min, f_max])

#     # Maybe rather use restrict_f_range?

#     assert_allclose(hp_f_M1, hp_f_M2, atol=0.0, rtol=0.01)
#     assert_allclose(hp_f_M2, hp_f_M3, atol=0.0, rtol=0.01)


# @pytest.mark.parametrize('target_unit_sys', ['SI', 'cosmo', 'geom'])
# def test_scaling_td(target_unit_sys):
#     import astropy.constants as const  # TODO: check if import outside of function. Then also define constants outside
#     mass1 = total_mass
#     mass2 = 0.5 * total_mass
#     mass3 = 0.25 * total_mass

#     # hp_f_M1, _ = fd_wf_gen(wf_params_with_total_mass | {'total_mass': mass1}, gen)
#     # hp_f_M2, _ = fd_wf_gen(wf_params_with_total_mass | {'total_mass': mass2})
#     # hp_f_M3, _ = fd_wf_gen(wf_params_with_total_mass | {'total_mass': mass3})

#     # hp_f_M1 = rescale_with_Mtotal(hp_f_M1, mass1, target_unit_sys)
#     # hp_f_M2 = rescale_with_Mtotal(hp_f_M2, mass2, target_unit_sys)
#     # hp_f_M3 = rescale_with_Mtotal(hp_f_M3, mass3, target_unit_sys)

#     deltaT = 2**-4 * 1./4096.*u.s

#     hp_t_M1 = get_mass_scaled_wf(wf_params_with_total_mass | {'total_mass': mass1, 'deltaT': deltaT * mass1.value}, 'TD', gen, target_unit_sys)
#     hp_t_M2 = get_mass_scaled_wf(wf_params_with_total_mass | {'total_mass': mass2, 'deltaT': deltaT * mass2.value}, 'TD', gen, target_unit_sys)
#     hp_t_M3 = get_mass_scaled_wf(wf_params_with_total_mass | {'total_mass': mass3, 'deltaT': deltaT * mass3.value}, 'TD', gen, target_unit_sys)

#     # df_interpolate = 2**-6
#     # hp_f_M1 = hp_f_M1.interpolate(df_interpolate)
#     # hp_f_M2 = hp_f_M2.interpolate(df_interpolate)
#     # hp_f_M3 = hp_f_M3.interpolate(df_interpolate)

#     # t_min = max(hp_t_M1.times[0], hp_t_M2.times[0], hp_t_M3.times[0])
#     # t_max = min(hp_t_M1.times[-1], hp_t_M2.times[-1], hp_t_M3.times[-1])

#     Msun_to_kg = const.M_sun / u.Msun
#     kg_to_s = const.G / const.c**3
    
#     if target_unit_sys == 'cosmo':
#         t_min, t_max = -0.01, 0.0001
#     elif target_unit_sys == 'SI':
#         t_min, t_max = -0.01 / Msun_to_kg.value, 0.0001 / Msun_to_kg.value
#     elif target_unit_sys == 'geom':
#         t_min, t_max = -0.01 / (Msun_to_kg * kg_to_s).value, 0.0001 / (Msun_to_kg * kg_to_s).value

#     hp_t_M1 = hp_t_M1.crop(start=t_min, end=t_max)
#     hp_t_M2 = hp_t_M2.crop(start=t_min, end=t_max)
#     hp_t_M3 = hp_t_M3.crop(start=t_min, end=t_max)

#     assert_allclose(hp_t_M1.value, hp_t_M2.value, atol=0.0, rtol=0.1)
#     assert_allclose(hp_t_M2.value, hp_t_M3.value, atol=0.0, rtol=0.1)


# def test_conversion_si_units():
#     hp_f, _ = fd_wf_gen(wf_params_with_total_mass)

#     hp_f_rescaled = rescale_with_Mtotal(
#         hp_f,
#         wf_params_with_total_mass['total_mass'],
#         target_unit_sys='si'
#     )

#     hp_f_v2 = scale_to_Mtotal(
#         hp_f_rescaled,
#         wf_params_with_total_mass['total_mass'],
#         unit_sys='si'
#     )


#     assert_allclose(hp_f, hp_f_v2, atol=0.0, rtol=0.01)


# def test_conversion_geom_units():
#     hp_f, _ = fd_wf_gen(wf_params_with_total_mass)

#     hp_f_rescaled = rescale_with_Mtotal(
#         hp_f,
#         wf_params_with_total_mass['total_mass'],
#         target_unit_sys='geom'
#     )

#     hp_f_v2 = scale_to_Mtotal(
#         hp_f_rescaled,
#         wf_params_with_total_mass['total_mass'],
#         unit_sys='geom'
#     )


#     assert_allclose(hp_f, hp_f_v2, atol=0.0, rtol=0.01)


# def test_conversion_cosmo_units():
#     hp_f, _ = fd_wf_gen(wf_params_with_total_mass)

#     hp_f_rescaled = rescale_with_Mtotal(
#         hp_f,
#         wf_params_with_total_mass['total_mass'],
#         target_unit_sys='cosmo'
#     )

#     hp_f_v2 = scale_to_Mtotal(
#         hp_f_rescaled,
#         wf_params_with_total_mass['total_mass'],
#         unit_sys='si'
#     )


#     assert_allclose(hp_f, hp_f_v2, atol=0.0, rtol=0.01)
# # %%
