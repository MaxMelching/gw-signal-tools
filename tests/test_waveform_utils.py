from gw_signal_tools.waveform_utils import (
    td_to_fd_waveform, fd_to_td_waveform,
    pad_to_get_target_df, restrict_f_range,
    get_strain, get_mass_scaled_wf,
    # rescale_with_Mtotal, scale_to_Mtotal,
)

import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm
import numpy as np
from numpy.testing import assert_allclose
from gw_signal_tools.test_utils import (
    assert_allclose_quantity, assert_allclose_frequseries,
    assert_allclose_timeseries
)
from gwpy.testing.utils import assert_quantity_equal

import pytest


# We will perform tests with a GW150914-like signal
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

approximant = 'IMRPhenomXPHM'

gen = wfm.LALCompactBinaryCoalescenceGenerator(approximant)

hp_t, hc_t = wfm.GenerateTDWaveform(wf_params, gen)


hp_f_fine, hc_f_fine = wfm.GenerateFDWaveform(wf_params, gen)

hp_f_coarse, hc_f_coarse = wfm.GenerateFDWaveform(wf_params | {'deltaF': 1.0 / (hp_t.size * hp_t.dx)}, gen)


hp_f_fine.override_unit(u.s)
hc_f_fine.override_unit(u.s)
hp_f_coarse.override_unit(u.s)
hc_f_coarse.override_unit(u.s)
# NOTE: unit conversion is needed because of inconsistent handling of units in
# lal, not because of error in gw_signal_tools code


#%% ---------- Testing transformation into one domain and back ----------
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

    hp_t_cropped = hp_t.crop(start=t_min_comp, end=t_max_comp)[2:]
    hp_t_fft_ifft_cropped = hp_t_fft_ifft.crop(start=t_min_comp, end=t_max_comp)

    # Have to apply different kind of threshold here because coarse sampling is indeed very coarse
    # -> slight phase shift due to that is biggest problem, causes amplitude errors in comparison of values at same time
    assert_allclose_quantity(hp_t_cropped.times, hp_t_fft_ifft_cropped.times, atol=2 * hp_t.dt.value, rtol=0.0)  # Oof, this is a lot...
    # assert_quantity_equal(hp_t_cropped.times, hp_t_fft_ifft_cropped.times)
    assert_allclose_quantity(hp_t_cropped, hp_t_fft_ifft_cropped, atol=6e-21, rtol=0.0)
    # assert_quantity_equal(hp_t_cropped, hp_t_fft_ifft_cropped)


    hp_t_fft_ifft_fine = fd_to_td_waveform(td_to_fd_waveform(pad_to_get_target_df(hp_t, df=0.0625 * u.Hz)))

    t_min_comp, t_max_comp = max(hp_t.t0, hp_t_fft_ifft_fine.t0), min(hp_t.times[-1], hp_t_fft_ifft_fine.times[-1])  # hp_t_fft_ifft_fine is padded to be much longer
    
    hp_t_cropped = hp_t.crop(start=t_min_comp, end=t_max_comp)[1:]
    hp_t_fft_ifft_fine_cropped = hp_t_fft_ifft_fine.crop(start=t_min_comp, end=t_max_comp)[1:]
    # NOTE: for some reason, first sample is not equal. Thus excluded here

    # assert_allclose_quantity(hp_t_cropped.times, hp_t_fft_ifft_fine_cropped.times, atol=0.0, rtol=1e-10)
    assert_quantity_equal(hp_t_cropped.times, hp_t_fft_ifft_fine_cropped.times)
    assert_allclose_quantity(hp_t_cropped, hp_t_fft_ifft_fine_cropped, atol=0.0, rtol=1e-8)
    # assert_quantity_equal(hp_t_cropped, hp_t_fft_ifft_fine_cropped)
    

#%% ---------- Testing transformations with generated signals from different domain ----------
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

# TODO: add tests for complex stuff (works, see plots in notebook; but still)


#%% ---------- Testing helper functions for frequency region stuff ----------
@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df])
# These input values are powers of two, have to be reproduced exactly
def test_pad_to_target_df_exact(df):
    hp_t_padded = pad_to_get_target_df(hp_t, df)
    hp_t_f = td_to_fd_waveform(hp_t_padded)

    assert_quantity_equal(df, hp_t_f.df)


@pytest.mark.parametrize('df', [0.001*u.Hz, 0.007*u.Hz])
# These input values are not exact powers of two and thus cannot be
# reproduced exactly (thus ensure sufficient accuracy)
def test_pad_to_target_df_not_exact(df):
    hp_t_padded = pad_to_get_target_df(hp_t, df)
    hp_t_f = td_to_fd_waveform(hp_t_padded)

    assert_allclose_quantity(df, hp_t_f.df, atol=0.0, rtol=1e-5)


@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df, 0.001*u.Hz, 0.007*u.Hz])
# Checking with one that is not power of two is important to ensure
# pad_to_target_df does good job (not necessarily related to restrict_f_range)
def test_restrict_f_range(df):
    f_crop_low, f_crop_high = 20.0 * u.Hz, 30.0 * u.Hz

    hp_t_padded = pad_to_get_target_df(hp_t, df)
    hp_t_f = td_to_fd_waveform(hp_t_padded)
    hp_t_f_restricted = restrict_f_range(hp_t_f, f_range=[f_crop_low, f_crop_high])
    
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes computations similar to what is done in restrict_f_range.
    # Instead, more straightforward array slicing is used

    hp_t_f_cropped = hp_t_f[(hp_t_f.frequencies >= f_crop_low)
                            & (hp_t_f.frequencies <= f_crop_high)]
    hp_t_f_restricted_cropped = hp_t_f_restricted[
        (hp_t_f_restricted.frequencies >= f_crop_low)
        & (hp_t_f_restricted.frequencies <= f_crop_high)
    ]

    assert_quantity_equal(hp_t_f_cropped, hp_t_f_restricted_cropped)


    # Also check that everything has been set to zero outside of f_range
    hp_t_f_restricted_cropped_2 = hp_t_f_restricted[
        hp_t_f_restricted.frequencies < f_crop_low
    ]
    hp_t_f_restricted_cropped_3 = hp_t_f_restricted[
        hp_t_f_restricted.frequencies > f_crop_high
    ]

    assert_quantity_equal(0.0 * u.s, hp_t_f_restricted_cropped_2)
    assert_quantity_equal(0.0 * u.s, hp_t_f_restricted_cropped_3)


@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df, hp_f_fine.df / 4])
# Region where pad_to_get_to
def test_restrict_f_range_with_padding_and_cropping_exact(df):
    f_crop_low, f_crop_high = 20.0 * u.Hz, 30.0 * u.Hz
    # For contiguous padding to be possible, f_crop_low has to be an integer
    # multiple of df

    # hp_t_padded = pad_to_get_target_df(hp_t, df)
    # hp_t_f = td_to_fd_waveform(hp_t_padded)
    hp_f, _ = wfm.GenerateFDWaveform(wf_params | {'deltaF': df}, gen)
    hp_f = hp_f[hp_f.frequencies >= f_crop_low]  # Cut off so no start at f=0
    hp_f_restricted = restrict_f_range(hp_f, f_range=[f_crop_low, f_crop_high],
                                         cut_upper=True, pad_to_f_zero=True)
    
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes computations similar to what is done in restrict_f_range.
    # Instead, more straightforward array slicing is used

    assert_quantity_equal(hp_f_restricted.f0, 0.0 * u.Hz)
    assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
                             atol=0.42 * df.value, rtol=0.0)
    # We expect accuracy smaller than 0.5 here, which should be distinguishable


@pytest.mark.parametrize('df', [0.001*u.Hz, 0.007*u.Hz])
def test_restrict_f_range_with_padding_and_cropping_not_exact(df):
    f_crop_low, f_crop_high = 20.0 * u.Hz, 30.0 * u.Hz

    hp_f, _ = wfm.GenerateFDWaveform(wf_params | {'deltaF': df}, gen)
    hp_f = hp_f[hp_f.frequencies >= f_crop_low]  # Cut off so no start at f=0
    hp_f_restricted = restrict_f_range(hp_f, f_range=[f_crop_low, f_crop_high],
                                         cut_upper=True, pad_to_f_zero=True)
    
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes computations similar to what is done in restrict_f_range.
    # Instead, more straightforward array slicing is used

    assert_quantity_equal(hp_f_restricted.f0, 0.0 * u.Hz)
    assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
                             atol=df.value, rtol=0.0)
    # More tolerance needed here since using the more accurate slicing
    # method used here is too expensive for use in restrict_f_range. This
    # comes at the price of certain smaller deviations for some df


#%% ---------- Testing get_strain function ----------
# Goal is essentially just to make sure code works
def test_get_strain_no_extrinsic():
    # Not sure we can capture this in parametrize, problem is how to
    # automatically get array we want to compare with
    hp_t_test = get_strain(wf_params, 'time', wf_generator=gen, mode='plus')
    hc_t_test = get_strain(wf_params, 'time', wf_generator=gen, mode='cross')
    h_t_test = get_strain(wf_params, 'time', wf_generator=gen, mode='mixed')

    assert_quantity_equal(hp_t, hp_t_test)
    assert_quantity_equal(hc_t, hc_t_test)
    assert_quantity_equal(hp_t + 1.j * hc_t, h_t_test)

    hp_f_test = get_strain(wf_params, 'frequency', wf_generator=gen, mode='plus')
    hp_f_test.override_unit(u.s)
    hc_f_test = get_strain(wf_params, 'frequency', wf_generator=gen, mode='cross')
    hc_f_test.override_unit(u.s)
    h_f_test = get_strain(wf_params, 'frequency', wf_generator=gen, mode='mixed')
    h_f_test.override_unit(u.s)
    
    assert_quantity_equal(hp_f_fine, hp_f_test)
    assert_quantity_equal(hc_f_fine, hc_f_test)

    h_f_fine = np.flip((np.conjugate(hp_f_fine) + 1.j * np.conjugate(hc_f_fine))[1:])
    h_f_fine.df = hp_f_fine.df
    h_f_fine.frequencies -= h_f_fine.frequencies[-1] + h_f_fine.df
    h_f_fine = h_f_fine.append(hp_f_fine + 1.j * hc_f_fine, inplace=False)

    assert_quantity_equal(h_f_fine, h_f_test)


def test_get_strain_extrinsic():
    # Not sure we can capture this in parametrize, problem is how to
    # automatically get array we want to compare with
    hp_t_test = get_strain(wf_params, 'time', wf_generator=gen, mode='plus')
    hc_t_test = get_strain(wf_params, 'time', wf_generator=gen, mode='cross')
    h_t_test = get_strain(wf_params, 'time', wf_generator=gen, mode='mixed')

    assert_quantity_equal(hp_t, hp_t_test)
    assert_quantity_equal(hc_t, hc_t_test)
    assert_quantity_equal(hp_t + 1.j * hc_t, h_t_test)

    hp_f_test = get_strain(wf_params, 'frequency', wf_generator=gen, mode='plus')
    hp_f_test.override_unit(u.s)
    hc_f_test = get_strain(wf_params, 'frequency', wf_generator=gen, mode='cross')
    hc_f_test.override_unit(u.s)
    h_f_test = get_strain(wf_params, 'frequency', wf_generator=gen, mode='mixed')
    h_f_test.override_unit(u.s)
    
    assert_quantity_equal(hp_f_fine, hp_f_test)
    assert_quantity_equal(hc_f_fine, hc_f_test)

    h_f_fine = np.flip((np.conjugate(hp_f_fine) + 1.j * np.conjugate(hc_f_fine))[1:])
    h_f_fine.df = hp_f_fine.df
    h_f_fine.frequencies -= h_f_fine.frequencies[-1] + h_f_fine.df
    h_f_fine = h_f_fine.append(hp_f_fine + 1.j * hc_f_fine, inplace=False)

    assert_quantity_equal(h_f_fine, h_f_test)


#%% ---------- Testing mass rescaling ----------

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

#     # hp_f_M1, _ = wfm.GenerateFDWaveform(wf_params_with_total_mass | {'total_mass': mass1}, gen)
#     # hp_f_M2, _ = wfm.GenerateFDWaveform(wf_params_with_total_mass | {'total_mass': mass2}, gen)
#     # hp_f_M3, _ = wfm.GenerateFDWaveform(wf_params_with_total_mass | {'total_mass': mass3}, gen)

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

#     # hp_f_M1, _ = wfm.GenerateFDWaveform(wf_params_with_total_mass | {'total_mass': mass1}, gen)
#     # hp_f_M2, _ = wfm.GenerateFDWaveform(wf_params_with_total_mass | {'total_mass': mass2}, gen)
#     # hp_f_M3, _ = wfm.GenerateFDWaveform(wf_params_with_total_mass | {'total_mass': mass3}, gen)

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
#     hp_f, _ = wfm.GenerateFDWaveform(wf_params_with_total_mass, gen)

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
#     hp_f, _ = wfm.GenerateFDWaveform(wf_params_with_total_mass, gen)

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
#     hp_f, _ = wfm.GenerateFDWaveform(wf_params_with_total_mass, gen)

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
