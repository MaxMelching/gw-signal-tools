# -- Standard Lib Imports
import unittest

# -- Third Party Imports
import numpy as np
import astropy.units as u
from lalsimulation.gwsignal import gwsignal_get_waveform_generator
import lalsimulation.gwsignal.core.waveform as wfm
from gwpy.testing.utils import assert_quantity_equal
from gwpy.types import Series
import pytest

# -- Local Package Imports
from gw_signal_tools.waveform.utils import (
    pad_to_dx, adjust_x_range,
    signal_at_dx, signal_at_xindex,
    get_strain, fill_x_range, get_wf_generator
)
from gw_signal_tools.waveform.ft import td_to_fd
from gw_signal_tools.test_utils import (
    assert_allclose_quantity, assert_allequal_series
)
from gw_signal_tools import enable_caching_locally, disable_caching_locally
from gw_signal_tools.types import HashableDict


#%% -- Initializing commonly used variables -----------------------------------
f_min = 20.*u.Hz
f_max = 1024.*u.Hz

wf_params = HashableDict({
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
})

# -- Make sure mass1 and mass2 are not in default_dict
import lalsimulation.gwsignal.core.parameter_conventions as pc
pc.default_dict.pop('mass1', None);
pc.default_dict.pop('mass2', None);

gen = gwsignal_get_waveform_generator('IMRPhenomXPHM')

from gw_signal_tools.waveform.utils import _CORRECT_H_UNIT_TIME, _CORRECT_H_UNIT_FREQU

with enable_caching_locally():
# with disable_caching_locally():
    from gw_signal_tools.caching import cache_func

    @cache_func
    def td_wf_gen(wf_params):
        hp, hc = wfm.GenerateTDWaveform(wf_params, gen)
        return hp*_CORRECT_H_UNIT_TIME, hc*_CORRECT_H_UNIT_TIME

    @cache_func
    def fd_wf_gen(wf_params):
        hp, hc = wfm.GenerateFDWaveform(wf_params, gen)
        return hp*_CORRECT_H_UNIT_FREQU, hc*_CORRECT_H_UNIT_FREQU
# -- NOTE: unit conversion is needed because of inconsistent handling of
# -- units in lal, not because of error in gw_signal_tools code


hp_t, hc_t = td_wf_gen(wf_params)
hp_f_fine, hc_f_fine = fd_wf_gen(wf_params)
hp_f_coarse, hc_f_coarse = fd_wf_gen(wf_params | {'deltaF': 1.0 / (hp_t.size * hp_t.dx)})


#%% -- Testing helper functions for frequency region stuff --------------------
@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df])
# -- These input values are powers of two, have to be reproduced exactly
def test_pad_to_dx_exact(df):
    hp_t_padded = pad_to_dx(hp_t, df)
    hp_t_f = td_to_fd(hp_t_padded)

    assert_quantity_equal(df, hp_t_f.df)


@pytest.mark.parametrize('df', [0.007*u.Hz, 0.001*u.Hz])
# -- These input values are not exact powers of two and thus cannot be
# -- reproduced exactly (thus ensure sufficient accuracy)
def test_pad_to_dx_not_exact(df):
    hp_t_padded = pad_to_dx(hp_t, df)
    hp_t_f = td_to_fd(hp_t_padded)

    assert df >= hp_t_f.df  # If not equal, must not be coarser
    assert_allclose_quantity(df, hp_t_f.df, atol=0.0, rtol=1e-5)


def test_pad_to_dx_too_large():
    df = 2.0*u.Hz
    # -- Above sampling frequency of signal, so padding is not
    # -- supposed to do anything
    hp_t_padded = pad_to_dx(hp_t, df)
    hp_t_f = td_to_fd(hp_t_padded)
    expected_df = 1.0 / (hp_t.dt * hp_t.size)

    assert_allequal_series(hp_t_padded, hp_t)

    assert_quantity_equal(expected_df, hp_t_f.df)


@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df, 0.007*u.Hz, 0.001*u.Hz])
@pytest.mark.parametrize('full_metadata', [False, True])
def test_signal_at_dx_exact(df, full_metadata):
    hp_f_interp = signal_at_dx(hp_f_fine, df, full_metadata=full_metadata)

    assert_quantity_equal(df, hp_f_interp.df)


@pytest.mark.parametrize('xindex', [[0, 0.1], [0, 0.125]])  # Test with and without interpolation
def test_signal_at_xindex_full_metadata(xindex):
    s1 = Series([0, 1], xindex=[0, 0.1], name='s1', channel='test', epoch=2*u.s)

    s2 = signal_at_xindex(s1, xindex)
    assert s2.name is None
    assert s2.channel is None
    assert s2.epoch is None

    s3 = signal_at_xindex(s1, xindex, full_metadata=True)
    assert s3.name == s1.name
    assert s3.channel == s1.channel
    assert s3.epoch == s1.epoch


@pytest.mark.parametrize('df', [hp_f_fine.df, hp_f_fine.df / 2, hp_f_fine.df / 4, 0.007*u.Hz, 0.001*u.Hz])  # hp_f_coarse.df too coarse for comparison to make sense
@pytest.mark.parametrize('f_low', [0.9 * f_min, f_min])
@pytest.mark.parametrize('f_high', [f_max, 1.1 * f_max])
def test_signal_at_xindex_interp_and_padding(f_low, f_high, df):
    target_frequs = np.arange(f_low.value, f_high.value , step=df.value) << u.Hz
    hp_f_at_target_frequs = signal_at_xindex(hp_f_fine, target_frequs, fill_val=0.0)

    assert_quantity_equal(hp_f_at_target_frequs.frequencies, target_frequs)


    hp_f_at_df, _ = fd_wf_gen(wf_params | {'deltaF': df})

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

    assert_quantity_equal(hp_f_at_target_frequs_restricted_2, 0.0 * _CORRECT_H_UNIT_FREQU)
    assert_quantity_equal(hp_f_at_target_frequs_restricted_3, 0.0 * _CORRECT_H_UNIT_FREQU)


@pytest.mark.parametrize('df', [hp_f_fine.df, hp_f_fine.df / 2, hp_f_fine.df / 4, 0.007 * u.Hz, 0.001 * u.Hz])  # hp_f_coarse.df too coarse for comparison to make sense
@pytest.mark.parametrize('f_low', [1.1 * f_min])
@pytest.mark.parametrize('f_high', [0.9 * f_max])
def test_signal_at_xindex_interp_and_filling(f_low, f_high, df):
    target_frequs = np.arange(f_min.value, f_max.value , step=df.value) << u.Hz
    hp_f_at_target_frequs = signal_at_xindex(hp_f_fine, target_frequs, fill_val=0.0, fill_bounds=[f_low, f_high])

    assert_quantity_equal(hp_f_at_target_frequs.frequencies, target_frequs)


    hp_f_at_df, _ = fd_wf_gen(wf_params | {'deltaF': df})

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

    assert_quantity_equal(hp_f_at_target_frequs_restricted_2, 0.0 * _CORRECT_H_UNIT_FREQU)
    assert_quantity_equal(hp_f_at_target_frequs_restricted_3, 0.0 * _CORRECT_H_UNIT_FREQU)


def test_adjust_x_range_copy():
    hf = type(hp_f_fine)(
        np.ones(hp_f_fine.size),
        xindex=hp_f_fine.frequencies.copy(),
        unit=hp_f_fine.unit,
    )
    hf_backup = hf.copy()

    # -- First test: if copy=True, filling must not edit original array
    # -- (if copy=False, the default, this is not the case, second test)
    hf_2 = adjust_x_range(hf, x_range=None, fill_range=[f_min, None], fill_val=42, copy=True)
    assert_allequal_series(hf, hf_backup)
    hf_2 = adjust_x_range(hf, x_range=None, fill_range=[f_min, None], fill_val=42, copy=False)
    assert_allequal_series(hf.crop(start=f_min), hf_2.crop(start=f_min))

    # -- If we pad something in beginning, a copy is already made, so
    # -- filling (even with False) should not edit original array
    hf = hf_backup.copy()
    hf_2 = adjust_x_range(hf, x_range=[-f_min, None], fill_range=[f_min, None], fill_val=42, copy=False)
    assert_allequal_series(hf, hf_backup)

    # -- If we fill, but on same frequency that padding started, no
    # -- inplace editing must take place
    hf = hf_backup.copy()
    hf_2 = adjust_x_range(hf, x_range=[-f_min, None], fill_range=[-f_min, None], fill_val=42, copy=False)
    assert_allequal_series(hf, hf_backup)

    # -- Even for no filling, a copy is made in certain circumstances,
    # -- e.g. filling.
    hf = hf_backup.copy()
    hf_2 = adjust_x_range(hf, x_range=[-f_min, None], fill_range=None, fill_val=42, copy=False)
    hf_2[100:] = np.full_like(100, 42) * hf_2.unit
    assert_allequal_series(hf, hf_backup)

    # -- Make sure copy is always made when true, not just when required
    # -- due to application of filling (because users will rely on that
    # -- when passing copy=True)
    hf = hf_backup.copy()
    # hf_2 = adjust_x_range(hf, x_range=None, fill_range=None, copy=True)
    hf_2 = adjust_x_range(hf, x_range=[f_min, None], fill_range=[f_min, None], fill_val=42, copy=True)
    hf_2[200:420] = np.full_like(-100, 42) * hf_2.unit
    assert_allequal_series(hf, hf_backup)


def test_adjust_x_range_none_args():
    hp_f, _ = fd_wf_gen(wf_params)

    hp_f_filtered = hp_f[hp_f != 0.0 * hp_f.unit]

    # First sanity check, otherwise errors later on might not be our fault
    assert_quantity_equal(hp_f.f0, 0.0 * u.Hz)
    assert_quantity_equal(hp_f_filtered.f0, f_min)
    assert_quantity_equal(hp_f.frequencies[-1], f_max)


    hp_f_restricted = adjust_x_range(hp_f)

    assert_quantity_equal(hp_f_restricted.f0, hp_f.f0)
    assert_quantity_equal(hp_f_restricted.frequencies[-1], hp_f.frequencies[-1])


    hp_f_restricted_2 = adjust_x_range(hp_f, x_range=[None, None])

    assert_quantity_equal(hp_f_restricted_2.f0, hp_f.f0)
    assert_quantity_equal(hp_f_restricted_2.frequencies[-1], hp_f.frequencies[-1])


    hp_f_restricted_2_v2 = adjust_x_range(hp_f, x_range=[f_min, f_max])

    assert_quantity_equal(hp_f_restricted_2_v2.f0, hp_f_filtered.f0)
    assert_quantity_equal(hp_f_restricted_2_v2.frequencies[-1], hp_f.frequencies[-1])


    hp_f_restricted_3 = adjust_x_range(hp_f, fill_range=[None, None])

    assert_quantity_equal(hp_f_restricted_3.f0, hp_f.f0)
    assert_quantity_equal(hp_f_restricted_3.frequencies[-1], hp_f.frequencies[-1])


    hp_f_restricted_3_v2 = adjust_x_range(hp_f, fill_range=[f_min, f_max])
    hp_f_restricted_3_v2_filtered = hp_f_restricted_3_v2[hp_f_restricted_3_v2 != 0.0 * hp_f_restricted_3_v2.unit]

    assert_quantity_equal(hp_f_restricted_3_v2.f0, hp_f.f0)
    assert_quantity_equal(hp_f_restricted_3_v2_filtered.f0, hp_f_filtered.f0)
    assert_quantity_equal(hp_f_restricted_3_v2.frequencies[-1], hp_f.frequencies[-1])


@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df, hp_f_fine.df / 4])
@pytest.mark.parametrize('f_crop_low', [0.9 * f_min, 1.1 * f_min])
@pytest.mark.parametrize('f_crop_high', [0.9 * f_max, 1.1 * f_max])
# Last one is there to demonstrate that it is not about size of df, behaviour
# is about nature of its value (power of two or not)
def test_adjust_x_range_cropping_and_padding_exact(df, f_crop_low, f_crop_high):
    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})
    
    hp_f_restricted = adjust_x_range(hp_f, x_range=[f_crop_low, f_crop_high])
    
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes a method similar to what is done in adjust_x_range.
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
# pad_to_dx does good job (not necessarily related to adjust_x_range)
def test_adjust_x_range_cropping_and_padding_not_exact(df, f_crop_low, f_crop_high):
    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})
    
    hp_f_restricted = adjust_x_range(hp_f, x_range=[f_crop_low, f_crop_high])
    
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes a method similar to what is done in adjust_x_range.
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


# @pytest.mark.parametrize('frequ_mode', ['log', 'two_df'])
@pytest.mark.parametrize('frequ_mode', ['two_df'])
@pytest.mark.parametrize('f_crop_low', [
    pytest.param([0.9 * f_min], marks=pytest.mark.xfail(raises=ValueError,
    strict=True, reason='Invalid f_lower for unequal sampling')),
    f_min,
    1.1 * f_min
    ]
)
@pytest.mark.parametrize('f_crop_high', [
    0.9 * f_max,
    f_max,
    pytest.param([1.1 * f_max], marks=pytest.mark.xfail(raises=ValueError,
    strict=True, reason='Invalid f_lower for unequal sampling'))
    ]
)
def test_adjust_x_range_cropping_and_padding_unequal(frequ_mode, f_crop_low, f_crop_high):
    if frequ_mode == 'log':
        # frequs = np.logspace(np.log10(f_crop_low.value), np.log10(f_crop_high.value), endpoint=True, num=hp_f_fine.size//2) << u.Hz
        frequs = np.logspace(np.log10(f_min.value), np.log10(f_max.value), endpoint=True, num=hp_f_fine.size//2) << u.Hz
    elif frequ_mode == 'two_df':
        # frequs = np.concatenate([np.linspace(f_crop_low.value, f_crop_high.value/2, endpoint=True, num=hp_f_fine.size//2),
        #                          np.linspace(f_crop_high.value/2, f_crop_high.value, endpoint=True, num=hp_f_fine.size//2)]) << u.Hz
        frequs = np.concatenate([np.linspace(f_min.value, f_max.value/2, endpoint=True, num=hp_f_fine.size//2),
                                 np.linspace(f_max.value/2, f_max.value, endpoint=True, num=hp_f_fine.size//2)]) << u.Hz

    hp_f = signal_at_xindex(hp_f_fine, frequs)
    hp_f_restricted = adjust_x_range(hp_f, x_range=[f_crop_low, f_crop_high])
    
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes a method similar to what is done in adjust_x_range.
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
                             atol=(frequs[1] - frequs[0]).value, rtol=0.0)
    assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
                             atol=(frequs[-1] - frequs[-2]).value, rtol=0.0)
    # -- Things are much more difficult for log...

    # assert_allclose_quantity(hp_f_restricted.f0, f_crop_low,
    #                          atol=0.5 * (frequs[1] - frequs[0]).value, rtol=0.0)
    # assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
    #                          atol=0.5 * (frequs[-1] - frequs[-2]).value, rtol=0.0)

    # assert_allclose_quantity(hp_f_restricted.f0, f_crop_low,
    #                          atol=0.9 * (frequs[1] - frequs[0]).value, rtol=0.0)
    # assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
    #                          atol=0.9 * (frequs[-1] - frequs[-2]).value, rtol=0.0)
    # -- Have full df difference if we choose side='left' in adjust_x_range


@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df])#, 0.007*u.Hz, 0.001*u.Hz])
# Checking with one that is not power of two is important to ensure
# pad_to_dx does good job (not necessarily related to adjust_x_range)
@pytest.mark.parametrize('f_fill_low', [-f_min, 0.8 * f_min, f_min, 1.2 * f_min])
@pytest.mark.parametrize('f_fill_high', [0.8 * f_max, f_max, 1.2 * f_max])
def test_adjust_x_range_filling(df, f_fill_low, f_fill_high):
    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})

    hp_f_restricted = adjust_x_range(hp_f, fill_range=[f_fill_low, f_fill_high])
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes a method similar to what is done in adjust_x_range.
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


    # Also check that everything has been set to zero outside of x_range
    hp_f_restricted_cropped_2 = hp_f_restricted[
        hp_f_restricted.frequencies < f_fill_low
    ]
    hp_f_restricted_cropped_3 = hp_f_restricted[
        hp_f_restricted.frequencies > f_fill_high
    ]

    if f_fill_low > hp_f.f0:
        assert_quantity_equal(0.0 * _CORRECT_H_UNIT_FREQU, hp_f_restricted_cropped_2)
    else:
        assert len(hp_f_restricted_cropped_2) == 0

    if f_fill_high < hp_f.frequencies[-1]:
        assert_quantity_equal(0.0 * _CORRECT_H_UNIT_FREQU, hp_f_restricted_cropped_3)
    else:
        assert len(hp_f_restricted_cropped_3) == 0


@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df])#, 0.007*u.Hz, 0.001*u.Hz])
# Checking with one that is not power of two is important to ensure
# pad_to_dx does good job (not necessarily related to adjust_x_range)
@pytest.mark.parametrize('f_crop_low', [0.9 * f_min, f_min])
@pytest.mark.parametrize('f_crop_high', [f_max, 1.1 * f_max])
@pytest.mark.parametrize('f_fill_low', [1.1 * f_min, f_min])
@pytest.mark.parametrize('f_fill_high', [0.9 * f_max, f_max])
def test_adjust_x_range_arg_interplay(df, f_crop_low, f_crop_high, f_fill_low, f_fill_high):
    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})

    hp_f_restricted = adjust_x_range(hp_f,
                                       x_range=[f_crop_low, f_crop_high],
                                       fill_range=[f_fill_low, f_fill_high])

    assert_allclose_quantity(hp_f_restricted.f0, f_crop_low,
                             atol=df.value, rtol=0.0)
    assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
                             atol=df.value, rtol=0.0)
    
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes a method similar to what is done in adjust_x_range.
    # Instead, more straightforward array slicing is used

    hp_f_cropped = hp_f[(hp_f.frequencies >= f_fill_low)
                            & (hp_f.frequencies <= f_fill_high)]
    hp_f_restricted_cropped = hp_f_restricted[
        (hp_f_restricted.frequencies >= f_fill_low)
        & (hp_f_restricted.frequencies <= f_fill_high)
    ]

    assert_quantity_equal(hp_f_cropped, hp_f_restricted_cropped)

    # assert_allclose_quantity(hp_f_restricted_cropped.f0, f_fill_low,
    #                          atol=df.value, rtol=0.0)
    # assert_allclose_quantity(hp_f_restricted_cropped.frequencies[-1], f_fill_high,
    #                          atol=df.value, rtol=0.0)


    # Also check that everything has been set to zero outside of x_range
    hp_f_restricted_cropped_2 = hp_f_restricted[
        hp_f_restricted.frequencies < f_fill_low
    ]
    hp_f_restricted_cropped_3 = hp_f_restricted[
        hp_f_restricted.frequencies > f_fill_high
    ]

    assert_quantity_equal(0.0 * _CORRECT_H_UNIT_FREQU, hp_f_restricted_cropped_2)
    assert_quantity_equal(0.0 * _CORRECT_H_UNIT_FREQU, hp_f_restricted_cropped_3)
    
    # assert_allclose_quantity(hp_f_restricted_cropped_2.f0, f_crop_low,
    #                          atol=df.value, rtol=0.0)
    # assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
    #                          atol=df.value, rtol=0.0)


@pytest.mark.parametrize('df', [hp_f_coarse.df, hp_f_fine.df, hp_f_fine.df / 4])
def test_adjust_x_range_with_padding_and_cropping_exact(df):
    f_crop_low, f_crop_high = 20.0 * u.Hz, 30.0 * u.Hz
    # For contiguous padding to be possible, f_crop_low has to be an integer
    # multiple of df

    # hp_t_padded = pad_to_dx(hp_t, df)
    # hp_t_f = td_to_fd(hp_t_padded)
    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})
    hp_f = hp_f[hp_f.frequencies >= f_crop_low]  # Cut off so no start at f=0
    hp_f_restricted = adjust_x_range(hp_f, x_range=[0.0, f_crop_high],
                                       fill_range=[f_crop_low, None])
    
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes computations similar to what is done in adjust_x_range.
    # Instead, more straightforward array slicing is used
    print(hp_f_restricted.frequencies[-1], hp_f_restricted.df, f_crop_high)
    assert_quantity_equal(hp_f_restricted.f0, 0.0 * u.Hz)
    assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
                             atol=0.8 * df.value, rtol=0.0)
    # -- Coarse requires 0.8, for fine 0.42 would be sufficient


@pytest.mark.parametrize('df', [0.001*u.Hz, 0.007*u.Hz])
def test_adjust_x_range_with_padding_and_cropping_not_exact(df):
    f_crop_low, f_crop_high = 20.0 * u.Hz, 30.0 * u.Hz

    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': df})
    hp_f = hp_f[hp_f.frequencies >= f_crop_low]  # Cut off so no start at f=0
    hp_f_restricted = adjust_x_range(hp_f, x_range=[0.0, f_crop_high],
                                       fill_range=[f_crop_low, None])
    
    # NOTE: we will not use Series.crop to get the comparisons because it
    # utilizes computations similar to what is done in adjust_x_range.
    # Instead, more straightforward array slicing is used

    assert_quantity_equal(hp_f_restricted.f0, 0.0 * u.Hz)
    assert_allclose_quantity(hp_f_restricted.frequencies[-1], f_crop_high,
                             atol=df.value, rtol=0.0)
    # More tolerance needed here since using the more accurate slicing
    # method used here is too expensive for use in adjust_x_range. This
    # comes at the price of certain smaller deviations for some df


@pytest.mark.parametrize('fill_val', [0., 2.])
def test_fill_x_range(fill_val):
    hf = hp_f_fine.copy()

    fill_x_range(hf, fill_val, [-f_min, 1.1*f_max])

    assert_allequal_series(hp_f_fine, hf)

    f_lower, f_upper = 30.*u.Hz, 50.*u.Hz
    fill_x_range(hf, fill_val, [f_lower, f_upper])

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
    # -- Note: overriding needed here because of some lal error that is
    # -- occuring. Maybe also due to insufficient lalsimultion version
    hp_t.override_unit(u.dimensionless_unscaled)
    hc_t.override_unit(u.dimensionless_unscaled)
    lal_t_out = GravitationalWavePolarizations(hp_t, hc_t).strain(**ext_params)
    lal_t_out.override_unit(_CORRECT_H_UNIT_TIME)
    assert_quantity_equal(lal_t_out, ht_test)

    hf_test = get_strain(wf_params | ext_params, 'frequency', generator=gen)
    hp_f_fine.override_unit(u.dimensionless_unscaled)
    hc_f_fine.override_unit(u.dimensionless_unscaled)
    lal_f_out = GravitationalWavePolarizations(hp_f_fine, hc_f_fine).strain(**ext_params)
    lal_f_out.override_unit(_CORRECT_H_UNIT_FREQU)
    assert_quantity_equal(lal_f_out, hf_test)


class GetStrainErrorRaising(unittest.TestCase):
    def test_domain_checking(self):
        with self.assertRaises(ValueError):
            get_strain(wf_params, 'domain', generator=gen)

    def test_mode_checking(self):
        with self.assertRaises(ValueError):
            get_strain(wf_params, 'time', generator=gen, mode='mode')
    
    def test_extr_params_checking(self):
        with self.assertRaises(ValueError):
            get_strain(wf_params | {'psi': 0.5*u.rad}, 'time', generator=gen)


def test_get_wf_generator_cache():
    # -- Creative test for whether caching is on or not: try accessing
    # -- an attribute that cache as wrapper defines, ".cache_info()"

    with enable_caching_locally():
        wf_gen = get_wf_generator('IMRPhenomXPHM', cache=True)
        wf_gen.cache_info()
    
        wf_gen = get_wf_generator('IMRPhenomXPHM', cache=False)
        with pytest.raises(AttributeError, match="'function' object has no attribute 'cache_info'"):
            wf_gen.cache_info()
        
        wf_gen = get_wf_generator('IMRPhenomXPHM')
        wf_gen.cache_info()

    with disable_caching_locally():
        wf_gen = get_wf_generator('IMRPhenomXPHM', cache=True)
        wf_gen.cache_info()
    
        wf_gen = get_wf_generator('IMRPhenomXPHM', cache=False)
        with pytest.raises(AttributeError, match="'function' object has no attribute 'cache_info'"):
            wf_gen.cache_info()
        
        wf_gen = get_wf_generator('IMRPhenomXPHM')
        with pytest.raises(AttributeError, match="'function' object has no attribute 'cache_info'"):
            wf_gen.cache_info()
