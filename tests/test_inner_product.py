import numpy as np
import unittest
from numpy.testing import assert_allclose

from gw_signal_tools.waveform_utils import (
    td_to_fd_waveform, pad_to_get_target_df, get_strain
)
from gw_signal_tools.test_utils import (
    allclose_quantity, assert_allclose_quantity,
    assert_allclose_frequseries, assert_allclose_timeseries
)
from gwpy.testing.utils import assert_quantity_equal

from gw_signal_tools.inner_product import inner_product, norm, overlap
from gw_signal_tools.PSDs import psd_gw150914, psd_no_noise

import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm

import pytest


#%% Initializing commonly used variables
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
    'distance': 1.*u.Mpc,
    'inclination': 0.0*u.rad,
    'eccentricity': 0.*u.dimensionless_unscaled,
    'longAscNodes': 0.*u.rad,
    'meanPerAno': 0.*u.rad,
    'condition': 0
}

approximant = 'IMRPhenomXPHM'

# Call the generator
gen = wfm.LALCompactBinaryCoalescenceGenerator(approximant)


# Generate time domain waveform
hp_t, _ = wfm.GenerateTDWaveform(wf_params, gen)

# Two waveforms will be generated in frequency domain, first with finer
# sampling and then with coarser one
hp_f_fine, _ = wfm.GenerateFDWaveform(wf_params, gen)

hp_f_coarse, _ = wfm.GenerateFDWaveform(wf_params | {'deltaF': 1.0 / (hp_t.size * hp_t.dx)}, gen)

# Make units consistent with gw_signal_tools
hp_f_fine *= u.s
hp_f_coarse *= u.s


#%% ---------- Technical test if signals are edited inplace ----------
@pytest.mark.parametrize('optimize_time_and_phase', [False, True])
def test_no_inplace_editing_of_signals(optimize_time_and_phase):
    from gw_signal_tools.PSDs import psd_no_noise
    psd = psd_no_noise.copy()
    hp_f_fine_2 = hp_f_fine.copy()
    # TODO: check they do not share memory

    norm(hp_f_fine_2, f_range=[2 * f_min, 0.5 * f_max], df=hp_f_fine_2.df,
         optimize_time_and_phase=optimize_time_and_phase)
    # We ensure some conversions take place, which could potentially change
    # signal inplace. Setting df so that no interpolation takes place is
    # important because otherwise, interpolate would copy

    assert_quantity_equal(hp_f_fine, hp_f_fine_2)

    # The following assertion is based on bug that was present for a short
    # time, where no copying of input PSD took place, so it was changed inplace
    hp_f_fine_2.frequencies *= u.s
    norm(hp_f_fine_2, optimize_time_and_phase=optimize_time_and_phase)

    from gw_signal_tools.PSDs import psd_no_noise
    assert_quantity_equal(psd, psd_no_noise)


#%% ---------- Consistency tests with inner_product function ----------
def test_fd_td_match_consistency():
    norm_td_coarse = norm(hp_t, df=2**-2, f_range=[f_min, None])
    norm_fd_coarse = norm(hp_f_coarse, df=2**-2, f_range=[f_min, None])

    assert_allclose_quantity(norm_td_coarse, norm_fd_coarse, atol=0.0, rtol=0.11)

    norm_td_fine = norm(hp_t, df=2**-4, f_range=[f_min, None])
    norm_fd_fine = norm(hp_f_fine, df=2**-4, f_range=[f_min, None])

    assert_allclose_quantity(norm_td_fine, norm_fd_fine, atol=0.0, rtol=0.005)


def test_fd_td_overlap_consistency():
    norm_td = overlap(hp_t, hp_t, df=2**-4, f_range=[f_min, None])
    norm_fd_coarse = overlap(hp_f_coarse, hp_f_coarse, df=2**-2, f_range=[f_min, None])
    norm_fd_fine = overlap(hp_f_fine, hp_f_fine, df=2**-4, f_range=[f_min, None])

    # assert_allclose_quantity(u.Quantity([norm_td, norm_fd_coarse, norm_fd_fine]), u.Quantity([1.0, 1.0, 1.0]), atol=0.0, rtol=0.005)
    assert_allclose_quantity(norm_td, 1.0 * u.dimensionless_unscaled, atol=0.0, rtol=0.005)
    assert_allclose_quantity(norm_fd_coarse, 1.0 * u.dimensionless_unscaled, atol=0.0, rtol=0.005)
    assert_allclose_quantity(norm_fd_fine, 1.0 * u.dimensionless_unscaled, atol=0.0, rtol=0.005)
    assert_allclose_quantity(norm_td, norm_fd_fine, atol=0.0, rtol=0.005)


def test_optimize_match_consistency():
    norm1_coarse = norm(hp_f_coarse)
    _, norm2_coarse, time_coarse = norm(hp_f_coarse, optimize_time_and_phase=True)

    assert_allclose_quantity(norm1_coarse, norm2_coarse, atol=0.0, rtol=0.11)
    assert_allclose_quantity(0.0 * u.s, time_coarse, atol=1e-10, rtol=0.0)


    norm1_fine = norm(hp_f_fine)
    _, norm2_fine, time_fine = norm(hp_f_fine, optimize_time_and_phase=True)

    assert_allclose_quantity(norm1_fine, norm2_fine, atol=0.0, rtol=0.001)
    assert_allclose_quantity(0.0 * u.s, time_fine, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize('f_min', [f_min, 30.0 * u.Hz])
@pytest.mark.parametrize('f_max', [50.0 * u.Hz, f_max])
def test_f_range(f_min, f_max):
    norm1 = norm(hp_f_fine, f_range=[f_min, f_max])
    norm_no_units = norm(hp_f_fine, f_range=[f_min.value, f_max.value])
    assert_quantity_equal(norm1, norm_no_units)

    hp_f_restricted, _ = wfm.GenerateFDWaveform(wf_params | {'f22_start': f_min, 'f_max': f_max}, gen)
    hp_f_restricted.override_unit(u.s)
    norm2 = norm(hp_f_restricted)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=1e-3)
    # Not fully equal due to potentially being one sample off when filling


def test_positive_negative_f_range_consistency():
    h = td_to_fd_waveform(pad_to_get_target_df(hp_t, df=hp_f_fine.df))
    h_symm = td_to_fd_waveform(pad_to_get_target_df(hp_t, df=hp_f_fine.df) + 1.j * 0.0)
    # h_symm has symmetric spectrum around f=0.0 and the same spectrum as h
    # for positive frequencies

    # f_max = min(h.frequencies[-1], h_symm.frequencies[-1])
    f_max = 1024.0 * u.Hz

    norm1 = norm(h, f_range=[0.0, f_max])
    _, norm1_opt, time_1 = norm(h, f_range=[0.0, f_max], optimize_time_and_phase=True)
    assert_allclose_quantity(norm1, norm1_opt, atol=0.0, rtol=1e-12)
    assert_allclose_quantity(0.0 * u.s, time_1, atol=1e-12, rtol=0.0)

    norm2 = norm(h_symm, f_range=[-f_max, f_max])
    _, norm2_opt, time_2 = norm(h_symm, f_range=[-f_max, f_max], optimize_time_and_phase=True)
    assert_allclose_quantity(norm2, norm2_opt, atol=0.0, rtol=1e-12)
    assert_allclose_quantity(0.0 * u.s, time_2, atol=1e-12, rtol=0.0)


    assert_quantity_equal(norm1, norm2)
    assert_allclose_quantity(norm1_opt, norm2_opt, atol=0.0, rtol=1e-12)


    norm_plus = norm(h_symm, f_range=[0.0, f_max])
    norm_minus = norm(h_symm, f_range=[-f_max, 0.0])

    assert_allclose_quantity(norm_plus, norm_minus, atol=0.0, rtol=1e-15)
    assert_allclose_quantity(norm_plus, norm2, atol=0.0, rtol=1e-15)
    assert_allclose_quantity(norm_minus, norm2, atol=0.0, rtol=1e-15)


def test_df_no_unit():
    df_val = 2**-5

    norm1 = norm(hp_f_fine, df=df_val)
    norm2 = norm(hp_f_fine, df=df_val * u.Hz)

    assert_quantity_equal(norm1, norm2)


def test_different_units():
    norm2 = norm(hp_f_fine, psd=psd_no_noise)

    hp_f_fine_rescaled = hp_f_fine.copy()
    hp_f_fine_rescaled.frequencies *= u.s
    hp_f_fine_rescaled /= u.s
    # NOTE: rescaling the amplitude this way is not strictly necessary, one
    # could also get a consistent result without this step. By doing that, we
    # merely ensure the resulting norm is dimensionless, making the subsequent
    # comparison easier

    norm1 = norm(hp_f_fine)
    norm2 = norm(hp_f_fine_rescaled)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=0.01)
    
    new_frequ_unit = hp_f_fine_rescaled.frequencies.unit

    psd_no_noise_rescaled = psd_no_noise.copy()  # Verify manually what happens
    # psd_no_noise_rescaled = psd_no_noise.__deepcopy__(None)  # Verify manually what happens
    psd_no_noise_rescaled.override_unit(1/new_frequ_unit)
    psd_no_noise_rescaled.frequencies *= (
        1/psd_no_noise_rescaled.frequencies.unit * new_frequ_unit
    )

    norm1 = norm(hp_f_fine_rescaled, psd=psd_no_noise_rescaled)
    norm2 = norm(hp_f_fine, psd=psd_no_noise)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=0.01)

# TODO (maybe): test with mass rescaled waveforms?


#%% Confirm that certain errors are raised
class ErrorRaising(unittest.TestCase): 
    def test_signal_type_checking(self):
        with self.assertRaises(TypeError):
            inner_product(np.array([42]), hp_f_fine)
    
        with self.assertRaises(TypeError):
            inner_product(hp_f_fine, np.array([42]))
    
        with self.assertRaises(TypeError):
            inner_product(hp_f_fine, hp_f_fine, psd=np.array([42]))

    
    def test_signal_unit_checking(self):
        with self.assertRaises(AssertionError):
            inner_product(hp_f_fine, hp_f_fine * u.m)
        
        with self.assertRaises(AssertionError):
            inner_product(hp_f_fine, hp_f_fine, psd=psd_no_noise * u.m)


    def test_frequ_unit_checking(self):
        with self.assertRaises(AssertionError):
            hp_f_fine_wrong = hp_f_fine.copy()
            hp_f_fine_wrong.frequencies *= u.m

            inner_product(hp_f_fine, hp_f_fine * u.m)

        with self.assertRaises(AssertionError):
            psd_no_noise_wrong = psd_no_noise.copy()
            psd_no_noise_wrong.frequencies *= u.m

            norm(hp_f_fine, psd=psd_no_noise_wrong)
    

    def test_df_unit_testing(self):
        with self.assertRaises(ValueError):
            norm(hp_f_fine, df=0.0625 * u.m)


    # def test_optimize_requirements(self):
    #     with self.assertRaises(ValueError):
    #         ...  # Generate using get_strain, then remove certain components -> with behaviour from now, it is intended that no error should be raised!


#%% ---------- Confirming results with PyCBC match function ----------
from pycbc.waveform import get_fd_waveform
from pycbc.filter import match
from pycbc.psd import aLIGOZeroDetHighPower

from gwpy.frequencyseries.frequencyseries import FrequencySeries

f_low, f_high = 20, 350  # f_min and some essentially arbitrary cutoff
sample_rate = 4096

# Enter some arbitrary parameters here
wfs_to_compare = {
    'signal1': {
        'mass1': 10,
        'mass2': 10,
        'spin1': 0.6,
        'spin2': 0.0
    },
    'signal2':{
        'mass1': 96,
        'mass2': 20,
        'spin1': 0.0,
        'spin2': 0.1
    }
}


hp_1_pycbc, _ = get_fd_waveform(
    approximant=approximant,
    **wfs_to_compare['signal1'],
    f_lower=f_low,
    f_upper=f_high,
    delta_f=1.0/sample_rate
)

hp_2_pycbc, _ = get_fd_waveform(
    approximant=approximant,
    **wfs_to_compare['signal2'],
    f_lower=f_low,
    f_upper=f_high,
    delta_f=1.0/sample_rate
)

tlen = max(len(hp_1_pycbc), len(hp_2_pycbc))
hp_1_pycbc.resize(tlen)
hp_2_pycbc.resize(tlen)

delta_f = 1.0 / hp_2_pycbc.duration
flen = tlen//2 + 1
psd_pycbc = aLIGOZeroDetHighPower(flen, delta_f, f_low)


hp_1_pycbc_converted = FrequencySeries.from_pycbc(hp_1_pycbc) * u.s
hp_2_pycbc_converted = FrequencySeries.from_pycbc(hp_2_pycbc) * u.s
psd_pycbc_converted = FrequencySeries.from_pycbc(psd_pycbc) / u.Hz


def test_match_pycbc():
    overlap_pycbc, _ = match(hp_1_pycbc, hp_2_pycbc, v1_norm=1.0, v2_norm=1.0, psd=psd_pycbc, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)

    _, overlap_gw_signal_tools, _ = inner_product(hp_1_pycbc_converted, hp_2_pycbc_converted, psd_pycbc_converted, f_range=[f_low, f_high], optimize_time_and_phase=True)

    assert_allclose(overlap_pycbc, overlap_gw_signal_tools, atol=0.0, rtol=2e-3)


def test_overlap_pycbc():
    overlap_normalized_pycbc, _ = match(hp_1_pycbc, hp_2_pycbc, psd=psd_pycbc, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)

    _, overlap_normalized_gw_signal_tools, _ = overlap(hp_1_pycbc_converted, hp_2_pycbc_converted, psd_pycbc_converted, f_range=[f_low, f_high], optimize_time_and_phase=True)

    assert_allclose(overlap_normalized_pycbc, overlap_normalized_gw_signal_tools, atol=0.0,rtol=2e-3)  # TODO: change rtol to 0.005?


def test_norm_optimized():
    _, norm1_gw_signal_tools, _ = overlap(hp_1_pycbc_converted, hp_1_pycbc_converted, psd_pycbc_converted, f_range=[f_low, f_high], optimize_time_and_phase=True)
    _, norm2_gw_signal_tools, _ = overlap(hp_2_pycbc_converted, hp_2_pycbc_converted, psd_pycbc_converted, f_range=[f_low, f_high], optimize_time_and_phase=True)

    assert_allclose(norm1_gw_signal_tools, 1.0, atol=0.0, rtol=1e-5)
    assert_allclose(norm2_gw_signal_tools, 1.0, atol=0.0, rtol=1e-5)
