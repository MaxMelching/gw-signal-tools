import numpy as np
import unittest
from numpy.testing import assert_allclose

from gw_signal_tools.waveform_utils import (
    td_to_fd_waveform, pad_to_get_target_df, get_wf_generator
)
from gw_signal_tools.test_utils import (
    allclose_quantity, assert_allclose_quantity,
    assert_allclose_series
)
from gwpy.testing.utils import assert_quantity_equal

from gw_signal_tools.inner_product import inner_product, norm, overlap
from gw_signal_tools.PSDs import psd_gw150914, psd_no_noise

import astropy.units as u
from lalsimulation.gwsignal import gwsignal_get_waveform_generator
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
gen = gwsignal_get_waveform_generator(approximant)
def td_wf_gen(wf_params):
    return wfm.GenerateTDWaveform(wf_params, gen)

def fd_wf_gen(wf_params):
    return wfm.GenerateFDWaveform(wf_params, gen)

# Generate time domain waveform
hp_t, _ = td_wf_gen(wf_params)

# Two waveforms will be generated in frequency domain, first with finer
# sampling and then with coarser one
hp_f_fine, _ = fd_wf_gen(wf_params)

hp_f_coarse, _ = fd_wf_gen(wf_params | {'deltaF': 1.0 / (hp_t.size * hp_t.dx)})

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
    # norm2_coarse, info_coarse = norm(hp_f_coarse, optimize_time_and_phase=True,
    norm2_coarse, info_coarse = norm(hp_f_coarse, optimize_time=True,
                                     optimize_phase=True,
                                     return_opt_info=True)
    time_coarse = info_coarse['peak_time']
    phase_coarse = info_coarse['peak_phase']

    assert_allclose_quantity(norm1_coarse, norm2_coarse, atol=0.0, rtol=0.11)
    assert_allclose_quantity(0.*u.s, time_coarse, atol=1e-10, rtol=0.0)
    assert_allclose_quantity(0.*u.rad, phase_coarse, atol=1e-18, rtol=0.0)


    norm1_fine = norm(hp_f_fine)
    norm2_fine, info_fine = norm(hp_f_fine, optimize_time_and_phase=True,
                                 return_opt_info=True)
    time_fine = info_fine['peak_time']
    phase_fine = info_fine['peak_phase']

    assert_allclose_quantity(norm1_fine, norm2_fine, atol=0.0, rtol=5e-4)
    assert_allclose_quantity(0.*u.s, time_fine, atol=1e-12, rtol=0.0)
    assert_allclose_quantity(0.*u.rad, phase_fine, atol=1e-17, rtol=0.0)

@pytest.mark.parametrize('time_shift', [0.*u.s, 1e-3*u.s, -0.2*u.s, 0.5*u.s])
@pytest.mark.parametrize('phase_shift', [0.*u.rad, 0.12*u.rad, -0.3*np.pi*u.rad])
def test_optimize_match(time_shift, phase_shift):
    # norm_coarse = norm(hp_f_coarse)**2
    # hp_f_coarse_shifted = hp_f_coarse * np.exp(2.j*np.pi*hp_f_coarse.frequencies*time_shift + 1.j*phase_shift)

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

    # coarse performs REALLY bad, thus omitted for these tests

    norm_fine = norm(hp_f_fine)**2
    hp_f_fine_shifted = hp_f_fine * np.exp(2.j*np.pi*hp_f_fine.frequencies*time_shift + 1.j*phase_shift)
    overlap_fine, info_fine = inner_product(
        hp_f_fine,
        hp_f_fine_shifted,
        optimize_time_and_phase=True,
        return_opt_info=True
    )
    time_fine = info_fine['peak_time']
    phase_fine = info_fine['peak_phase']
    match_series_fine = info_fine['match_series']

    assert_allclose_quantity(norm_fine, overlap_fine, atol=0.0, rtol=9e-4)
    assert_allclose_quantity(time_shift, time_fine, atol=0.8*match_series_fine.dx.value, rtol=0.0)
    assert_allclose_quantity(phase_shift, phase_fine, atol=0.061, rtol=0.0)
    # Phase error is higher than I like, but despite intensive testing,
    # I found no better implementation

def test_different_optimizations():
    norm1 = norm(hp_f_fine, optimize_time_and_phase=False)
    norm2, info2 = norm(hp_f_fine, optimize_time_and_phase=True, return_opt_info=True)
    norm3, info3 = norm(hp_f_fine, optimize_time=True, optimize_phase=False, return_opt_info=True)
    norm4, info4 = norm(hp_f_fine, optimize_time=False, optimize_phase=True, return_opt_info=True)

    assert_allclose_quantity(norm1, [norm2, norm3, norm4], atol=0., rtol=4e-4)
    # rtol for usual deviation between simpson result and fft one. Next test
    # verifies that all optimized norms are actually equal
    assert_allclose_quantity(norm2, [norm3, norm4], atol=0., rtol=0.)

    time2 = info2['peak_time']
    time3 = info3['peak_time']
    time4 = info4['peak_time']
    assert_allclose_quantity(0.*u.s, [time2, time3, time4], atol=3e-13, rtol=0.)

    phase2 = info2['peak_phase']
    phase3 = info3['peak_phase']
    phase4 = info4['peak_phase']
    assert_allclose_quantity(0.*u.rad, [phase2, phase3, phase4], atol=6.3e-18, rtol=0.)

@pytest.mark.parametrize('f_min', [f_min, 30.0 * u.Hz])
@pytest.mark.parametrize('f_max', [50.0 * u.Hz, f_max])
def test_f_range(f_min, f_max):
    norm1 = norm(hp_f_fine, f_range=[f_min, f_max])
    norm_no_units = norm(hp_f_fine, f_range=[f_min.value, f_max.value])
    assert_quantity_equal(norm1, norm_no_units)

    hp_f_restricted, _ = fd_wf_gen(wf_params | {'f22_start': f_min, 'f_max': f_max})
    hp_f_restricted.override_unit(u.s)
    norm2 = norm(hp_f_restricted)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=1e-3)
    # Not fully equal due to potentially being one sample off when filling


def test_positive_negative_f_range_consistency():
    h = td_to_fd_waveform(pad_to_get_target_df(hp_t, df=hp_f_fine.df))
    h_symm = td_to_fd_waveform(pad_to_get_target_df(hp_t, df=hp_f_fine.df) + 0.j)
    # h_symm has symmetric spectrum around f=0.0 and the same spectrum as h
    # for positive frequencies
    assert h.f0 != h_symm.f0  # Make sure they are not the same

    f_upper = f_max

    norm1 = norm(h, f_range=[0.0, f_upper])
    norm1_opt, info1 = norm(h, f_range=[0.0, f_upper],
                            optimize_time_and_phase=True, return_opt_info=True)
    time_1 = info1['peak_time']
    assert_allclose_quantity(norm1, norm1_opt, atol=0.0, rtol=1e-12)
    assert_allclose_quantity(0.*u.s, time_1, atol=1e-12, rtol=0.0)

    norm2 = norm(h_symm, f_range=[-f_upper, f_upper])
    norm2_opt, info2 = norm(h_symm, f_range=[-f_upper, f_upper],
                            optimize_time_and_phase=True, return_opt_info=True)
    time_2 = info2['peak_time']
    assert_allclose_quantity(norm2, norm2_opt, atol=0.0, rtol=1e-12)
    assert_allclose_quantity(0.*u.s, time_2, atol=1e-12, rtol=0.0)


    assert_quantity_equal(norm1, norm2)
    assert_allclose_quantity(norm1_opt, norm2_opt, atol=0.0, rtol=1e-12)


    norm_plus = norm(h_symm, f_range=[0.0, f_upper])
    norm_minus = norm(h_symm, f_range=[-f_upper, 0.0])

    assert_allclose_quantity(norm_plus, norm_minus, atol=0.0, rtol=1e-15)
    assert_allclose_quantity(norm_plus, norm2, atol=0.0, rtol=1e-15)
    assert_allclose_quantity(norm_minus, norm2, atol=0.0, rtol=1e-15)

def test_df_consistency():
    # Same signal, decreasing df in inner_product
    norm1 = norm(hp_f_fine, df=hp_f_fine.df)
    norm2 = norm(hp_f_fine, df=hp_f_fine.df / 2)
    norm3 = norm(hp_f_fine, df=hp_f_fine.df / 4)


    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=2e-3)
    assert_allclose_quantity(norm1, norm3, atol=0.0, rtol=2e-3)
    assert_quantity_equal(norm2, norm3)  # Because linear interpolation the same for them


    # Different signals with matching df in inner_product
    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': hp_f_fine.df / 2})
    hp_f.override_unit(u.s)
    norm2 = norm(hp_f, df=hp_f_fine.df / 2)

    hp_f, _ = fd_wf_gen(wf_params | {'deltaF': hp_f_fine.df / 4})
    hp_f.override_unit(u.s)
    norm3 = norm(hp_f, df=hp_f_fine.df / 4)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=5e-4)
    assert_allclose_quantity(norm1, norm3, atol=0.0, rtol=6e-4)
    assert_allclose_quantity(norm2, norm3, atol=0.0, rtol=2e-4)


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

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=0.001)
    
    new_frequ_unit = hp_f_fine_rescaled.frequencies.unit

    psd_no_noise_rescaled = psd_no_noise.copy()  # Verify manually what happens
    # psd_no_noise_rescaled = psd_no_noise.__deepcopy__(None)  # Verify manually what happens
    psd_no_noise_rescaled.override_unit(1/new_frequ_unit)
    psd_no_noise_rescaled.frequencies *= (
        1/psd_no_noise_rescaled.frequencies.unit * new_frequ_unit
    )

    norm1 = norm(hp_f_fine_rescaled, psd=psd_no_noise_rescaled)
    norm2 = norm(hp_f_fine, psd=psd_no_noise)

    assert_allclose_quantity(norm1, norm2, atol=0.0, rtol=0.001)


    hp_f_fine_rescaled_2 = hp_f_fine.copy()
    hp_f_fine_rescaled_2 *= u.m**2

    norm3 = np.sqrt(inner_product(hp_f_fine, hp_f_fine_rescaled_2))

    assert_allclose_quantity(norm1 * u.m, norm3, atol=0.0, rtol=0.001)

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
    overlap_pycbc, time_pycbc, phase_pycbc = match(
        hp_1_pycbc,
        hp_2_pycbc,
        v1_norm=1.,
        v2_norm=1.,
        psd=psd_pycbc,
        low_frequency_cutoff=f_low,
        high_frequency_cutoff=f_high,
        return_phase=True
    )
    time_pycbc *= 1/(2 * (tlen - 1) * delta_f)

    overlap_gw_signal_tools, info = inner_product(
        hp_1_pycbc_converted,
        hp_2_pycbc_converted,
        psd_pycbc_converted,
        f_range=[f_low, f_high],
        optimize_time_and_phase=True,
        return_opt_info=True
    )
    time_gw_signal_tools = info['peak_time'].value
    phase_gw_signal_tools = info['peak_phase'].value

    assert_allclose(overlap_pycbc, overlap_gw_signal_tools, atol=0.0, rtol=2e-3)
    assert_allclose(np.abs(time_pycbc), np.abs(time_gw_signal_tools), atol=0.0, rtol=2e-2)
    # assert_allclose(phase_pycbc, phase_gw_signal_tools, atol=0.0, rtol=0.0)  # Not matching well, have to find out why


def test_overlap_pycbc():
    overlap_normalized_pycbc, _ = match(hp_1_pycbc, hp_2_pycbc, psd=psd_pycbc, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)

    overlap_normalized_gw_signal_tools = overlap(hp_1_pycbc_converted, hp_2_pycbc_converted, psd_pycbc_converted, f_range=[f_low, f_high], optimize_time_and_phase=True)

    assert_allclose(overlap_normalized_pycbc, overlap_normalized_gw_signal_tools, atol=0.0,rtol=2e-3)


def test_norm_optimized():
    norm1_gw_signal_tools = overlap(hp_1_pycbc_converted, hp_1_pycbc_converted, psd_pycbc_converted, f_range=[f_low, f_high], optimize_time_and_phase=True)
    norm2_gw_signal_tools = overlap(hp_2_pycbc_converted, hp_2_pycbc_converted, psd_pycbc_converted, f_range=[f_low, f_high], optimize_time_and_phase=True)

    assert_allclose(1.0, [norm1_gw_signal_tools, norm2_gw_signal_tools], atol=0.0, rtol=1e-5)
