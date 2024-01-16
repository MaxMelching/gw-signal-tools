import numpy as np

from gw_signal_tools.inner_product import inner_product, norm, overlap
from gw_signal_tools.PSDs import psd_gw150914

import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm


#%% ---------- Consistency tests with inner_product function ----------

deltaT = 1./2048.*u.s
f_min = 20.*u.Hz  # Cutoff frequency
f_ref = 20.*u.Hz  # Frequency where we specify spins
distance = 440.*u.Mpc
inclination = 2.7*u.rad  # Value taken from posteriors.ipynb, where posterior of inclination is plotted
phiRef = 0.*u.rad
eccentricity = 0.*u.dimensionless_unscaled
longAscNodes = 0.*u.rad
meanPerAno = 0.*u.rad


wf_params = {
    'mass1' : 36*u.solMass,
    'mass2' : 29*u.solMass,
    'deltaT' : deltaT,  # Needed even for FDWaveform, otherwise f_max very small
    # 'deltaF' : deltaT / u.s * u.Hz,  # Does work
    'f22_start' : f_min,
    'f22_ref': f_ref,
    'phi_ref' : phiRef,
    'distance' : distance,
    'inclination' : inclination,
    'eccentricity' : eccentricity,
    'longAscNodes' : longAscNodes,
    'meanPerAno' : meanPerAno,
    'condition' : 0
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


def test_fd_td_match_consistency():
    match_td = norm(hp_t, psd_gw150914, df=2**-2, f_range=[f_min, None])
    match_fd = norm(hp_f_coarse, psd_gw150914, df=2**-2, f_range=[f_min, None])

    assert np.isclose(match_td, match_fd, atol=0.0, rtol=0.005)

    match_td = norm(hp_t, psd_gw150914, df=2**-4, f_range=[f_min, None])
    match_fd = norm(hp_f_fine, psd_gw150914, df=2**-4, f_range=[f_min, None])

    assert np.isclose(match_td, match_fd, atol=0.0, rtol=0.005)


def test_norm():
    norm_td = overlap(hp_t, hp_t, psd_gw150914, df=2**-4, f_range=[f_min, None])
    norm_fd_coarse = overlap(hp_f_coarse, hp_f_coarse, psd_gw150914, df=2**-2, f_range=[f_min, None])
    norm_fd_fine = overlap(hp_f_fine, hp_f_fine, psd_gw150914, df=2**-4, f_range=[f_min, None])

    assert np.all(np.isclose([norm_td, norm_fd_coarse, norm_fd_fine], [1.0, 1.0, 1.0], atol=0.0, rtol=0.005))


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


hp_1_pycbc_converted = FrequencySeries.from_pycbc(hp_1_pycbc)
hp_2_pycbc_converted = FrequencySeries.from_pycbc(hp_2_pycbc)
psd_pycbc_converted = FrequencySeries.from_pycbc(psd_pycbc)


def test_match_pycbc():
    overlap_pycbc, _ = match(hp_1_pycbc, hp_2_pycbc, v1_norm=1.0, v2_norm=1.0, psd=psd_pycbc, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)

    _, overlap_gw_signal_tools, _ = inner_product(hp_1_pycbc_converted, hp_2_pycbc_converted, psd_pycbc_converted, f_range=[f_low, f_high], optimize_time_and_phase=True)

    assert np.isclose(overlap_pycbc, overlap_gw_signal_tools, atol=0.0, rtol=0.01)


def test_overlap_pycbc():
    overlap_normalized_pycbc, _ = match(hp_1_pycbc, hp_2_pycbc, psd=psd_pycbc, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)

    _, overlap_normalized_gw_signal_tools, _ = overlap(hp_1_pycbc_converted, hp_2_pycbc_converted, psd_pycbc_converted, f_range=[f_low, f_high], optimize_time_and_phase=True)

    # assert np.abs((overlap_normalized_pycbc - overlap_normalized) / overlap_normalized_pycbc) < 0.01
    assert np.isclose(overlap_normalized_pycbc, overlap_normalized_gw_signal_tools, atol=0.0,rtol=0.01)  # TODO: change rtol to 0.005?


def test_norm_optimized():
    _, norm1_gw_signal_tools, _ = overlap(hp_1_pycbc_converted, hp_1_pycbc_converted, psd_pycbc_converted, f_range=[f_low, f_high], optimize_time_and_phase=True)
    _, norm2_gw_signal_tools, _ = overlap(hp_2_pycbc_converted, hp_2_pycbc_converted, psd_pycbc_converted, f_range=[f_low, f_high], optimize_time_and_phase=True)

    assert np.isclose(norm1_gw_signal_tools, 1.0, atol=0.0, rtol=0.01)
    assert np.isclose(norm2_gw_signal_tools, 1.0, atol=0.0, rtol=0.01)
