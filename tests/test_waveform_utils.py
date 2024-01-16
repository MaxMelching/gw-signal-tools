from gw_signal_tools.waveform_utils import (
    td_to_fd_waveform, fd_to_td_waveform,
    pad_to_get_target_df, restrict_f_range
)

import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm
import numpy as np


# We will perform tests with a GW150914-like signal
deltaT = 1./2048.*u.s
f_min = 20.*u.Hz
f_ref = 20.*u.Hz
distance = 440.*u.Mpc
inclination = 2.7*u.rad
phiRef = 0.*u.rad
eccentricity = 0.*u.dimensionless_unscaled
longAscNodes = 0.*u.rad
meanPerAno = 0.*u.rad


wf_params = {
    'mass1' : 36*u.solMass,
    'mass2' : 29*u.solMass,
    'deltaT' : deltaT,
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

gen = wfm.LALCompactBinaryCoalescenceGenerator(approximant)

hp_t, _ = wfm.GenerateTDWaveform(wf_params, gen)


hp_f_fine, _ = wfm.GenerateFDWaveform(wf_params, gen)

hp_f_coarse, _ = wfm.GenerateFDWaveform(wf_params | {'deltaF': 1.0 / (hp_t.size * hp_t.dx)}, gen)


#%% ---------- Testing transformation into one domain and back ----------
def test_ifft_fft_consistency():
    f_min_comp, f_max_comp = 20.0 * u.Hz, 512.0 * u.Hz  # Restrict to interesting region, elsewhere only values close to zero and thus numerical errors might occur

    hp_f_coarse_ifft_fft = td_to_fd_waveform(fd_to_td_waveform(hp_f_coarse))

    hp_f_coarse_cropped = hp_f_coarse.crop(start=f_min_comp, end=f_max_comp)
    hp_f_coarse_ifft_fft_cropped = hp_f_coarse_ifft_fft.crop(start=f_min_comp, end=f_max_comp)

    assert np.all(np.isclose(np.real(hp_f_coarse_cropped), np.real(hp_f_coarse_ifft_fft_cropped), atol=0.0, rtol=0.001))

    hp_f_fine_ifft_fft = td_to_fd_waveform(fd_to_td_waveform(hp_f_fine))

    hp_f_fine_cropped = hp_f_fine.crop(start=f_min_comp, end=f_max_comp)
    hp_f_fine_ifft_fft_cropped = hp_f_fine_ifft_fft.crop(start=f_min_comp, end=f_max_comp)

    assert np.all(np.isclose(np.real(hp_f_fine_cropped), np.real(hp_f_fine_ifft_fft_cropped), atol=0.0, rtol=0.001))


def test_fft_ifft_consistency():
    hp_t_fft_ifft = fd_to_td_waveform(td_to_fd_waveform(hp_t))

    t_min_comp, t_max_comp = max(hp_t.t0, hp_t_fft_ifft.t0), min(hp_t.times[-1], hp_t_fft_ifft.times[-1])  # hp_t_fft_ifft_fine is padded to be much longer
    # t_min_comp, t_max_comp = -0.5, 0.01  # hp_t_fft_ifft_fine is padded to be much longer

    hp_t_cropped = hp_t.crop(start=t_min_comp, end=t_max_comp)[2:]
    hp_t_fft_ifft_cropped = hp_t_fft_ifft.crop(start=t_min_comp, end=t_max_comp)

    assert np.all(np.isclose(hp_t_cropped, hp_t_fft_ifft_cropped, atol=1.2e-23, rtol=0.0))
    # Have to apply different kind of threshold here because coarse sampling is indeed very coarse

    hp_t_fft_ifft_fine = fd_to_td_waveform(td_to_fd_waveform(pad_to_get_target_df(hp_t, df=0.0625 * u.Hz)))

    t_min_comp, t_max_comp = max(hp_t.t0, hp_t_fft_ifft_fine.t0), min(hp_t.times[-1], hp_t_fft_ifft_fine.times[-1])  # hp_t_fft_ifft_fine is padded to be much longer
    
    hp_t_cropped = hp_t.crop(start=t_min_comp, end=t_max_comp)[1:]
    hp_t_fft_ifft_fine_cropped = hp_t_fft_ifft_fine.crop(start=t_min_comp, end=t_max_comp)[1:]
    # NOTE: for some reason, first sample is not equal. Thus excluded here

    assert np.all(np.isclose(hp_t_cropped, hp_t_fft_ifft_fine_cropped, atol=0.0, rtol=0.001))
    

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

    assert np.all(np.isclose(hp_f_coarse_cropped, hp_t_f_coarse_cropped, atol=0.0, rtol=0.05))

    # For a finer resolution, we have to pad signal
    hp_t_padded = pad_to_get_target_df(hp_t, df=hp_f_fine.df)
    hp_t_f_fine = td_to_fd_waveform(hp_t_padded)

    hp_f_fine_cropped = hp_f_fine.crop(start=f_min_comp, end=f_max_comp)
    hp_t_f_fine_cropped = hp_t_f_fine.crop(start=f_min_comp, end=f_max_comp)

    assert np.all(np.isclose(hp_f_fine_cropped, hp_t_f_fine_cropped, atol=0.0, rtol=0.05))



    f_min_comp, f_max_comp = 25.0 * u.Hz, 512.0 * u.Hz  # Restrict to interesting region, elsewhere only values close to zero and thus numerical errors might occur
    
    hp_t_f_coarse = td_to_fd_waveform(hp_t)

    hp_f_coarse_cropped = hp_f_coarse.crop(start=f_min_comp, end=f_max_comp)
    hp_t_f_coarse_cropped = hp_t_f_coarse.crop(start=f_min_comp, end=f_max_comp)

    assert np.all(np.isclose(hp_f_coarse_cropped, hp_t_f_coarse_cropped, atol=0.0, rtol=0.01))

    # For a finer resolution, we have to pad signal
    hp_t_padded = pad_to_get_target_df(hp_t, df=hp_f_fine.df)
    hp_t_f_fine = td_to_fd_waveform(hp_t_padded)

    hp_f_fine_cropped = hp_f_fine.crop(start=f_min_comp, end=f_max_comp)
    hp_t_f_fine_cropped = hp_t_f_fine.crop(start=f_min_comp, end=f_max_comp)

    assert np.all(np.isclose(hp_f_fine_cropped, hp_t_f_fine_cropped, atol=0.0, rtol=0.01))


#%% ---------- Testing helper functions ----------
    
def test_restrict_f_range():
    f_crop_low, f_crop_high = 20.0 * u.Hz, 30.0 * u.Hz

    hp_t_f_coarse = td_to_fd_waveform(hp_t)
    hp_t_f_coarse_restricted = restrict_f_range(hp_t_f_coarse, f_range=[f_crop_low, f_crop_high])
    
    f_min_comp, f_max_comp = f_crop_low + hp_t_f_coarse_restricted.df * 3.0 / 2.0, f_crop_high
    # Series.crop() uses floor also for lower limit, but restrict_f_range uses ceil. Have to correct for that
    hp_t_f_coarse_cropped = hp_t_f_coarse.crop(start=f_min_comp, end=f_max_comp)
    hp_t_f_coarse_rescricted_cropped = hp_t_f_coarse_restricted.crop(start=f_min_comp, end=f_max_comp)

    # f_min_comp, f_max_comp = 20.0 * u.Hz, 30.0 * u.Hz
    # hp_t_f_coarse_cropped = hp_t_f_coarse[(hp_t_f_coarse.frequencies > f_min_comp) & (hp_t_f_coarse.frequencies < f_max_comp)]
    # hp_t_f_coarse_rescricted_cropped = hp_t_f_coarse_restricted[(hp_t_f_coarse_restricted.frequencies > f_min_comp) & (hp_t_f_coarse_restricted.frequencies < f_max_comp)]

    assert np.all(np.isclose(hp_t_f_coarse_cropped, hp_t_f_coarse_rescricted_cropped, atol=0.0, rtol=0.001))

    # Also check that everything has been set to zero outside of f_range
    f_min_comp, f_max_comp = f_crop_low, f_crop_high + hp_t_f_coarse_restricted.df * 3.0 / 2.0
    hp_t_f_coarse_rescricted_cropped_2 = hp_t_f_coarse_restricted.crop(end=f_min_comp)
    hp_t_f_coarse_rescricted_cropped_3 = hp_t_f_coarse_restricted.crop(start=f_max_comp)

    assert np.all(np.isclose(0.0, hp_t_f_coarse_rescricted_cropped_2, atol=0.0, rtol=0.001))
    assert np.all(np.isclose(0.0, hp_t_f_coarse_rescricted_cropped_3, atol=0.0, rtol=0.001))