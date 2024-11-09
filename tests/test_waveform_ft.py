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
    pad_to_target_df, restrict_f_range,
    get_signal_at_target_df, get_signal_at_target_frequs,
    get_strain, fill_f_range, get_wf_generator
)
from gw_signal_tools.waveform.ft import (
    td_to_fd_waveform, fd_to_td_waveform,
)
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
    'condition': 1
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


# TODO: remove tests here altogether? And just rely on lalsimulation ones?


#%% -- Testing transformation into one domain and back ------------------------
# def test_fft_ifft_consistency():
#     hp_t_fft_ifft = fd_to_td_waveform(td_to_fd_waveform(hp_t))

#     assert_allclose_quantity(hp_t.times, hp_t_fft_ifft.times, atol=0.5*hp_t.dt.value, rtol=0.0)
#     h_strength = hp_t.max().value
#     assert_allclose_quantity(hp_t, hp_t_fft_ifft, atol=h_strength*1e-15, rtol=0.0)
#     # -- Reason for non-zero atol are numerical errors for parts where
#     # -- hp_t is zero. We show that by demonstrating they are on scale
#     # -- of numerical accuracy (1e-15) of strain amplitude (represented
#     # -- by maximum).


#     hp_t_padded = pad_to_target_df(hp_t, df=0.0625*u.Hz)
#     hp_t_fft_ifft_fine = fd_to_td_waveform(td_to_fd_waveform(hp_t_padded))

#     assert_allclose_quantity(hp_t_padded.times, hp_t_fft_ifft_fine.times, atol=0.5*hp_t.dt.value, rtol=0.0)
#     h_strength = hp_t_padded.max().value
#     assert_allclose_quantity(hp_t_padded, hp_t_fft_ifft_fine, atol=h_strength*1e-15, rtol=0.0)


# def test_ifft_fft_consistency():
#     hp_f_coarse_ifft_fft = td_to_fd_waveform(fd_to_td_waveform(hp_f_coarse))

#     assert_allclose_quantity(hp_f_coarse.frequencies, hp_f_coarse_ifft_fft.frequencies, atol=0.0, rtol=0.0)
#     h_strength = hp_f_coarse.max().value
#     assert_allclose_quantity(hp_f_coarse, hp_f_coarse_ifft_fft, atol=h_strength*1e-15, rtol=0.0)


#     hp_f_fine_ifft_fft = td_to_fd_waveform(fd_to_td_waveform(hp_f_fine))

#     assert_allclose_quantity(hp_f_fine.frequencies, hp_f_fine_ifft_fft.frequencies, atol=0.0, rtol=1e-10)
#     h_strength = hp_f_fine.max().value
#     assert_allclose_quantity(hp_f_fine, hp_f_fine_ifft_fft, atol=h_strength*1e-15, rtol=0.0)


@pytest.mark.parametrize('ht', [hp_t, pad_to_target_df(hp_t, df=hp_f_fine.df)])
def test_fft_ifft_consistency(ht):
    ht_fft_ifft = fd_to_td_waveform(td_to_fd_waveform(ht))

    assert_allclose_quantity(ht.times, ht_fft_ifft.times, atol=0.5*ht.dt.value, rtol=0.0)
    h_strength = ht.max().value
    assert_allclose_quantity(ht, ht_fft_ifft, atol=h_strength*1e-15, rtol=0.0)
    # -- Reason for non-zero atol are numerical errors for parts where
    # -- ht is zero. We show that by demonstrating they are on scale of
    # -- numerical accuracy (1e-15) of strain amplitude (represented by
    # -- maximum of ht).


@pytest.mark.parametrize('hf', [hp_f_coarse, hp_f_fine])
def test_ifft_fft_consistency(hf):
    hf_ifft_fft = td_to_fd_waveform(fd_to_td_waveform(hf))

    assert_allclose_quantity(hf.frequencies, hf_ifft_fft.frequencies, atol=0.0, rtol=0.0)
    h_strength = hf.max().value
    assert_allclose_quantity(hf, hf_ifft_fft, atol=h_strength*1e-15, rtol=0.0)
    # -- Reason for non-zero atol are numerical errors for parts where
    # -- hf is zero. We show that by demonstrating they are on scale of
    # -- numerical accuracy (1e-15) of strain amplitude (represented by
    # -- maximum of ht).


#%% -- Testing transformations with generated signals from different domain ---
import matplotlib.pyplot as plt
from gw_signal_tools.waveform.ft import zero_pad

def test_fd_td_consistency():
    # hp_t_f_coarse = td_to_fd_waveform(hp_t)

    # plt.plot(hp_f_coarse, 'x-')
    # plt.plot(hp_t_f_coarse, '+--')
    # plt.show()
    # # plt.close()

    # assert_allclose_quantity(
    #     hp_f_coarse.frequencies,
    #     hp_t_f_coarse.frequencies,
    #     atol=0.0,
    #     rtol=0.0
    # )
    # assert_allclose_quantity(
    #     # (hp_f_coarse - hp_t_f_coarse).crop(start=f_min, end=256*u.Hz),
    #     # 0.*u.strain*u.s,
    #     hp_f_coarse.crop(start=f_min, end=f_max),
    #     hp_t_f_coarse.crop(start=f_min, end=f_max),
    #     atol=0.0,
    #     rtol=1e-2
    # )

    hp_t_f_fine = td_to_fd_waveform(zero_pad(hp_t, df=hp_f_fine.df))

    assert_allclose_quantity(
        hp_f_fine.frequencies,
        hp_t_f_fine.frequencies,
        atol=0.0,
        rtol=0.0
    )
    assert_allclose_quantity(
        hp_f_fine.crop(start=f_min, end=f_max),
        hp_t_f_fine.crop(start=f_min, end=f_max),
        atol=1e-30,
        rtol=6e-3
    )
    # -- Restrict to f_min due to conditioning.
    # -- atol for numerical errors at large frequencies (basically zero
    # -- there, thus we do not care too much)
    # -- rtol quantifies "real" deviations

def test_fd_td_consistency():
    hp_t_padded = zero_pad(hp_t, df=hp_f_fine.df)
    hp_f_t_fine = fd_to_td_waveform(hp_f_fine)
    # hp_f_t_fine.t0 = -hp_f_t_fine.duration
    hp_f_t_fine.t0 = hp_t_padded.t0

    plt.plot(hp_t_padded, 'x-')
    plt.plot(hp_f_t_fine, '+--')
    plt.show()

    assert_allclose_quantity(
        hp_t_padded,
        hp_f_t_fine,
        atol=0.0,
        rtol=0.0
    )

# def test_fd_td_consistency():
#     # NOTE: we have to apply different thresholds for certain frequency regions here.
#     # For f_min_comp close to f_min from the parameter dictionary above, the threshold
#     # has to be chosen a bit higher than the usual 1%. Here, it comes into play
#     # that tapering is applied to TDWaveform that we do FFT of, while this is not
#     # done for FDWaveform. This causes certain differences in the Fourier components

#     f_min_comp, f_max_comp = 20.0 * u.Hz, 512.0 * u.Hz  # Restrict to interesting region, elsewhere only values close to zero and thus numerical errors might occur
    
#     hp_t_f_coarse = td_to_fd_waveform(hp_t)

#     hp_f_coarse_cropped = hp_f_coarse.crop(start=f_min_comp, end=f_max_comp)
#     hp_t_f_coarse_cropped = hp_t_f_coarse.crop(start=f_min_comp, end=f_max_comp)

#     assert_allclose_quantity(hp_f_coarse_cropped.frequencies, hp_t_f_coarse_cropped.frequencies, atol=0.0, rtol=1e-14)
#     # assert_quantity_equal(hp_f_coarse_cropped.frequencies, hp_t_f_coarse_cropped.frequencies)
#     assert_allclose_quantity(hp_f_coarse_cropped, hp_t_f_coarse_cropped, atol=0.0, rtol=0.05)
#     # assert_quantity_equal(hp_f_coarse_cropped, hp_t_f_coarse_cropped)

#     # For a finer resolution, we have to pad signal
#     hp_t_padded = pad_to_target_df(hp_t, df=hp_f_fine.df)
#     hp_t_f_fine = td_to_fd_waveform(hp_t_padded)

#     hp_f_fine_cropped = hp_f_fine.crop(start=f_min_comp, end=f_max_comp)
#     hp_t_f_fine_cropped = hp_t_f_fine.crop(start=f_min_comp, end=f_max_comp)

#     # assert_allclose_quantity(hp_f_fine_cropped.frequencies, hp_t_f_fine_cropped.frequencies, atol=0.0, rtol=0.05)
#     assert_quantity_equal(hp_f_fine_cropped.frequencies, hp_t_f_fine_cropped.frequencies)
#     assert_allclose_quantity(hp_f_fine_cropped, hp_t_f_fine_cropped, atol=0.0, rtol=0.01)
#     # assert_quantity_equal(hp_f_fine_cropped, hp_t_f_fine_cropped)


#     f_min_comp, f_max_comp = 25.0 * u.Hz, 512.0 * u.Hz  # Restrict to interesting region, elsewhere only values close to zero and thus numerical errors might occur
    
#     hp_t_f_coarse = td_to_fd_waveform(hp_t)

#     hp_f_coarse_cropped = hp_f_coarse.crop(start=f_min_comp, end=f_max_comp)
#     hp_t_f_coarse_cropped = hp_t_f_coarse.crop(start=f_min_comp, end=f_max_comp)

#     assert_allclose(hp_f_coarse_cropped, hp_t_f_coarse_cropped, atol=0.0, rtol=0.01)

#     # For a finer resolution, we have to pad signal
#     hp_t_padded = pad_to_target_df(hp_t, df=hp_f_fine.df)
#     hp_t_f_fine = td_to_fd_waveform(hp_t_padded)

#     hp_f_fine_cropped = hp_f_fine.crop(start=f_min_comp, end=f_max_comp)
#     hp_t_f_fine_cropped = hp_t_f_fine.crop(start=f_min_comp, end=f_max_comp)

#     assert_allclose(hp_f_fine_cropped, hp_t_f_fine_cropped, atol=0.0, rtol=0.01)


# #%% -- Verification of complex transformation ---------------------------------
# def test_complex_fft_ifft_consistency():
#     h_symm = td_to_fd_waveform(pad_to_target_df(hp_t, df=hp_f_fine.df) + 0.j)
#     # Padding hp_t to make sure resolution is sufficient and to avoid
#     # wrap-around due to insufficient length
#     h_symm_t = fd_to_td_waveform(h_symm)

#     t_min, t_max = max(hp_t.times[0], h_symm_t.times[0]), min(hp_t.times[-1], h_symm_t.times[-1])
#     assert_allclose(hp_t.crop(t_min, t_max).times, h_symm_t.crop(t_min, t_max).times, atol=5e-12, rtol=0)
#     assert_allclose(hp_t.crop(t_min, t_max).value, h_symm_t.crop(t_min, t_max).value, atol=5e-27, rtol=0)


# def test_complex_ifft_fft_consistency():
#     h_symm = FrequencySeries(
#         np.flip(np.conjugate(hp_f_fine+0.j)[1:]),
#         f0=-hp_f_fine.frequencies[-1],
#         df=hp_f_fine.df
#     ).append(hp_f_fine+0.j, inplace=True)

#     h_symm_t = fd_to_td_waveform(h_symm)

#     # t_min, t_max = max(hp_t.times[0], h_symm_t.times[0]), min(hp_t.times[-1], h_symm_t.times[-1])
    
#     h_symm_f = td_to_fd_waveform(h_symm_t)

#     # assert_allclose_series(h_symm, h_symm_f, atol=0., rtol=0.)
#     assert_allclose_quantity(h_symm.frequencies, h_symm_f.frequencies, atol=3e-11, rtol=0.)
#     assert_allclose_quantity(h_symm.value, h_symm_f.value, atol=4e-33, rtol=0.)
#     # Some residual imaginary part is there, but of negligible amplitude
#     # f_min_comp, f_max_comp = 25.0 * u.Hz, 512.0 * u.Hz  # Restrict to interesting region, elsewhere only values close to zero and thus numerical errors might occur
#     # assert_allclose_series(h_symm.crop(start=f_min_comp, end=f_max_comp), h_symm_f.crop(start=f_min_comp, end=f_max_comp), atol=0., rtol=0.)


# def test_ifft_frequency_checking():
#     with pytest.raises(ValueError, match='Signal starts at positive frequency'):
#         fd_to_td_waveform(hp_f_fine[1:])
    
#     h_symm = FrequencySeries(
#         np.flip(np.conjugate(hp_f_fine+0.j)[1:]),
#         f0=-hp_f_fine.frequencies[-1],
#         df=hp_f_fine.df
#     ).append(hp_f_fine+0.j, inplace=True)

#     with pytest.raises(ValueError, match='`signal` does not have correct format for ifft'):
#         fd_to_td_waveform(h_symm[1:])


# def test_complex_and_real_fft_consistency():
#     h_symm = td_to_fd_waveform(pad_to_target_df(hp_t, df=hp_f_fine.df) + 0.j)
#     # Padding hp_t to make sure resolution is sufficient and to avoid
#     # wrap-around due to insufficient length
#     h_neg = h_symm[h_symm.frequencies < 0.*u.Hz][1:]  # Filter somehow includes 0 as well, thus excluding first one
#     h_pos = h_symm[h_symm.frequencies > 0.*u.Hz]

#     assert_allclose(-h_neg.frequencies[::-1], h_pos.frequencies, atol=0, rtol=0)
#     assert_allclose(h_neg[::-1].real, h_pos.real, atol=3e-38, rtol=0)
#     assert_allclose(-h_neg[::-1].imag, h_pos.imag, atol=3e-28, rtol=0)
#     # Note: as https://en.wikipedia.org/wiki/Fourier_transform#Conjugation
#     # shows, real signals have the following property: real part of Fourier
#     # spectrum is symmetric around f=0, while imaginary part is antisymmetric.


# def test_complex_and_real_ifft_consistency():
#     h_symm = FrequencySeries(
#         np.flip(np.conjugate(hp_f_fine)[1:]),
#         f0=-hp_f_fine.frequencies[-1],
#         df=hp_f_fine.df
#     ).append(hp_f_fine, inplace=True)

#     h_symm_t = fd_to_td_waveform(h_symm)
#     h_t = fd_to_td_waveform(hp_f_fine)

#     assert_allclose_quantity(h_t.times, h_symm_t[:-1].times, atol=5e-4, rtol=0)  # Deviations on scale of dt are ok
#     assert_allclose_quantity(h_t, h_symm_t[:-1], atol=2e-24, rtol=0)  # Context: peaks are at roughly 1e-21
