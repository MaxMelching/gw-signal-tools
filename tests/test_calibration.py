import astropy.units as u
from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator
import lalsimulation.gwsignal.core.waveform as wfm

from gw_signal_tools.calibration import *
from gw_signal_tools.test_utils import assert_allequal_series



# f_min = 20.*u.Hz  # Cutoff frequency -> usual cutoff
# f_min = 25.*u.Hz  # Cutoff frequency for 50 Msun
f_min = 15.*u.Hz  # Cutoff frequency for 100 Msun
f_max = 1024. * u.Hz  # Cutoff from PSD
delta_f = 2**-6 * u.Hz
delta_t = 1./4096.*u.s
f_ref = f_min  # Frequency where we specify spins

wf_params = {
    'total_mass': 100.*u.Msun,
    'mass_ratio': 0.5*u.dimensionless_unscaled,
    'f22_start': f_min,
    'f_max': f_max,
    'deltaF': delta_f,
    'f22_ref': f_ref,
    'phi_ref': 0.*u.rad,
    'distance': 440.*u.Mpc,
    'inclination': 0.*u.rad,
    'eccentricity': 0.*u.dimensionless_unscaled,
    'longAscNodes': 0.*u.rad,
    'meanPerAno': 0.*u.rad,
    'condition': 0
}

import lalsimulation.gwsignal.core.parameter_conventions as pc
pc.default_dict.pop('mass1', None);
pc.default_dict.pop('mass2', None);



approximant = 'IMRPhenomXPHM'

gen = gwsignal_get_waveform_generator(approximant)
wrap_gen = CalibrationWrapper(gen)
cal_gen = CalibrationGenerator(approximant)


def test_getattr():
    gen.some_attr = 42
    cal_gen = CalibrationWrapper(gen)
    
    assert gen.some_attr == cal_gen.some_attr

    cal_gen.another_attr = 96
    cal_gen.another_attr

    try:
        gen.another_attr

        raise Exception('There should be an AttributeError.')
    except AttributeError:
        pass

    try:
        cal_gen.invalid_attr
        
        raise Exception('There should be an AttributeError.')
    except AttributeError as err:
        # -- To test error message
        assert "'CalibrationWrapper' object has no attribute 'invalid_attr'" in str(err)



def test_generate_fd_waveform_no_mod():
    wf_gen = gen.generate_fd_waveform(**wf_params)
    wf_wrap_gen = wrap_gen.generate_fd_waveform(**wf_params)
    wf_cal_gen = cal_gen.generate_fd_waveform(**wf_params)

    for i in [0, 1]:
        assert_allequal_series(wf_gen[i], wf_wrap_gen[i])
        assert_allequal_series(wf_gen[i], wf_cal_gen[i])


def test_generatefdwaveform_no_mod():
    wf_gen = wfm.GenerateFDWaveform(wf_params, gen)
    wf_wrap_gen = wfm.GenerateFDWaveform(wf_params, wrap_gen)
    wf_cal_gen = wfm.GenerateFDWaveform(wf_params, cal_gen)

    for i in [0, 1]:
        assert_allequal_series(wf_gen[i], wf_wrap_gen[i])
        assert_allequal_series(wf_gen[i], wf_cal_gen[i])


def test_domain():
    assert gen.domain == cal_gen.domain == wrap_gen.domain


def test_metadata():
    for attr in [
        '_implemented_domain',
        '_generation_domain',
    ]:
        assert getattr(gen, attr) == getattr(wrap_gen, attr)
        assert getattr(gen, attr) == getattr(cal_gen, attr)


def test_cal_polarizations_init():
    CalGravitationalWavePolarizations(*wfm.GenerateFDWaveform(wf_params, cal_gen))
    CalGravitationalWavePolarizations(wfm.GenerateFDWaveform(wf_params, cal_gen))


# def test_plugin():
#     from lalsimulation.gwsignal.models import list_models_plugins

#     print(list_models_plugins())

#     # gwsignal_get_waveform_generator('wferror', baseline_approximant=appr)
#     gwsignal_get_waveform_generator('wferror', plugin=True, approximant=appr)
#     # gwsignal_get_waveform_generator('wferrors')
# TODO: activate once this is in lalsimulation
