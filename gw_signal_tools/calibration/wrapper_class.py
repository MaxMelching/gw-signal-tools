from lalsimulation.gwsignal.core.waveform import GravitationalWaveGenerator
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
import numpy as np


# class CalibrationGenerator(GravitationalWaveGenerator):
class CalibrationWrapper(GravitationalWaveGenerator):
    # -- Inherit from most basic class
    # -- -> intended usage: call CalibrationGenerator(phenom_gen)
    # def __init__(self, *args, **kwargs):
    #     # self.mod_type = kwargs.pop('mod_type')
    #     ...
    #     # super().__init__(*args, **kwargs)
    #     super().__init__()

    # TODO: this is not the place to extract mod_type etc, should be
    # done in generate_f(t)d_waveform

    # def __new__(cls, *args, **kwargs):
    #     return super().__new__()

    # def __init__(self, gen):
    #     # gen.__init__()
    #     self = gen

    # def __new__(cls, gen):
    #     return gen.__new__()

    # def __new__(cls, gen):
    #     return gen.__new__(cls)

    def __init__(self, gen):
        self.gen = gen
        # self.__getattribute__ = self.gen.__getattribute__
        # self.__getattr__ = self.gen.__getattr__
        # self.__setattr__ = self.gen.__setattr__

    @staticmethod
    def _get_fd_amplitude(hf: FrequencySeries) -> FrequencySeries:
        return np.abs(hf)

    @staticmethod
    def _get_fd_phase(hf: FrequencySeries) -> FrequencySeries:
        return np.unwrap(np.angle(hf))

    def _calibrate_f_series(
        self,
        hf: FrequencySeries,
        modification=None,  # TODO: make dict, config file, or even custom class?
    ) -> FrequencySeries:
        """
        Idea: only this function has to be able to parse whatever is in
        modification. Through clever call structure, we can do
        calibration of hp, hc in signal frame and calibration of h in
        detector frame using this function.
        """
        if modification is None:
            return hf

        hf_amp = self._get_fd_amplitude(hf)
        hf_phase = self._get_fd_phase(hf)

        # TODO: apply calibration here

        hf_cal = hf_amp*np.exp(1.j*hf_phase)

        return hf_cal

    @staticmethod
    def _get_td_amplitude(ht: TimeSeries) -> TimeSeries:
        return NotImplemented

    @staticmethod
    def _get_td_phase(hf: TimeSeries) -> TimeSeries:
        return NotImplemented

    def _calibrate_t_series(
        self,
        ht: TimeSeries,
        modification=None,
    ) -> TimeSeries:
        if modification is None:
            return ht

        return NotImplemented

    def generate_fd_waveform(self, **kwargs):
        # TODO: apply calibration
        # return super().generate_fd_waveform(**kwargs)
        return self.gen.generate_fd_waveform(**kwargs)

    def generate_td_waveform(self, **kwargs):
        # TODO: apply calibration
        # return super().generate_td_waveform(**kwargs)
        return self.gen.generate_td_waveform(**kwargs)

    @property
    def gen(self):
        return self._gen

    @gen.setter
    def gen(self, value):
        self._gen = value

    def __getattr__(self, name):
        if name == 'gen':
            return self._gen
        else:
            try:
                return self.__getattribute__(name)
            except AttributeError as e:
                # -- Maybe gen had this defined. If not, throw error
                return self.gen.__getattribute__(name)
                # -- Thought following might be required, but is not
                # try:
                #     return self.gen.__getattribute__(name)
                # except AttributeError as e:
                #     raise AttributeError(
                #         # f'Class ``{self.__class__.__name__}`` has no attribute {name}.'
                #         f'``{self.__class__.__name__}`` has no attribute {name}.'
                #     )

    # def __setattr__(self, name, value):
    #     if name == 'gen':
    #     # if name in ['gen', '_gen']:
    #         self._gen = value
    #     else:
    #         return self._gen.__setattr__(name, value)

    # We do not need that, right? We can simply set on this here

    # def __getattribute__(self, name):
    #     if name == 'gen':
    #         return self._gen
    #     else:
    #         return self._gen.__getattribute__(name)



# -- How to adjust GWPolarizations based on this
from lalsimulation.gwsignal.core.gw import GravitationalWavePolarizations

class CalGravitationalWavePolarizations(GravitationalWavePolarizations):
    # _inherit_cal_gen = CalibrationGenerator  # Where we get functions to apply calibrations from
    _inherit_cal_gen = CalibrationWrapper  # Where we get functions to apply calibrations from

    _get_fd_amplitude = _inherit_cal_gen._get_fd_amplitude
    _get_fd_phase = _inherit_cal_gen._get_fd_phase
    _calibrate_f_series = _inherit_cal_gen._calibrate_f_series
    _get_td_amplitude = _inherit_cal_gen._get_td_amplitude
    _get_td_phase = _inherit_cal_gen._get_td_phase
    _calibrate_t_series = _inherit_cal_gen._calibrate_t_series

    def strain(self, det, ra, dec, psi, tgps, **cal_kwargs):
        h = super().strain(det, ra, dec, psi, tgps)
        if self.domain() == 'time':
            return self._calibrate_t_series(ht=h, modification=cal_kwargs)
        elif self.domain() == 'frequency':
            return self._calibrate_f_series(hf=h, modification=cal_kwargs)
        else:
            raise ValueError('Cannot apply calibration to mixed polarizations.')
            # return ValueError('hp and hc must both be either TimeSeries or FrequencySeries')



# TODO: could also register custom approximant. And then use baseline_approximant
# keyword, like PyCBC version.
# -> via plugin structure
# -> we must adhere to https://git.ligo.org/waveforms/reviews/lalsuite/-/blob/32a73da89d3d3638d4a24a1df80aa946fae964c9/lalsimulation/python/lalsimulation/gwsignal/models/__init__.py

from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator

# class CalibrationGenerator(GravitationalWaveGenerator):
class CalibrationGenerator:
    # def __init__(self, **kwargs):
    def __new__(cls, *args, **kwargs):
        # super().__init__()
        # _appr = kwargs.pop('baseline_approximant')
        # gen = gwsignal_get_waveform_generator(_appr, **kwargs)
        if not isinstance(args[0], str):
            args = [kwargs.pop('baseline_approximant')].append(args)
        gen = gwsignal_get_waveform_generator(*args, **kwargs)
        return CalibrationWrapper(gen)
        # return CalibrationWrapper.__new__(gen)


if __name__ == '__main__':
    appr = 'IMRPhenomXPHM'
    test = CalibrationGenerator(appr)

    gen = gwsignal_get_waveform_generator(appr)
    cal_gen = CalibrationWrapper(gen)


    import astropy.units as u
    # f_min = 20.*u.Hz  # Cutoff frequency -> usual cutoff
    # f_min = 25.*u.Hz  # Cutoff frequency for 50 Msun
    f_min = 15.*u.Hz  # Cutoff frequency for 100 Msun
    f_max = 1024. * u.Hz  # Cutoff from PSD
    delta_f = 2**-6 * u.Hz
    delta_t = 1./4096.*u.s
    f_ref = f_min  # Frequency where we specify spins
    # f_ref = 0.8*f_min  # Frequency where we specify spins
    # f_ref = 1.2*f_min  # Frequency where we specify spins


    wf_params = {
        # 'total_mass': 50.*u.Msun,
        'total_mass': 100.*u.Msun,  # To get into range of NRSur validity (above 60)
        # 'mass_ratio': 0.05*u.dimensionless_unscaled,
        # 'mass_ratio': 0.15*u.dimensionless_unscaled,
        'mass_ratio': 0.5*u.dimensionless_unscaled,
        'f22_start': f_min,
        'f_max': f_max,
        'deltaF': delta_f,
        'f22_ref': f_ref,
        'phi_ref': 0.*u.rad,
        # 'distance': 1.*u.Mpc,
        'distance': 440.*u.Mpc,  # As expected, systematic error is independent of SNR and thus amplitude given by D_L
        'inclination': 0.*u.rad,
        'eccentricity': 0.*u.dimensionless_unscaled,
        'longAscNodes': 0.*u.rad,
        'meanPerAno': 0.*u.rad,
        'condition': 0
    }

    import lalsimulation.gwsignal.core.parameter_conventions as pc
    pc.default_dict.pop('mass1', None);
    pc.default_dict.pop('mass2', None);

    print(gen.domain, cal_gen.domain)

    gen.some_attr = 42

    cal_gen = CalibrationWrapper(gen)
    print(gen.some_attr, cal_gen.some_attr)

    cal_gen.another_attr = 96
    print(cal_gen.another_attr)
    print(cal_gen.invalid_attr)  # To test error message

    # print(gen.generate_fd_waveform(**wf_params))
    # print(cal_gen.generate_fd_waveform(**wf_params))
