# -- Standard Lib Imports
from typing import Any

# -- Third Party Imports
import numpy as np
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from lalsimulation.gwsignal.core.waveform import GravitationalWaveGenerator

# -- Local Package Imports
from gw_signal_tools.types import WFGen, FDWFGen, TDWFGen  # To run as py file
# from ..types import WFGen, FDWFGen, TDWFGen


__doc__ = """
Module for waveform generators that allow application of systematic
error corrections.
"""

__all__ = ('CalibrationWrapper', 'CalibrationGenerator', )  # TODO: rename?


# class CalibrationGenerator(GravitationalWaveGenerator):
class CalibrationWrapper(GravitationalWaveGenerator):
# TODO: could WFModWrapper be a better name? Could then also rename _calibrate_f_series to _apply_fd_mod
    def __init__(self, gen):
        self.gen = gen

        # -- Initialize some important GravitationalWaveGenerator attributes
        self._generation_domain = None
        self._update_domains()
    # def __new__(cls, gen):
    #     cls.gen = gen

    #     # -- Initialize some important GravitationalWaveGenerator attributes
    #     cls._generation_domain = None
    #     cls._update_domains()

    #     return cls

    @staticmethod
    def _get_fd_amplitude(hf: FrequencySeries) -> FrequencySeries:
        """Extract the amplitude of a waveform on the whole frequency range."""
        return np.abs(hf)

    @staticmethod
    def _get_fd_phase(hf: FrequencySeries) -> FrequencySeries:
        """Extract the phase of a waveform on the whole frequency range."""
        return np.unwrap(np.angle(hf))

    @staticmethod
    def _recombine_to_fd_wf(ampl: FrequencySeries, phase: FrequencySeries) -> FrequencySeries:
        """Recombine a given amplitude and phase into a frequency domain waveform."""
        # TODO: assert compatible frequencies?
        return ampl * np.exp(1.j * phase)

    def _calibrate_f_series(
        self,
        hf: FrequencySeries,
        modification: dict[str, Any] = None,
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

        hf_cal = self._recombine_to_fd_wf(hf_amp, hf_phase)

        return hf_cal

    # -- Note: keeping separate _get_<>_amplitude etc. makes subclassing
    # -- with adjustments to selected functionality much easier.
    @staticmethod
    def _get_td_amplitude(ht: TimeSeries) -> TimeSeries:
        return NotImplemented

    @staticmethod
    def _get_td_phase(hf: TimeSeries) -> TimeSeries:
        return NotImplemented

    def _calibrate_t_series(
        self,
        ht: TimeSeries,
        modification: dict[str, Any] = None,
    ) -> TimeSeries:
        if modification is None:
            return ht

        return NotImplemented

    def _extract_calib_kwds(
        self, **kwargs
    ) -> tuple[dict[str, u.Quantity], dict[str, u.Quantity]]:
        """Helper function to separate waveform arguments from systematics arguments."""
        wf_params = {}
        calib_params = {}

        for key, val in kwargs.items():
            if key in [
                'modification_type',
                'error_in_phase',
                'delta_amplitude',
                'delta_phase',
                'nodal_points',
                'config',  # TODO: do this? And try to use bilby parser for example?
            ]:
                calib_params[key] = val
            else:
                wf_params[key] = val

        # TODO: potentially already apply checks whether given calib_params make sense?

        return wf_params, calib_params

    def generate_fd_waveform(self, **kwargs):
        # wf_params, calib_params = self._extract_calib_kwds(kwargs=kwargs)
        wf_params, calib_params = self._extract_calib_kwds(**kwargs)
        wf = self.gen.generate_fd_waveform(**wf_params)
        return self._calibrate_f_series(hf=wf, modification=calib_params)

    def generate_td_waveform(self, **kwargs):
        # wf_params, calib_params = self._extract_calib_kwds(kwargs=kwargs)
        wf_params, calib_params = self._extract_calib_kwds(**kwargs)
        wf = self.gen.generate_td_waveform(**wf_params)
        return self._calibrate_t_series(hf=wf, modification=calib_params)

    @property
    def gen(self) -> FDWFGen:
        """
        Generator that is wrapper around in this class, i.e. that the
        calibration is applied to.
        """
        return self._gen

    @gen.setter
    def gen(self, value: FDWFGen) -> None:
        self._gen = value

        # -- Make sure domains are still correct and match the ones of
        # -- self.gen. Since these are meant to be set in the generator,
        # -- we can simply use update_domains.
        self._update_domains = value._update_domains

    @property
    def metadata(self):
        return self.gen.metadata | {
            'implemented_domain': 'freq',
            'generation_domain': 'freq',
        }
        # TODO: remove this additional stuff once TD calibration becomes available

    def __getattr__(self, name) -> Any:
        try:
            return self.__getattribute__(name)
        except AttributeError:
            # -- Maybe gen had this defined. If not, throw error
            return self.gen.__getattribute__(name)


# -- How to adjust GWPolarizations based on this
from lalsimulation.gwsignal.core.gw import GravitationalWavePolarizations
from typing import Union

class CalGravitationalWavePolarizations(GravitationalWavePolarizations):
    hp: Union[TimeSeries, FrequencySeries]
    hc: Union[TimeSeries, FrequencySeries]

    # _inherit_cal_gen = CalibrationGenerator  # Where we get functions to apply calibrations from
    _inherit_cal_gen = CalibrationWrapper  # Where we get functions to apply calibrations from
    
    # _get_fd_amplitude = _inherit_cal_gen._get_fd_amplitude
    # _get_fd_phase = _inherit_cal_gen._get_fd_phase
    # _recombine_to_fd_wf = _inherit_cal_gen._recombine_to_fd_wf
    # _calibrate_f_series = _inherit_cal_gen._calibrate_f_series
    # _get_td_amplitude = _inherit_cal_gen._get_td_amplitude
    # _get_td_phase = _inherit_cal_gen._get_td_phase
    # _calibrate_t_series = _inherit_cal_gen._calibrate_t_series

    # def __new__(cls, hp, hc):
    def __new__(cls, *args):
        # -- Doing it here allows for easier subclassing, where only
        # -- _inherit_cal_gen must be replaced in the subclass
        cls._get_fd_amplitude = cls._inherit_cal_gen._get_fd_amplitude
        cls._get_fd_phase = cls._inherit_cal_gen._get_fd_phase
        cls._recombine_to_fd_wf = cls._inherit_cal_gen._recombine_to_fd_wf
        cls._calibrate_f_series = cls._inherit_cal_gen._calibrate_f_series
        cls._get_td_amplitude = cls._inherit_cal_gen._get_td_amplitude
        cls._get_td_phase = cls._inherit_cal_gen._get_td_phase
        cls._calibrate_t_series = cls._inherit_cal_gen._calibrate_t_series

        # return super().__new__(cls, hp, hc)

        if len(args) == 1:
            args = args[0]  # Is already tuple of polarizations
            # TODO: should we allow for this? Or demand passing of both polarizations?
        return super().__new__(cls, *args)


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

class CalibrationGenerator(GravitationalWaveGenerator):
# class CalibrationGenerator:
    # # def __init__(self, **kwargs):
    # def __new__(cls, *args, **kwargs):
    #     # super().__init__()
    #     # _appr = kwargs.pop('baseline_approximant')
    #     # gen = gwsignal_get_waveform_generator(_appr, **kwargs)
    #     if not isinstance(args[0], str):
    #         args = [kwargs.pop('baseline_approximant')].append(args)
    #     gen = gwsignal_get_waveform_generator(*args, **kwargs)
    #     return CalibrationWrapper(gen)
    #     # return CalibrationWrapper.__new__(gen)

    # def __new__(cls, baseline_approximant, *args, **kwargs):
    #     gen = gwsignal_get_waveform_generator(baseline_approximant, *args, **kwargs)
    #     return CalibrationWrapper(gen)
    def __new__(cls, approximant, *args, **kwargs):
        gen = gwsignal_get_waveform_generator(approximant, *args, **kwargs)
        return CalibrationWrapper(gen)


if __name__ == '__main__':
    appr = 'IMRPhenomXPHM'

    gen = gwsignal_get_waveform_generator(appr)
    cal_gen = CalibrationWrapper(gen)
    test_cal_gen = CalibrationGenerator(appr)


    import astropy.units as u
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

    print(gen.domain, cal_gen.domain)

    gen.some_attr = 42

    cal_gen = CalibrationWrapper(gen)
    print(gen.some_attr, cal_gen.some_attr)

    cal_gen.another_attr = 96
    print(cal_gen.another_attr)
    # print(cal_gen.invalid_attr)  # To test error message

    print(cal_gen.gen)

    print(gen.generate_fd_waveform(**wf_params))
    print(cal_gen.generate_fd_waveform(**wf_params))
    print(test_cal_gen.generate_fd_waveform(**wf_params))

    import lalsimulation.gwsignal.core.waveform as wfm
    print(wfm.GenerateFDWaveform(wf_params, gen))
    print(wfm.GenerateFDWaveform(wf_params, cal_gen))
    print(wfm.GenerateFDWaveform(wf_params, test_cal_gen))

    CalGravitationalWavePolarizations(*wfm.GenerateFDWaveform(wf_params, test_cal_gen))
    CalGravitationalWavePolarizations(wfm.GenerateFDWaveform(wf_params, test_cal_gen))


    # -- Demonstrate how subclassing can work by provoking error
    class WrongWrapper(CalGravitationalWavePolarizations):
        _inherit_cal_gen = CalibrationGenerator

    # WrongWrapper(*wfm.GenerateFDWaveform(wf_params, test_cal_gen))  # Throws error, as it should, nice!!!


    assert False

    # from importlib.metadata import entry_points
    # for model in entry_points(group='gwsignal_models'):
    #     print(model.name)
    # print('These are all models')

    from lalsimulation.gwsignal.models import list_models_plugins

    print(list_models_plugins())

    # gwsignal_get_waveform_generator('wferror', baseline_approximant=appr)
    gwsignal_get_waveform_generator('wferror', plugin=True, approximant=appr)
    # gwsignal_get_waveform_generator('wferrors')

    from gw_signal_tools.waveform import inner_product
