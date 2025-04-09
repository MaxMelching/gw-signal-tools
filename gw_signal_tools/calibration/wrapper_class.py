# -- Standard Lib Imports
from typing import Any

# -- Third Party Imports
import numpy as np
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from lalsimulation.gwsignal.core.waveform import GravitationalWaveGenerator
from scipy.interpolate import CubicSpline

# -- Local Package Imports
from gw_signal_tools.types import WFGen, FDWFGen, TDWFGen  # To run as py file
# from ..types import WFGen, FDWFGen, TDWFGen


__doc__ = """
Module for waveform generators that allow application of systematic
error corrections.
"""

__all__ = ('CalibrationWrapper', 'CalibrationGenerator', 'CalGravitationalWavePolarizations', )  # TODO: rename?


# class CalibrationGenerator(GravitationalWaveGenerator):
class CalibrationWrapper(GravitationalWaveGenerator):
# TODO: could WFModWrapper be a better name? Could then also rename calibrate_f_series to _apply_fd_mod
    def __init__(self, gen=None):
        """
        gen=None means that people intend to use parser capabilities only
        """
        if gen is None:
            # return None
            # -- Not setting attributes would be bad, thus use basic gen as default
            gen = GravitationalWaveGenerator()

        self.gen = gen

        # -- Initialize some important GravitationalWaveGenerator attributes
        self._generation_domain = None
        self._update_domains()

    @staticmethod
    def get_fd_amplitude(hf: FrequencySeries) -> FrequencySeries:
        """Extract the amplitude of a waveform on the whole frequency range."""
        return np.abs(hf)

    @staticmethod
    def get_fd_phase(hf: FrequencySeries) -> FrequencySeries:
        """Extract the phase of a waveform on the whole frequency range."""
        return np.unwrap(np.angle(hf))

    @staticmethod
    def recombine_to_fd_wf(ampl: FrequencySeries, phase: FrequencySeries) -> FrequencySeries:
        """Recombine a given amplitude and phase into a frequency domain waveform."""
        # TODO: assert compatible frequencies?
        return ampl * np.exp(1.j * phase)

    @staticmethod
    def calibrate_f_series(
        hf: FrequencySeries,
        modification: dict[str, Any] = None,
    ) -> FrequencySeries:
        """
        Idea: only this function has to be able to parse whatever is in
        modification. Through clever call structure, we can do
        calibration of hp, hc in signal frame and calibration of h in
        detector frame using this function.

        Note that modification here expects three keys: delta_amplitude,
        delta_phase, error_in_phase. The first two can be either Callables
        like CubicSplines that return interpolated errors or it must
        return the errors themselves (e.g. constant ones).
        """
        if modification is None:
            return hf
        else:
            # if isinstance(modification['delta_amplitude'], CubicSpline):
            if callable(modification['delta_amplitude']):  # Some interpolant, e.g. CubicSpline
                delta_amplitude = modification['delta_amplitude'](hf.frequencies)
            else:
                delta_amplitude = modification['delta_amplitude']

            # if isinstance(modification['delta_phase'], CubicSpline):
            if callable(modification['delta_phase']):  # Some interpolant, e.g. CubicSpline
                delta_phase = modification['delta_phase'](hf.frequencies)
            else:
                delta_phase = modification['delta_phase']

        hf_amp = CalibrationWrapper.get_fd_amplitude(hf)
        hf_phase = CalibrationWrapper.get_fd_phase(hf)

        # -- Applying the modifications
        hf_amp *= delta_amplitude

        if modification['error_in_phase'] == 'absolute':
            hf_phase += delta_phase
        if modification['error_in_phase'] == 'relative':
            hf_phase *= 1.0 + delta_phase
        else:
            raise ValueError('Invalid `\'modification_type\'` given.')

        hf_cal = CalibrationWrapper.recombine_to_fd_wf(hf_amp, hf_phase)

        return hf_cal

    # -- Note: keeping separate _get_<>_amplitude etc. makes subclassing
    # -- with adjustments to selected functionality much easier.
    @staticmethod
    def get_td_amplitude(ht: TimeSeries) -> TimeSeries:
        return NotImplemented

    @staticmethod
    def get_td_phase(ht: TimeSeries) -> TimeSeries:
        return NotImplemented

    @staticmethod
    def calibrate_t_series(
        ht: TimeSeries,
        modification: dict[str, Any] = None,
    ) -> TimeSeries:
        if modification is None:
            return ht

        return NotImplemented

    @staticmethod
    def parse_calib_kwds(
        **kwargs
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
                'f_low_wferror',
                'f_high_wferror',
                'n_nodes_wferror',
            ]:
                calib_params[key] = val
            else:
                wf_params[key] = val

        # -- Check if any calibration parameters are given at all
        if len(calib_params) == 0:
            return wf_params, None

        # -- Parse modifications. Here is more efficient than in _calibrate_series
        if calib_params['modification_type'] == 'cubic_spline':
            wf_nodal_points = calib_params['nodal_points']
            delta_amplitude_arr = calib_params['delta_amplitude']
            delta_phase_arr = calib_params['delta_phase']

            delta_amplitude_interp = CubicSpline(wf_nodal_points, delta_amplitude_arr)
            delta_phase_interp = CubicSpline(wf_nodal_points, delta_phase_arr)

            delta_amplitude = delta_amplitude_interp
            delta_phase = delta_phase_interp
        elif calib_params['modification_type'] == 'cubic_spline_nodes':
            f_lower = calib_params['f_low_wferror']
            f_high_wferror = calib_params['f_high_wferror']
            n_nodes_wferror = int(calib_params['n_nodes_wferror'])
            wf_nodal_points = np.logspace(
                np.log10(f_lower), np.log10(f_high_wferror), n_nodes_wferror
            )

            delta_amplitude_arr = np.hstack(
                [
                    calib_params['wferror_amplitude_{}'.format(i)]
                    for i in range(len(wf_nodal_points))
                ]
            )
            delta_phase_arr = np.hstack(
                [
                    calib_params['wferror_phase_{}'.format(i)]
                    for i in range(len(wf_nodal_points))
                ]
            )

            delta_amplitude_interp = CubicSpline(wf_nodal_points, delta_amplitude_arr)
            delta_phase_interp = CubicSpline(wf_nodal_points, delta_phase_arr)

            delta_amplitude = delta_amplitude_interp
            delta_phase = delta_phase_interp
        elif calib_params['modification_type'] == 'constant_shift':
            delta_amplitude = calib_params['delta_amplitude']
            delta_phase = calib_params['delta_phase']
        else:
            raise ValueError('Invalid `\'modification_type\'` given.')

        return wf_params, {
            'delta_amplitude': delta_amplitude,
            'delta_phase': delta_phase,
            'error_in_phase': calib_params['error_in_phase'],
        }

    def generate_fd_waveform(self, **kwargs):
        # wf_params, calib_params = self.parse_calib_kwds(kwargs=kwargs)
        wf_params, calib_params = self.parse_calib_kwds(**kwargs)
        wf = self.gen.generate_fd_waveform(**wf_params)

        if isinstance(wf, GravitationalWavePolarizations):
            return GravitationalWavePolarizations(
                self.calibrate_f_series(hf=wf[0], modification=calib_params),
                self.calibrate_f_series(hf=wf[1], modification=calib_params)
            )
        elif (
            isinstance(wf, tuple) and len(wf) == 2
            and isinstance(wf[0], FrequencySeries)
            and isinstance(wf[1], FrequencySeries)
        ):
            return (
                self.calibrate_f_series(hf=wf[0], modification=calib_params),
                self.calibrate_f_series(hf=wf[1], modification=calib_params)
            )
        elif isinstance(wf, FrequencySeries):
            return self.calibrate_f_series(hf=wf, modification=calib_params)
        else:
            # TODO: do this? Or try to calibrate anyway?
            raise ValueError(f'Output type of waveform generator is unknown.')

    def generate_td_waveform(self, **kwargs):
        # wf_params, calib_params = self.parse_calib_kwds(kwargs=kwargs)
        wf_params, calib_params = self.parse_calib_kwds(**kwargs)
        wf = self.gen.generate_td_waveform(**wf_params)
        return self.calibrate_t_series(hf=wf, modification=calib_params)

    @property
    def gen(self) -> GravitationalWaveGenerator | FDWFGen:
        """
        Generator that is wrapped around in this class, i.e. that the
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
        except AttributeError as err:
            # -- No use of another try except here to avoid chained error message
            if hasattr(self.gen, name):
                return self.gen.__getattribute__(name)
            else:
                raise err


# -- How to adjust GWPolarizations based on this
from lalsimulation.gwsignal.core.gw import GravitationalWavePolarizations
from typing import Union

class CalGravitationalWavePolarizations(GravitationalWavePolarizations):
    hp: Union[TimeSeries, FrequencySeries]
    hc: Union[TimeSeries, FrequencySeries]

    # _inherit_cal_gen = CalibrationGenerator  # Where we get functions to apply calibrations from
    _inherit_cal_gen = CalibrationWrapper  # Where we get functions to apply calibrations from

    # def __new__(cls, hp, hc):
    def __new__(cls, *args):
        # -- Doing it here allows for easier subclassing, where only
        # -- _inherit_cal_gen must be replaced in the subclass
        cls.get_fd_amplitude = cls._inherit_cal_gen.get_fd_amplitude
        cls.get_fd_phase = cls._inherit_cal_gen.get_fd_phase
        cls.recombine_to_fd_wf = cls._inherit_cal_gen.recombine_to_fd_wf
        cls.calibrate_f_series = cls._inherit_cal_gen.calibrate_f_series
        cls.get_td_amplitude = cls._inherit_cal_gen.get_td_amplitude
        cls.get_td_phase = cls._inherit_cal_gen.get_td_phase
        cls.calibrate_t_series = cls._inherit_cal_gen.calibrate_t_series

        # return super().__new__(cls, hp, hc)

        if len(args) == 1:
            args = args[0]  # Is already tuple of polarizations
            # TODO: should we allow for this? Or demand passing of both polarizations?
        return super().__new__(cls, *args)


    def strain(self, det, ra, dec, psi, tgps, **cal_kwargs):
        h = super().strain(det, ra, dec, psi, tgps)
        if self.domain() == 'time':
            return self.calibrate_t_series(ht=h, modification=cal_kwargs)
        elif self.domain() == 'frequency':
            return self.calibrate_f_series(hf=h, modification=cal_kwargs)
        else:
            raise ValueError('Cannot apply calibration to mixed polarizations.')
            # return ValueError('hp and hc must both be either TimeSeries or FrequencySeries')



# TODO: could also register custom approximant. And then use baseline_approximant
# keyword, like PyCBC version.
# -> via plugin structure
# -> we must adhere to https://git.ligo.org/waveforms/reviews/lalsuite/-/blob/32a73da89d3d3638d4a24a1df80aa946fae964c9/lalsimulation/python/lalsimulation/gwsignal/models/__init__.py

from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator

class CalibrationGenerator(GravitationalWaveGenerator):
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


    # -- Test operations with numpy array -> works
    # wf = wfm.GenerateFDWaveform(wf_params, gen)[0]
    # print(wf + wf.value)
    # print(wf * wf.value)


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
