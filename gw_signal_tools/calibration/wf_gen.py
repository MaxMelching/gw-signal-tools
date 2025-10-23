# -- Standard Lib Imports
from __future__ import annotations  # Enables type hinting own type in a class
from typing import Literal, Optional

# -- Third Party Imports
import numpy as np
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
from lalsimulation.gwsignal import GravitationalWaveGenerator
import lalsimulation.gwsignal.core.waveform as wfm
from lalsimulation.gwsignal.core.gw import GravitationalWavePolarizations

# -- Local Package Imports
# from ..logging import logger


__doc__: str = """
A collection of waveform generators that incorporate calibration uncertainties
in both the waveforms and the detector outputs.
"""

__all__ = ('calib_error_generator', 'no_error_generator',
           'det_calib_error_generator', 'wf_calib_error_generator')


def calib_error_generator(
    wf_params: dict[str, u.Quantity],
    gen: GravitationalWaveGenerator,
    error_kind: Literal['wf_calib', 'det_calib'] = None,
    amp_error: Optional[Literal['abs', 'rel']] = None,
    phase_error: Optional[Literal['abs', 'rel']] = None,
) -> GravitationalWavePolarizations | FrequencySeries:
    """
    _summary_

    Parameters
    ----------
    wf_params : dict[str, u.Quantity]
        _description_
    gen: ~lalsimulation.gwsignal.waveform.GravitationalWaveGenerator
        ...
    error_kind : Literal['wf_calib', 'det_calib']
        _description_
    amp_error : Optional[Literal['abs', 'rel']], optional
        _description_, by default None
        None means no error correction will be applied
    phase_error : Optional[Literal['abs', 'rel']], optional
        _description_, by default None

    Raises
    ------
    ValueError
        _description_
    """
    # -- Check defaults
    if isinstance(error_kind, str):
        assert (error_kind == 'wf_calib') or (error_kind == 'det_calib'), \
            'Invalid `error_kind` given.'
    
    if isinstance(amp_error, str):
        assert (amp_error == 'abs') or (amp_error == 'rel'), \
            'Invalid `amp_error` given.'
        
    if isinstance(phase_error, str):
        assert (phase_error == 'abs') or (phase_error == 'rel'), \
            'Invalid `phase_error` given.'

    # -- Param handling
    _wf_params = wf_params.copy()
    delta_amplitude = _wf_params.pop('delta_amplitude')
    delta_phase = _wf_params.pop('delta_phase')
    _ext_params = {key: _wf_params.pop(key) for key in ['det', 'ra', 'dec', 'psi', 'tgps']}
    
    _ext_params_numb = len(_ext_params)
    if _ext_params_numb == 0:
        _project_into_det = False
    elif _ext_params_numb == 5:
        _project_into_det = True
    elif (_ext_params_numb > 0) and (_ext_params_numb < 5):
        # logger.info('Invalid set of external parameters given. '
        #             'Cannot perform detector projection.')
        
        _project_into_det = False
        
        if error_kind == 'det_calib':
            raise ValueError()

    try:
        _dist_val = _wf_params.pop('log_distance')
        _wf_params['distance'] = 10**_dist_val*u.Mpc
    except KeyError:
        pass

    try:
        _incl_val = _wf_params.pop('cos_inclination')
        _wf_params['inclination'] = np.arccos(_incl_val)*u.rad
    except KeyError:
        pass


    # wf = get_strain(_wf_params, 'frequency', gen)
    hp, hc = wfm.GenerateFDWaveform(_wf_params, gen)

    if error_kind == 'wf_calib':
        hp_amp, hp_phase = np.abs(hp), np.unwrap(np.angle(hp))
        hc_amp, hc_phase = np.abs(hc), np.unwrap(np.angle(hc))

        if amp_error == 'abs':
            hp_amp += delta_amplitude
            hc_amp += delta_amplitude
        elif amp_error == 'rel':
            hp_amp *= 1. + delta_amplitude
            hc_amp *= 1. + delta_amplitude
        
        if phase_error == 'abs':
            hp_phase += delta_phase
            hc_phase += delta_phase
        elif phase_error == 'rel':
            hp_phase *= 1. + delta_phase
            hc_phase *= 1. + delta_phase
        
        hp = hp_amp*np.exp(1.j*hp_phase)
        hc = hc_amp*np.exp(1.j*hc_phase)
        # TODO: use np.vectorize(complex)(...) here?

    if _project_into_det:
        h = GravitationalWavePolarizations(hp, hc).strain(**_ext_params)
    else:
        # TODO: check units here?
        hp.override_unit(u.strain*u.s)
        hc.override_unit(u.strain*u.s)
        return hp, hc

    if error_kind == 'det_calib':
        h_amp, h_phase = np.abs(h), np.unwrap(np.angle(h))

        if amp_error == 'abs':
            h_amp += delta_amplitude
        elif amp_error == 'rel':
            h_amp *= 1. + delta_amplitude
        
        if phase_error == 'abs':
            h_phase += delta_phase
        elif phase_error == 'rel':
            h_phase *= 1. + delta_phase
        
        h = h_amp*np.exp(1.j*h_phase)
    # TODO: check units here?
    h.override_unit(u.strain*u.s)
    return h


# -- Define wrappers for this function
def no_error_generator(
    wf_params: dict[str, u.Quantity],
    gen: GravitationalWaveGenerator
) -> GravitationalWavePolarizations | FrequencySeries:
    return calib_error_generator(
        wf_params=wf_params,
        gen=gen,
        error_kind=None,
        amp_error=None,
        phase_error=None
    )

def det_calib_error_generator(
    wf_params: dict[str, u.Quantity],
    gen: GravitationalWaveGenerator
) -> GravitationalWavePolarizations | FrequencySeries:
    return calib_error_generator(
        wf_params=wf_params,
        gen=gen,
        error_kind='det_calib',
        amp_error='rel',
        phase_error='rel'
    )

def wf_calib_error_generator(
    wf_params: dict[str, u.Quantity],
    gen: GravitationalWaveGenerator
) -> GravitationalWavePolarizations | FrequencySeries:
    return calib_error_generator(
        wf_params=wf_params,
        gen=gen,
        error_kind='wf_calib',
        amp_error='rel',
        phase_error='rel'
    )


# ----------------------------------------------------------------------
# from ..waveform import fd_to_td, td_to_fd
# from ..logging import logger
from gw_signal_tools.waveform import fd_to_td, td_to_fd  # To be able to run this file
from gw_signal_tools.logging import logger
# from gwpy.types import Series
from typing import Union#, Tuple
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries


# TODO: decide name. CalibratedGWPolarizations also nice
class CalGravitationalWavePolarizations(GravitationalWavePolarizations):
    """
    Apply different frequency domain models for errors to waveforms.
    Includes (systematic) waveform errors and calibration errors.
    """
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    #     assert self.domain == 'frequency', (
    #         'Calibration only works with ``FrequencySeries`` for now.'
    #     )

    # def __new__(cls, *args, **kwargs):
    #     out = super().__new__(cls ,*args, **kwargs)

    #     # if out.domain() == 'time':
    #     #     out.hpt = out.hp
    #     #     out.hct = out.hc

    #     #     out.hpf = td_to_fd(out.hp)
    #     #     out.hcf = td_to_fd(out.hc)
    #     # elif out.domain() == 'frequency':
    #     #     out.hpf = out.hp
    #     #     out.hcf = out.hc

    #     #     out.hpt = fd_to_td(out.hp)
    #     #     out.hct = fd_to_td(out.hc)
    #     # else:
    #     #     # -- To make sure hp, hc have been produced consistently in
    #     #     # -- the respective domain, assert they have same type
    #     #     raise ValueError('Please provide either two ``TimeSeries`` or two '
    #     #                      '``FrequencySeries``.')


    #     print(type(out))
        
    #     # out._domain = ''

    #     if isinstance(out.hp, TimeSeries) and isinstance(out.hc, TimeSeries):
    #         out._domain = 'TD'
    #     elif isinstance(out.hp, FrequencySeries) and isinstance(out.hc, FrequencySeries):
    #         out._domain = 'FD'
    #     else:
    #         # -- To make sure hp, hc have been produced consistently in
    #         # -- the respective domain, assert they have same type
    #         raise ValueError('Please provide either two ``TimeSeries`` or two '
    #                          '``FrequencySeries``.')


    #     # if out.domain() == 'mixed':
    #     #     # -- To make sure hp, hc have been produced consistently in
    #     #     # -- the respective domain, assert they have same type
    #     #     raise ValueError('Please provide either two ``TimeSeries`` or two '
    #     #                      '``FrequencySeries``.')
        
    #     # -- Call setters for polarizations again, to set hp(c)t, hp(c)f
    #     out.hp = out.hp
    #     out.hc = out.hc

    #     return out
    def __init__(self, hp, hc):
        super().__init__()

        if isinstance(hp, TimeSeries) and isinstance(hc, TimeSeries):
            self._domain = 'TD'

            logger.warning(
                'Calibration errors are applied in frequency domain, so please'
                ' make sure the time domain waveforms have been generated in a'
                ' way that will yield sensible results upon a Fourier '
                'transform (i.e. with some kind of conditioning).'
            )
        elif isinstance(hp, FrequencySeries) and isinstance(hc, FrequencySeries):
            self._domain = 'FD'
        else:
            # -- To make sure hp, hc have been produced consistently in
            # -- the respective domain, assert they have same type
            raise ValueError('Please provide either two ``TimeSeries`` or two '
                             '``FrequencySeries``.')
        
        # -- Call setters for polarizations again, to set hp(c)t, hp(c)f
        self.hp = hp
        self.hc = hc

    @property
    def hp(self) -> Union[TimeSeries, FrequencySeries]:
        # return self._hp
        # if self.domain() == 'time':
        if self._domain == 'TD':
            return self.hpt
        # elif self.domain() == 'frequency':
        elif self._domain == 'FD':
            return self.hpf

    @hp.setter
    def hp(self, h):
        if isinstance(h, TimeSeries):
            self.hpt = h
            self.hpf = td_to_fd(h)
        elif isinstance(h, FrequencySeries):
            self.hpf = h
            self.hpt = fd_to_td(h)
        else:
            raise ValueError('Invalid type given for waveform in `h`.')

    @property
    def hc(self) -> Union[TimeSeries, FrequencySeries]:
        # return self._hc
        # if self.domain() == 'time':
        if self._domain == 'TD':
            return self.hct
        # elif self.domain() == 'frequency':
        elif self._domain == 'FD':
            return self.hcf

    @hc.setter
    def hc(self, h):
        if isinstance(h, TimeSeries):
            self.hct = h
            self.hcf = td_to_fd(h)
        elif isinstance(h, FrequencySeries):
            self.hcf = h
            self.hct = fd_to_td(h)
        else:
            raise ValueError('Invalid type given for waveform in `h`.')
    
    # def apply_signal_frame_calibration(
    #     self,
    #     modification,  # dict or config file
    # # ) -> None:
    # ) -> CalGravitationalWavePolarizations:
    #     # self.hp = ...
    #     # self.hc = ...
    #     # TODO: is inplace a good idea? Rather return the calibrated signals?

    #     hp = ...
    #     hc = ...

    #     return CalGravitationalWavePolarizations(hp, hc)
    # TODO: if at some point TD calibration shall be supported, we can
    # create functions _apply_signal_frame_calibration_td, ..._fd that
    # are then called by this function here


    # -- New structure from here on

    @staticmethod
    def _get_fd_amplitude(hf: FrequencySeries) -> FrequencySeries:
        return np.abs(hf)

    @staticmethod
    def _get_fd_phase(hf: FrequencySeries) -> FrequencySeries:
        return np.unwrap(np.angle(hf))

    def _calibrate_f_series(
        self,
        hf: FrequencySeries,
        modification,  # TODO: make dict, config file, or even custom class?
    ) -> FrequencySeries:
        """
        Idea: only this function has to be able to parse whatever is in
        modification. Through clever call structure, we can do
        calibration of hp, hc in signal frame and calibration of h in
        detector frame using this function.
        """
        hf_amp = self._get_fd_amplitude(hf)
        hf_phase = self._get_fd_phase(hf)

        # TODO: apply calibration here

        hf_cal = hf_amp*np.exp(1.j*hf_phase)

        return hf_cal

    # def _get_fd_amplitude(self) -> tuple[FrequencySeries, FrequencySeries]:
    #     return np.abs(self.hpf), np.abs(self.hcf)

    # def _get_fd_phase(self) -> tuple[FrequencySeries, FrequencySeries]:
    #     return np.unwrap(np.angle(self.hpf)), np.unwrap(np.angle(self.hcf))

    @staticmethod
    def signal_frame_calibration(
        hp: Union[TimeSeries, FrequencySeries],
        hc: Union[TimeSeries, FrequencySeries],
        modification,
    ) -> CalGravitationalWavePolarizations:
        """Return calibrated polarizations from given ones."""
        out = CalGravitationalWavePolarizations(hp, hc)  # To go through init
        out.apply_signal_frame_calibration(modification)
        return out.hp, out.hc

    def apply_signal_frame_calibration(
        self,
        modification,
    ) -> None:
        """Apply calibration to polarizations in this class."""
        # hp_amp, hc_amp = self._get_fd_amplitude()
        # hp_phase, hc_phase = self._get_fd_phase()

        # # TODO: apply calibration here

        # self.hp = hp_amp*np.exp(1.j*hp_phase)
        # self.hc = hc_amp*np.exp(1.j*hc_phase)

        self.hp = self._calibrate_f_series(self.hp, modification)
        self.hc = self._calibrate_f_series(self.hc, modification)

    @staticmethod
    def detector_frame_calibration(
        hp: Union[TimeSeries, FrequencySeries],
        hc: Union[TimeSeries, FrequencySeries],
        # ext_params: dict[str, u.Quantity],
        modification,
        **ext_params
    ) -> Union[TimeSeries, FrequencySeries]:
        out = CalGravitationalWavePolarizations(hp, hc)
        # out.apply_detector_frame_calibration(ext_params, modification)
        out.apply_detector_frame_calibration(modification, **ext_params)
        return out.hp, out.hc

    def apply_detector_frame_calibration(
        self,
        # ext_params: dict[str, u.Quantity],
        modification,
        **ext_params
    ) -> Union[TimeSeries, FrequencySeries]:
        """
        This time we have to return something
        """
        h = self.strain(**ext_params)

        if isinstance(h, TimeSeries):
            hf = td_to_fd(h)
        elif isinstance(h, FrequencySeries):
            hf = h

        hf = self._calibrate_f_series(hf, modification)

        if isinstance(h, TimeSeries):
            return fd_to_td(hf)
        elif isinstance(h, FrequencySeries):
            return hf

    @staticmethod
    def calibration(
        hp: Union[TimeSeries, FrequencySeries],
        hc: Union[TimeSeries, FrequencySeries],
        # ext_params: dict[str, u.Quantity],
        signal_frame_mod,
        det_frame_mod,
        **ext_params
    ) -> Union[TimeSeries, FrequencySeries]:
        out = CalGravitationalWavePolarizations(hp, hc)
        # out.apply_calibration(ext_params, signal_frame_mod, det_frame_mod)
        out.apply_calibration(signal_frame_mod, det_frame_mod, **ext_params)
        return out.hp, out.hc

    def apply_calibration(
        self,
        # ext_params,
        signal_frame_mod,
        det_frame_mod,
        **ext_params
    ) -> Union[TimeSeries, FrequencySeries]:
        """
        This time we have to return something
        """
        cal_hpc = self.apply_signal_frame_calibration(signal_frame_mod)

        return cal_hpc.apply_detector_frame_calibration(ext_params, det_frame_mod)


# -- For quick testing
if __name__ == '__main__':
    # hp = FrequencySeries(np.ones(10), epoch=0)
    # hc = FrequencySeries(np.ones(10), epoch=0)
    hp = TimeSeries(np.ones(10), epoch=0)
    hc = TimeSeries(np.ones(10), epoch=0)

    hpc = CalGravitationalWavePolarizations(hp, hc)

    print(hpc.apply_signal_frame_calibration(None))
    print(CalGravitationalWavePolarizations.signal_frame_calibration(hp, hc, None))
    print(CalGravitationalWavePolarizations.detector_frame_calibration(hp, hc, None, det='H1', ra=0*u.rad, dec=0*u.rad, psi=0*u.rad, tgps=0*u.s))
