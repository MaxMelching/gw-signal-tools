# ----- Third Party Imports -----
from gwpy.frequencyseries import FrequencySeries


class Detector:
    """
    Basic representation of a gravitational wave (GW) detector for use
    in the context of waveforms.

    Parameters
    ----------
    name : str
        Name of the detector. Is used during waveform generation as
        input to the `'det'` parameter in LAL dictionaries.
    psd : ~gwpy.frequencyseries.FrequencySeries
        Power spectral density of the detector.
    kw_args :
        All other keyword arguments will be interpreted as arguments
        that are supposed to be used in waveform generation. This allows
        to specify certain properties that are distinct for detector
        that this instance represents, e.g. a certain starting frequency
        (note: this is not required for the `'det'` parameter, which is
        set automatically based on `name`).
    """
    def __init__(self, name: str, psd: FrequencySeries, **kw_args) -> None:
        """Initializa a ``Detector``."""
        self.name = name
        self.psd = psd
        # TODO: make default psd? No noise one?
        self.wf_args = kw_args
    
    @property
    def name(self):
        """Name of the detector."""
        return self._name
    
    @name.setter
    def name(self, name: str) -> None:
        assert isinstance(name, str), 'New `name` must be a string.'
        self._name = name
    
    @name.deleter
    def name(self) -> None:
        try:
            del self._name
        except AttributeError:
            pass
    
    @property
    def psd(self):
        """Power spectral density (PSD) of the detector."""
        return self._psd
    
    @psd.setter
    def psd(self, psd: FrequencySeries) -> None:
        assert isinstance(psd, FrequencySeries), (
            'New `psd` must be a GWpy ``FrequencySeries``.')
        self._psd = psd
    
    @psd.deleter
    def psd(self) -> None:
        try:
            del self._psd
        except AttributeError:
            pass

    @property
    def wf_args(self) -> dict:
        """Arguments for waveform generation that shall be used
        specifically for this detector.
        """
        return self._wf_args
    
    @wf_args.setter
    def wf_args(self, wf_args) -> None:
        self._wf_args = dict(wf_args)

    @wf_args.deleter
    def wf_args(self) -> None:
        try:
            del self._wf_args
        except AttributeError:
            pass
    
    def __repr__(self) -> str:
        # TODO: make better
        return 'Detector: ' + self.name + ' with PSD ' + self.psd.name
