# -- Standard Lib Imports -----------------------
from typing import Any

# -- Third Party Imports ------------------------
from gwpy.frequencyseries import FrequencySeries
import numpy as np

# -- Local Package Imports ----------------------
from .series_matrix_with_units import SeriesMatrixWithUnits


__doc__ = """
Module for the ``Detector`` class that is intended to provide a simple
representation of gravitational wave detectors, with all information
needed by functions in `gw_signal_tools`.
"""


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
        that are supposed to be used in inner product calculations. This
        allows to specify certain properties that are distinct for
        detector that this instance represents, e.g. a certain starting
        frequency (note: this is not required for the `'det'` parameter,
        which is set automatically based on `name`).

        Note: since the PSD is already an attribute of this class, it
        does not need to be given here (only relevant is `kw_args` is
        passed as a dictionary and not via keyword arguments).
    """
    def __init__(self, name: str, psd: FrequencySeries, **kw_args) -> None:
        """Initializa a ``Detector``."""
        self.name = name
        self.psd = psd
        # TODO: make default psd? No noise one?
        self.inner_prod_kwargs = kw_args
    
    @property
    def name(self) -> str:
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
        except AttributeError:  # pragma: no cover
            pass
    
    @property
    def psd(self) -> FrequencySeries:
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
        except AttributeError:  # pragma: no cover
            pass

    @property
    def inner_prod_kwargs(self) -> dict:
        """Arguments for inner product calculations that shall be used
        specifically for this detector.
        """
        return self._inner_prod_kwargs
    
    @inner_prod_kwargs.setter
    def inner_prod_kwargs(self, kw_args: dict[str, Any]) -> None:
        self._inner_prod_kwargs = kw_args | {'psd': self.psd}

    @inner_prod_kwargs.deleter
    def inner_prod_kwargs(self) -> None:
        try:
            del self._inner_prod_kwargs
        except AttributeError:  # pragma: no cover
            pass
    
    def __repr__(self) -> str:
        # TODO: make better
        return 'Detector name: ' + self.name + '\nPSD description: ' + self.psd.name


class DetectorNetwork:
    # Essentially list of detectors. But some additional goodies could be added too, think about that
    def __init__(self, *dets: Any) -> None:
        self.detectors = dets

        self._detector_indices = {}
        for i, det in enumerate(self.detectors):
            self._detector_indices[det.name] = i
    
    @property
    def detectors(self) -> Any:
        """List of detectors stored in this network."""
        return self._detectors

    @detectors.setter
    def detectors(self, dets) -> None:
        for det in dets:
            assert isinstance(det, Detector), (
                'Need each detector to be of type ``Detector``.'
            )
        
        self._detectors = dets
        self._ndet = len(dets)
    
    @detectors.deleter
    def detectors(self) -> None:
        try:
            del self._detectors
        except AttributeError:  # pragma: no cover
            pass
    
    @property
    def ndet(self) -> int:
        return self._ndet
    
    def index_from_detector(self, det: Detector | str):
        """Get index for a detector or detector name."""
        if isinstance(det, Detector):
            return self._detector_indices[det.name]
        elif isinstance(det, str):
            return self._detector_indices[det]
        else:  # pragma: no cover
            raise ValueError(
                '`det` must be a ``Detector`` instance or a string.'
            )
    
    def detector_from_index(self, index: int) -> Detector:
        """Get detector for a given index."""
        return self.detectors[index]
    
    @property
    def name(self):
        """Name of the detector network."""
        return self._name
    
    @name.setter
    def name(self, name: str) -> None:
        assert isinstance(name, str), 'New `name` must be a string.'
        self._name = name
    
    @name.deleter
    def name(self) -> None:
        try:
            del self._name
        except AttributeError:  # pragma: no cover
            pass
    
    @property
    def psd(self) -> SeriesMatrixWithUnits:
        """Power spectral density (PSD) matrix of the detector network."""
        return self._psd
    
    @psd.setter
    def psd(self, psd: SeriesMatrixWithUnits) -> None:
        assert isinstance(psd, SeriesMatrixWithUnits), (
            'New `psd` must be a ``SeriesMatrixWithUnits``.')
        self._psd = psd
    
    @psd.deleter
    def psd(self) -> None:
        try:
            del self._psd
        except AttributeError:  # pragma: no cover
            pass
    
    def _psd_matrix_from_dets(self) -> None:
        """Set PSD matrix from given detectors PSDs."""
        psd = SeriesMatrixWithUnits(np.zeros(2*(self.ndet, )))
        for i, det in enumerate(self.detectors):
            psd[i, i] = det.psd
        
        self._psd = psd

        # TODO: how to set CSDs?
