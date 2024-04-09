# ----- Standard Lib Imports -----
from __future__ import annotations  # Enables type hinting own type in a class
import logging
import warnings
from typing import Optional, Any, Literal, Self, Callable

# ----- Third Party Imports -----
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from gwpy.frequencyseries import FrequencySeries
import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm

# ----- Local Package Imports -----
from ..inner_product import inner_product
from ..waveform_utils import get_strain
from ..matrix_with_units import MatrixWithUnits
from .fisher_utils import fisher_matrix
from .fisher import FisherMatrix
from gw_signal_tools import preferred_unit_system


class Detector:
    # Very basic, should have psd and name
    def __init__(self, name: str, psd: FrequencySeries) -> None:
        self._name = name
        self._psd = psd
    
    @property
    def name(self):
        return self._name
    
    @property
    def psd(self):
        return self._psd

class FisherMatrixNetwork(FisherMatrix):
    def __init__(self,
        detectors: list[Detector],
        wf_params_at_point: dict[str, u.Quantity],
        params_to_vary: str | list[str],
        wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
        _copy: bool = False,
        **metadata
    ) -> None:
        self._detectors = detectors

        self._detector_indices = {}
        self._fisher_for_dets = []
        for i, det in enumerate(self.detectors):
            self._detector_indices[det.name] = i

            psd = det.psd
            self._fisher_for_dets += fisher_matrix(psd=psd)
    
    @property
    def detectors(self):
        return self._detectors

    @property    
    def _fisher_for_dets(self):
        # list of fisher matrices
        ...
    
    @property
    def _index_from_det(self, det: Detector | str):
        if isinstance(det, Detector):
            return self._detector_indices[det.name]
        elif isinstance(det, str):
            return self._detector_indices[det]
        else:
            raise ValueError('`det` must be an instance of the ``Detector`` class or a string.')
    
    def detector_fisher(self, det: Detector | str | int):
        if isinstance(det, (Detector, str)):
            return self._fisher_for_dets[self._index_from_det(det)]
        elif isinstance(det, int):
            return self._fisher_for_dets[det]
        else:
            raise ValueError('`det` must be an instance of the ``Detector`` class, and index to pick from the list of detectors or a string.')

