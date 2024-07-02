# ----- Standard Lib Imports -----
from __future__ import annotations  # Enables type hinting own type in a class
import logging
from typing import Optional, Any, Literal, Self, Callable

# ----- Third Party Imports -----
import numpy as np

from gwpy.frequencyseries import FrequencySeries
import astropy.units as u

# ----- Local Package Imports -----
from ..inner_product import inner_product
from ..matrix_with_units import MatrixWithUnits
from .fisher import FisherMatrix


class Detector:
    # Very basic, should have psd and name
    def __init__(self, name: str, psd: FrequencySeries) -> None:
        self._name = name
        self._psd = psd

        # TODO: make default psd? No noise one?
    
    @property
    def name(self):
        return self._name
    
    @property
    def psd(self):
        return self._psd
    
    # TODO: think about whether or not this has to be immutable. Maybe
    # we can also work with self.name and self.psd directly in __init__

class FisherMatrixNetwork(FisherMatrix):
    def __init__(self,
        wf_params_at_point: dict[str, u.Quantity],
        params_to_vary: str | list[str],
        wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
        detectors: Detector | list[Detector],
        direct_computation: bool = True,
        **metadata
    ) -> None:
        # Standard setup, same as in FisherMatrix
        self.wf_params_at_point = wf_params_at_point
        if isinstance(params_to_vary, str):
            self.params_to_vary = [params_to_vary]
        else:
            self.params_to_vary = params_to_vary.copy()
        self.wf_generator = wf_generator
        self.metadata = self.default_metadata | metadata

        # Setup for Network specifically
        if isinstance(detectors, Detector):
            self._detectors = [detectors]
        else:
            self._detectors = detectors
        
        self._detector_indices = {}
        self._fisher_for_dets = []  # TODO: make dict too?
        self._fisher = MatrixWithUnits.from_numpy_array(np.zeros(2*(len(params_to_vary),)))
        
        for i, det in enumerate(self.detectors):
            self._detector_indices[det.name] = i

            # psd = det.psd
            # self._fisher_for_dets += [fisher_matrix(psd=psd)]
            self._fisher_for_dets += [
                FisherMatrix(
                    wf_params_at_point=wf_params_at_point,
                    params_to_vary=params_to_vary,
                    wf_generator=wf_generator,
                    direct_computation=direct_computation,
                    psd=det.psd
                    **metadata)
            ]
    
        if direct_computation:
            self._calc_fisher()
    
    # ----- Adding Network specific properties -----
    @property
    def detectors(self):
        return self._detectors

    # @property    
    # def _fisher_for_dets(self):
    #     """List of Fisher matrices for detectors."""
    #     ...
    # TODO: check if needed. This job is taken by detector_fisher, right?
    
    @property
    def _index_from_det(self, det: Detector | str):
        """Get index for detector name."""
        if isinstance(det, Detector):
            return self._detector_indices[det.name]
        elif isinstance(det, str):
            return self._detector_indices[det]
        else:
            raise ValueError('`det` must be an instance of the ``Detector`` class or a string.')
    
    def detector_fisher(self, det: Detector | str | int):
        """Get Fisher matrix for detector name or index."""
        if isinstance(det, (Detector, str)):
            return self._fisher_for_dets[self._index_from_det(det)]
        elif isinstance(det, int):
            return self._fisher_for_dets[det]
        else:
            raise ValueError('`det` must be an instance of the ``Detector`` class, and index to pick from the list of detectors or a string.')
    
    # ----- Overwriting certain FisherMatrix properties -----
    def _calc_fisher(self):
        # TODO: check if we can make calculations more efficient. Maybe
        # by caching or maybe by realizing that derivatives will not
        # differ much in different detectors (only difference is PSD
        # used to check convergence)
        for i, det in enumerate(self.detectors):
            # self._fisher += self._fisher_for_dets[i].fisher
            self._fisher += self.detector_fisher[det].fisher

            # TODO: decide which one is better
    
    def systematic_error(self,
        reference_wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
        params: str | list[str] | None = None,
        optimize: bool | str | list[str] = True,
        optimize_fisher: str | list[str] | None = None,
        return_opt_info: bool = True,
        **inner_prod_kwargs
    ) -> MatrixWithUnits | tuple[MatrixWithUnits, dict[str, Any]]:
        if isinstance(optimize, str):
            optimize = [optimize]
        
        if isinstance(optimize_fisher, str):
            optimize_fisher = [optimize_fisher]
        
        # Goal of this function: duplicate as little code as possible.
        # -> the operations required for this (mainly matrix multiplications)
        #    do not add significant overhead compared to putting
        #    adjusted version of code from FisherMatrix.sys_error here
        #    (main cost is waveform generation and thus optimization)

        sys_error_list = []
        # vector_list = []
        vector = MatrixWithUnits.from_numpy_array(np.zeros((len(self.params_to_vary), 1)))
        fisher = MatrixWithUnits.from_numpy_array(np.zeros(2*(len(self.params_to_vary),)))
        opt_bias = MatrixWithUnits.from_numpy_array(np.zeros((len(self.params_to_vary), 1)))
        optimization_info = {}

        for i, det in enumerate(self.detectors):
            sys_error_list += [
                self.detector_fisher[det].systematic_error(
                    reference_wf_generator=reference_wf_generator,
                    params=None,  # Get all for now, filter before return
                    optimize=optimize,
                    optimize_fisher=optimize_fisher,
                    return_opt_info=True,
                    **inner_prod_kwargs
                )
            ]
            # NOTE: every element is now tuple, so pay attention to indices

            optimization_info[det] = sys_error_list[-1][1]

            if isinstance(optimize, bool) and not optimize:
                # used_fisher = self.detector_fisher[det].fisher
                used_fisher = self.detector_fisher[i].fisher  # Should be faster
            # elif isinstance(optimize, bool) and not optimize:
            else:
                # Some kind of optimization was carried out, thus we
                # can access
                used_fisher = sys_error_list[-1][1].opt_fisher.fisher
            
            # TODO: do same check for projection
            
            # vector_list += [
            #     used_fisher @ sys_error_list[-1][0]
            # ]
            vector += used_fisher @ sys_error_list[-1][0]
            # Ah shit, the idea is good, but we have added opt_bias,
            # which was not obtained via calculation with inverse matrix.
            # -> maybe give opt_bias as output into info, so that we can
            #    subtract it here?

            opt_bias += 0.

        fisher_bias = MatrixWithUnits.inv(fisher) @ vector + opt_bias
        
        
        # Check which params shall be returned
        if params is not None:
            if isinstance(params, str):
                params = [params]
            
            if optimize_fisher is None:
                param_indices = self.get_param_indices(params)
            else:
                # Take difference of parameters
                _params = params.copy()
                for param in optimize_fisher:
                    _params.remove(param)
        
            fisher_bias = fisher_bias[param_indices]
        
        if optimize is False or return_opt_info is False:
            return fisher_bias
        else:
            return fisher_bias, optimization_info
