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
from ..types import Detector


class FisherMatrixNetwork(FisherMatrix):
    """
    _summary_

    Parameters
    ----------
    wf_generator :
        Must accept extrinsic parameters now, otherwise notion of
        multiple detectors does not make sense.
    """
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
        # self._fisher = MatrixWithUnits.from_numpy_array(np.zeros(2*(len(params_to_vary),)))
        
        for i, det in enumerate(self.detectors):
            self._detector_indices[det.name] = i

            self._fisher_for_dets += [
                FisherMatrix(
                    wf_params_at_point=wf_params_at_point | det.wf_args,
                    params_to_vary=params_to_vary,
                    wf_generator=wf_generator,
                    direct_computation=direct_computation,
                    psd=det.psd,
                    **metadata
                )
            ]
    
        if direct_computation:
            self._calc_fisher()
    
    # ----- Adding Network specific properties -----
    @property
    def detectors(self):
        return self._detectors
    
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
        self._fisher = self.detector_fisher(self.detectors[0]).fisher
        # Needed to have correct units. Setting just with zeros does not work
        for i, det in enumerate(self.detectors[1:]):
            # self._fisher += self._fisher_for_dets[i].fisher
            self._fisher += self.detector_fisher(det).fisher

            # TODO: decide which one is better
    
    def systematic_error(self,
        reference_wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
        params: str | list[str] | None = None,
        optimize: bool | str | list[str] = True,
        optimize_fisher: str | list[str] | None = None,
        return_opt_info: bool = False,
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

        # sys_error_list = []
        # # vector_list = []
        # fisher = MatrixWithUnits(np.zeros(2*(len(self.params_to_vary),)), self.fisher.unit)
        # opt_bias = MatrixWithUnits(np.zeros(len(self.params_to_vary)),
        #     [self.wf_params_at_point[param].unit for param in self.params_to_vary]).reshape((len(self.params_to_vary), 1))
        # vector = MatrixWithUnits(np.zeros((len(self.params_to_vary), 1)),
        #     (fisher @ opt_bias).unit)
        sys_error = 0.
        fisher = 0.
        opt_bias = 0.
        vector = 0.
        # 0. is most convenient way to initialize here, adding a
        # MatrixWithUnits on top is allowed

        optimization_info = {}

        for i, det in enumerate(self.detectors):
            # sys_error_list += [
                # self.detector_fisher(det).systematic_error(
            sys_error = self.detector_fisher(i).systematic_error(
                    reference_wf_generator=reference_wf_generator,
                    params=None,  # Get all for now, filter before return
                    optimize=optimize,
                    optimize_fisher=optimize_fisher,
                    return_opt_info=True,
                    **inner_prod_kwargs
                )
            # ]
            # NOTE: every element is now tuple, so pay attention to indices

            # optimization_info[det] = sys_error_list[-1][1]
            optimization_info[det] = sys_error[1]

            if isinstance(optimize, bool) and not optimize:
                # used_fisher = self.detector_fisher(det)
                # used_fisher = self.detector_fisher(i)  # Should be faster
                # used_opt_bias = MatrixWithUnits(np.zeros(sys_error_list[-1][0].shape), sys_error_list[-1][0].unit)
                used_opt_bias = 0.
            
                if optimize_fisher is not None:
                    # used_fisher = used_fisher.project_fisher(optimize_fisher).fisher
                    used_fisher = sys_error[1]['opt_fisher'].fisher
                else:
                    # used_fisher = used_fisher.fisher
                    used_fisher = self.detector_fisher(i).fisher
            else:
                # Some kind of optimization was carried out, thus we
                # can access attribute in info dictionary
                # used_fisher = sys_error_list[-1][1]['opt_fisher']
                # used_opt_bias = sys_error_list[-1][1]['opt_bias']
                used_fisher = sys_error[1]['opt_fisher'].fisher
                used_opt_bias = sys_error[1]['opt_bias']
            
            # vector += used_fisher @ (sys_error_list[-1][0] - used_opt_bias)
            vector += used_fisher @ (sys_error[0] - used_opt_bias)
            opt_bias += used_opt_bias
            fisher += used_fisher

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
        
        if return_opt_info is False:
            return fisher_bias
        else:
            return fisher_bias, optimization_info
