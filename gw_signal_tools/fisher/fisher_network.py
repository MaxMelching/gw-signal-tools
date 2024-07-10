# ----- Standard Lib Imports -----
from __future__ import annotations  # Enables type hinting own type in a class
import logging
from typing import Optional, Any, Literal, Self, Callable

# ----- Third Party Imports -----
import numpy as np

from gwpy.frequencyseries import FrequencySeries
import astropy.units as u

# ----- Local Package Imports -----
from ..inner_product import norm
from .fisher import FisherMatrix, fisher_matrix
from ..types import MatrixWithUnits, Detector


class FisherMatrixNetwork(FisherMatrix):
    """
    _summary_

    Parameters
    ----------
    wf_generator :
        Must accept extrinsic parameters now, otherwise having
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
        """Initialize a ``FisherMatrixNetwork``."""
        # Setup for Network specifically
        if isinstance(detectors, Detector):
            self._detectors = [detectors]
        else:
            self._detectors = detectors

        self._detector_indices = {}
        for i, det in enumerate(self.detectors):
            self._detector_indices[det.name] = i
        
        # Now we can proceed with standard Fisher setup
        # Note that handling of detectors prior to this call is
        # detrimental because self._calc_fisher needs it, which is
        # potentially called in the following.
        _metadata = metadata.copy()
        _metadata.pop('psd', None)
        # Make sure no psd keyword is present, this is always taken from
        # detectors. Would not make sense to pass single PSD for a
        # network of multiple detectors anyway.

        super().__init__(
            wf_params_at_point=wf_params_at_point,
            params_to_vary=params_to_vary,
            wf_generator=wf_generator,
            direct_computation=direct_computation,
            **_metadata
        )

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
    
    def detector_fisher(self, det: Detector | str | int) -> FisherMatrix:
        """Get Fisher matrix for detector, detector name or index."""
        if isinstance(det, (Detector, str)):
            return self._fisher_for_dets[self._index_from_det(det)]
        elif isinstance(det, int):
            return self._fisher_for_dets[det]
        else:
            raise ValueError(
                '`det` must be an instance of the ``Detector`` class, a'
                ' string representing a detector name from `self.'
                'detectors` or an index to pick from the list of '
                'detectors.'
            )
    
    # ----- Overwriting certain FisherMatrix properties -----
    def _calc_fisher(self):
        # TODO: check if we can make calculations more efficient. Maybe
        # by caching or maybe by realizing that derivatives will not
        # differ much in different detectors (only difference is PSD
        # used to check convergence)

        self._fisher_for_dets = []
        self._fisher = 0.
        # for i, det in enumerate(self.detectors):
        for det in self.detectors:
            det_fisher = FisherMatrix(
                wf_params_at_point=self.wf_params_at_point | det.wf_args,
                params_to_vary=self.params_to_vary,
                wf_generator=self.wf_generator,
                direct_computation=True,
                psd=det.psd,
                **self.metadata
            )

            self._fisher_for_dets += [det_fisher]
            self._fisher += det_fisher.fisher
            
            if self.metadata['return_info']:
                self.deriv_info[det.name] = det_fisher.deriv_info
                # Note: this works even if self.deriv_info is not
                # initialized and despite it having no setter. The
                # reason is that only elements are set, so the property
                # is accessed first, which then initializes an instance.
                # Commands like self.deriv_info = 0 do throw an error.
    
    # TODO: make fisher_for_dets property? Because problem is that we sometimes
    # try to access it
        
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
        fisher = 0.*self.fisher
        # Instead of just initializing with zeros, this makes sure
        # fisher was calculated (important for certain attributes to be
        # accessible)
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

            # optimization_info[det.name] = sys_error_list[-1][1]
            optimization_info[det.name] = sys_error[1]

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
    
    def snr(self, **inner_prod_kwargs):
        """
        Calculate the signal-to-noise ratio (SNR) of the signal that
        `self.wf_generator` produces at `self.wf_params_at_point`.

        Parameters
        ----------
        inner_prod_kwargs :
            Any keyword argument given here will be passed to the inner
            product calculation. Enables e.g. testing SNR with different
            PSD while leaving all other arguments the same.

        Returns
        -------
        ~astropy.units.Quantity :
            SNR, i.e. norm of signal, in the given detector network.
        """
        _inner_prod_kwargs = self._inner_prod_kwargs | inner_prod_kwargs
        _inner_prod_kwargs.pop('psd', None)  # Make sure no PSD given

        snr = 0.
        for det in self.detectors:
            signal = self.wf_generator(self.wf_params_at_point | det.wf_args)
            snr += norm(signal, psd=det.psd, **_inner_prod_kwargs)**2
        
        return snr**.5
        # TODO: check if normalization factor like 1/len(self.detectors) is needed
