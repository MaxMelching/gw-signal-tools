# -- Standard Lib Imports
# from functools import cached_property  # TODO: use for some stuff?
from typing import Optional, Literal, Any

# -- Third Party Imports
import numpy as np
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import matplotlib as mpl

# -- Local Package Imports
from ...logging import logger
from ..inner_product import norm, inner_product, param_bounds
from ...types import WFGen


__doc__ = """Module for the ``WaveformDerivativeBase`` class."""

__all__ = ('WaveformDerivativeBase',)


class WaveformDerivativeBase:
    def __init__(
        self,
        # point: dict[str, u.Quantity],
        # param_to_vary: str,
        # wf_generator: WFGen,
        # *args,
        # **kwds,
    ) -> None:
        pass

    def __call__(self, x=None) -> Any:
        return NotImplementedError

    _param_bound_storage = param_bounds.copy()

    @property
    def param_bounds(self, param: str) -> tuple[float, float]:
        return self._param_bound_storage[param]

    @param_bounds.setter
    def param_bounds(self, param: str, bounds: list[float, float]) -> None:
        """
        Specify bounds for a parameter that does not have registered
        bounds yet, or update parameter bounds.

        Parameters
        ----------
        param : str
            The parameter for which bounds shall be specified.
        bounds : list[float, float]
            Lower and upper bound.
        """
        assert len(bounds) == 2, 'Need exactly one lower and one upper bound.'
        self._param_bound_storage[param] = bounds

    # TODO: do we need test_point? I guess yeah; for autodiff, it can just check if the point itself is valid I guess?
    def test_point(self) -> None:
        """
        Check if `self.point` contains potentially tricky values, e.g.
        mass ratios close to 1. If yes, a subsequent adjustment takes
        place.
        """
        default_bounds = (-np.inf, np.inf)
        lower_bound, upper_bound = self._param_bound_storage.get(
            self.param_to_vary, default_bounds
        )
        if self.param_to_vary == 'mass_ratio':
            # -- Depending on chosen convention, bounds might have to be corrected
            if self.param_center_val > 1:
                lower_bound, upper_bound = self._param_bound_storage.get(
                    'inverse_mass_ratio', default_bounds
                )

        _base_step = ...  # TODO: this is where you have to replace with the step size you use
        _par_val = self.param_center_val.value
        lower_violation = _par_val - _base_step <= lower_bound
        upper_violation = _par_val + _base_step >= upper_bound

        # -- Check if base_step needs change
        if lower_violation and upper_violation:
            self.step.base_step = min(_base_step / 2.0, self._default_base_step)
        elif lower_violation and not upper_violation:
            # -- Can only happen if method is not forward yet
            self.method = 'forward'
            self.step.base_step = min(_base_step / 2.0, self._default_base_step)
        elif not lower_violation and upper_violation:
            # -- Can only happen if method is not backward yet
            self.method = 'backward'
            self.step.base_step = min(_base_step / 2.0, self._default_base_step)

    # -- In case calling seems unintiutive, create attribute
    @property
    def deriv(self) -> Any:
        """Alias for calling with no arguments."""
        return self.__call__()

    # -- Define certain properties. These have NO setters, on purpose!
    @property
    def param_to_vary(self) -> str:
        """
        Parameter that derivative is taken with respect to.

        :type: `str`
        """
        return self._param_to_vary

    @property
    def param_center_val(self) -> u.Quantity:
        """
        Value of `self.param_to_vary` at which derivative is taken by
        default.

        :type: `~astropy.units.Quantity`
        """
        return self._param_center_val

    @property
    def wf_generator(self) -> WFGen:
        """
        Generator for waveform model that is differentiated.

        :type: `~gw_signal_tools.types.WFGen`
        """
        return self._wf_generator

    @property
    def point(self) -> dict[str, u.Quantity]:
        """
        Point in parameter space at which waveform is differentiated,
        encoded as key-value pairs representing parameter-value pairs.

        :type: `dict[str, ~astropy.units.Quantity]`
        """
        return self._point

    @property
    def deriv_info(self) -> dict[str, Any]:
        """
        Information about the calculated derivative, given as a
        dictionary. All keys from this dictionary are also accessible
        as a class attribute.
        """
        return self._deriv_info

    @deriv_info.setter
    def deriv_info(self, info: dict[str, Any]):
        for key, val in info.items():
            # -- Make each key from deriv_info available as attribute
            setattr(self, key, val)

        self._deriv_info = info
