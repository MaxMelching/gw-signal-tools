# -- Standard Lib Imports
# from functools import cached_property  # TODO: use for some stuff?
from __future__ import annotations  # Needed for "if TYPE_CHECKING" block
from typing import Any, TYPE_CHECKING, Optional, NamedTuple

# -- Third Party Imports
import numpy as np
import astropy.units as u

if TYPE_CHECKING:
    from collections import namedtuple
    from gwpy.types import Series

# -- Local Package Imports
from .ana_derivs import ana_deriv_map
from ..inner_product import param_bounds as _param_bounds
from ...types import WFGen


__doc__ = """Module for the ``WaveformDerivativeBase`` class."""

__all__ = ('WaveformDerivativeBase',)


class WaveformDerivativeBase:
    """
    Base class for derivatives to inherit from. Defines useful
    attributes and the initialization signature expected for a
    derivative class in downstream applications (in particular Fisher
    matrix calculations).
    """

    def __init__(
        self,
        point: dict[str, u.Quantity],
        param_to_vary: str,
        wf_generator: WFGen,
        *args,
        **kwds,
    ) -> None:
        # -- We have typical arguments here, more can be added in child classes
        self._point = point
        self._param_to_vary = param_to_vary
        self._wf_generator = wf_generator
        self._param_bound_storage = _param_bounds.copy()
        self._ana_derivs = ana_deriv_map.copy()

    def __call__(
        self, x: Optional[float | u.Quantity] = None
    ) -> Series:  # pragma: no cover - meant to be overridden
        """
        Get derivative with respect to `self.param_to_vary` at :code:`x`.

        Parameters
        ----------
        x : float or ~astropy.units.Quantity, optional, default = None
            Parameter value at which the derivative is calculated. By
            default, the corresponding value from :code:`self.point`
            is chosen.

        Returns
        -------
        Derivative, put into whatever type :code:`self.wf_generator`
        has. This should, in principle, be either a GWpy
        :code:``FrequencySeries`` or a GWpy :code:``TimeSeries`` (in
        accordance with standard LAL output types), but this function
        only relies on GWPy :code:``Series`` properties being defined and
        thus the output could also be of this type.

        Notes
        -----
        Information gathered during calculation is stored in the
        :code:`self.info` property.
        """
        return NotImplemented

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return self._param_bound_storage

    def test_point(
        self, point: dict[str, u.Quantity], step: Optional[float] = None
    ) -> None:  # pragma: no cover - meant to be overridden
        """
        Check if `point` contains potentially tricky values, e.g.
        mass ratios close to 1. If yes, a subsequent adjustment of step
        sizes etc may be performed.
        """
        default_bounds = (-np.inf, np.inf)
        lower_bound, upper_bound = self.param_bounds.get(
            self.param_to_vary, default_bounds
        )
        if (self.param_to_vary == 'mass_ratio') and (self.param_center_val > 1):
            # -- In this convention, bounds have to be corrected
            lower_bound, upper_bound = self.param_bounds.get(
                'inverse_mass_ratio', default_bounds
            )

        if step is None:
            step = self.step.base_step
        _par_val = point[self.param_to_vary].value

        def violation(step):
            return (
                _par_val - step <= lower_bound,
                _par_val + step >= upper_bound,
            )

        lower_violation, upper_violation = violation(step)

        # -- Potentially more code that you want to have here

    # -- In case calling seems unintuitive, create attribute
    @property
    def deriv(self) -> Series:
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
        return self.point[self.param_to_vary]

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

    class DerivInfo(NamedTuple):
        """Namedtuple for derivative information with default values."""

        is_exact_deriv: bool = False
        """Indicates whether the derivative is analytical."""

    @property
    def info(self) -> namedtuple:
        f"""
        Information about the calculated derivative, given as a
        namedtuple. All fields from this namedtuple are also accessible
        as a class attribute. Available keys are those from the
        class's ``DerivInfo`` namedtuple, i.e. {self.DerivInfo._fields}.
        """
        return self._info

    @info.setter
    def info(self, info: dict[str, Any] | namedtuple) -> None:
        if isinstance(info, dict):
            info = self.__class__.DerivInfo(**info)
        elif not isinstance(info, self.DerivInfo):
            raise TypeError(
                f'`info` must be a dict or an instance of {self.DerivInfo.__name__}.'
            )

        self._info = info

    def __getattr__(self, name: str) -> Any:
        # -- Allow access to info fields as instance attributes.
        # -- This is more memory efficient than setting them individually.
        # -- Note that this function is only called if the attribute
        # -- wasn't found normally.

        # -- Avoid infinite recursion when accessing _info
        if name == '_info':
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # -- Check if info exists and has this field
        if hasattr(self, '_info') and hasattr(self.info, name):
            return getattr(self.info, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
