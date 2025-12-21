# -- Standard Lib Imports
from __future__ import annotations  # Needed for "if TYPE_CHECKING" block
from typing import TYPE_CHECKING, Callable

# -- Third Party Imports
import astropy.units as u
import numpy as np

if TYPE_CHECKING:
    from gwpy.types import Series

# -- Local Package Imports
from ...types import WFGen


__doc__ = """Module for analytical waveform derivatives."""

__all__ = ('distance_deriv', 'time_deriv', 'phase_deriv', 'ana_deriv_map')


def distance_deriv(
    point: dict[str, u.Quantity],
    wf_generator: WFGen,
) -> Series:
    """Analytical derivative of waveform with respect to distance."""
    wf = wf_generator(point)
    deriv = (-1.0 / point['distance']) * wf
    return deriv


def time_deriv(
    point: dict[str, u.Quantity],
    wf_generator: WFGen,
) -> Series:
    """Analytical derivative of waveform with respect to time."""
    wf = wf_generator(point)
    deriv = wf * (-1.0j * 2.0 * np.pi * wf.frequencies)
    return deriv


def phase_deriv(
    point: dict[str, u.Quantity],
    wf_generator: WFGen,
) -> Series:
    """Analytical derivative of waveform with respect to phase."""
    wf = wf_generator(point)
    deriv = -1.0j * wf / u.rad
    return deriv


ana_deriv_map: dict[str, Callable[[dict[str, u.Quantity], WFGen], Series]] = {
    'distance': distance_deriv,
    'time': time_deriv,
    'phase': phase_deriv,
}
