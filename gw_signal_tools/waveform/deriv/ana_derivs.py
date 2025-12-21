# -- Standard Lib Imports
from __future__ import annotations  # Needed for "if TYPE_CHECKING" block
from typing import TYPE_CHECKING, Callable

# -- Third Party Imports
import astropy.units as u
import numpy as np

if TYPE_CHECKING:
    from gwpy.types import Series
    from .base import WaveformDerivativeBase

# -- Local Package Imports
from ...types import WFGen


__doc__ = """Module for analytical waveform derivatives."""

__all__ = ('distance_deriv', 'time_deriv', 'phase_deriv', 'ana_deriv_map')


def distance_deriv(
    deriv_class: WaveformDerivativeBase,
    eval_point: dict[str, u.Quantity],
) -> Series:
    """Analytical derivative of waveform with respect to distance."""
    wf = deriv_class.wf_generator(eval_point)
    deriv = wf * (-1.0 / eval_point['distance'])
    return deriv


def time_deriv(
    deriv_class: WaveformDerivativeBase,
    eval_point: dict[str, u.Quantity],
) -> Series:
    """Analytical derivative of waveform with respect to time."""
    wf = deriv_class.wf_generator(eval_point)
    deriv = wf * (-1.0j * 2.0 * np.pi * wf.frequencies)
    return deriv


def phase_deriv(
    deriv_class: WaveformDerivativeBase,
    eval_point: dict[str, u.Quantity],
) -> Series:
    """Analytical derivative of waveform with respect to phase."""
    wf = deriv_class.wf_generator(eval_point)
    deriv = wf * (-1.0j / u.rad)
    return deriv


ana_deriv_map: dict[str, Callable[[dict[str, u.Quantity], WFGen], Series]] = {
    'distance': distance_deriv,
    'time': time_deriv,
    'phase': phase_deriv,
}
