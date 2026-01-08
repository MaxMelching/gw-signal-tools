from typing import Callable, Union, TypeAlias
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries


__doc__ = """File for custom type definitions related to gwsignal."""

__all__ = (
    'FDWFGen',
    'TDWFGen',
    'WFGen',
)


# GWParams: TypeAlias = dict[str, u.Quantity]
# FDWFGen: TypeAlias = Callable[[GWParams], FrequencySeries]
# TDWFGen: TypeAlias = Callable[[GWParams], TimeSeries]
# WFGen: TypeAlias = Union[FDWFGen, TDWFGen]

# TODO: should we define GWParams?

FDWFGen: TypeAlias = Callable[[dict[str, u.Quantity]], FrequencySeries]
"""Frequency-domain gwsignal waveform generator."""
TDWFGen: TypeAlias = Callable[[dict[str, u.Quantity]], TimeSeries]
"""Time-domain gwsignal waveform generator."""
WFGen: TypeAlias = Union[FDWFGen, TDWFGen]
"""General gwsignal waveform generator, either frequency or time domain."""
