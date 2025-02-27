# -- Standard Lib Imports
from typing import Optional

# -- Third Party Imports
import astropy.units as u
from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries
import numpy as np

# -- Local Package Imports
from ..test_utils import allclose_quantity


__all__ = ('_UNIT_CONV_ERR', '_q_convert', '_compare_series_xindex', '_assert_ft_compatible')

__doc__ = """
Little helper file containing functions and other definitions that help
dealing with errors in package functions.
"""


_UNIT_CONV_ERR = '''
Need consistent (i.e. convertible) units for `%s` (%s) and `%s` (%s).
'''


def _q_convert(
    quant: float | u.Quantity,
    target_unit: u.Unit,
    arg1name: str,
    arg2name: str,
    err_msg: Optional[str] = None,
) -> u.Quantity:
    """Wrapper function for quantity conversion with custom error raising."""
    try:
        return u.Quantity(quant, unit=target_unit)
        # -- No use of quant.to because input might not be Quantity
    except u.UnitConversionError:
        # -- Only happens when quant is already Quantity, so we can
        # -- assume that accessing the unit property works
        if err_msg is not None:
            raise ValueError(err_msg)
        else:
            raise ValueError(
                _UNIT_CONV_ERR % (arg1name, quant.unit, arg2name, target_unit)
            )


def _compare_series_xindex(*s: list[Series], enforce_dx: bool = True) -> None:
    """
    Checks if input is mutually compatible, raises error if not.

    The only argument is `enforce_dx`, which determines whether the `dx`
    attribute is accessed (bad for signals with unequal sampling).
    """
    # -- Checks: equal length, sufficiently equal spacing, sufficiently
    # -- equal index (making sure small errors do not accumulate)
    if len(s) < 2:
        return None
    if len(s) > 2:
        # for val in s[:-1]:
        #     _compare_series_xindex(val, s[-1])
        # -- Idea: performing every mutual comparison is not required
        _compare_series_xindex(s[-1], s[-2], enforce_dx=enforce_dx)
        _compare_series_xindex(s[:-1], enforce_dx=enforce_dx)
        return None

    # -- Leaves case with len(s)s == 2
    s1, s2 = s
    if enforce_dx and not allclose_quantity(s1.dx, s2.dx, atol=0.0, rtol=1e-5):
        # -- Some arbitrary deviation allowed, in case of numerical differences
        # raise ValueError('Signals must have equal spacing on x-axis.')
        # raise ValueError('Signals do not have equal spacing on x-axis.')
        raise ValueError(
            f'Signals do not have equal spacing on x-axis ({s1.dx} vs. {s2.dx}).'
        )

    # TODO: decide whether to compare whole frequency array or just
    # start and end frequency. But then things in between could go wrong
    # (and also checking with something like is_contiguous does not make
    # sense because this operation is also O(n), one can also just use
    # checks on all frequencies)

    if enforce_dx and not allclose_quantity(s1.xindex, s2.xindex, atol=0.5 * s1.dx.value, rtol=0.0):
        # -- Note: this atol checks for equality up to sampling accuracy
        # -- Note: this automatically checks for equal size
        raise ValueError(
            'Signals must have sufficiently equal xindex. '
            'Maximum allowed deviation is 0.5*dx.'
        )

    if not enforce_dx and not np.all(abs(s1.xindex - s2.xindex)[:-1] <= 0.5*abs(np.diff(s1.xindex))):
        # -- Unequal spacing was given, allowed for inner_product_computation
        raise ValueError(
            'Signals must have sufficiently equal xindex. '
            'Maximum allowed deviation is 0.5*dx.'
        )


def _assert_ft_compatible(*fs: list[FrequencySeries]) -> None:
    """
    Checks if input has correct format for a Fourier transformation,
    raises error if not.
    """
    if len(fs) > 1:
        for val in fs:
            _assert_ft_compatible(val)
        return None

    fs = fs[0]
    assert allclose_quantity(fs.f0.value, 0.0, atol=0.0, rtol=0.0) or allclose_quantity(
        -(fs.f0 + fs.df), fs.frequencies[-1], atol=fs.df.value, rtol=0.0
    ), (
        'All signals must start either at f=0 or be symmetric around f=0, '
        'where the latter refers to the case of an odd sample size. For an '
        'even sample size, on the other hand, the number of samples for '
        'positive frequencies is expected to be one less than the number of '
        'samples for negative frequencies, in accordance with the format '
        'expected by `~numpy.fft.ifftshift`. Note that the conditions just'
        'mentioned do not apply to the case of a starting frequency f=0, '
        'where both even and odd sample sizes are accepted.'
    )
