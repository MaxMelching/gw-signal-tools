# -- Standard Lib Imports
from typing import Optional

# -- Third Party Imports
import astropy.units as u
from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries

# -- Local Package Imports
from ..test_utils import allclose_quantity


__all__ = ('_UNIT_CONV_ERR', '_q_convert', '_compare_series', '_assert_ft_compatible')

__doc__: str = """
Little helper file containing functions and other definitions that help
dealing with errors in package functions.
"""


_UNIT_CONV_ERR = '''
Need consistent (i.e. convertible) units for `%s` (%s) and `%s` (%s).
'''


def _q_convert(quant: float | u.Quantity, target_unit: u.Unit, arg1name: str,
               arg2name: str, err_msg: Optional[str] = None) -> u.Quantity:
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
            raise ValueError(_UNIT_CONV_ERR % (arg1name, quant.unit, arg2name, target_unit))


# def _compare_series(s1: Series, s2: Series) -> tuple[bool, str]:
#     """Return whether equal, error message with reason if not."""
#     if not s1.unit._is_equivalent(s2.unit):
#         return False, 'Signals must have equivalent units.'

#     if not allclose_quantity(s1.df, s2.df, atol=0., rtol=1e-5):
#         return False, 'Signals must have equal spacing on x-axis.'

#     # TODO: decide whether to compare whole frequency array or just
#     # start and end frequency. But then things in between could go wrong

#     if not allclose_quantity(s1.xindex, s2.xindex, atol=0.5*s1.dx.value, rtol=0.):
#         # -- Note: this automatically checks for equal size
#         return False, ('Signals must have sufficiently equal xindex. '
#                        'Maximum allowed deviation is 0.5*dx.')

#     # -- Default return is that series are equal
#     return True, ''

# def _compare_series(s1: Series, s2: Series, test_unit: bool = True) -> None:
#     """Checks input for compatibility, raises error if not."""
#     if test_unit and not s1.unit._is_equivalent(s2.unit):
#         # raise ValueError(f'Signals must have equivalent units.')
#         # raise ValueError(f'Signals do not have equivalent units.')
#         raise ValueError(f'Signals do not have equivalent units ({s1.unit} vs. {s2.unit}).')
#         # raise ValueError(f'Signals do not have equivalent units ({s1.unit:latex} vs. {s2.unit:latex}).')

#     if not allclose_quantity(s1.dx, s2.dx, atol=0., rtol=1e-5):
#         # raise ValueError('Signals must have equal spacing on x-axis.')
#         # raise ValueError('Signals do not have equal spacing on x-axis.')
#         raise ValueError(f'Signals do not have equal spacing on x-axis ({s1.dx} vs. {s2.dx}).')

#     # TODO: decide whether to compare whole frequency array or just
#     # start and end frequency. But then things in between could go wrong

#     if not allclose_quantity(s1.xindex, s2.xindex, atol=0.5*s1.dx.value, rtol=0.):
#         # -- Note: this automatically checks for equal size
#         raise ValueError('Signals must have sufficiently equal xindex. '
#                          'Maximum allowed deviation is 0.5*dx.')

# def _compare_series(s1: Series, s2: Series) -> None:
def _compare_series(*s: list[Series]) -> None:
    """Checks if input is mutually compatible, raises error if not."""
    # -- Checks: equal length, sufficiently equal spacing, sufficiently
    # -- equal index (making sure small errors do not accumulate)
    if len(s) < 2:
        return None
    if len(s) > 2:
        # for val in s[:-1]:
        #     _compare_series(val, s[-1])
        # -- Idea: performing every mutual comparison is not required
        _compare_series(s[-1], s[-2])
        _compare_series(s[:-1])
        return None    
    
    # -- Leaves case with len(s)s == 2
    s1, s2 = s
    if not allclose_quantity(s1.dx, s2.dx, atol=0., rtol=1e-5):
        # -- Some arbitrary deviation allowed, in case of numerical differences
        # raise ValueError('Signals must have equal spacing on x-axis.')
        # raise ValueError('Signals do not have equal spacing on x-axis.')
        raise ValueError(f'Signals do not have equal spacing on x-axis ({s1.dx} vs. {s2.dx}).')

    # TODO: decide whether to compare whole frequency array or just
    # start and end frequency. But then things in between could go wrong

    if not allclose_quantity(s1.xindex, s2.xindex, atol=0.5*s1.dx.value, rtol=0.):
        # -- Note: this atol checks for equality up to sampling accuracy
        # -- Note: this automatically checks for equal size
        raise ValueError('Signals must have sufficiently equal xindex. '
                         'Maximum allowed deviation is 0.5*dx.')


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
    assert (
        allclose_quantity(fs.f0.value, 0.0, atol=0.0, rtol=0.0)
        or allclose_quantity(-(fs.f0 + fs.df), fs.frequencies[-1],
                             atol=fs.df.value, rtol=0.0)
    ), ('All signals must start either at f=0 or be symmetric around f=0, '
        'where the latter refers to the case of an odd sample size. For an '
        'even sample size, on the other hand, the number of samples for '
        'positive frequencies is expected to be one less than the number of '
        'samples for negative frequencies, in accordance with the format '
        'expected by `~numpy.fft.ifftshift`. Note that the conditions just'
        'mentioned do not apply to the case of a starting frequency f=0, '
        'where both even and odd sample sizes are accepted.')
