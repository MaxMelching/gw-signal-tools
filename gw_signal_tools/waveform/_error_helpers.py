# -- Standard Lib Imports
from typing import Optional

# -- Third Party Imports
import astropy.units as u


__all__ = ('_UNIT_CONV_ERR', '_q_convert')

__doc__: str = """
Little helper file containing functions and other definitions that help
dealing with errors in package functions.
"""


# _UNIT_CONV_ERR = '''
# Need consistent (i.e. convertible) units for `%(arg1)s` (%(arg1unit)s)
# and `%(arg2)s` (%(arg2unit)s).
# '''
_UNIT_CONV_ERR = '''
Need consistent (i.e. convertible) units for `%s` (%s) and `%s` (%s).
'''
# _FREQU_UNIT_CONV_ERR = '''
# Need consistent (i.e. convertible) frequency units for `%s` (%s) and `%s` (%s).
# '''


# def _convert(q: float | u.Quantity, target_unit: u.Unit, err_msg: str) -> u.Quantity:
#     """Wrapper function for unit conversion with custom error raising."""
#     try:
#         return u.Quantity(q, unit=target_unit)
#         # -- No use of q.to because input might not be Quantity
#     except u.UnitConversionError:
#         raise ValueError(err_msg)

def _q_convert(quant: float | u.Quantity, target_unit: u.Unit,
               arg1name: str, arg2name: str, err_msg: Optional[str] = None) -> u.Quantity:
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

# def _convert(q1: str, q2: str) -> u.Quantity:
#     """Wrapper function for unit conversion with custom error raising."""
#     _q1 = locals()[q1]
#     _q2 = locals()[q2]
#     try:
#         return u.Quantity(_q1, unit=_q2.unit)
#         # -- No use of q.to because input might not be Quantity
#     except u.UnitConversionError:
#         # -- Only happens when q is already Quantity, so we can assume
#         # -- that accessing the unit property works
#         raise ValueError(_UNIT_CONV_ERR % (q1, _q1.unit, q2, _q2.unit))
# -- Does not work
