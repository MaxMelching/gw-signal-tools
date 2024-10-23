# -- Standard Lib Imports
from typing import Optional

# -- Third Party Imports
import astropy.units as u


__all__ = ('_UNIT_CONV_ERR', '_q_convert')

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
