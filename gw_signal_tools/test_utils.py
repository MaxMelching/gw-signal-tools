import numpy as np
from numpy.testing import assert_allclose

from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
import astropy.units as u


def allclose_quantity(arr1: u.Quantity, arr2: u.Quantity, *args,
                      **kwargs) -> bool:
    """
    Wrapper to apply numpy function `isclose` to astropy Quantities
    (a `~numpy.all` wrapper is also added to support arrays as well).
    Natively, numpy does not support this due to the units attached
    to them. All arguments besides the Quantities to compare are passed
    to `~numpy.isclose`.

    Parameters
    ----------
    arr1 : ~astropy.units.Quantity
        First value to compare.
    arr2 : ~astropy.units.Quantity
        Second value to compare.
    
    Returns
    -------
    bool
        Result of the comparison of `arr1` and `arr2`.

    Notes
    -----
    This function is heavily inspired by a function from GWPy, see
    https://github.com/gwpy/gwpy/blob/v3.0.7/gwpy/testing/utils.py#L131.
    """

    if not isinstance(arr1, u.Quantity):
        arr1 = u.Quantity(arr1)

    if not isinstance(arr2, u.Quantity):
        arr2 = u.Quantity(arr2)
    
    assert arr1.unit == arr2.unit, \
        f'Cannot compare unequal units, {arr1.unit} != {arr2.unit}.'
    

    return np.all(np.isclose(arr1.value, arr2.value, *args, **kwargs))


def assert_allclose_quantity(arr1: u.Quantity, arr2: u.Quantity, *args,
                             **kwargs) -> None:
    """
    Wrapper to apply numpy function `assert_allclose` to astropy
    Quantities. Natively, numpy does not support this due to the units
    attached to them. All arguments besides the Quantities to compare
    are passed to `assert_allclose`.

    Parameters
    ----------
    arr1 : ~astropy.units.Quantity
        First value to compare.
    arr2 : ~astropy.units.Quantity
        Second value to compare.

    Notes
    -----
    This function is heavily inspired by a function from GWPy, see
    https://github.com/gwpy/gwpy/blob/v3.0.7/gwpy/testing/utils.py#L131.
    """

    if not isinstance(arr1, u.Quantity):
        arr1 = u.Quantity(arr1)

    if not isinstance(arr2, u.Quantity):
        arr2 = u.Quantity(arr2)
    
    assert arr1.unit == arr2.unit, f'{arr1.unit} != {arr2.unit}'

    assert_allclose(arr1.value, arr2.value, *args, **kwargs)


def assert_allclose_frequseries(
    series1: FrequencySeries,
    series2: FrequencySeries,
    *args,
    **kwargs
) -> None:
    assert_allclose_quantity(series1.frequencies, series2.frequencies,
                             *args, **kwargs)
    assert_allclose_quantity(series1, series2, *args, **kwargs)


def assert_allclose_timeseries(
    series1: TimeSeries,
    series2: TimeSeries,
    *args,
    **kwargs
) -> None:
    assert_allclose_quantity(series1.frequencies, series2.frequencies,
                             *args, **kwargs)
    assert_allclose_quantity(series1, series2, *args, **kwargs)
