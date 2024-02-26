# ----- Third Party Imports -----
import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u

from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

# ----- Local Package Imports -----
from gw_signal_tools.matrix_with_units import MatrixWithUnits


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
    
    assert arr1.unit == arr2.unit, f'{arr1.unit} != {arr2.unit}'

    assert_allclose(arr1.value, arr2.value, *args, **kwargs)


def assert_allclose_MatrixWithUnits(
    arr1: MatrixWithUnits,
    arr2: MatrixWithUnits,
    *args, **kwargs
) -> None:
    """
    Wrapper to apply numpy function `assert_allclose` to a
    ``MatrixWithUnit``. All arguments besides the matrices to compare
    are passed to `assert_allclose`.

    Parameters
    ----------
    arr1 : ~gw_signal_tools.matrix_with_unit.MatrixWithUnit
        First value to compare.
    arr2 : ~gw_signal_tools.matrix_with_unit.MatrixWithUnit
        Second value to compare.

    Notes
    -----
    This function is heavily inspired by a function from GWPy, see
    https://github.com/gwpy/gwpy/blob/v3.0.7/gwpy/testing/utils.py#L131.
    """
    
    assert np.all(arr1.unit == arr2.unit)

    assert_allclose(arr1.value, arr2.value, *args, **kwargs)


def assert_allequal_MatrixWithUnits(
    arr1: MatrixWithUnits,
    arr2: MatrixWithUnits
) -> None:
    """
    Wrapper to assert equality of two ``MatrixWithUnit`` instances.

    Parameters
    ----------
    arr1 : ~gw_signal_tools.matrix_with_unit.MatrixWithUnit
        First value to compare.
    arr2 : ~gw_signal_tools.matrix_with_unit.MatrixWithUnit
        Second value to compare.
    """
    
    assert np.all(arr1.value == arr2.value)
    assert np.all(arr1.unit == arr2.unit)


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
