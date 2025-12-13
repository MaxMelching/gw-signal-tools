# -- Third Party Imports
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gwpy.types import Series

# -- Local Package Imports
from .types import MatrixWithUnits


__doc__ = """
Convenient wrappers to test closeness or equality of various types.
"""

__all__ = (
    'allclose_quantity',
    'assert_allclose_quantity',
    'assert_allclose_MatrixWithUnits',
    'assert_allequal_MatrixWithUnits',
    'assert_allclose_series',
    'assert_allequal_series',
)


# -- Quantity equalities
def allclose_quantity(arr1: u.Quantity, arr2: u.Quantity, *args, **kwargs) -> np.bool_:
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

    assert (
        arr1.unit == arr2.unit
    ), f'Cannot compare unequal units, {arr1.unit} != {arr2.unit}.'

    return np.all(np.isclose(arr1.value, arr2.value, *args, **kwargs))


def assert_allclose_quantity(
    arr1: u.Quantity, arr2: u.Quantity, *args, **kwargs
) -> None:
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


# -- MatrixWithUnits equalities
def assert_allclose_MatrixWithUnits(
    matr1: MatrixWithUnits, matr2: MatrixWithUnits, *args, **kwargs
) -> None:
    """
    Wrapper to apply numpy function `assert_allclose` to a
    ``MatrixWithUnit``. All arguments besides the matrices to compare
    are passed to `assert_allclose`.

    Parameters
    ----------
    matr1 : ~gw_signal_tools.matrix_with_unit.MatrixWithUnit
        First value to compare.
    matr2 : ~gw_signal_tools.matrix_with_unit.MatrixWithUnit
        Second value to compare.

    Raises
    ------
    AssertionError
        If the inputs are not equal to the desired accuracy.

    Notes
    -----
    This function is heavily inspired by a function from GWPy, see
    https://github.com/gwpy/gwpy/blob/v3.0.7/gwpy/testing/utils.py#L131.
    """

    assert np.all(np.equal(matr1.unit, matr2.unit))
    # NOT equivalent to ==, equal has better behaviour (also able to compare
    # unit arrays and scalar units in way we intend to)

    assert_allclose(matr1.value, matr2.value, *args, **kwargs)


def assert_allequal_MatrixWithUnits(
    matr1: MatrixWithUnits, matr2: MatrixWithUnits
) -> None:
    """
    Wrapper to assert equality of two ``MatrixWithUnit`` instances.

    Parameters
    ----------
    matr1 : ~gw_signal_tools.matrix_with_unit.MatrixWithUnit
        First value to compare.
    matr2 : ~gw_signal_tools.matrix_with_unit.MatrixWithUnit
        Second value to compare.

    Raises
    ------
    AssertionError
        If the inputs are not equal.
    """

    assert np.all(matr1 == matr2)  # Uses np.equal already, this is sufficient


# -- GWpy type equalities
def assert_allclose_series(series1: Series, series2: Series, *args, **kwargs) -> None:
    """
    Wrapper for application of `~gw_signal_tools.
    assert_allclose_quantity` to a GWPy ``Series`` instance (includes
    ``FrequencySeries`` and ``TimeSeries``).

    Parameters
    ----------
    series1 :  ~gwpy.types.series.Series
        First value to compare.
    series2 : ~gwpy.types.series.Series
        Second value to compare.
    """
    assert_allclose_quantity(series1.xindex, series2.xindex, *args, **kwargs)
    assert_allclose_quantity(series1, series2, *args, **kwargs)


def assert_allequal_series(series1: Series, series2: Series) -> None:
    """
    Assert that two GWPy ``Series`` instances are equal (includes
    ``FrequencySeries`` and ``TimeSeries``).

    Parameters
    ----------
    series1 : ~gwpy.types.series.Series
        First value to compare.
    series2 : ~gwpy.types.series.Series
        Second value to compare.
    """
    assert series1.xindex.unit == series2.xindex.unit
    assert np.all(np.equal(series1.xindex.value, series2.xindex.value))

    assert series1.unit == series2.unit
    assert np.all(np.equal(series1.value, series2.value))
