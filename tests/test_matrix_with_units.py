# -- Standard Lib Imports
import unittest

# -- Third Party Imports
import numpy as np
import astropy.units as u
import pytest
import matplotlib.pyplot as plt

# -- Local Package Imports
from gw_signal_tools.types import MatrixWithUnits
from gw_signal_tools.test_utils import (
    assert_allclose_MatrixWithUnits, assert_allequal_MatrixWithUnits
)
from gw_signal_tools import preferred_unit_system


example_values = np.array([[42, 24], [18, 96]])
example_units = np.array([[u.s, u.m], [u.m, u.s]])
example_non_si_units = np.array([[2.0 * u.s, u.m], [u.m**2, u.s]])
example_scaled_units = np.array([[u.Quantity(1e-3, u.s), u.Quantity(1.0, u.m)], [u.Quantity(1e2, u.m), u.Quantity(1.0, u.s)]], dtype=object)


#%% -- Test cornerstone properties, value and unit ----------------------------
def test_different_inits():
    value_arr = np.array([[42., 24.], [18., 96.]])  # Quantity converts to float anyway!
    unit_arr = np.array([[u.s, u.pc], [u.m, u.kg]])
    matrix_1 = MatrixWithUnits(value_arr, unit_arr)
    assert matrix_1.value.dtype == float

    value_unit_arr = value_arr*unit_arr
    matrix_2 = MatrixWithUnits(value_unit_arr)
    assert matrix_2.value.dtype == float

    value_unit_list = [[42*u.s, 24*u.pc], [18*u.m, 96*u.kg]]
    matrix_3 = MatrixWithUnits(value_unit_list)
    assert matrix_3.value.dtype == float

    assert_allequal_MatrixWithUnits(matrix_1, matrix_2)
    assert_allequal_MatrixWithUnits(matrix_1, matrix_3)


def test_convert_int():
    values = [[42, 24], [18, 96]]
    matrix_float = MatrixWithUnits(values, example_units, convert_int=True)
    matrix_int = MatrixWithUnits(values, example_units, convert_int=False)

    print(matrix_float.value.dtype)

    assert matrix_float.value.dtype == float
    assert matrix_int.value.dtype == int
test_convert_int()

@pytest.mark.parametrize('units', [example_units, example_non_si_units, example_scaled_units])
def test_unit_matrix_reading(units):
    matrix = MatrixWithUnits(example_values, units)

    assert np.all(matrix.unit == units)


@pytest.mark.parametrize('unit', [u.s, 2.0 * u.s])
def test_unit_scalar_reading(unit):
    matrix = MatrixWithUnits(example_values, unit)

    if isinstance(unit, u.Quantity):
        # unit = u.CompositeUnit(unit.value, [unit.unit], [1.0])
        unit = u.Unit(unit)

    # np.all(matrix.unit == np.full(example_values.shape, unit, dtype=object))
    assert np.all(np.equal(matrix.unit, unit))


def test_scalar_unit():
    matrix = MatrixWithUnits(example_values, example_units)
    matrix.unit = u.s

    assert_allequal_MatrixWithUnits(
        matrix,
        MatrixWithUnits(example_values, u.s)
    )

    matrix = MatrixWithUnits(example_values, u.s)
    matrix.unit = example_units

    assert_allequal_MatrixWithUnits(
        matrix,
        MatrixWithUnits(example_values, example_units)
    )


#%% -- Test standard class functions ------------------------------------------
def test_getting_and_slicing():
    matrix = MatrixWithUnits(example_values, example_units)

    assert_allequal_MatrixWithUnits(matrix[0], MatrixWithUnits([42, 24], [u.s, u.m]))

    assert_allequal_MatrixWithUnits(matrix[0, 0], MatrixWithUnits(42, u.s))

    assert_allequal_MatrixWithUnits(matrix[:1, :], MatrixWithUnits([42, 24], [u.s, u.m]))
    assert_allequal_MatrixWithUnits(matrix[:, :1], MatrixWithUnits([[42], [18]], [[u.s], [u.m]]))
    
    
    matrix2 = MatrixWithUnits(example_values, example_non_si_units)

    assert_allequal_MatrixWithUnits(matrix2[0], MatrixWithUnits([42, 24], [2.0 * u.s, u.m]))
    assert_allequal_MatrixWithUnits(matrix2[1, 0], MatrixWithUnits(18, u.m**2))


@pytest.mark.parametrize('set_val', [-2*u.dimensionless_unscaled, 3*u.pc, 42])
@pytest.mark.parametrize('units', [example_units, u.s])
def test_setting(set_val, units):
    matrix = MatrixWithUnits(example_values, units).copy()

    # We know that getting works because previous test passed, can rely on that
    matrix[0, 0] = set_val

    assert matrix[0, 0] == set_val
    assert matrix[1, 1].unit == u.s
    # Only 1,1 component tested because it is same one for example_units and
    # scalar case u.s (otherwise values are unequal or not indexable)


def test_eq():
    matrix_in_s1 = MatrixWithUnits(example_values, u.s)

    assert np.all(np.equal(matrix_in_s1 == matrix_in_s1, True))  # Trivial, but must work

    # -- Test that equality between scalar and array unit works
    matrix_in_s2 = MatrixWithUnits(example_values, np.full(example_values.shape, u.s, dtype=object))
    assert np.all(np.equal(matrix_in_s1 == matrix_in_s2, True))

    matrix_in_m = MatrixWithUnits(example_values, u.m)
    assert np.all(np.equal(matrix_in_s1 == matrix_in_m, False))

    # -- Test where not necessarily all units are unequal, but some
    matrix = MatrixWithUnits(example_values, example_units)

    assert np.all(matrix_in_s1 == matrix) == False
    assert np.any(matrix_in_s1 != matrix) == True  # Test not equal operator
    
    matrix_dim_less = MatrixWithUnits(example_values)
    assert np.all(matrix_dim_less == example_values)

    matrix_single_val = MatrixWithUnits(42, u.dimensionless_unscaled)
    assert np.all(matrix_single_val == 42)


def test_float():
    matrix = MatrixWithUnits(42, u.dimensionless_unscaled)

    assert float(matrix) == 42


def test_copy():
    matrix = MatrixWithUnits(example_values, example_units)
    component_val_copy = matrix[0, 0]

    matrix_copy = matrix.copy()  # Calls __copy__, so this is tested automatically

    assert matrix[0, 0] == matrix_copy[0, 0]

    matrix_copy[0, 0] = -2*u.dimensionless_unscaled  # Inplace setting

    assert (matrix[0, 0] == component_val_copy) and (matrix[0, 0] != matrix_copy[0, 0])
    
    # -- Test if equivalent calls work
    matrix_copy = MatrixWithUnits.copy(matrix)
    from copy import copy, deepcopy
    matrix_copy = copy(matrix)
    matrix_copy = deepcopy(matrix)


def test_dict_conversion():
    matrix = MatrixWithUnits(example_values, example_units)

    test_dict = {'matrix': matrix}
    assert_allequal_MatrixWithUnits(test_dict['matrix'], matrix)
    # Apparently, __hash__ is not needed, good


#%% -- Test common operations -------------------------------------------------
def test_addition():
    matrix = MatrixWithUnits(example_values, example_units)
    
    # -- Float addition
    assert_allequal_MatrixWithUnits(
        matrix + 2.0,
        MatrixWithUnits(example_values + 2.0, example_units)
    )

    assert_allequal_MatrixWithUnits(
        2.0 + matrix,
        MatrixWithUnits(2.0 + example_values, example_units)
    )

    # -- Quantitiy addition -> only works with scalar unit
    matrix_in_s = MatrixWithUnits(example_values, u.s)
    
    assert_allequal_MatrixWithUnits(
        matrix_in_s + 2.0 * u.s,
        MatrixWithUnits(example_values + 2.0, u.s)
    )
    
    assert_allequal_MatrixWithUnits(
        2.0 * u.s + matrix_in_s,
        MatrixWithUnits(2.0 + example_values, u.s)
    )

    # -- MatrixWithUnits addition -> similar to test_multiplication
    assert_allequal_MatrixWithUnits(
        matrix + matrix,
        MatrixWithUnits(example_values + example_values, example_units)
    )


def test_subtraction():
    matrix = MatrixWithUnits(example_values, example_units)

    # -- Float subtraction
    assert_allequal_MatrixWithUnits(
        matrix - 2.0,
        MatrixWithUnits(example_values - 2.0, example_units)
    )

    assert_allequal_MatrixWithUnits(
        2.0 - matrix,
        MatrixWithUnits(2.0 - example_values, example_units)
    )

    # -- Quantity subtraction
    matrix_in_s = MatrixWithUnits(example_values, u.s)
    
    assert_allequal_MatrixWithUnits(
        matrix_in_s - 2.0 * u.s,
        MatrixWithUnits(example_values - 2.0, u.s)
    )
    
    assert_allequal_MatrixWithUnits(
        2.0 * u.s  + matrix_in_s,
        MatrixWithUnits(2.0 + example_values, u.s)
    )

    # -- MatrixWithUnits subtraction -> similar to test_multiplication
    assert_allequal_MatrixWithUnits(
        matrix - 0.5 * matrix,
        MatrixWithUnits(0.5 * example_values, example_units)
    )


def test_multiplication():
    matrix = MatrixWithUnits(example_values, example_units)
    matrix_in_s = MatrixWithUnits(example_values, u.s)

    # -- Float multiplication
    assert_allequal_MatrixWithUnits(
        matrix * 2.0,
        MatrixWithUnits(example_values * 2.0, example_units)
    )

    assert_allequal_MatrixWithUnits(
        2.0 * matrix,
        MatrixWithUnits(2.0 * example_values, example_units)
    )

    # -- Verify results of float multiplication coincide with add/sub
    assert_allequal_MatrixWithUnits(2 * matrix, matrix + matrix)

    assert_allequal_MatrixWithUnits(0.5 * matrix, matrix - 0.5 * matrix)

    # -- Quantity multiplication
    example_units_times_s = np.array([[u.s**2, u.m*u.s], [u.m*u.s, u.s**2]])
    
    assert_allequal_MatrixWithUnits(
        matrix_in_s * (2.0 * u.s),
        MatrixWithUnits(example_values * 2.0, u.s * u.s)
    )
    
    assert_allequal_MatrixWithUnits(
        matrix * (2.0 * u.s),
        MatrixWithUnits(example_values * 2.0, example_units_times_s)
    )
    
    assert_allequal_MatrixWithUnits(
        (2.0 * u.s) * matrix_in_s,
        MatrixWithUnits(2.0 * example_values, u.s * u.s)
    )
    
    assert_allequal_MatrixWithUnits(
        (2.0 * u.s) * matrix,
        MatrixWithUnits(2.0 * example_values, example_units_times_s)
    )

    # -- Unit multiplication
    assert_allequal_MatrixWithUnits(
        matrix_in_s * u.s,
        MatrixWithUnits(example_values, u.s**2)
    )
    assert_allequal_MatrixWithUnits(
        matrix * u.s,
        MatrixWithUnits(example_values, example_units_times_s)
    )

    assert_allequal_MatrixWithUnits(
        u.s * matrix_in_s,
        MatrixWithUnits(example_values, u.s**2)
    )

    assert_allequal_MatrixWithUnits(
        u.s * matrix,
        MatrixWithUnits(example_values, example_units_times_s)
    )

    # -- MatrixWithUnit Multiplication -> similar to test_power
    assert_allequal_MatrixWithUnits(
        matrix * matrix,
        MatrixWithUnits(example_values**2, example_units**2)
    )


def test_division():
    matrix = MatrixWithUnits(example_values, example_units)
    matrix_in_s = MatrixWithUnits(example_values, u.s)

    # -- Float division
    assert_allequal_MatrixWithUnits(
        matrix / 2.0,
        MatrixWithUnits(example_values / 2.0, example_units)
    )

    assert_allequal_MatrixWithUnits(
        2.0 / matrix,
        MatrixWithUnits(2.0 / example_values, 1 / example_units)
    )

    # -- Quantity division
    example_units_by_s = np.array([[u.dimensionless_unscaled, u.m/u.s],
                                   [u.m/u.s, u.dimensionless_unscaled]])
    s_by_example_units = np.array([[u.dimensionless_unscaled, u.s/u.m],
                                   [u.s/u.m, u.dimensionless_unscaled]])
    
    assert_allequal_MatrixWithUnits(
        matrix_in_s / (2.0 * u.s),
        MatrixWithUnits(example_values / 2.0, u.dimensionless_unscaled)
    )
    
    assert_allequal_MatrixWithUnits(
        matrix / (2.0 * u.s),
        MatrixWithUnits(example_values / 2.0, example_units_by_s)
    )
    
    assert_allequal_MatrixWithUnits(
        (2.0 * u.s) / matrix_in_s,
        MatrixWithUnits(2.0 / example_values, u.dimensionless_unscaled)
    )
    
    assert_allequal_MatrixWithUnits(
        (2.0 * u.s) / matrix,
        MatrixWithUnits(2.0 / example_values, s_by_example_units)
    )

    # -- Unit division
    assert_allequal_MatrixWithUnits(
        matrix_in_s / u.s,
        MatrixWithUnits(example_values, u.dimensionless_unscaled)
    )

    assert_allequal_MatrixWithUnits(
        matrix / u.s,
        MatrixWithUnits(example_values, example_units_by_s)
    )

    assert_allequal_MatrixWithUnits(
        u.s / matrix_in_s,
        MatrixWithUnits(1/example_values, u.dimensionless_unscaled)
    )

    assert_allequal_MatrixWithUnits(
        u.s / matrix,
        MatrixWithUnits(1/example_values, s_by_example_units)
    )

    # -- MatrixWithUnit Division -> similar to test_power
    assert_allequal_MatrixWithUnits(
        matrix / matrix,
        MatrixWithUnits(np.ones((2, 2)),
                        np.full((2, 2), u.dimensionless_unscaled))
    )


def test_power():
    matrix = MatrixWithUnits(example_values, example_units)
    
    assert_allequal_MatrixWithUnits(
        matrix**2,
        MatrixWithUnits(example_values**2, example_units**2)
    )

    assert_allequal_MatrixWithUnits(matrix * matrix, matrix**2)


    assert_allequal_MatrixWithUnits(
        matrix**0,
        MatrixWithUnits(np.ones((2, 2)),
                        np.full((2, 2), u.dimensionless_unscaled))
    )

    assert_allclose_MatrixWithUnits(
        matrix**(1/2),
        MatrixWithUnits(example_values**(1/2), example_units**(1/2))
    )


def test_matmul():
    matrix = MatrixWithUnits(example_values, example_units)
    example_units_2 = np.array([[u.s, u.m], [u.s, u.m]], dtype=object)
    matrix2 = MatrixWithUnits(example_values, example_units_2)

    matrix_in_s = MatrixWithUnits(example_values, u.s)

    example_units_s = np.full((2, 2), u.s, dtype=object)
    matrix_in_s_2 = MatrixWithUnits(example_values, example_units_s)

    # -- Test units support, i.e. multiplication with scalar units etc
    assert_allequal_MatrixWithUnits(
        matrix_in_s @ matrix_in_s,
        matrix_in_s @ matrix_in_s_2
    )

    assert_allequal_MatrixWithUnits(
        matrix_in_s @ matrix_in_s,
        MatrixWithUnits(example_values @ example_values, u.s**2)
    )

    assert_allequal_MatrixWithUnits(
        matrix_in_s @ matrix_in_s_2,
        MatrixWithUnits(example_values @ example_values, example_units_s**2)
    )

    assert_allequal_MatrixWithUnits(
        matrix_in_s_2 @ matrix_in_s,
        matrix_in_s @ matrix_in_s_2,
    )

    # -- Now tests for matrices that do not have single unit
    assert_allequal_MatrixWithUnits(
        matrix_in_s @ matrix2,
        # MatrixWithUnits(example_values @ example_values, example_units_2 * example_units_s)
        MatrixWithUnits(example_values @ example_values, np.array([[u.s**2, u.m * u.s], [u.s**2, u.m * u.s]], dtype=object))
    )

    assert_allequal_MatrixWithUnits(
        matrix2.T @ matrix_in_s,
        MatrixWithUnits(example_values.T @ example_values, np.array([[u.s**2, u.s**2], [u.m * u.s, u.m * u.s]], dtype=object))
    )

    # -- Test with rows and columns
    matrix_col = matrix_in_s[:, 0]
    matrix_col = MatrixWithUnits.reshape(matrix_col, (2, 1))
    matrix_row = matrix_in_s[0, :]
    matrix_row = MatrixWithUnits.reshape(matrix_row, (1, 2))

    assert_allequal_MatrixWithUnits(
        matrix_row @ matrix_col,  # Inner product
        MatrixWithUnits(42**2 + 24 * 18, u.s**2)
    )

    assert_allequal_MatrixWithUnits(
        matrix_col @ matrix_row,  # Outer product
        MatrixWithUnits(np.array([[42*42, 42*24], [18*42, 18*24]]), u.s**2)
    )

    assert_allequal_MatrixWithUnits(
        matrix_in_s @ matrix_col,  # Matrix and column
        MatrixWithUnits(np.array([[42*42 + 24*18], [18*42 + 96*18]]), u.s**2)
    )

    assert_allequal_MatrixWithUnits(
        matrix2.T @ matrix_col,  # Matrix and column
        MatrixWithUnits(np.array([[42*42 + 18*18], [24*42 + 96*18]]), np.array([[u.s**2], [u.s*u.m]], dtype=object))
    )

    assert_allequal_MatrixWithUnits(
        matrix_row @ matrix_in_s,  # Row and matrix
        MatrixWithUnits(np.array([42*42 + 24*18, 42*24 + 24*96]), u.s**2)
    )

    assert_allequal_MatrixWithUnits(
        matrix_row @ matrix2,  # Row and matrix
        MatrixWithUnits(np.array([42*42 + 24*18, 42*24 + 24*96]), np.array([u.s**2, u.s * u.m], dtype=object))
    )

    # -- Now column/row with non-scalar unit
    matrix_col = MatrixWithUnits(example_values[:, 0], np.array([u.s, u.m]))
    matrix_col = MatrixWithUnits.reshape(matrix_col, (2, 1))
    matrix_row = MatrixWithUnits(example_values[0, :], np.array([u.m, u.s]))
    matrix_row = MatrixWithUnits.reshape(matrix_row, (1, 2))

    assert_allequal_MatrixWithUnits(
        matrix_row @ matrix_col,  # Inner product
        MatrixWithUnits(42**2 + 24 * 18, u.s*u.m)
    )

    assert_allequal_MatrixWithUnits(
        matrix_col @ matrix_row,  # Outer product
        MatrixWithUnits(np.array([[42*42, 42*24], [18*42, 18*24]]),
                        np.array([[u.s*u.m, u.s**2], [u.m**2, u.m*u.s]], dtype=object))
    )


@pytest.mark.parametrize('sign', [+1, -1])
def test_abs(sign):
    matrix = MatrixWithUnits(sign*example_values, example_units)
    matrix_abs = MatrixWithUnits(np.abs(example_values), example_units)

    assert_allequal_MatrixWithUnits(abs(matrix), matrix_abs)


#%% -- Test numpy functions ---------------------------------------------------
def test_transposing():
    matrix = MatrixWithUnits(example_values, example_units)

    assert_allequal_MatrixWithUnits(
        matrix.T,
        MatrixWithUnits(
            np.array([[42, 18], [24, 96]]),
            example_units  # Is symmetric
        )
    )

    matrix2 = MatrixWithUnits(example_values, u.s)
    
    assert_allequal_MatrixWithUnits(
        matrix2.T,
        MatrixWithUnits(
            np.array([[42, 18], [24, 96]]),
            u.s
        )
    )


def test_array_conversion():
    matrix = MatrixWithUnits(example_values, example_units)
    matrix = MatrixWithUnits(example_values, example_units)
    matrix_array = np.array(matrix)

    np.all(matrix_array == matrix.value)


@pytest.mark.parametrize('units', [example_units, u.s])
def test_len(units):
    matrix = MatrixWithUnits(example_values, units)

    assert len(matrix) == 2


@pytest.mark.parametrize('units', [example_units, u.s])
def test_size(units):
    matrix = MatrixWithUnits(example_values, units)

    assert matrix.size == 4


@pytest.mark.parametrize('units', [example_units, u.s])
def test_shape(units):
    matrix = MatrixWithUnits(example_values, units)
    
    assert matrix.shape == (2, 2)


@pytest.mark.parametrize('new_shape', [(2, 2), (1, 4), (4, 1), -1])
def test_reshape(new_shape):
    matrix = MatrixWithUnits(example_values, example_units)
    
    matrix2 = MatrixWithUnits(np.reshape(example_values, new_shape),
                              np.reshape(example_units, new_shape))
    
    assert_allequal_MatrixWithUnits(
        MatrixWithUnits.reshape(matrix, new_shape),
        matrix2
    )


@pytest.mark.parametrize('units', [example_units, u.s])
def test_ndim(units):
    matrix = MatrixWithUnits(example_values, units)
    
    assert matrix.ndim == 2


@pytest.mark.parametrize('values', [example_values, np.array([42], dtype=int),
    np.array([42.], dtype=float), np.array([42.j], dtype=complex)])
def test_dtype(values):
    matrix = MatrixWithUnits(values, u.s, convert_int=False)
    
    assert matrix.value.dtype == values.dtype
    assert matrix.dtype == u.Quantity
    assert type(matrix.reshape(-1)[0]) == u.Quantity


def test_reading_from_array():
    matrix = MatrixWithUnits(example_values)

    assert np.all(np.equal(matrix.value, example_values))
    assert np.all(np.equal(matrix.unit, u.dimensionless_unscaled))


def test_inv():
    matrix = MatrixWithUnits(example_values, u.s)
    matrix_inv = MatrixWithUnits.inv(matrix)

    assert_allclose_MatrixWithUnits(
        matrix @ matrix_inv,
        MatrixWithUnits(np.eye(2))
    )

    test_units_arr = np.array([u.s, u.m, u.kg], dtype=object)
    test_units = np.outer(test_units_arr, test_units_arr)
    test_values = [[1, 2, 1], [5, 6, 7], [9, 8, 7]]
    matrix2 = MatrixWithUnits(test_values, test_units)
    # matrix2_inv = MatrixWithUnits(np.linalg.inv(test_values), test_units**-1)
    matrix2_inv = MatrixWithUnits.inv(matrix2)

    # assert_allclose(np.eye(3), matrix2 @ matrix2_inv, atol=1e-15, rtol=0.0)
    assert_allclose_MatrixWithUnits(
        matrix2 @ matrix2_inv,
        MatrixWithUnits(
            np.eye(3), 
            np.array([[u.dimensionless_unscaled, u.s / u.m, u.s / u.kg],
                      [u.m / u.s, u.dimensionless_unscaled, u.m / u.kg],
                      [u.kg / u.s, u.kg / u.m, u.dimensionless_unscaled]]),
            # Unit result calculated by hand
        ),
        atol=1e-15,  # Account for numerical errors
        rtol=0.0
    )


def test_diagonal():
    matrix = MatrixWithUnits(example_values, example_units)

    assert_allequal_MatrixWithUnits(
        matrix.diagonal(),
        MatrixWithUnits([42, 96], u.s)
    )

    matrix_in_s = MatrixWithUnits(example_values, u.s)

    assert_allequal_MatrixWithUnits(
        matrix_in_s.diagonal(),
        MatrixWithUnits([42, 96], u.s)
    )

    # -- Testing args
    assert_allequal_MatrixWithUnits(
        matrix_in_s.diagonal(1),
        MatrixWithUnits([24], u.s)
    )


def test_sqrt():
    matrix = MatrixWithUnits(example_values, example_units)

    assert_allequal_MatrixWithUnits(
        matrix.sqrt(),
        MatrixWithUnits(np.sqrt(example_values), example_units**(1/2))
    )


def test_cond():
    # -- Here we just make sure that calling works. Results are assumed
    # -- to be correct because we just pass to numpy
    matrix = MatrixWithUnits(example_values, example_units)

    matrix.cond()
    matrix.cond(2)
    matrix.cond('nuc')


#%% -- Test astropy functions -------------------------------------------------
@pytest.mark.parametrize('units', [example_units, u.s])
def test_to_system(units):
    matrix = MatrixWithUnits(example_values, units)

    matrix.to_system(u.si)
    matrix.to_system(preferred_unit_system)


def test_to():
    matrix = MatrixWithUnits([42., 96.], [u.m, u.km])

    matrix[0] = matrix[0].to(u.km)
    assert_allequal_MatrixWithUnits(matrix, MatrixWithUnits([0.042, 96], [u.km, u.km]))

    # -- Now test for whole matrix
    matrix_nm = matrix.to(u.nm)
    assert_allclose_MatrixWithUnits(
        matrix_nm,
        MatrixWithUnits([0.042e12, 96e12], [u.nm, u.nm]),
        atol=0.0,
        rtol=1e-15  # Numerical errors, I think in initialization
    )


def test_decompose():
    matrix = MatrixWithUnits(example_values, [[u.Msun, u.pc], [u.km, u.h]])
    
    matrix_dec = matrix.decompose(bases=preferred_unit_system.bases)
    assert_allclose_MatrixWithUnits(
        matrix_dec,
        MatrixWithUnits(example_values*np.array([[1., 1.], [(u.km/u.pc).si.scale, (u.h/u.s).si.scale]]), [[u.Msun, u.pc], [u.pc, u.s]]),
        atol=1e-12, rtol=0.
    )

    matrix_dec_2 = matrix.decompose(bases=[u.Unit(1e-6*u.Msun), u.km, u.h])
    assert_allclose_MatrixWithUnits(
        matrix_dec_2,
        MatrixWithUnits(example_values*np.array([[1e6, (u.pc/u.km).si.scale], [1., 1.]]), [[u.Unit(1e-6*u.Msun), u.km], [u.km, u.h]]),
        atol=0., rtol=1e-15
    )

    matrix_dec_3 = matrix.copy()
    matrix_dec_3[0, :] = matrix_dec_3[0, :].decompose(bases=[u.Unit(1e-6*u.Msun), u.km])
    assert_allclose_MatrixWithUnits(
        matrix_dec_3,
        MatrixWithUnits(example_values*np.array([[1e6, (u.pc/u.km).si.scale], [1., 1.]]), [[u.Unit(1e-6*u.Msun), u.km], [u.km, u.h]]),
        atol=0., rtol=1e-15
    )


#%% -- Test custom additions --------------------------------------------------
@pytest.mark.parametrize('given_ax', [True, False])
def test_plot(given_ax):
    if given_ax:
        fig, ax = plt.subplots()
    else:
        ax = None

    matrix = MatrixWithUnits(example_values, example_units)

    matrix.plot(ax=ax)
    plt.close()

    MatrixWithUnits.plot(matrix)
    plt.close()
    
    matrix[:, 0].reshape((1, 2)).plot()
    plt.close()

    # Main goal is to make sure there are no errors

@pytest.mark.parametrize('units', [example_units, u.s])
def test_to_row_to_col(units):
    matrix = MatrixWithUnits(example_values, units)

    row = matrix.to_row()
    assert row.shape == (1, 4)

    col = matrix.to_col()
    assert col.shape == (4, 1)


#%% -- Test error raising -----------------------------------------------------
class Errors(unittest.TestCase):
    matrix = MatrixWithUnits(example_values, example_units)
    matrix_in_s = MatrixWithUnits(example_values, u.s)

    def test_wrong_unit_setting(self):
        with self.assertRaises(AssertionError):
            self.matrix.unit = example_units[:, 0]
    
    # -- Following does not work as intended
    # def test_forcing_incompatible_matmul_unit(self):
    #     with self.assertRaises(ValueError):
    #         matrix = self.matrix.copy()
    #         matrix._unit = 1.
    #         matrix @ matrix

    def test_operations_with_wrong_type(self):
        # -- Setting
        with self.assertRaises(TypeError):
            self.matrix[0] = {'matrix': self.matrix}
        
        # -- Equality testing
        with self.assertRaises(TypeError):
            self.matrix == {'key': 1}
        
        # -- Operations
        with self.assertRaises(TypeError):
            self.matrix + {'key': 1}
        with self.assertRaises(TypeError):
            {'key': 1} + self.matrix

        with self.assertRaises(TypeError):
            self.matrix - {'key': 1}
        with self.assertRaises(TypeError):
            {'key': 1} - self.matrix

        with self.assertRaises(TypeError):
            self.matrix * {'key': 1}
        with self.assertRaises(TypeError):
            {'key': 1} * self.matrix

        with self.assertRaises(TypeError):
            self.matrix / {'key': 1}
        with self.assertRaises(TypeError):
            {'key': 1} / self.matrix
    
    def test_hash(self):
        with self.assertRaises(TypeError):
            hash(self.matrix)

    def test_list_operations(self):
        with self.assertRaises(TypeError):
            self.matrix + [1, 2, 3]

        with self.assertRaises(TypeError):
            self.matrix * [1, 2, 3]

        with self.assertRaises(TypeError):
            self.matrix / [1, 2, 3]

        with self.assertRaises(TypeError):
            self.matrix**[1, 2, 3]
    
    def test_quantitiy_addition(self):
        # -- Test that unequal units throw error
        with self.assertRaises(AssertionError):
            self.matrix + (2.0 * u.s)

        with self.assertRaises(AssertionError):
            (2.0 * u.s) + self.matrix
    
    def test_quantitiy_subtraction(self):
        # -- Test that unequal units throw error
        with self.assertRaises(AssertionError):
            self.matrix - (2.0 * u.s)

        with self.assertRaises(AssertionError):
            (2.0 * u.s) - self.matrix
    
    def test_matmul_wrong_units(self):
        with self.assertRaises(AssertionError):
            self.matrix @ self.matrix  # Due to units not fitting together
    
    def test_matmul_wrong_shapes(self):
        # -- Scalars
        with self.assertRaises(ValueError):
            self.matrix @ MatrixWithUnits(42, u.s)

        with self.assertRaises(ValueError):
            MatrixWithUnits(42, u.s) @ self.matrix

        # -- Other 1D input
        with self.assertRaises(ValueError):
            self.matrix @ MatrixWithUnits([42, 96], [u.m, u.s])

        with self.assertRaises(ValueError):
            MatrixWithUnits([42, 96], [u.m, u.s]) @ self.matrix
    
    def test_matmul_wrong_types(self):
        with self.assertRaises(TypeError):
            self.matrix @ np.array([42, 96])

        with self.assertRaises(TypeError):
            self.matrix @ u.Quantity([42, 96], unit=u.s)
    
    def test_inv_wrong_units(self):
        with self.assertRaises(AssertionError):
            matrix2 = self.matrix.copy()
            matrix2.unit = example_non_si_units
            MatrixWithUnits.inv(matrix2)

    def test_invalid_instance(self):
        matrix = MatrixWithUnits(example_values, u.s)
        matrix._unit = np.array([u.s, u.m])

        with self.assertRaises(AssertionError):
            matrix.size

        with self.assertRaises(AssertionError):
            matrix.shape

        with self.assertRaises(AssertionError):
            matrix.ndim

        matrix._unit = 42  # Has no size etc, so this does not fail.
                           # But is also not valid unit, other error origin

        with self.assertRaises(ValueError):
            matrix.size

        with self.assertRaises(ValueError):
            matrix.shape

        with self.assertRaises(ValueError):
            matrix.ndim
