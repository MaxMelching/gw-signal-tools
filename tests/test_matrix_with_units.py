# ----- Standard Lib Imports -----
import unittest

# ----- Third Party Imports -----
import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u

import pytest

# ----- Local Package Imports -----
from gw_signal_tools.matrix_with_units import MatrixWithUnits
from gw_signal_tools.test_utils import (
    assert_allclose_MatrixWithUnits, assert_allequal_MatrixWithUnits
)
import gw_signal_tools.cosmo as cosmo


example_values = np.array([[42, 24], [18, 96]])
example_units = np.array([[u.s, u.m], [u.m, u.s]])
example_non_si_units = np.array([[2.0 * u.s, u.m], [u.m**2, u.s]])
example_scaled_units = np.array([[u.Quantity(1e-3, u.s), u.Quantity(1.0, u.m)], [u.Quantity(1e2, u.m), u.Quantity(1.0, u.s)]], dtype=object)


# ----- Test cornerstone properties, value and unit -----
@pytest.mark.parametrize('units', [example_units, example_scaled_units])
def test_unit_matrix_reading(units):
    matrix = MatrixWithUnits(example_values, units)

    np.all(matrix.unit == units)

@pytest.mark.parametrize('unit', [u.s, 2.0 * u.s])
def test_unit_scalar_reading(unit):
    matrix = MatrixWithUnits(example_values, unit)

    if isinstance(unit, u.Quantity):
        unit = u.CompositeUnit(unit.value, [unit.unit], [1.0])

    np.all(matrix.unit == np.full(example_values.shape, unit, dtype=object))

def test_scalar_unit_setting():
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


# ----- Test standard class functions -----
def test_slicing():
    matrix = MatrixWithUnits(example_values, example_units)

    assert_allequal_MatrixWithUnits(matrix[0], MatrixWithUnits([42, 24], [u.s, u.m]))

    assert_allequal_MatrixWithUnits(matrix[0, 0], MatrixWithUnits(42, u.s))

    assert_allequal_MatrixWithUnits(matrix[:1, :], MatrixWithUnits([42, 24], [u.s, u.m]))
    assert_allequal_MatrixWithUnits(matrix[:, :1], MatrixWithUnits([[42], [18]], [[u.s], [u.m]]))
    
    
    matrix2 = MatrixWithUnits(example_values, example_non_si_units)

    assert_allequal_MatrixWithUnits(matrix2[0], MatrixWithUnits([42, 24], [2.0 * u.s, u.m]))
    assert_allequal_MatrixWithUnits(matrix2[1, 0], MatrixWithUnits(18, u.m**2))

def test_copy():
    ...


# ----- Test common operations -----
def test_addition():
    matrix = MatrixWithUnits(example_values, example_units)
    
    # Float addition
    assert_allequal_MatrixWithUnits(
        matrix + 2.0,
        MatrixWithUnits(example_values + 2.0, example_units)
    )

    assert_allequal_MatrixWithUnits(
        2.0 + matrix,
        MatrixWithUnits(2.0 + example_values, example_units)
    )

    # Quantitiy addition -> only works with scalar unit
    matrix_in_s = MatrixWithUnits(example_values, u.s)
    
    assert_allequal_MatrixWithUnits(
        matrix_in_s + 2.0 * u.s,
        MatrixWithUnits(example_values + 2.0, u.s)
    )
    
    # Following produces astropy error
    # assert_allequal_MatrixWithUnits(
    #     2.0 * u.s  + matrix_in_s,
    #     MatrixWithUnits(2.0 + example_values, u.s)
    # )

    # MatrixWithUnits addition -> similar to test_multiplcation
    assert_allequal_MatrixWithUnits(
        matrix + matrix,
        MatrixWithUnits(example_values + example_values, example_units)
    )

def test_subtraction():
    matrix = MatrixWithUnits(example_values, example_units)

    # Float subtraction
    assert_allequal_MatrixWithUnits(
        matrix - 2.0,
        MatrixWithUnits(example_values - 2.0, example_units)
    )

    assert_allequal_MatrixWithUnits(
        2.0 - matrix,
        MatrixWithUnits(2.0 - example_values, example_units)
    )

    # Quantity subtraction
    matrix_in_s = MatrixWithUnits(example_values, u.s)
    
    assert_allequal_MatrixWithUnits(
        matrix_in_s - 2.0 * u.s,
        MatrixWithUnits(example_values - 2.0, u.s)
    )
    
    # Following produces astropy error
    # assert_allequal_MatrixWithUnits(
    #     2.0 * u.s  + matrix_in_s,
    #     MatrixWithUnits(2.0 + example_values, u.s)
    # )

    # MatrixWithUnits subtraction -> similar to test_multiplication
    assert_allequal_MatrixWithUnits(
        matrix - 0.5 * matrix,
        MatrixWithUnits(0.5 * example_values, example_units)
    )

def test_multiplication():
    matrix = MatrixWithUnits(example_values, example_units)

    # Float multiplication
    assert_allequal_MatrixWithUnits(
        matrix * 2.0,
        MatrixWithUnits(example_values * 2.0, example_units)
    )

    assert_allequal_MatrixWithUnits(
        2.0 * matrix,
        MatrixWithUnits(2.0 * example_values, example_units)
    )

    # Also verify results of float multiplcation coincide with add/sub
    assert_allequal_MatrixWithUnits(2 * matrix, matrix + matrix)

    assert_allequal_MatrixWithUnits(0.5 * matrix, matrix - 0.5 * matrix)

    # Quantity multiplication -> only works with scalar unit
    matrix_in_s = MatrixWithUnits(example_values, u.s)
    
    assert_allequal_MatrixWithUnits(
        matrix_in_s * (2.0 * u.s),
        MatrixWithUnits(example_values * 2.0, u.s * u.s)
    )
    
    assert_allequal_MatrixWithUnits(
        (2.0 * u.s) * matrix_in_s,
        MatrixWithUnits(2.0 * example_values, u.s * u.s)
    )

    # MatrixWithUnit Multiplication -> similar to test_power
    assert_allequal_MatrixWithUnits(
        matrix * matrix,
        MatrixWithUnits(example_values**2, example_units**2)
    )

def test_division():
    matrix = MatrixWithUnits(example_values, example_units)

    # Float division
    assert_allequal_MatrixWithUnits(
        matrix / 2.0,
        MatrixWithUnits(example_values / 2.0, example_units)
    )

    assert_allequal_MatrixWithUnits(
        2.0 / matrix,
        MatrixWithUnits(2.0 / example_values, 1 / example_units)
    )

    # Quantity multiplication -> only works with scalar unit
    matrix_in_s = MatrixWithUnits(example_values, u.s)
    
    assert_allequal_MatrixWithUnits(
        matrix_in_s / (2.0 * u.s),
        MatrixWithUnits(example_values / 2.0, u.dimensionless_unscaled)
    )
    
    assert_allequal_MatrixWithUnits(
        (2.0 * u.s) / matrix_in_s,
        MatrixWithUnits(2.0 / example_values, u.dimensionless_unscaled)
    )


    # MatrixWithUnit Division -> similar to test_power
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

    # Test support for units, i.e. multiplication with scalar units etc
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


    # Now tests for matrices that do not have single unit
    assert_allequal_MatrixWithUnits(
        matrix_in_s @ matrix2,
        # MatrixWithUnits(example_values @ example_values, example_units_2 * example_units_s)
        MatrixWithUnits(example_values @ example_values, np.array([[u.s**2, u.m * u.s], [u.s**2, u.m * u.s]], dtype=object))
    )

    assert_allequal_MatrixWithUnits(
        matrix2.T @ matrix_in_s,
        MatrixWithUnits(example_values.T @ example_values, np.array([[u.s**2, u.s**2], [u.m * u.s, u.m * u.s]], dtype=object))
    )


    # Test with rows and columns
    matrix_col = MatrixWithUnits(example_values[:, 0], u.s)
    matrix_row = MatrixWithUnits(example_values[0 ,:], u.s)

    assert_allequal_MatrixWithUnits(
        matrix_col @ matrix_row,  # Inner product
        MatrixWithUnits(42**2 + 24 * 18, u.s**2)
    )

    assert_allequal_MatrixWithUnits(
        matrix_row @ matrix_col,  # Outer product
        MatrixWithUnits(example_values[0, :] @ example_values[:, 0], u.s**2)
    )

    assert_allequal_MatrixWithUnits(
        matrix_in_s @ matrix_col,  # Matrix and column
        MatrixWithUnits(example_values @ example_values[:, 0], u.s**2)
    )

    assert_allequal_MatrixWithUnits(
        matrix2.T @ matrix_col,  # Matrix and column
        MatrixWithUnits(example_values.T @ example_values[:, 0], np.array([u.s**2, u.s * u.m], dtype=object))
    )

    assert_allequal_MatrixWithUnits(
        matrix_row @ matrix_in_s,  # Row and matrix
        MatrixWithUnits(example_values[0, :] @ example_values, u.s**2)
    )

    assert_allequal_MatrixWithUnits(
        matrix_row @ matrix2,  # Row and matrix
        MatrixWithUnits(example_values[0, :] @ example_values, np.array([u.s**2, u.s * u.m], dtype=object))
    )


# ----- Test numpy functions -----
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
    matrix = MatrixWithUnits(values, u.s)
    
    assert matrix.value.dtype == values.dtype
    assert matrix.dtype == u.Quantity

def test_reading_from_array():
    matrix = MatrixWithUnits.from_numpy_array(example_values)

    assert np.all(np.equal(matrix.value, example_values))
    assert np.all(np.equal(matrix.unit, u.dimensionless_unscaled))

def test_inv():
    matrix = MatrixWithUnits(example_values, u.s)
    matrix_inv = MatrixWithUnits.inv(matrix)

    assert_allclose_MatrixWithUnits(
        matrix @ matrix_inv,
        MatrixWithUnits.from_numpy_array(np.eye(2))
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


# ----- Test numpy functions -----
def test_to_system():
    matrix = MatrixWithUnits(example_values, example_units)

    matrix.to_system(u.si)
    matrix.to_system(cosmo)


# ----- Test error raising -----
class Errors(unittest.TestCase):
    matrix = MatrixWithUnits(example_values, example_units)
    matrix_in_s = MatrixWithUnits(example_values, u.s)
    
    def test_quantitiy_addition(self):
        # Test that unequal units throw error
        with self.assertRaises(AssertionError):
            self.matrix + (2.0 * u.s)

        # astropy throws AttributeError
        with self.assertRaises(AttributeError):
            (2.0 * u.s) + self.matrix
    
    def test_quantitiy_subtraction(self):
        # Test that unequal units throw error
        with self.assertRaises(AssertionError):
            self.matrix - (2.0 * u.s)

        # astropy throws AttributeError
        with self.assertRaises(AttributeError):
            (2.0 * u.s) - self.matrix
    
    def test_quantitiy_multiplication(self):
        # astropy throws UnitConversionError
        with self.assertRaises(u.UnitConversionError):
            (2.0 * u.s) * self.matrix
    
    def test_matmul_wrong_units(self):
        with self.assertRaises(AssertionError):
            self.matrix @ self.matrix  # Due to units not fitting together
    
    def test_matmul_wrong_shapes(self):
        with self.assertRaises(ValueError):
            self.matrix @ MatrixWithUnits(42, u.s)

        with self.assertRaises(ValueError):
            MatrixWithUnits(42, u.s) @ self.matrix
    
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
