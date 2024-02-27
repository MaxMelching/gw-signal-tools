# ----- Standard Lib Imports -----
import unittest

# ----- Third Party Imports -----
import numpy as np

import astropy.units as u

import pytest

# ----- Local Package Imports -----
from gw_signal_tools.matrix_with_units import MatrixWithUnits
from gw_signal_tools.test_utils import (
    assert_allclose_MatrixWithUnits, assert_allequal_MatrixWithUnits
)


example_values = np.array([[42, 24], [18, 96]])
example_units = np.array([[u.s, u.m], [u.m, u.s]])
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



# ----- Test numpy functions -----
def test_array_conversion():
    matrix = MatrixWithUnits(example_values, example_units)
    matrix_array = np.array(matrix)

    np.all(matrix_array == matrix.value)

def test_shape():
    matrix = MatrixWithUnits(example_values, example_units)
    
    assert matrix.shape == (2, 2)


@pytest.mark.parametrize('new_shape', [(2, 2), (1, 4), (4, 1), -1])
def test_reshape(new_shape):
    matrix = MatrixWithUnits(example_values, example_units)
    
    matrix2 = MatrixWithUnits(np.reshape(example_values, new_shape),
                              np.reshape(example_units, new_shape))
    
    assert_allequal_MatrixWithUnits(
        matrix.reshape(new_shape),
        matrix2
    )

def test_ndim():
    matrix = MatrixWithUnits(example_values, example_units)
    
    assert matrix.ndim == 2

def test_dtype():
    matrix = MatrixWithUnits(example_values, example_units)
    
    assert matrix.dtype == example_values.dtype


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
    
    def test_matmul(self):
        with self.assertRaises(TypeError):
            self.matrix @ self.matrix


matrix = MatrixWithUnits(example_values, u.s)
