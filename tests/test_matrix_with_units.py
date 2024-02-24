import unittest

import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u

import pytest

from gw_signal_tools.matrix_with_units import MatrixWithUnits
from gw_signal_tools.test_utils import assert_allclose_quantity


example_values = np.array([[42, 24], [18, 96]])
example_units = np.array([[u.s, u.m], [u.m, u.s]])
example_scaled_units = np.array([[u.Quantity(1e-3, u.s), u.Quantity(1.0, u.m)], [u.Quantity(1e2, u.m), u.Quantity(1.0, u.s)]], dtype=object)

print(example_scaled_units)
print(MatrixWithUnits(example_values, example_scaled_units))
print(MatrixWithUnits(example_values, 2.0 * u.s))



@pytest.mark.parametrize('units', [example_units, example_scaled_units])
def test_unit_matrix_reading(units):
    matrix = MatrixWithUnits(example_values, units)

    assert np.all(matrix.unit == units)


@pytest.mark.parametrize('unit', [u.s, 2.0 * u.s])
def test_unit_scalar_reading(unit):
    # matrix1 = MatrixWithUnits(example_values, unit)
    
    # if isinstance(unit, u.Quantity):
    #     scale = unit.value
    #     unit = u.Quantity(1.0, unit.unit)
    # else:
    #     scale = 1.0
    #     unit = u.Quantity(1.0, unit)
    
    # matrix2 = MatrixWithUnits(scale * example_values, unit)

    # # assert np.all(matrix1 == matrix2)
    # # assert np.all(np.equal(matrix1, matrix2))

    matrix = MatrixWithUnits(example_values, unit)

    if isinstance(unit, u.Quantity):
        unit = u.CompositeUnit(unit.value, [unit.unit], [1.0])

    assert np.all(matrix.unit == np.full(example_values.shape, unit, dtype=object))
