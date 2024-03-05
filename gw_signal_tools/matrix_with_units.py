# ----- Standard Lib Imports -----
from __future__ import annotations  # Enables type hinting own type in a class
import logging
import warnings
from typing import Optional, Any, Literal, Self

# ----- Third Party Imports -----
import numpy as np
from numpy.typing import ArrayLike
import astropy.units as u


class MatrixWithUnits:
    """
    Class for a matrix where entries can have differing units, following
    the spirit of astropy Quantities.

    Parameters
    ----------
    value : 
        ...
    unit : 
        ...
        Note that no care is taken to produce irreducible units (i.e.
        unscaled ones, while applying the scale from units to the
        values). This is because units of sun masses etc. that are
        given as numbers of the kind 10e30*kg would then also be
        converted. For our usecases of this class, such a behaviour is
        unwanted.

    Notes
    -----
    This class was written to represent a Fisher matrix, which is a
    matrix that has units. Astropy quanitities do not allow for an
    object that carries different units and while it is possible to do
    in numpy by creating arrays with dtype=object, this does not allow
    for further calculations like matrix inversion etc.

    The idea of this class is to allow for both and this is achieved
    by separating values and units into different class properties. Upon
    reasonable initialization as numpy arrays of the same shape, one
    can then access the values only using `MatrixWithUnits.value`, while
    the units are stored in `MatrixWithUnits.unit` (where the names are
    chosen to match the corresponding astropy Quantity names). This
    allows for a flexible usage, where operations can be carried out
    with or without units by using either `MatrixWithUnits.value` or
    `MatrixWithUnits.value * MatrixWithUnits.unit` (where the latter
    is also the output when printing a class instance).

    -> say that scalar unit also works, whence a u.Quantity is mimicked

    An important point of emphasis, however, is that unit handling is
    not an easy task, especially not for matrices. For this reason,
    operations like matrix multiplication are not supported, which means
    one has to resort to applying these operations to `MatrixWithUnits.
    value`. Simpler operations like addition and multiplication, on the
    other hand, are defined.

    Examples
    --------
    >>> value_matrix = np.array([[42, 96], [96, 42]])
    >>> unit_matrix = np.array([[u.s, u.m], [u.m, u.s]], dtype=object)
    >>> matrix = MatrixWithUnits(value_matrix, unit_matrix)
    >>> print(matrix)
    array([[<Quantity 42. s>, <Quantity 96. m>],
           [<Quantity 96. m>, <Quantity 42. s>]], dtype=object)
    >>> np.all(matrix.value == value_matrix)
    True
    >>> np.all(matrix.unit == unit_matrix)
    True

    Alternatively, one can extract the values by converting to an array,
    which is supposed to simplify usage and provide an easy way to
    convert this class into more common data types:
    >>> np.array(matrix)
    array([[42, 96],
           [96, 42]])
    
    This enables calling numpy functions directly on instances of
    ``MatrixWithUnits``, e.g.
    >>> np.linalg.inv(matrix)
    array([[-0.00563607,  0.01288245],
           [ 0.01288245, -0.00563607]])
    
    Note, however, that this only uses the values and does not check
    whether such an operation would make sense to do with the units
    of matrix. Figuring out if this is the case is left to the user
    because implementing a consistent matrix multiplication for units
    is beyond the scope of this class, which is mainly made to provide
    a convenient representation with units. For physically motivated
    matrices like the Fisher matrix, however, this consistency is
    usually given, so that an inversion is indeed meaningful.
    

    In order to get the printed representation, we can simply multiply
    values and units:
    >>> matrix.value * matrix.unit
    array([[<Quantity 42. s>, <Quantity 96. m>],
           [<Quantity 96. m>, <Quantity 42. s>]], dtype=object)
    
    Note, however, that this only works because the class has been
    initialized from two numpy arrays, where multiplication is
    supported, and also that the object printed here is now also a
    numpy array, not a ``MatrixWithUnits`` anymore.

    It is also possible to initialize using a single unit, e.g.
    >>> MatrixWithUnits(np.array([[42, 96], [96, 42]]), u.s)
    array([[<Quantity 42. s>, <Quantity 96. s>],
           [<Quantity 96. s>, <Quantity 42. s>]], dtype=object)
    """

    _allowed_numeric_types = (int, float, complex, np.number)
    _pure_unit_types = (u.IrreducibleUnit, u.CompositeUnit, u.Unit)
    _allowed_unit_types = _pure_unit_types + (u.Quantity,)


    def  __init__(self, value: ArrayLike, unit: ArrayLike) -> None:
        # Internally, value and unit are stored as numpy arrays due to their
        # versatility. Now we have to make sure the conversion works
        try:
            # Will work for arrays
            value_dtype = value.dtype  # type: ignore
            # Explanation of ignore: we cover case where no dtype exists
        except AttributeError:
            if type(value) in self._allowed_numeric_types:
                value_dtype = type(value)
            else:
                # Default numeric type
                value_dtype = float

        value = np.asarray(value, dtype=value_dtype)


        # Scalar unit is allowed input even for value array, handled here
        if isinstance(unit, self._pure_unit_types):
            unit = u.CompositeUnit(1.0, [unit], [1.0])
        elif isinstance(unit, u.Quantity):
            unit = u.CompositeUnit(unit.value, [unit.unit], [1.0])
        elif isinstance(unit, self._allowed_numeric_types):
            # Could occur as result of performing operations
            unit = u.CompositeUnit(unit, [u.dimensionless_unscaled], [1.0])
        else:
            assert np.shape(value) == np.shape(unit), \
                ('`value` and `unit` must have equal shape if unit is an '
                'array of astropy units.')
            
            unit = np.asarray(unit, dtype=object)

        # Setters will take care of checking each element for correct type
        self.value = value
        self.unit = unit
        # NOTE: setting "private" properties here already is not good practice.
        # Instead go through setters of attributes, where these are set
        

    # ----- Define cornerstone properties, value and unit -----
    # TODO: add deleters?
    @property
    def value(self) -> np.ndarray:
        """
        Numeric values of the matrix.

        :type:`~numpy.ndarray`
        """
        return self._value  # type: ignore
        # Explanation of ignore: we convert to array in __init__, but mypy
        # performs static type checking and does not recognize this
    
    @value.setter
    def value(self, value: ArrayLike) -> None:
        try:
            assert np.shape(value) == np.shape(self.value), \
                'New and old `value` must have equal shape'
        except AttributeError:
            pass  # New class instance is created, nothing to check
        

        for _, val in np.ndenumerate(value):
            assert (isinstance(val, self._allowed_numeric_types)
                    and not isinstance(val, bool)), \
                f'Need valid numeric types for all members of `value` (not {type(val)}).'
        
        self._value = value

    @property
    def unit(self) -> np.ndarray:
        """
        Units of the matrix.

        :type:`~numpy.ndarray`
        """
        return self._unit
    
    @unit.setter
    def unit(self, unit: ArrayLike):
        try:
            if not (unit not in self._pure_unit_types
                or self.unit not in self._pure_unit_types):
                assert np.shape(unit) == np.shape(self.unit), \
                    'New and old `unit` must have equal shape'
        except AttributeError:
            pass  # New class instance is created, nothing to check
                  # or unit is scalar, also ok
        

        if not isinstance(unit, self._pure_unit_types):
            # Unit is also array (otherwise would have been converted
            # in __init__)
            for i, val in np.ndenumerate(unit):
                assert isinstance(val, self._allowed_unit_types), \
                    f'Need valid unit types for all members of `unit` (not {type(val)}).'
                
                if isinstance(val, self._pure_unit_types):
                    unit[i] = u.CompositeUnit(1.0, [val], [1.0])
                else:
                    unit[i] = u.CompositeUnit(val.value, [val.unit], [1.0])

            
        self._unit = unit


    # ----- Deal with certain standard class functions -----
    def __repr__(self) -> str:
        # return (self.value * self.unit).__repr__()
        return (np.array(self.value) * np.array(self.unit)).__repr__()  # Are converted in init -> now not anymore
    
    def __float__(self) -> float:
        # if np.shape(self.value)
        # Will throw TypeError in most cases, but not because of our class,
        # but because of way it has been set by user (e.g. to array)
        return float(self.value * self.unit)
    
    def __copy__(self) -> MatrixWithUnits:
        return MatrixWithUnits(self.value.__copy__(), self.unit.__copy__())
    
    def copy(self) -> MatrixWithUnits:
        return self.__copy__()
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MatrixWithUnits):
            return NotImplemented
        else:
            return self.value.__eq__(other.value) & self.unit.__eq__(other.unit)
    
    def __hash__(self) -> int:
        return hash(self.value) ^ hash(self.unit)
    
    def __getitem__(self, key: Any) -> MatrixWithUnits:
        # return self.value.__getitem__(key) * self.unit.__getitem__(key)
        if isinstance(self.unit, self._pure_unit_types):
            return MatrixWithUnits(self.value.__getitem__(key), self.unit)
        else:
            return MatrixWithUnits(self.value.__getitem__(key), self.unit.__getitem__(key))
    
    def __setitem__(self, key: Any, value: Any) -> None:
        try:
            self.value.__setitem__(key, value.value)
            # self.unit.__setitem__(key, value.unit)  # From array version
        
            if isinstance(self.unit, self._pure_unit_types):
                self.unit = value.unit
            else:
                self.unit.__setitem__(key, value.unit)
        except AttributeError:
            try:
                value = u.Quantity(value)
            except TypeError:
                raise TypeError(
                    'Can only set items to data types that have members'
                    '`value` and `unit` (such as astropy Quantities or '
                    'MatrixWithUnits) or can be converted into a Quantity.'
                )
            
            self.value.__setitem__(key, value.value)
            self.unit.__setitem__(key, value.unit)
    
    def __len__(self):
        return self.value.__len__()
    

    # ----- Common operations -----
    def __neg__(self) -> MatrixWithUnits:
        return MatrixWithUnits(-self.value, self.unit)
    
    def __add__(self, other: Any) -> MatrixWithUnits:
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnits(self.value + other, self.unit)
        elif isinstance(other, u.Quantity):
            assert np.all(np.equal(other.unit, self.unit))

            return MatrixWithUnits(self.value + other.value, self.unit)
        elif isinstance(other, MatrixWithUnits):
            assert np.all(np.equal(other.unit, self.unit))
            
            return MatrixWithUnits(self.value + other.value, self.unit)
        else:
            raise TypeError(
                f'Addition between {type(other)} and `MatrixWithUnit` is not '
                'supported.'
            )
    
    def __radd__(self, other: Any) -> MatrixWithUnits:
        # if isinstance(other, u.Quantity):
        #     raise NotImplementedError(
        #         'Right addition with astropy Quantities is currently not '
        #         'supported. Consider changing this to left addition.'
        #     )
        # else:
        #     return self.__add__(other)
        # Not raised anyway, astropy tries to do it and fails
        return self.__add__(other)
    
    def __sub__(self, other: Any) -> MatrixWithUnits:
        return self.__add__(other.__neg__())
    
    def __rsub__(self, other: Any) -> MatrixWithUnits:
        # if isinstance(other, u.Quantity):
        #     raise NotImplementedError(
        #         'Right subtraction with astropy Quantities is currently not '
        #         'supported. Consider changing this to left addition.'
        #     )
        # else:
        #     return self.__neg__().__add__(other)
        # Not raised anyway, astropy tries to do it and fails
        return self.__neg__().__add__(other)
            
    def __mul__(self, other: Any) -> MatrixWithUnits:
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnits(self.value * other, self.unit)
        elif isinstance(other, self._pure_unit_types):
            # return MatrixWithUnits(self.value, self.unit * np.asarray(other, dtype=object))  # From array version
            return MatrixWithUnits(self.value, self.unit * other)
        elif isinstance(other, u.Quantity):
            # Fall back to multiplication with unit and value
            return self * other.value * other.unit
        # Harder to implement than it seems
        elif isinstance(other, MatrixWithUnits):
            return MatrixWithUnits(self.value * other.value, self.unit * other.unit)
        else:
            raise TypeError(
                f'Multiplication between {type(other)} and `MatrixWithUnit`'
                ' is not supported.'
            )

    def __rmul__(self, other: Any) -> MatrixWithUnits:
        # if isinstance(other, u.Quantity):
        #     raise NotImplementedError(
        #         'Right multiplication with astropy Quantities is currently '
        #         'not supported. Consider changing this to left addition.'
        #     )
        # else:
        #     return self.__mul__(other)
        # Not raised anyway, astropy tries to do it and fails
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnits(self.value / other, self.unit)
        elif isinstance(other, self._pure_unit_types):
            return MatrixWithUnits(self.value, self.unit / np.asarray(other, dtype=object))  # from array version
        elif isinstance(other, u.Quantity):
            return MatrixWithUnits(self.value / other.value, self.unit / other.unit)
        elif isinstance(other, MatrixWithUnits):
            return MatrixWithUnits(self.value / other.value, self.unit / other.unit)
        else:
            try:
                return MatrixWithUnits(self.value / other, self.unit)
            except:
                raise TypeError(
                    f'Division of `MatrixWithUnit` and {type(other)}'
                    ' is not supported.'
                )
    
    def __rtruediv__(self, other: Any) -> MatrixWithUnits:
        # Important: 1/u.Unit becomes quantity, we have to use power -1 so that
        # unit actually remains a unit
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnits(other / self.value, self.unit**-1)
        elif isinstance(other, self._pure_unit_types):
            return MatrixWithUnits(1.0 / self.value, np.asarray(other, dtype=object) / self.unit)  # from array version
        elif isinstance(other, u.Quantity):
            return MatrixWithUnits(other.value / self.value, other.unit / self.unit)
        elif isinstance(other, MatrixWithUnits):
            return MatrixWithUnits(other.value / self.value, other.unit / self.unit)
        else:
            try:
                return MatrixWithUnits(other / self.value, self.unit**-1)
            except:
                raise TypeError(
                    f'Division of {type(other)} and `MatrixWithUnit`'
                    ' is not supported.'
                )
            
    def __pow__(self, other: Any) -> MatrixWithUnits:
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnits(self.value.__pow__(other), self.unit.__pow__(other))
        else:
            raise TypeError(
                'Raising of `MatrixWithUnit` to a non-numeric type like '
                f'{type(other)} is not supported.'
            )
    
    def __matmul__(self, other):
        # Problem we have to circumvent:
        # return MatrixWithUnits(self.value @ other.value, self.unit @ other.unit)
        # does not work because CompositeUnits cannot be added (and adding
        # them would also change value, which is not intended). Thus we have
        # to handle unit manually (also for compatibility with scalar unit)

        if isinstance(other, MatrixWithUnits):
            # Step 1: apply to values. This is useful because it already checks
            # compatibility of shapes
            new_value = self.value @ other.value
            new_shape = new_value.shape

            # Hack for 1D output
            new_shape_1 = new_shape
            if len(new_shape) == 1:
                if len(self.value.shape) == 1:
                    # self is row vector
                    new_shape = (1, new_shape[0])
                else:
                    # other is column vector
                    new_shape = (new_shape[0], 1)
            

            # print(new_shape)

            # Step 2: handle unit.
            # TODO: handle case of scalar unit for one of them
            new_unit = np.empty(new_shape, dtype=object)

            # V1, not suitable for scalar units
            # for i in range(new_shape[0]):
            #     for j in range(new_shape[1]):
            #         unit_test = self.unit[i, 0] * other.unit[0, j]

            #         # print(unit_test)
            #         # print(type(unit_test))

            #         assert np.all(np.equal(self.unit[i, :] * other.unit[:, j], unit_test)), \
            #             'Need consistent units for matrix multiplication.'
            #             # TODO: mention more explicitly which units are incompatible? And also indices for which it occurs
            #         # NOTE: do NOT replace with == here, does not what we want
                    
            #         new_unit[i, j] = unit_test
            
            # V2, trying to account for scalar unit
            # try:
            #     # test_unit_shape = self.unit.shape
            #     # test_unit_shape = other.unit.shape

            #     # If we can access both, they are not scalar
            #     # -> would not be proof against setting unit to [u.s] or so, right?
            #     # -> just apply test from unit property

            #     for i in range(new_shape[0]):
            #         for j in range(new_shape[1]):
            #             unit_test = self.unit[i, 0] * other.unit[0, j]

            #             # print(unit_test)
            #             # print(type(unit_test))

            #             assert np.all(np.equal(self.unit[i, :] * other.unit[:, j], unit_test)), \
            #                 'Need consistent units for matrix multiplication.'
            #                 # TODO: mention more explicitly which units are incompatible? And also indices for which it occurs
            #             # NOTE: do NOT replace with == here, does not what we want
                        
            #             new_unit[i, j] = unit_test
            # except AttributeError:
            #     ...
            
            # We have to distinguish several cases
            # TODO (idea): reduce scalar unit cases for only one of the
            # matrices to first one by following:
            # new_self_unit = np.full(new_shape, self.unit, dtype=object)
            # return MatrixWithUnits(self.value, new_self_unit) @ other
            # -> not strictly necessary, though
            # print(isinstance(self.unit, self._pure_unit_types))
            # print(isinstance(other.unit, other._pure_unit_types))

            if (not isinstance(self.unit, self._pure_unit_types) and
                not isinstance(other.unit, other._pure_unit_types)):
                # Both arrays

                # for i in range(new_shape[0]):
                #     for j in range(new_shape[1]):
                for index in np.ndindex(new_shape):
                    i, j = index
                    unit_test = self.unit[i, 0] * other.unit[0, j]

                    # print(unit_test)
                    # print(type(unit_test))

                    assert np.all(np.equal(self.unit[i, :] * other.unit[:, j], unit_test)), \
                        'Need consistent units for matrix multiplication.'
                        # TODO: mention more explicitly which units are incompatible? And also indices for which it occurs
                    # NOTE: do NOT replace with == here, does not what we want
                    
                    new_unit[i, j] = unit_test
            elif (isinstance(self.unit, self._pure_unit_types) and
                  isinstance(other.unit, self._pure_unit_types)):
                # Both scalar units
                new_unit = np.full(new_shape, self.unit * other.unit, dtype=object)
                # new_unit = self.unit * other.unit
            elif isinstance(self.unit, self._pure_unit_types):
                # One scalar unit
                # for i in range(new_shape[0]):
                #     for j in range(new_shape[1]):
                for index in np.ndindex(new_shape):
                    i, j = index
                    unit_test = other.unit[0, j]

                    # print(unit_test)
                    # print(type(unit_test))

                    assert np.all(np.equal(other.unit[:, j], unit_test)), \
                        'Need consistent units for matrix multiplication.'
                        # TODO: mention more explicitly which units are incompatible? And also indices for which it occurs
                    # NOTE: do NOT replace with == here, does not what we want
                    
                    new_unit[i, j] = self.unit * unit_test
            elif isinstance(other.unit, self._pure_unit_types):
                # One scalar unit
                # for i in range(new_shape[0]):
                #     for j in range(new_shape[1]):
                for index in np.ndindex(new_shape):
                    i, j = index
                    unit_test = self.unit[i, 0]

                    # print(unit_test)
                    # print(type(unit_test))

                    assert np.all(np.equal(self.unit[i, :], unit_test)), \
                        'Need consistent units for matrix multiplication.'
                        # TODO: mention more explicitly which units are incompatible? And also indices for which it occurs
                    # NOTE: do NOT replace with == here, does not what we want
                    
                    new_unit[i, j] = unit_test * other.unit

            if new_shape != new_shape_1:
                new_unit = np.reshape(new_unit, new_shape_1)

            return MatrixWithUnits(new_value, new_unit)
        else:
            raise TypeError(
                'Cannot perform matrix multiplication between '
                f'``MatrixWithUnits`` and ``{type(other)}``.'
                )
    
    # TODO: implement iadd, isub, imul etc. for inplace operations
    
    # ----- Deal with selected useful numpy functions/attributes -----
    def __array__(self) -> np.ndarray:
        return self.value

    @property
    def T(self):
        """
        Transposed Matrix.

        Returns
        -------
        :type:`~gw_signal_tools.matrix_with_units.MatrixWithUnits`
        """
        if isinstance(self.unit, self._pure_unit_types):
            return MatrixWithUnits(self.value.T, self.unit)
        else:
            return MatrixWithUnits(self.value.T, self.unit.T)
        
    @property
    def size(self):
        value_size = self.value.size

        try:
            unit_size = self.unit.size

            assert value_size == unit_size, \
                'Instance is invalid, `value` and `unit` have incompatible sizes.'

            return value_size
        except AttributeError:
            # Might be scalar unit, then everything is fine
            if isinstance(self.unit, self._pure_unit_types):
                return value_size
            else:
                raise ValueError(
                    'Instance is invalid, `value` and `unit` have incompatible sizes.'
                )

    @property
    def shape(self):
        value_shape = self.value.shape

        try:
            unit_shape = self.unit.shape

            assert value_shape == unit_shape, \
                'Instance is invalid, `value` and `unit` have incompatible shapes.'

            return value_shape
        except AttributeError:
            # Might be scalar unit, then everything is fine
            if isinstance(self.unit, self._pure_unit_types):
                return value_shape
            else:
                raise ValueError(
                    'Instance is invalid, `value` and `unit` have incompatible shapes.'
                )

    @property
    def ndim(self) -> int:
        value_ndim = self.value.ndim

        try:
            unit_ndim = self.unit.ndim

            assert value_ndim == unit_ndim, \
                'Instance is invalid, `value` and `unit` have incompatible ndim.'

            return value_ndim
        except AttributeError:
            # Might be scalar unit, then everything is fine
            if isinstance(self.unit, self._pure_unit_types):
                return value_ndim
            else:
                raise ValueError(
                    'Instance is invalid, `value` and `unit` have incompatible ndim.'
                )
    
    @property
    def dtype(self) -> Any:
        return u.Quantity
    
    @staticmethod
    def from_numpy_array(arr: np.ndarray) -> MatrixWithUnits:
        return MatrixWithUnits(arr, u.dimensionless_unscaled)
    
    @staticmethod
    def reshape(matrix: MatrixWithUnits, new_shape: Any) -> MatrixWithUnits:
        return MatrixWithUnits(np.reshape(matrix.value, new_shape),
                               np.reshape(matrix.unit, new_shape))

    @staticmethod
    def inv(matrix: MatrixWithUnits) -> MatrixWithUnits:
        assert np.all(np.equal(matrix.unit, matrix.T.unit)), \
            'Need symmetric unit for inversion.'

        return MatrixWithUnits(np.linalg.inv(matrix.value), matrix.unit**-1)
