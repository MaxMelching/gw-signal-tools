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
# class MatrixWithUnits(u.Quantity):
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

    # def __new__(cls, value, unit):
    #     # return MatrixWithUnits(value, unit)
    #     return super().__new__(cls, value)


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
        # if isinstance(unit, self._allowed_unit_types):
        #     # Quantities are not accepted as elements of array, except unit is
        #     # dimensionless_unscaled, so we convert them to CompositeUnits
        #     if isinstance(unit, u.Quantity):
        #         unit = u.CompositeUnit(unit.value, [unit.unit], [1.0])
            
        #     unit = np.full(np.shape(value), unit, dtype=object)
        # elif isinstance(unit, self._allowed_numeric_types):
        #     # Could occur as result of performing operations
        #     unit = np.full(
        #         np.shape(value),
        #         u.CompositeUnit(unit, [u.dimensionless_unscaled], [1.0]),
        #         dtype=object
        #     )
        # else:
        #     assert np.shape(value) == np.shape(unit), \
        #         ('`value` and `unit` must have equal shape if unit is an '
        #         'array of astropy units.')

        # unit = np.asarray(unit, dtype=object)


        # V2, allowing for scalar units as well
        if isinstance(unit, self._allowed_unit_types):
            if isinstance(unit, u.Quantity):
                unit = u.CompositeUnit(unit.value, [unit.unit], [1.0])
            
            # unit = np.array([unit], dtype=object)
        elif isinstance(unit, self._allowed_numeric_types):
            # Could occur as result of performing operations
            # unit = np.array(
            #     [u.CompositeUnit(unit, [u.dimensionless_unscaled], [1.0])],
            #     dtype=object
            # )
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
            
        for val in np.reshape(value, -1):
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
            assert np.shape(unit) == np.shape(self.unit), \
                'New and old `unit` must have equal shape'
        except AttributeError:
            pass  # New class instance is created, nothing to check -> or scalar unit, also ok

        
        if not isinstance(unit, self._pure_unit_types):  # array version was without instance check
            reshaped_unit = np.reshape(unit, -1)
            for i, val in enumerate(reshaped_unit):
                assert isinstance(val, self._allowed_unit_types), \
                    f'Need valid unit types for all members of `unit` (not {type(val)}).'

                # if isinstance(val, u.Quantity):
                #     # reshaped_unit[i] = u.Quantity(1.0, val.unit)
                #     # IMPORTANT: scale can also be part of unit (e.g. in Quantity),
                #     # so we better not replace value with 1, right?
                #     reshaped_unit[i] = val
                # else:
                #     reshaped_unit[i] = u.Quantity(1.0, val)

                if isinstance(unit, u.Quantity):
                    reshaped_unit[i] = u.CompositeUnit(unit.value, [unit.unit], [1.0])
                elif isinstance(unit, (u.Unit, u.IrreducibleUnit)):
                    reshaped_unit[i] = u.CompositeUnit(1.0, [unit], [1.0])
            
            unit = np.reshape(reshaped_unit, np.shape(unit))

        # TODO (potentially): handle this in self._unit_to_quantity?
        # And rather store as Unit or ComposedUnit only? E.g. using self._unit_from quantity function
        # -> should then adjust unit handling in matrix mult,
        # self.unit_to_quantity * other.unit_to_quantity, then call .unit_from_quantity in init?

        # for i, val in enumerate(reshaped_unit):
        #     # print(val, np.reshape(self.value, -1)[i])
            
        #     # assert (isinstance(val, self._allowed_unit_types)
        #     #         or (isinstance(val, u.Quantity)
        #     #             and (val.value == 1.0) or np.isclose(np.reshape(self.value, -1)[i], 0.0, rtol=0.0, atol=1e-15))), \
        #     #     f'Need valid unit types for all members of `unit` (not {type(val)}).'

        #     if isinstance(val, u.Quantity):
        #         if val.value != 1.0:
        #             # self.value *= val.value  # Hm, how to map back to index?
        #             # val.value /= val.value

        #             # # Have to access indices
        #             # new_val = np.reshape(self.value, -1)
        #             # new_val[i] *= val.value
        #             # self.value = np.reshape(new_val, self.value.shape)

        #             # new_unit = np.reshape(unit, -1)
        #             # new_unit[i] /= val.value
        #             # unit = np.reshape(new_unit, unit.shape)


        #             # Ahhh no, values are already added, adding units is "bug"
        #             new_unit = np.reshape(unit, -1)
        #             new_unit[i] = 1.0
        #             unit = np.reshape(new_unit, unit.shape)
        #     else:
        #         assert isinstance(val, self._allowed_unit_types), \
        #         f'Need valid unit types for all members of `unit` (not {type(val)}).'
        
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

        # if np.all(np.equal(np.shape(self.value), 1)):
        #     return np.reshape(self.value, -1)[0] * np.reshape(self.unit, -1)[0]
        # else:
        #     raise TypeError(
        #         'Cannot convert arrays with more than one value to a scalar.'
        #         )
    
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
            # if isinstance(value, u.core.IrreducibleUnit):
            #     # raise TypeError(
            #     #     'Cannot set just unit.'
            #     # )
            #     self.unit.__setitem__(key, value)
            # else:
            #     self.value.__setitem__(key, value)

            # Hmmm, setting only value or unit should not be allowed, right?

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
    

    # ----- Common operations -----
    def __neg__(self) -> MatrixWithUnits:
        return MatrixWithUnits(-self.value, self.unit)
    
    def __add__(self, other: Any) -> MatrixWithUnits:
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnits(self.value + other, self.unit)
            # TODO: rethink if this should be allowed
        # elif isinstance(other, self._allowed_unit_types):
        #     return MatrixWithUnits(self.value, self.unit + np.asarray(other, dtype=object))
        # Adding just units without values does not make much sense, right?
        elif isinstance(other, u.Quantity):
            assert np.all(np.equal(other.unit, self.unit))
            # assert np.all(np.equal(MatrixWithUnits(other.value, other.unit).unit, self.unit))
            # -> not even needed, above works for more complicated shapes too

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
            # return MatrixWithUnits(self.value * other.value, self.unit * other.unit)
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
    
    # def __matmul__(self, other):
    #     return MatrixWithUnits(self.value.__matmul__(other.value),
    #                            self.unit.__matmul__(other.unit))
    # Problem: adding units of same quantity should not result in overall
    # factor of two to be added to unit, but controlling this by setting the
    # scale to 1 would disallow adding u.Msun.si for example (what we want)
    # -> thus, not supporting it seems to be best choice

    # TODO: implement iadd, isub, imul etc. for inplace operations
    
    # ----- Deal with selected useful numpy functions/attributes -----
    def __array__(self) -> np.ndarray:
        # return self.value.__array__()  # Lists or ints do not have __array__
        # return np.array(self.value)
        return self.value

    # def __array_ufunc__(self, function, method, *inputs, **kwargs):
    #     return self.value. __array_ufunc__(function, method, *inputs, **kwargs) #* \
    #         #    self.unit. __array_ufunc__(function, method, *inputs, **kwargs)
    #     # return np.ndarray.array_ufunc(function, method, *inputs, **kwargs)

    @property
    def shape(self):
        value_shape = self.value.shape
        unit_shape = self.unit.shape

        assert value_shape == unit_shape, \
            'Instance is invalid, `value` and `unit` have incompatible shapes.'

        return value_shape
    
    def reshape(self, new_shape: Any) -> MatrixWithUnits:
        return MatrixWithUnits(np.reshape(self.value, new_shape),
                               np.reshape(self.unit, new_shape))

    @property
    def ndim(self) -> int:
        value_ndim = self.value.ndim
        unit_ndim = self.unit.ndim

        assert value_ndim == unit_ndim, \
            'Instance is invalid, `value` and `unit` have incompatible ndim.'

        return value_ndim
    
    @property
    def dtype(self) -> Any:
        return self.value.dtype  # TODO: do this or Quantity?
