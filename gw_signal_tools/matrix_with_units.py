import numpy as np
import astropy.units as u


# class MatrixWithUnit(np.ndarray):
#     def __new__(cls, value, **kwargs):
#         new.value = super().__new__(np.shape(value), dtype=float, **kwargs)

#         new.unit = super().__new__(np.shape(value), dtype=object)
        # return self._value * self._unit



class MatrixWithUnits:
    _allowed_numeric_types = (int, float, complex, np.number)
    _allowed_unit_types = (u.IrreducibleUnit, u.CompositeUnit, u.Unit)


    def  __init__(self, value, unit) -> None:
        # assert np.shape(value) == np.shape(unit), \
        #         ('`value` and `unit` must have equal shape if unit is an '
        #         'array of astropy units.')
        

        # Idea: add support for given unit that is "scalar"
        # -> should we also make sure value is np.array?

        # value = np.asarray(value, dtype=np.number)
        # value = np.asarray(value, dtype=float)

        if isinstance(unit, self._allowed_unit_types):
            if not isinstance(value, self._allowed_numeric_types):
                unit = np.full(np.shape(value), unit, dtype=object)
        else:
            assert np.shape(value) == np.shape(unit), \
                ('`value` and `unit` must have equal shape if unit is an '
                'array of astropy units.')
            
        if not isinstance(value, (np.ndarray, u.Quantity)):
            value = np.asarray(value, dtype=float)

        if not isinstance(unit, (np.ndarray, u.Quantity)):
            unit = np.asarray(unit, dtype=object)

        self.value = value
        self.unit = unit
        # NOTE: setting "private" properties here already is not good practice.
        # Instead go through setters of attributes, where these are set
        

    # def __new__(cls, value, unit):
    #     cls._value = value
    #     cls._unit = unit

    #     """
    #     TODOS
    #     - make sure they have same shape
    #     - maybe make sure unit has dtype object? Or that all members are of type u.Quantity?
    #     """
        
    #     return cls._value * cls._unit


    @property
    def value(self):
        ...
        return self._value
    
    @value.setter
    def value(self, value):
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
    def unit(self):
        ...
        return self._unit
    
    @unit.setter
    def unit(self, unit):
        try:
            assert np.shape(unit) == np.shape(self.unit), \
                'New and old `unit` must have equal shape'
        except AttributeError:
            pass  # New class instance is created, nothing to check
        
        for val in np.reshape(unit, -1):
            assert isinstance(val, self._allowed_unit_types), \
                f'Need valid unit types for all members of `unit` (not {type(val)}).'
        
        self._unit = unit


    def __repr__(self) -> str:
        # return (self.value * self.unit).__repr__()
        return (np.array(self.value) * np.array(self.unit)).__repr__()
    

    def __float__(self):
        # if np.shape(self.value)
        # Will throw error in most cases, but not because of our class, but
        # because of way it has been set by user (e.g. to array)
        return float(self.value)
    

    def __array__(self):
        # return self.value.__array__()  # Lists or ints do not have __array__
        return np.array(self.value)
    

    # def __array_ufunc__(self, function, method, *inputs, **kwargs):
    #     return self.value. __array_ufunc__(function, method, *inputs, **kwargs) #* \
    #         #    self.unit. __array_ufunc__(function, method, *inputs, **kwargs)
    #     # return np.ndarray.array_ufunc(function, method, *inputs, **kwargs)
    

    def __copy__(self):
        return MatrixWithUnits(self.value.__copy__(), self.unit.__copy__())
    
    def copy(self):
        return self.__copy__()
    

    def __eq__(self, other):
        return self.value.__eq__(other.value) and self.unit.__eq__(other.unit)
    

    def __hash__(self):
        return hash(self.value) ^ hash(self.unit)
    

    def __getitem__(self, key):
        return self.value.__getitem__(key) * self.unit.__getitem__(key)
    

    def __setitem__(self, key, value):
        try:
            self.value.__setitem__(key, value.value)
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
    

    # Add properties for add, sub, mult, truediv, rtruediv; matmul maybe as well?
    # -> also iadd, isub, imul? For inplace stuff
    def __neg__(self):
        return MatrixWithUnits(-self.value, self.unit)
    

    def __add__(self, other):
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnits(self.value + other, self.unit)
        # elif isinstance(other, self._allowed_unit_types):
        #     return MatrixWithUnits(self.value, self.unit + np.asarray(other, dtype=object))
        # Adding just units without values does not make much sense, right?
        elif isinstance(other, u.Quantity):
            assert np.all(np.equal(other.unit, self.unit))

            return MatrixWithUnits(self.value + other.value, self.unit)
        elif isinstance(other, MatrixWithUnits):
            assert np.all(np.equal(other.unit, self.unit))
            
            return MatrixWithUnits(self.value + other.value, self.unit)
        else:
            try:
                return MatrixWithUnits(self.value * other, self.unit)
            except:
                raise TypeError(
                    f'Addition between {type(other)} and `MatrixWithUnit`'
                    ' is not supported.'
                )
    

    def __radd__(self, other):
        return self.__add__(other)
            

    def __mul__(self, other):
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnits(self.value * other, self.unit)
        elif isinstance(other, self._allowed_unit_types):
            return MatrixWithUnits(self.value, self.unit * np.asarray(other, dtype=object))
        elif isinstance(other, u.Quantity):
            return MatrixWithUnits(self.value * other.value, self.unit * other.unit)
        elif isinstance(other, MatrixWithUnits):
            return MatrixWithUnits(self.value * other.value, self.unit * other.unit)
        else:
            try:
                return MatrixWithUnits(self.value * other, self.unit)
            except:
                raise TypeError(
                    f'Multiplication between {type(other)} and `MatrixWithUnit`'
                    ' is not supported.'
                )
    

    def __rmul__(self, other):
        return self.__mul__(other)


    def __truediv__(self, other):
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnits(self.value / other, self.unit)
        elif isinstance(other, self._allowed_unit_types):
            return MatrixWithUnits(self.value, self.unit / np.asarray(other, dtype=object))
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
    

    def __rtruediv__(self, other):
        # Important: 1/u.Unit becomes quantity, we have to use power -1 so that
        # unit actually remains a unit
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnits(other / self.value, self.unit**-1)
        elif isinstance(other, self._allowed_unit_types):
            return MatrixWithUnits(1.0 / self.value, np.asarray(other, dtype=object) / self.unit)
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
            
    def __pow__(self, other):
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnits(self.value.__pow__(other), self.unit.__pow__(other))
        else:
            raise TypeError(
                'Raising of `MatrixWithUnit` to a non-numeric type like '
                f'{type(other)} is not supported.'
            )

    # Do we have to make manual slicing?
        



# ---------- First version ----------
# class MatrixWithUnit():
# # class MatrixWithUnit(np.ndarray):
#     # def __init__(self, shape):
#     #     self._values = np.ndarray(shape, dtype=float)
#     #     self._units = np.ndarray(shape, dtype=object)

#     #     # return self._values * self._units
    
#     # def __new__(cls, shape, **kwargs) -> Self:
#     #     cls._values = np.ndarray(shape, dtype=float, **kwargs)
#     #     cls._units = np.ndarray(shape, dtype=object, **kwargs)
#     #     return cls._values * cls._units
    
#     def __new__(cls, values, units, **kwargs):
#         # cls._values = np.ndarray(values, dtype=float, **kwargs)
#         # cls._units = np.ndarray(units, dtype=object, **kwargs)
#         cls._values = np.array(values, dtype=float, **kwargs)
#         cls._units = np.array(units, dtype=object, **kwargs)
#         return cls._values * cls._units

#     # def __new__(cls: type[_ArraySelf], shape: SupportsIndex | Sequence[SupportsIndex], dtype: dtype[Any] | type[Any] | _SupportsDType[dtype[Any]] | str | tuple[Any, int] | tuple[Any, SupportsIndex | Sequence[SupportsIndex]] | list[Any] | _DTypeDict | tuple[Any, Any] | None = ..., buffer: Buffer | None = ..., offset: np.SupportsIndex = ..., strides: SupportsIndex | Sequence[SupportsIndex] | None = ..., order: Literal['K', 'A', 'C', 'F'] | None = ...) -> _ArraySelf:
#     #     return super().__new__(shape, dtype, buffer, offset, strides, order)
    
#     def __float__(self):
#         return self._values
    
#     def __add__(self, other):
#         if np.all(self._units == other._units):
#             return (self._values + other._values) * self._units
#         else:
#             raise ValueError(
#                 'Cannot add matrices with different units.'
#             )

#     def __sub__(self, other):
#         if np.all(self._units == other._units):
#             return (self._values - other._values) * self._units
#         else:
#             raise ValueError(
#                 'Cannot add matrices with different units.'
#             )
        
#     def __mult__(self, other):
#         out = MatrixWithUnit()

#         out._values = self._values * other._values
#         out._units = self._units * other._units

#         return out
    
#     def __truediv__(self, other):
#         out = MatrixWithUnit()

#         out._values = self._values * other._values
#         out._units = self._units * other._units

#         return out
    
#     # TODO: make setters and getters