import numpy as np
import astropy.units as u


# class MatrixWithUnit(np.ndarray):
#     def __new__(cls, value, **kwargs):
#         new.value = super().__new__(np.shape(value), dtype=float, **kwargs)

#         new.unit = super().__new__(np.shape(value), dtype=object)
        # return self._value * self._unit



class MatrixWithUnit:
    _allowed_numeric_types = (int, float, complex, np.number)
    _allowed_unit_types = (u.IrreducibleUnit, u.CompositeUnit, u.Unit)


    def  __init__(self, value, unit) -> None:
        assert np.shape(value) == np.shape(unit), \
                ('`value` and `unit` must have equal shape if unit is an '
                'array of astropy units.')
        

        # Idea: add support for given unit that is "scalar"
        # -> should we also make sure value is np.array?

        # value = np.asarray(value, dtype=np.number)
        # value = np.asarray(value, dtype=float)

        # if isinstance(unit, self._allowed_unit_types):
        #     if not isinstance(value, self._allowed_numeric_types):
        #         unit = np.full(np.shape(value), unit, dtype=object)
        # else:
        #     assert np.shape(value) == np.shape(unit), \
        #         ('`value` and `unit` must have equal shape if unit is an '
        #         'array of astropy units.')
            

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
        return (self.value * self.unit).__repr__()
    

    def __float__(self):
        # if np.shape(self.value)
        return float(self.value)
    

    def __copy__(self):
        return MatrixWithUnit(self.value.__copy__(), self.unit.__copy__())
    
    def copy(self):
        return self.__copy__()
    

    def __eq__(self, other):
        return self.value.__eq__(self.other.value) and self.unit.__eq__(self.other.unit)
    

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
    def __add__(self, other):
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnit(self.value + other, self.unit)
        # elif isinstance(other, self._allowed_unit_types):
        #     return MatrixWithUnit(self.value, self.unit + np.asarray(other, dtype=object))
        # Adding just units without values does not make much sense, right?
        elif isinstance(other, u.Quantity):
            assert np.all(other.unit == self.unit)

            return MatrixWithUnit(self.value + other.value, self.unit)
        elif isinstance(other, MatrixWithUnit):
            assert np.all(other.unit == self.unit)
            
            return MatrixWithUnit(self.value + other.value, self.unit)
        else:
            try:
                return MatrixWithUnit(self.value * other, self.unit)
            except:
                raise TypeError(
                    f'Addition between {type(other)} and `MatrixWithUnit`'
                    ' is not supported.'
                )
    

    def __radd__(self, other):
        return self.__add__(other)
            

    def __mul__(self, other):
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnit(self.value * other, self.unit)
        elif isinstance(other, self._allowed_unit_types):
            return MatrixWithUnit(self.value, self.unit * np.asarray(other, dtype=object))
        elif isinstance(other, u.Quantity):
            return MatrixWithUnit(self.value * other.value, self.unit * other.unit)
        elif isinstance(other, MatrixWithUnit):
            return MatrixWithUnit(self.value * other.value, self.unit * other.unit)
        else:
            try:
                return MatrixWithUnit(self.value * other, self.unit)
            except:
                raise TypeError(
                    f'Multiplication between {type(other)} and `MatrixWithUnit`'
                    ' is not supported.'
                )
    

    def __rmul__(self, other):
        return self.__mul__(other)


    def __truediv__(self, other):
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnit(self.value / other, self.unit)
        elif isinstance(other, self._allowed_unit_types):
            return MatrixWithUnit(self.value, self.unit / np.asarray(other, dtype=object))
        elif isinstance(other, u.Quantity):
            return MatrixWithUnit(self.value / other.value, self.unit / other.unit)
        elif isinstance(other, MatrixWithUnit):
            return MatrixWithUnit(self.value / other.value, self.unit / other.unit)
        else:
            try:
                return MatrixWithUnit(self.value / other, self.unit)
            except:
                raise TypeError(
                    f'Division of `MatrixWithUnit` and {type(other)}'
                    ' is not supported.'
                )
    

    def __rtruediv__(self, other):
        # Important: 1/u.Unit becomes quantity, we have to use power -1 so that
        # unit actually remains a unit
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnit(other / self.value, self.unit**-1)
        elif isinstance(other, self._allowed_unit_types):
            return MatrixWithUnit(1.0 / self.value, np.asarray(other, dtype=object) / self.unit)
        elif isinstance(other, u.Quantity):
            return MatrixWithUnit(other.value / self.value, other.unit / self.unit)
        elif isinstance(other, MatrixWithUnit):
            return MatrixWithUnit(other.value / self.value, other.unit / self.unit)
        else:
            try:
                return MatrixWithUnit(other / self.value, self.unit**-1)
            except:
                raise TypeError(
                    f'Division of {type(other)} and `MatrixWithUnit`'
                    ' is not supported.'
                )
            
    def __pow__(self, other):
        if isinstance(other, self._allowed_numeric_types):
            return MatrixWithUnit(self.value.__pow__(other), self.unit.__pow__(other))
        else:
            raise TypeError(
                'Raising of `MatrixWithUnit` to a non-numeric type like '
                f'{type(other)} is not supported.'
            )

    # Do we have to make manual slicing?