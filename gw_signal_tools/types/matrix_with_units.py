# ----- Standard Lib Imports -----
from __future__ import annotations  # Enables type hinting own type in a class
from typing import Optional, Any, Literal

# ----- Third Party Imports -----
import numpy as np
from numpy.typing import ArrayLike
import astropy.units as u


__doc__ = """
Module for the ``MatrixWithUnits`` class that is intended to enable the
use of astropy units with matrices.
"""


class MatrixWithUnits:
    """
    Class for a matrix where entries can have differing units, following
    the spirit of astropy Quantities.

    Parameters
    ----------
    value : 
        Matrix-like object with numerical values.
    unit : 
        Matrix-like object with corresponding units to :code:`value`.
        
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
    not an easy task, especially not for matrices. In particular, some
    operations might not even be possible (e.g. matrix multiplcation
    with arbitrary units or matrix inversion, both of which only work
    in certain circumstances). If possible, ``MatrixWithUnits``
    -compatible versions of many numpy functions can be accessed via
    `MatrixWithUnits.funtion` (if you think an implementation would be
    possible, but is not available, please contact us).

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
    of matrix. In case this is your goal, calling `MatrixWithUnits.inv
    (matrix)` would be the way to go (automatically checks whether or
    not units are consistent).
    

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
    
           
    -> mention MatrixWithUnits.from_numpy_array? Can also say that this just
    creates instance with unit dimensionless, but is there for convenience
    in case you don't want to understand more of inner workings of class
    """
    # -- Set array priority so that Quantity left addition and
    # -- multiplication with MatrixWithUnits are superseded. Otherwise,
    # -- there will be errors (very unintuitive behaviour).
    __array_priority__ = u.Quantity.__array_priority__ + 10

    _allowed_value_types = (int, float, complex, np.number)
    _pure_unit_types = (u.IrreducibleUnit, u.CompositeUnit, u.Unit)
    _allowed_unit_types = _pure_unit_types + (u.Quantity,)
    _allowed_input_types = _allowed_unit_types + _allowed_value_types


    def  __init__(self, value: ArrayLike, unit: ArrayLike) -> None:
        """Initialize a ``MatrixWithUnits``."""
        # Internally, value and unit are stored as numpy arrays due to their
        # versatility. Now we have to make sure the conversion works
        try:
            # Will work for arrays
            value_dtype = value.dtype  # type: ignore
            # Explanation of ignore: we cover case where no dtype exists
        except AttributeError:
            if type(value) in self._allowed_value_types:
                value_dtype = type(value)
            else:
                # Default numeric type
                value_dtype = float

        value = np.asarray(value, dtype=value_dtype)


        # Scalar unit is allowed input even for value array, handled here
        if isinstance(unit, self._allowed_unit_types):
            unit = u.Unit(unit)
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

        :type: `~numpy.ndarray`
        """
        return self._value  # type: ignore
        # Explanation of ignore: we convert to array in __init__, but mypy
        # performs static type checking and does not recognize this
    
    @value.setter
    def value(self, value: ArrayLike) -> None:
        try:
            assert np.shape(value) == np.shape(self.value), \
                'New and old `value` must have equal shape'

            # TODO: also check that len of shape (thus ndim) is not greater than 2?
            # This class is not really made to handle more than that, not
            # sure how this could be handled
        except AttributeError:
            pass  # New class instance is created, nothing to check
        
        # TODO: replace with ndindex[shape] (?) For sure check only
        # elements up to self._ndim (or maybe up to 2, fixed)
        # -> this is NOT needed for unit loops. Only value potentially
        #    has non-zero ndim
        # -> ah shit, we often use MatrixWithUnits that have ndim
        #    smaller than 2, for example row/column vectors... How do we
        #    deal with that? Maybe just allow it, but build it so that
        #    it is possible to allow Series as well
        # for _, val in np.ndenumerate(value):
        for i in np.ndindex(np.shape(value)[:2]):  # [:-1] does not work
            # print(i)
            val = value[i]
            assert (isinstance(val, self._allowed_value_types)
                    and not isinstance(val, bool)), \
                f'Need valid numeric types for all members of `value` (not {type(val)}).'
        
        self._value = value

    @property
    def unit(self) -> np.ndarray:
        """
        Units of the matrix.

        :type: `~numpy.ndarray`
        """
        return self._unit
    
    @unit.setter
    def unit(self, unit: ArrayLike) -> None:
        try:
            if (not isinstance(unit, self._pure_unit_types)
                and not isinstance(self.unit, self._pure_unit_types)):
                assert np.shape(unit) == np.shape(self.unit), \
                    ('New and old `unit` must have equal shape '
                     '(if both are not a scalar unit).')
        except AttributeError:
            pass  # New class instance is created
        
        if not isinstance(unit, self._pure_unit_types):
            # Unit is also array (otherwise would have been converted
            # in __init__)
            for i, val in np.ndenumerate(unit):
                assert isinstance(val, self._allowed_unit_types), \
                    f'Need valid unit types for all members of `unit` (not {type(val)}).'

                unit[i] = u.Unit(val)
                # unit[i] = u.CompositeUnit(1, [val], [1])
                # Enforces keeping prefixes, needed to avoid numerical errors
                # that happen at times (apparently during conversion using,
                # u.Unit, presumably due to scale that is not 1)
                # -> had other reason, Unit and CompositeUnit are equivalent

        self._unit = unit


    # ----- Deal with certain standard class functions -----
    def __repr__(self) -> str:
        return (np.array(self.value) * np.array(self.unit)).__repr__()
        # return (self.value * self.unit).__repr__()
        # return (np.array(self.value) * np.array(self.unit)).tolist().__repr__()  # Are converted in init -> now not anymore
    
    def __float__(self) -> float:
        # Will throw TypeError in most cases, but not because of our class,
        # but because of way it has been set by user (e.g. to array)
        return float(self.value * self.unit)
    
    def __copy__(self) -> MatrixWithUnits:
        # Is called for matrix_copy = copy(matrix)
        if isinstance(self.unit, self._pure_unit_types):
            return MatrixWithUnits(self.value.__copy__(), self.unit)
        else:
            return MatrixWithUnits(self.value.__copy__(), self.unit.__copy__())
    
    def copy(self) -> MatrixWithUnits:
        # Is called for matrix_copy = matrix.copy() or
        # matrix_copy = MatrixWithUnits.copy(matrix)
        return self.__copy__()
    
    def __eq__(self, other: Any) -> np.ndarray:
        if not isinstance(other, (MatrixWithUnits, u.Quantity, np.ndarray, self._allowed_value_types)):
            # Quantities are included here because slicing sometimes returns
            # them, so throwing error here would not be good
            raise TypeError(
                f'Cannot compare ``MatrixWithUnits`` with type {type(other)}.'
            )
        else:
            if isinstance(other, (np.ndarray, self._allowed_value_types)):
                other = other * u.dimensionless_unscaled

            return np.equal(self.value, other.value) & np.equal(self.unit, other.unit)
            # NOT equivalent to == or .__eq__, np.equal has better behaviour
            # (compares unit arrays and scalar units in way we intend to)
    
    def __ne__(self, other: Any) -> np.ndarray:
        # Has to be implemented, applying not operator to array is not working
        return np.logical_not(self == other)
    
    def __hash__(self) -> int:
        raise TypeError(
            '`MatrixWithUnits` instances cannot be hashed because they are '
            'based on numpy arrays, which are in turn unhashable.'
        )
    
    def __getitem__(self, key: Any) -> MatrixWithUnits:
        new_value = self.value.__getitem__(key)
        if isinstance(self.unit, self._pure_unit_types):
            if isinstance(new_value, self._allowed_value_types):
                # Scalar
                return new_value * self.unit
            else:
                return MatrixWithUnits(new_value, self.unit)
        else:
            if isinstance(new_value, self._allowed_value_types):
                return new_value * self.unit.__getitem__(key)
            else:
                return MatrixWithUnits(new_value, self.unit.__getitem__(key))
    
    # -- How GWpy does it for Index. Do the same?
    # def __getitem__(self, key):
    #     item = super().__getitem__(key)
    #     if item.isscalar:
    #         return item.view(Quantity)
    #     return item
    
    def __setitem__(self, key: Any, value: Any) -> None:
        try:
            self.value.__setitem__(key, value.value)
            
            if isinstance(self.unit, self._pure_unit_types):
                # Make sure not unit of whole matrix is replaced,
                # could easily happen otherwise for scalar unit case
                self.unit = np.full(self.shape, self.unit, dtype=object)
            
            self.unit.__setitem__(key, value.unit)
        except AttributeError:
            try:
                value = u.Quantity(value)

                self.__setitem__(key, value)  # Otherwise scalar case would
                                              # require special handling again
            except TypeError:
                raise TypeError(
                    'Can only set items to data types that have members '
                    '`value` and `unit` (such as astropy Quantities or '
                    'MatrixWithUnits) or can be converted into a Quantity.'
                )
    
    def __len__(self):
        return self.value.__len__()
    

    # ----- Common operations -----
    def __neg__(self) -> MatrixWithUnits:
        return MatrixWithUnits(-self.value, self.unit)
    
    def __abs__(self) -> MatrixWithUnits:
        return MatrixWithUnits(self.value.__abs__(), self.unit)
        # return MatrixWithUnits(np.abs(self.value), self.unit)
    
    def __add__(self, other: Any) -> MatrixWithUnits:
        if isinstance(other, self._allowed_value_types):
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
        # Not used anyway, astropy tries to do it and fails
        return self.__add__(other)
    
    def __sub__(self, other: Any) -> MatrixWithUnits:
        try:
            return self.__add__(other.__neg__())
        except AttributeError:  # no __neg__ for example
            raise TypeError(
                f'Addition between {type(other)} and `MatrixWithUnit` is not '
                'supported.'
            )
    
    def __rsub__(self, other: Any) -> MatrixWithUnits:
        # Not used anyway, astropy tries to do it and fails
        return self.__neg__().__add__(other)
        # return (self.__sub__(other)).__neg__()  # Equivalent
            
    def __mul__(self, other: Any) -> MatrixWithUnits:
        if isinstance(other, self._allowed_value_types):
            return MatrixWithUnits(self.value * other, self.unit)
        elif isinstance(other, self._pure_unit_types):
            # ndarray times Unit would produce error, thus do manually
            new_unit = np.empty(self.shape, dtype=object)
            for i, val in np.ndenumerate(self.unit):
                new_unit[i] = u.Unit(val*other)
            return MatrixWithUnits(self.value, new_unit)
        elif isinstance(other, u.Quantity):
            # Fall back to multiplication with unit and value
            return self * other.value * other.unit
        elif isinstance(other, MatrixWithUnits):
            return MatrixWithUnits(self.value * other.value, self.unit * other.unit)
        else:
            raise TypeError(
                f'Multiplication between {type(other)} and `MatrixWithUnit`'
                ' is not supported.'
            )

    def __rmul__(self, other: Any) -> MatrixWithUnits:
        # Not used anyway, astropy tries to do it and fails
        return self.__mul__(other)

    def __truediv__(self, other):
        # if isinstance(other, self._allowed_value_types):
        #     return MatrixWithUnits(self.value / other, self.unit)
        # elif isinstance(other, self._pure_unit_types):
        #     # ndarray times Unit would produce error, thus do manually
        #     new_unit = np.empty(self.shape, dtype=object)
        #     for i, val in np.ndenumerate(self.unit):
        #         new_unit[i] = u.Unit(val/other)
        #     return MatrixWithUnits(self.value, new_unit)
        # elif isinstance(other, u.Quantity):
        #     return self / other.value / other.unit
        # elif isinstance(other, MatrixWithUnits):
        #     return MatrixWithUnits(self.value / other.value, self.unit / other.unit)
        # else:
        #     try:
        #         return MatrixWithUnits(self.value / other, self.unit)
        #     except:
        #         raise TypeError(
        #             f'Division of `MatrixWithUnit` and {type(other)}'
        #             ' is not supported.'
        #         )

        try:
            return self * (1/other)
        except:
            raise TypeError(
                f'Division of `MatrixWithUnit` and {type(other)}'
                ' is not supported.'
            )
    
    def __rtruediv__(self, other: Any) -> MatrixWithUnits:
        # if isinstance(other, self._allowed_value_types):
        #     return MatrixWithUnits(other / self.value, 1/self.unit)
        # # Following two are actually handled by astropy (correctly), are left
        # # here as backup (to show how they work). Thus excluded from coverage
        # elif isinstance(other, self._pure_unit_types):
        #     # ndarray times Unit would produce error, thus do manually
        #     new_unit = np.empty(self.shape, dtype=object)
        #     for i, val in np.ndenumerate(self.unit):
        #         new_unit[i] = u.Unit(other/val)
        #     return MatrixWithUnits(1. / self.value, new_unit)
        # elif isinstance(other, u.Quantity):
        #     return MatrixWithUnits(other.value / self.value, new_unit)
        # else:
        #     try:
        #         return MatrixWithUnits(other / self.value, 1/self.unit)
        #     except:
        #         raise TypeError(
        #             f'Division of {type(other)} and `MatrixWithUnit`'
        #             ' is not supported.'
        #         )

        try:
            # return other * (1/self)
            return other * MatrixWithUnits(1/self.value, 1/self.unit)
        except:
            raise TypeError(
                f'Division of `MatrixWithUnit` and {type(other)}'
                ' is not supported.'
            )
            
    def __pow__(self, other: Any) -> MatrixWithUnits:
        if isinstance(other, self._allowed_value_types):
            return MatrixWithUnits(self.value.__pow__(other), self.unit.__pow__(other))
        else:
            raise TypeError(
                'Raising of `MatrixWithUnit` to a non-numeric type like '
                f'{type(other)} is not supported.'
            )
    
    def __matmul__(self, other):
        # Problem we have to circumvent: the code "return MatrixWithUnits(
        # self.value @ other.value, self.unit @ other.unit)" does not work
        # because astropy units cannot be added (and adding them would also
        # change value, which is not intended). Thus we have to handle unit
        # manually (also for compatibility with scalar unit of this class).

        if isinstance(other, MatrixWithUnits):
            # Step 1: apply to values. This is useful because it already checks
            # compatibility of shapes
            new_value = self.value @ other.value
            new_shape = new_value.shape

            # Need at least 1D output
            if len(new_shape) == 1:
                raise ValueError(
                    'For the provided shapes, only ``MatrixWithUnits``'
                    'instances initialized with a scalar unit are permitted. '
                    'If the intention was to perform matrix multiplication '
                    'with a row/column vector, please reshape the instance '
                    'from the current shape `(n,)` to `(n, 1)` or `(1, n)`.'
                )
            
            # Step 2: handle units (array or scalar are possible for both)
            new_unit = np.empty(new_shape, dtype=object)
            
            if (isinstance(self.unit, self._pure_unit_types) and
                  isinstance(other.unit, self._pure_unit_types)):
                # Both units are scalars
                new_unit = np.full(new_shape, self.unit * other.unit, dtype=object)
                # Not setting new scalar unit here because of reshaping
            elif isinstance(self.unit, self._pure_unit_types):
                # One scalar unit, one array
                for index in np.ndindex(new_shape):
                    i, j = index
                    unit_test = other.unit[0, j]

                    assert np.all(np.equal(other.unit[:, j], unit_test)), \
                        'Need consistent units for matrix multiplication.'
                        # TODO: mention more explicitly which units are incompatible? And also indices for which it occurs
                    # NOTE: do NOT replace with == here, does not what we want
                    
                    new_unit[i, j] = self.unit * unit_test
            elif isinstance(other.unit, self._pure_unit_types):
                # One scalar unit, one array
                for index in np.ndindex(new_shape):
                    i, j = index
                    unit_test = self.unit[i, 0]

                    assert np.all(np.equal(self.unit[i, :], unit_test)), \
                        'Need consistent units for matrix multiplication.'
                        # TODO: mention more explicitly which units are incompatible? And also indices for which it occurs
                    # NOTE: do NOT replace with == here, does not what we want
                    
                    new_unit[i, j] = unit_test * other.unit
            else:
                # Both units are arrays
                for index in np.ndindex(new_shape):
                    i, j = index
                    unit_test = self.unit[i, 0] * other.unit[0, j]
                    
                    assert np.all(np.equal(self.unit[i, :] * other.unit[:, j], unit_test)), \
                        'Need consistent units for matrix multiplication.'
                        # TODO: mention more explicitly which units are incompatible? And also indices for which it occurs
                    # NOTE: do NOT replace with == here, does not what we want
                    
                    new_unit[i, j] = unit_test

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
        :type: `~gw_signal_tools.matrix_with_units.MatrixWithUnits`
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
    
    def reshape(self, new_shape: Any) -> MatrixWithUnits:
        if isinstance(self.unit, self._pure_unit_types):
            return MatrixWithUnits(np.reshape(self.value, new_shape),
                                   self.unit)
        else:
            return MatrixWithUnits(np.reshape(self.value, new_shape),
                                   np.reshape(self.unit, new_shape))

    @staticmethod
    def inv(matrix: MatrixWithUnits) -> MatrixWithUnits:
        assert np.all(np.equal(matrix.unit, matrix.T.unit)), \
            'Need symmetric unit for inversion.'

        return MatrixWithUnits(np.linalg.inv(matrix.value), matrix.unit**-1)
    
    def diagonal(self, *args, **kwargs):
        if isinstance(self.unit, self._pure_unit_types):
            return MatrixWithUnits(
                np.diagonal(self.value,*args, **kwargs).copy(),
                self.unit
            )
        else:
            return MatrixWithUnits(
                np.diagonal(self.value, *args, **kwargs).copy(),
                np.diagonal(self.unit, *args, **kwargs).copy()
            )
        # TODO: test if copying is required
    
    def sqrt(self):
        return MatrixWithUnits(np.sqrt(self.value), self.unit**(1/2))

    def cond(self,
        matrix_norm: float | Literal['fro', 'nuc'] = 'fro'
    ) -> float:
        """
        Condition number of the matrix.

        Parameters
        ----------
        matrix_norm : float | Literal['fro', 'nuc'], optional, default = 'fro'
            Matrix norm that shall be used for the calculation. Must be
            compatible with argument `p` of `~numpy.linalg.cond`.

        Returns
        -------
        float
            Condition number of `self.value`.

        See Also
        --------
        numpy.linalg.cond : Routine used for calculation.
        """
        return np.linalg.cond(self.value, p=matrix_norm)

    # ----- Deal with selected useful astropy functions/attributes -----
    def to_system(self, system: Any) -> MatrixWithUnits:
        if isinstance(self.unit, self._pure_unit_types):
            return MatrixWithUnits(self.value, self.unit.to_system(system)[0])
        else:
            new_unit = self.unit
            for index, val in np.ndenumerate(self.unit):
                new_unit[index] = val.to_system(system)[0]
            
            return MatrixWithUnits(self.value, new_unit)
        
        # Trying to set values as well -> to_system only operates on units,
        # so while syntax is correct now, code is not working
        # new_matrix = self.copy()
        # for index in np.ndindex(new_matrix.shape):
        #     new_matrix[index] = new_matrix[index].to_system(system)[0]
        #     # new_matrix[index] = (self.value[index] * self.unit[index]).to_system(system)[0]
        
        # return new_matrix
    
    def to(self, new_unit: u.Unit) -> MatrixWithUnits:
        new_matrix = self.copy()

        for index in np.ndindex(new_matrix.shape):
            new_matrix[index] = new_matrix[index].to(new_unit)
        
        return new_matrix
    
    def decompose(self, bases: Any) -> MatrixWithUnits:
        new_matrix = self.copy()

        for index in np.ndindex(new_matrix.shape):
            new_matrix[index] = new_matrix[index].decompose(bases=bases)
        
        return new_matrix


    # ---------- Some custom additions -----
    def plot(self, ax: Optional[Any] = None):
        # NOTE: all of this code is inspired by heatmap in seaborn, in fact
        # the relative_luminosity function is copied from there
        # -> is done to avoid additional dependencies (pandas, seaborn)
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        def relative_luminance(color):  # pragma: no cover
            """Calculate the relative luminance of a color according to W3C standards

            Parameters
            ----------
            color : matplotlib color or sequence of matplotlib colors
                Hex code, rgb-tuple, or html color name.

            Returns
            -------
            luminance : float(s) between 0 and 1

            """
            rgb = mpl.colors.colorConverter.to_rgba_array(color)[:, :3]
            rgb = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            lum = rgb.dot([.2126, .7152, .0722])
            try:
                return lum.item()
            except ValueError:
                return lum

        if ax is None:
            fig, ax = plt.subplots()

        non_zero_mask = np.not_equal(self.value, 0.)
        mesh = ax.pcolormesh(np.log10(np.abs(self), where=non_zero_mask),
                             cmap='magma')
        mesh.update_scalarmappable()

        ax.invert_yaxis()  # Otherwise indices would start at bottom

        for (index), color in zip(np.ndindex(self.shape), mesh.get_facecolors()):
            i, j = index
            val = self[index].value
            unit = self[index].unit

            lum = relative_luminance(color)
            text_color = '.15' if lum > .408 else 'w'

            ax.text(
                x=j+0.5,
                y=i+0.5,
                s=f'{val:.3e}$\\,${unit:latex}',
                ha='center',
                va='center',
                color=text_color
            )

        if (shape := self.shape)[0] == 1 and shape[0] < shape[1]:
            cbar = ax.figure.colorbar(mesh, location='top')
        else:
            cbar = ax.figure.colorbar(mesh)

        cbar.set_label(r'$\log_{10}(|M_{ij}|)$', labelpad=24.0)
        # \cdot instead of M_{ij}?

        ax.set_aspect(1)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        return ax
