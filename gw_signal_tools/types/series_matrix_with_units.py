# -- Standard Lib Imports -----------------------
from __future__ import annotations
from typing import Any, Optional

# -- Third Party Imports ------------------------
import astropy.units as u
import numpy as np
from gwpy.types import Series, Index

# -- Local Package Imports ----------------------
from .matrix_with_units import MatrixWithUnits


class SeriesMatrixWithUnits(MatrixWithUnits):
# class SeriesMatrix(MatrixWithUnits):  # Second idea for name
    """
    Basic idea of class: each Series is treated as element, not
    """
    # _allowed_value_types = (Index, )
    _allowed_value_types = (np.ndarray, np.number)  # Number might be needed for init with row of Series
    _pure_unit_types = (u.IrreducibleUnit, u.CompositeUnit, u.Unit)
    _allowed_unit_types = _pure_unit_types + (u.Quantity,)
    _allowed_input_types = (Series, ) + _allowed_unit_types + _allowed_value_types

    _ndim = 2

    # Maybe make use of property _max_ndim, which really measures ndim
    # of value? Because should not be larger than this, even if Series
    # are element of it


    _meta_data_slots = ('epoch', )  # Exclude name

    def __init__(self, value: Any, unit: Optional[Any] = None) -> None:
        self.value = np.asarray(value)  # TODO: set proper dtype

        if unit is None:
            self.unit = u.dimensionless_unscaled
        else:
            self.unit = np.asarray(unit, dtype=object)
        # Maybe set to units of each Series? Not really helpful, but
        # better than having to avoid removing this property or so
    
    # -- Define index etc that all elements have in common
    xindex = Series.xindex
    xunit = Series.xunit
    xspan = Series.xspan

    def _compatible_index(self, other):
        # return self.xindex == other.xindex
        from gw_signal_tools.test_utils import allclose_quantity
        return allclose_quantity(self.xindex, other.xindex)

    # Then define functions like this:
    # -> explanation: unit checks are performed in MatrixWithUnits, all
    #    that we need is additional check for compatible indices
    def __matmul__(self, other):
        self._compatible_index(self, other)
        return super().__matmul__(other)
    
    @property
    def shape(self):
        return np.shape(self.value)
    # Something like this, maybe don't even overwrite MatrixWithUnits one
    
    @property
    def outer_shape(self):
        return self.shape[:self._ndim]
    
    @property
    def inner_shape(self):
        return self.shape[self._ndim:]
    
    # outer for comparison between value matrix and unit matrix.
    # inner for comparison between elements of value matrix and
    # xindex
    
    def __getitem__(self, key: Any) -> SeriesMatrixWithUnits:
        new_value = self.value.__getitem__(key)
        
        if isinstance(self.unit, self._pure_unit_types):
            new_unit = self.unit
        else:
            new_unit = self.unit.__getitem__(key)

        if isinstance(new_value, self._allowed_value_types):
            out = Series(new_value, unit=new_unit, xindex=self.xindex)
        else:
            out = SeriesMatrixWithUnits(new_value, unit=new_unit,
                                        xindex=self.xindex)

        out.__metadata_finalize__(self)
        return out
