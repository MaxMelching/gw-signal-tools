from typing import Any
import astropy.units as u
import numpy as np
from .matrix_with_units import MatrixWithUnits


from gwpy.types import Series, Index

class SeriesMatrixWithUnits(MatrixWithUnits):
# class SeriesMatrix(MatrixWithUnits):  # Second idea for name
    """
    Basic idea of class: each Series is treated as element, not
    """
    _allowed_value_types = (Index, )
    _pure_unit_types = (u.IrreducibleUnit, u.CompositeUnit, u.Unit)
    _allowed_unit_types = _pure_unit_types + (u.Quantity,)
    _allowed_input_types = (Series, ) + _allowed_unit_types + _allowed_value_types

    def __init__(self, value: Any) -> None:
        self.value = value

        self.unit = None
        # Maybe set to units of each Series? Not really helpful, but
        # better than having to avoid removing this property or so


y_vals = np.array([1, 2, 3])
x_vals = np.array([0, 1, 2])
test = Series(value=y_vals, xindex=x_vals, unit=u.s)

test_matrix = np.array([test, 2*test])

print(test_matrix)
print(test_matrix.T)
print(test_matrix @ test_matrix.T)  # Works, but returns numeric values
print(y_vals @ y_vals.T)
print(test*test)

# print(SeriesMatrixWithUnits(test_matrix, u.dimensionless_unscaled))

# Problem: np.ndenumerate goes through every value. For np.ndindex, we
# can make control the indices since shape is passed. If we pass
# self.shape, then controlling this attribute appropriately for
# SerierMatrixWithUnits should yield desired behaviour, that each
# element is Series then

# Idea: make a single index? Because numpy arrays throw it away, no?
# And then control that (i) every element has same one in initialization
# and (ii) that when setting new element, this has same as all others
# (which could just be handled via one that is stored in attribute)

# Uhhh, could be nice because of following: then no need to inherit
# from MatrixWithUnits, unit property would be kind of useless anyway
# (maybe not; but certain functions like plot also don't make sense).
# Instead, we make xindex property in the described manner and then
# have as values the Series values only (ah, so I guess unit needed
# too). But then we can really use numpy array multiplication that
# also works in much more complicated shapes, right? And we add as
# additional test for SeriesMatrixWithUnits that the respective xindex
# are compatible
# -> since we do need unit I think (for value=Series.value to work). So
#    maybe make BaseMatrix class where all operations are defined and
#    then we have MatrixWithUnits as instance that holds numeric values
#    (with plotting and all of this stuff), while SeriesMatrixWithUnits
#    is more basic and only allows operations with suitable types


# Perhaps even more convenient: just subclass Series? Alternative would
# be to add self.xindex = property(Series.xindex) (no idea if this is
# correct syntax, but that's not the point) to new subclass of
# MatrixWithUnits

from gwpy.frequencyseries import FrequencySeries
from gwpy.types import Series

# class SeriesVector(FrequencySeries):
# class SeriesVector(Series):
class NDSeries(Series):
    _ndim = 2

    # Could be datatype for NetworkWaveform
    # -> I guess SeriesMatrix should have _ndim = 3 then

    # Hmm, should SeriesVector[i] return the i-th column? I.e. values
    # for all series at i-th sample. And to really get i-th row, one
    # has to use SeriesVector[i, :] -> I think would be worth it and
    # also make sense from idea of class
    

    # Convenient: we can just add a matmul operation that works in way
    # we intend it to work

# test = NDSeries(data=np.zeros((2, 2)), xindex=[0, 1], unit=u.s)
test = NDSeries(value=np.zeros((2, 2)), xindex=[0, 1], unit=u.s)

print(test)
print(FrequencySeries._ndim)
print(Series._ndim)


class NDFrequencySeries(NDSeries):
    # -- Copy FrequencySeries properties --------
    _default_xunit = u.Unit('Hz')
    _print_slots = ['f0', 'df', 'epoch', 'name', 'channel']

    def __new__(cls, data, unit=None, f0=None, df=None, frequencies=None,
                name=None, epoch=None, channel=None, **kwargs):
        """Generate a new NDFrequencySeries.
        """
        if f0 is not None:
            kwargs['x0'] = f0
        if df is not None:
            kwargs['dx'] = df
        if frequencies is not None:
            kwargs['xindex'] = frequencies

        # generate FrequencySeries
        return super().__new__(
            cls, data, unit=unit, name=name, channel=channel,
            epoch=epoch, **kwargs)

    f0 = property(Series.x0.__get__, Series.x0.__set__, Series.x0.__delete__,
                  """Starting frequency for this `FrequencySeries`

                  :type: `~astropy.units.Quantity` scalar
                  """)

    df = property(Series.dx.__get__, Series.dx.__set__, Series.dx.__delete__,
                  """Frequency spacing of this `FrequencySeries`

                  :type: `~astropy.units.Quantity` scalar
                  """)

    frequencies = property(fget=Series.xindex.__get__,
                           fset=Series.xindex.__set__,
                           fdel=Series.xindex.__delete__,
                           doc="""Series of frequencies for each sample""")
    

from gwpy.timeseries import TimeSeries
from gwpy.timeseries.core import _format_time
from gwpy.time import Time, to_gps

class NDTimeSeries(NDSeries):
    # -- Copy TimeSeries properties -------------
    _default_xunit = u.second
    _print_slots = ('t0', 'dt', 'name', 'channel')

    def __new__(cls, data, unit=None, t0=None, dt=None, sample_rate=None,
                times=None, channel=None, name=None, **kwargs):
        """Generate a new `TimeSeriesBase`.
        """
        # parse t0 or epoch
        epoch = kwargs.pop('epoch', None)
        if epoch is not None and t0 is not None:
            raise ValueError("give only one of epoch or t0")
        if epoch is None and t0 is not None:
            kwargs['x0'] = _format_time(t0)
        elif epoch is not None:
            kwargs['x0'] = _format_time(epoch)
        # parse sample_rate or dt
        if sample_rate is not None and dt is not None:
            raise ValueError("give only one of sample_rate or dt")
        if sample_rate is None and dt is not None:
            kwargs['dx'] = dt
        # parse times
        if times is not None:
            kwargs['xindex'] = times

        # generate TimeSeries
        new = super().__new__(cls, data, name=name, unit=unit,
                              channel=channel, **kwargs)

        # manually set sample_rate if given
        if sample_rate is not None:
            new.sample_rate = sample_rate

        return new

    # -- TimeSeries properties ------------------

    # rename properties from the Series
    t0 = Series.x0
    dt = Series.dx
    span = Series.xspan
    times = Series.xindex

    # -- epoch
    # this gets redefined to attach to the t0 property
    @property
    def epoch(self):
        """GPS epoch for these data.

        This attribute is stored internally by the `t0` attribute

        :type: `~astropy.time.Time`
        """
        try:
            return Time(self.t0, format='gps', scale='utc')
        except AttributeError:
            return None

    @epoch.setter
    def epoch(self, epoch):
        if epoch is None:
            del self.t0
        elif isinstance(epoch, Time):
            self.t0 = epoch.gps
        else:
            try:
                self.t0 = to_gps(epoch)
            except TypeError:
                self.t0 = epoch

    # -- sample_rate
    @property
    def sample_rate(self):
        """Data rate for this `TimeSeries` in samples per second (Hertz).

        This attribute is stored internally by the `dx` attribute

        :type: `~astropy.units.Quantity` scalar
        """
        return (1 / self.dt).to('Hertz')

    @sample_rate.setter
    def sample_rate(self, val):
        if val is None:
            del self.dt
            return
        self.dt = (1 / u.Quantity(val, u.Hertz)).to(self.xunit)

    # -- duration
    @property
    def duration(self):
        """Duration of this series in seconds

        :type: `~astropy.units.Quantity` scalar
        """
        return u.Quantity(self.span[1] - self.span[0], self.xunit,
                              dtype=float)


class NDWaveform(NDSeries):
    def __init__(self) -> None:
        super().__init__()
