# -- Standard Lib Imports -----------------------
from __future__ import annotations
from typing import Any

# -- Third Party Imports ------------------------
import astropy.units as u
import numpy as np
from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from gwpy.timeseries.core import _format_time
# from gwpy.time import Time, to_gps


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

    @staticmethod
    def from_series_list(val: list[Series]) -> NDSeries:
        assert len(val) > 0, 'Need non-empty `val`.'
        values = []
        units = []
        xindex_vals = []
        xindex_units = []
        epochs = []
        # for i, series in enumerate(val):
        #     values[i] = series.value
        #     units[i] = series.unit
        #     xindices[i] = series.xindex
        #     epochs[i] = series.epoch
        for series in val:
            values += [series.value]
            units += [series.unit]
            # xindices += [series.xindex]
            xindex_vals += [series.xindex.value]
            xindex_units += [series.xindex.unit]
            # epochs += [series.epoch.value]
            # epochs += [series.epoch.__getattribute__('value', None)]
            try:
                epochs += [series.epoch.value]
            except AttributeError:
                epochs += [np.nan]
        

        # from numpy.testing import assert_allclose
        # assert_allclose(units, units[0])
        # assert_allclose(xindices, xindices[0])
        # assert_allclose(epochs, epochs[0])

        assert np.all(np.equal(units, units[0]))
        # assert np.all(np.isclose(xindices, xindices[0]))
        assert np.all(np.isclose(xindex_vals, xindex_vals[0]))
        assert np.all(np.equal(xindex_units, xindex_units[0]))
        assert np.all(np.isclose(epochs, epochs[0], equal_nan=True))

        # return NDSeries(value=values, unit=units[0], xindex=xindices[0],
        return NDSeries(value=values, unit=units[0], xindex=val[0].xindex)#,
                        # epoch=epochs[0] if (epochs[0] != np.nan) else None)

    def __getitem__(self, key: Any) -> NDSeries:
        # print(key)
        out = super().__getitem__(key)

        # -- V1
        # if out.value.ndim == 1:
        #     # return Series(out)
        #     return Series(out.value, out.unit, xindex=out.xindex)
        # elif out.shape[1] == 1:
        #     return NDSeries(out.value, out.unit, xindex=out.xindex)
        # else:
        #     # return NDSeries(out)
        #     return out

        # -- V2
        # if isinstance(key, int):  # Does not catch np.int64...
        #     out = Series(out.value, out.unit, xindex=out.xindex)
        # elif len(key) == 2 and key[0] == slice(None, None, None):
        #     out = NDSeries(out.value.reshape((len(out.value), 1)), out.unit,
        #                    xindex=out.xindex[key[1]])
        #                 #    xindex=out.xindex)
        # if not isinstance(out, u.Quantity):
        #     out.__metadata_finalize__(self)  # Copy metadata that was not set yet
        # return out
    
        # -- V3, final
        if np.isscalar(key):
            out = Series(out.value, out.unit, xindex=out.xindex)
            out.__metadata_finalize__(self)  # Copy metadata that was not set yet
        elif len(key) == 2 and key[0] == slice(None, None, None):
            # -- Column of array means we return values at the same
            # -- xindex value and therefore, do not have a Series
            # -- anymore. Similarly to the GWpy Series, this results in
            # -- a Quantity being returned.
            # -- This also determines return of value_at
            out = u.Quantity(out.value, out.unit)
        return out
    
    def value_at(self, x):
        # -- Essentially a copy of Series.value_at, just with an
        # -- adjusted changed indexing at the end
        x = u.Quantity(x, self.xindex.unit).value
        try:
            idx = (self.xindex.value == x).nonzero()[0][0]
        except IndexError as e:
            e.args = ("Value %r not found in array index" % x,)
            raise
        return self[:, idx]

    # Convenient: we can just add a matmul operation that works in way
    # we intend it to work


class NDFrequencySeries(NDSeries):
    # -- Copy FrequencySeries properties --------
    # _default_xunit = u.Unit('Hz')
    # _print_slots = ['f0', 'df', 'epoch', 'name', 'channel']

    def __new__(cls, data, unit=None, f0=None, df=None, frequencies=None,
                name=None, epoch=None, channel=None, **kwargs):
        """Generate a new NDFrequencySeries.
        """
        # -- Copy code from FrequencySeries.__new__, bust now super()
        # -- calls NDSeries and not Series
        if f0 is not None:
            kwargs['x0'] = f0
        if df is not None:
            kwargs['dx'] = df
        if frequencies is not None:
            kwargs['xindex'] = frequencies

        return super().__new__(
            cls, data, unit=unit, name=name, channel=channel,
            epoch=epoch, **kwargs)

    # f0 = property(Series.x0.__get__, Series.x0.__set__, Series.x0.__delete__,
    #               """Starting frequency for this `FrequencySeries`

    #               :type: `~astropy.units.Quantity` scalar
    #               """)

    # df = property(Series.dx.__get__, Series.dx.__set__, Series.dx.__delete__,
    #               """Frequency spacing of this `FrequencySeries`

    #               :type: `~astropy.units.Quantity` scalar
    #               """)

    # frequencies = property(fget=Series.xindex.__get__,
    #                        fset=Series.xindex.__set__,
    #                        fdel=Series.xindex.__delete__,
    #                        doc="""Series of frequencies for each sample""")
    
    # This here shold be sufficient, right? -> nope, calls Series.__new__
    # def __new__(cls, *args, **kw_args):
    #     return FrequencySeries.__new__(cls, *args, **kw_args)
    # __new__ = FrequencySeries.__new__  # Also not correct
    
    # -- Get properties from FrequencySeries
    _default_xunit = FrequencySeries._default_xunit
    _print_slots = FrequencySeries._print_slots
    f0 = FrequencySeries.f0
    df = FrequencySeries.df
    frequencies = FrequencySeries.frequencies
    

class NDTimeSeries(NDSeries):
    # -- Copy TimeSeries properties -------------
    # _default_xunit = u.second
    # _print_slots = ('t0', 'dt', 'name', 'channel')

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

        new = super().__new__(cls, data, name=name, unit=unit,
                              channel=channel, **kwargs)

        # manually set sample_rate if given
        if sample_rate is not None:
            new.sample_rate = sample_rate

        return new

    # # -- TimeSeries properties ------------------

    # # rename properties from the Series
    # t0 = Series.x0
    # dt = Series.dx
    # span = Series.xspan
    # times = Series.xindex

    # # -- epoch
    # # this gets redefined to attach to the t0 property
    # @property
    # def epoch(self):
    #     """GPS epoch for these data.

    #     This attribute is stored internally by the `t0` attribute

    #     :type: `~astropy.time.Time`
    #     """
    #     try:
    #         return Time(self.t0, format='gps', scale='utc')
    #     except AttributeError:
    #         return None

    # @epoch.setter
    # def epoch(self, epoch):
    #     if epoch is None:
    #         del self.t0
    #     elif isinstance(epoch, Time):
    #         self.t0 = epoch.gps
    #     else:
    #         try:
    #             self.t0 = to_gps(epoch)
    #         except TypeError:
    #             self.t0 = epoch

    # # -- sample_rate
    # @property
    # def sample_rate(self):
    #     """Data rate for this `TimeSeries` in samples per second (Hertz).

    #     This attribute is stored internally by the `dx` attribute

    #     :type: `~astropy.units.Quantity` scalar
    #     """
    #     return (1 / self.dt).to('Hertz')

    # @sample_rate.setter
    # def sample_rate(self, val):
    #     if val is None:
    #         del self.dt
    #         return
    #     self.dt = (1 / u.Quantity(val, u.Hertz)).to(self.xunit)

    # # -- duration
    # @property
    # def duration(self):
    #     """Duration of this series in seconds

    #     :type: `~astropy.units.Quantity` scalar
    #     """
    #     return u.Quantity(self.span[1] - self.span[0], self.xunit,
    #                           dtype=float)


    # This here should be sufficient, right? -> nope, calls Series.__new__
    # def __new__(cls, *args, **kw_args):
    #     return TimeSeries.__new__(*args, **kw_args)
    # __new__ = TimeSeries.__new__  # Also not correct
    
    # -- Get properties from TimeSeries
    _default_xunit = TimeSeries._default_xunit
    _print_slots = TimeSeries._print_slots
    t0 = TimeSeries.t0
    dt = TimeSeries.dt
    span = TimeSeries.span
    times = TimeSeries.times
    epoch = TimeSeries.epoch
    sample_rate = TimeSeries.sample_rate
    duration = TimeSeries.duration
