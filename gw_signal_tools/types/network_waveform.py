# -- Standard Lib Imports -----------------------
from __future__ import annotations

# -- Third Party Imports ------------------------
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from typing import Callable

# -- Local Package Imports ----------------------
from .ndseries import NDSeries
from .detector import Detector, DetectorNetwork


# class NDWaveform(NDSeries):
class NetworkWaveform(NDSeries):
    # def __init__(self,
    def __new__(cls,
        # waveform: NDSeries,
        detectors: DetectorNetwork,
        *args,
        **kw_args
    ) -> None:
        # print(args)
        # print(kw_args)
        # super().__init__(*args, **kw_args)
        # # super().__init__( **kw_args)

        # # self.waveform = waveform
        # self.detectors = detectors

        new = super().__new__(cls, *args, **kw_args)
        assert len(detectors) == new.shape[1], (
            # 'Number of waveforms must match number of detectors.'
            'Number of rows in `value` must match number of detectors.'
        )
        new.detectors = detectors
        return new
    
    @staticmethod
    def from_ndseries(
        val: NDSeries,
        detectors: DetectorNetwork,
        make_freq_series: bool = False,
        make_time_series: bool = False,
        #  *args, **kw_args
    ) -> NetworkWaveform:
        # out = NetworkWaveform(
        #     detectors=detectors,
        #     *args,
        #     **(kw_args | dict(
        #         value=val.value,
        #         unit=val.unit,
        #         xindex=val.xindex,
        #         epoch=val.epoch
        #     ))
        # )
        out = NetworkWaveform(
            detectors=detectors,
            # detectors,
            value=val.value,
            # val.value,
            unit=val.unit,
            # val.unit,
            # *args,
            xindex=val.xindex,
            epoch=val.epoch,
            # **kw_args
        )

        out.__metadata_finalize__(val)  # TODO: decide if this is required
        
        # TODO: instead of this, maybe rather make sure we copy correctly?
        # Because the plan is that waveform generators give NDFrequencySeries,
        # NDTimeSeries, so carrying all properties of input correctly
        # should already be sufficient
        # -> or check isinstance(val, NDFrequencySeries) etc and then
        #    set properties based on that?
        if make_freq_series:
            assert not make_time_series, (
                'Cannot convert simultaneously to ``NDFrequencySeries`` and'
                '``NDTimeSeries``.'
            )

            out._default_xunit = FrequencySeries._default_xunit
            out._print_slots = FrequencySeries._print_slots
            out.f0 = FrequencySeries.f0
            out.df = FrequencySeries.df
            out.frequencies = FrequencySeries.frequencies
        elif make_time_series:
            out._default_xunit = TimeSeries._default_xunit
            out._print_slots = TimeSeries._print_slots
            out.t0 = TimeSeries.t0
            out.dt = TimeSeries.dt
            out.span = TimeSeries.span
            out.times = TimeSeries.times
            out.epoch = TimeSeries.epoch
            out.sample_rate = TimeSeries.sample_rate
            out.duration = TimeSeries.duration

        return out

    
    # TODO: add something like get_detector_strain, where we return
    # NDWaveform[i] where i is index for detector -> should utilize
    # getting detectors from indices (and vice versa) and I think it
    # makes most sense to make this a method of DetectorNetwork

    # def get_detector_strain(self, det: Detector | str) -> FrequencySeries | TimeSeries:
    def get_detector_waveform(self, det: Detector | str) -> FrequencySeries | TimeSeries:
        return self[self.detectors.index_from_detector(det)]


class NetworkWaveformGenerator:  # Perhaps more obvious what is done then
    def __init__(self,
        wf_generator: Callable[[dict[str, u.Quantity]],
                               FrequencySeries | TimeSeries],
        detectors: DetectorNetwork
    ) -> None:
        self.wf_generator = wf_generator
        self.detectors = detectors.detectors

    def __call__(self, wf_params: dict[str, u.Quantity]) -> NetworkWaveform:
        waveforms = [self.wf_generator(wf_params | {'det': det.name})
                     for det in self.detectors]
        waveform_nd_series = NDSeries.from_series_list(waveforms)
        
        return NetworkWaveform.from_ndseries(waveform_nd_series, self.detectors)
        # Idea: return vector of waveforms or whatever data type I came up with here
        # -> but is nice that one can then simply call this like usual generator
