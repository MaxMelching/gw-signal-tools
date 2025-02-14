from typing import Callable, Union
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries


# class GWParams(dict[str, u.Quantity]):
#     """
#     A custom type to represent parameters required for waveform
#     generation in gw-signal-tools.
#     """


# class FDWFGen(Callable[[GWParams], FrequencySeries]):
#     """
#     A custom type to represent frequency domain waveform generators.
#     """


# class TDWFGen(Callable[[GWParams], TimeSeries]):
#     """
#     A custom type to represent time domain waveform generators.
#     """


# # class WFGen(FDWFGen | TDWFGen):
# # class WFGen(Union[FDWFGen, TDWFGen]):
# class WFGen(Callable[[GWParams], Union[FrequencySeries, TimeSeries]]):
#     """
#     A custom type to represent waveform generators in arbitrary domain.
#     """


from typing import TypeVar, TypeAlias

# # GWParams = TypeVar('GWParams', dict[str, u.Quantity])

GWParams: TypeAlias = dict[str, u.Quantity]
FDWFGen = Callable[[GWParams], FrequencySeries]
TDWFGen = Callable[[GWParams], TimeSeries]
WFGen = Union[FDWFGen, TDWFGen]


if __name__ == '__main__':
    test: GWParams = {'mass1': 20*u.Msun}
    # test: dict = {'mass1': 20*u.Msun}

    gen: WFGen = lambda x: x**2

    print(test, type(test))  # Still prints dict... But anyway