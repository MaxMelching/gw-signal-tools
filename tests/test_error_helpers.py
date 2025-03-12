# -- Third Party Imports
import pytest
import astropy.units as u
from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries

# -- Local Package Imports
from gw_signal_tools.waveform._error_helpers import *


@pytest.mark.parametrize(
    'q, unit', [
        (10*u.km, u.m),
        (1*u.pc, u.Mpc),
        pytest.param(1.*u.m, u.kg, marks=pytest.mark.xfail(raises=ValueError,
        strict=True, reason='Invalid conversion unit')),
    ],
)
def test_q_convert(q, unit):
    _q_convert(q, unit, '', '')


n = 42
@pytest.mark.parametrize(
    's', [
        (Series(n*[1, ], x0=0, dx=0.5),),
        (Series(n*[1, ], x0=0, dx=0.5), Series(n*[2, ], x0=0, dx=0.5)),
        (Series(n*[1, ], x0=0, dx=0.5), Series(n*[2, ], x0=0, dx=0.5), Series(n*[3, ], x0=0, dx=0.5)),
        pytest.param((Series(n*[1, ], x0=0, dx=0.5), Series(n*[1, ], x0=1, dx=0.5)), marks=pytest.mark.xfail(raises=ValueError,
        strict=True, reason='Non-matching x0')),
        pytest.param((Series(n*[1, ], x0=0, dx=0.5), Series(n*[1, ], x0=0, dx=0.25)), marks=pytest.mark.xfail(raises=ValueError,
        strict=True, reason='Non-matching dx')),
    ],
)
def test_compare_series_xindex(s):
    _compare_series_xindex(*s)

df = 0.5
@pytest.mark.parametrize(
    's', [
        (FrequencySeries(n*[1, ], f0=0, df=0.5),),
        (FrequencySeries((n-1)*[1, ], f0=0, df=0.5),),
        (FrequencySeries(n*[1, ], f0=0, df=0.5), FrequencySeries(n*[2, ], f0=0, df=0.5)),
        (FrequencySeries(n*[1, ], f0=0, df=0.5), FrequencySeries(n*[2, ], f0=0, df=0.5), FrequencySeries(n*[3, ], f0=0, df=0.5)),
        (FrequencySeries(n*[1, ], f0=-n//2*df, dx=df), FrequencySeries((n-1)*[1, ], f0=-n//2*df, df=df)),
        pytest.param((FrequencySeries(n*[1, ], f0=-n, df=df), ), marks=pytest.mark.xfail(raises=AssertionError,
        strict=True, reason='Invalid x0 (non-symmetric)')),
        pytest.param((FrequencySeries((n-1)*[1, ], f0=-n, df=df), ), marks=pytest.mark.xfail(raises=AssertionError,
        strict=True, reason='Invalid x0 (non-symmetric)')),  # Should not play role, but still check
        pytest.param((FrequencySeries(n*[1, ], f0=-n//2*df, df=df), FrequencySeries(n*[1, ], f0=n//2*df, df=df)), marks=pytest.mark.xfail(raises=AssertionError,
        strict=True, reason='Invalid x0')),  # Only second one invalid, must still recognize
    ],
)
def test_ft_compatible(s):
    _assert_ft_compatible(*s)
