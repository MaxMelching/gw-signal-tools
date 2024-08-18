import numpy as np
from gw_signal_tools.psd import psd_to_file
from gwpy.frequencyseries import FrequencySeries


# ----- Saving a PSD -----
frequs = np.arange(-2048., 2048.06, step=0.0625)


test = FrequencySeries(
    np.ones(frequs.size),
    frequencies=frequs
)

# psd_to_file(test, 'no_psd.txt')


# ----- Selectively adjusting units (e.g. to improve condition number) -----
from gw_signal_tools.types import MatrixWithUnits
import astropy.units as u

test = MatrixWithUnits([42, 96], [u.m, u.s])

print(test)

test[0] = test[0].to(u.km)

print(test)
print(test[0].value)
print(test[0].unit)
