# ----- Local Package Imports -----
import astropy.units as u

# ----- Local Package Imports -----
from gw_signal_tools.types import Detector
from gw_signal_tools.PSDs import psd_no_noise, psd_o3_h1, psd_o3_l1


def test_name():
    hanford = Detector('H1', psd_o3_h1)
    hanford.name = 'h1'
    del hanford.name

def test_psd():
    livingston = Detector('L1', psd_o3_l1)
    livingston.psd = psd_no_noise
    del livingston.psd

def test_wf_args():
    livingston = Detector('L1', psd_o3_l1)
    livingston.wf_args = {'total_mass': 50.*u.solMass}
    assert livingston.wf_args == {'total_mass': 50.*u.solMass, 'det': 'L1'}
    del livingston.wf_args

def test_repr():
    hanford = Detector('H1', psd_o3_h1)
    print(hanford)
