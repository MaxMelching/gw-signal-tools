# -- Third Party Imports
import astropy.units as u
import numpy as np

# -- Local Package Imports
from gw_signal_tools.types import Detector
from gw_signal_tools.PSDs import psd_no_noise, psd_o3_h1, psd_o3_l1


def test_name():
    hanford = Detector('H1', psd_o3_h1)

    assert hanford.name == 'H1'
    assert np.all(np.equal(hanford.psd, psd_o3_h1))
    assert hanford.inner_prod_kwargs == {'psd': psd_o3_h1}


def test_psd():
    livingston = Detector('L1', psd_o3_l1)

    assert livingston.name == 'L1'
    assert np.all(np.equal(livingston.psd, psd_o3_l1))
    assert livingston.inner_prod_kwargs == {'psd': psd_o3_l1}


def test_inner_prod_kwargs():
    livingston = Detector('L1', psd_o3_l1, f_range=[10.*u.Hz, 1024.*u.Hz])

    assert livingston.name == 'L1'
    assert np.all(np.equal(livingston.psd, psd_o3_l1))
    assert livingston.inner_prod_kwargs == {'f_range': [10.*u.Hz, 1024.*u.Hz],
                                            'psd': psd_o3_l1}


def test_repr():
    hanford = Detector('H1', psd_o3_h1)
    print(hanford)

def test_equal():
    hanford = Detector('H1', psd_o3_h1)
    livingston = Detector('L1', psd_o3_l1)

    assert hanford == Detector('H1', psd_o3_h1)
    assert hanford != livingston
    assert hanford != Detector('H1', psd_o3_l1)
    assert hanford != Detector('L1', psd_o3_h1)
    assert hanford != Detector('H1', psd_o3_h1, f_range=[10.*u.Hz, 1024.*u.Hz])

def test_update():
    hanford = Detector('H1', psd_o3_h1)

    updated_hanford = hanford.update(new_name='H1-updated')
    assert updated_hanford == Detector('H1-updated', psd_o3_h1)

    updated_hanford = hanford.update(new_psd=psd_no_noise)
    assert updated_hanford == Detector('H1', psd_no_noise)

    updated_hanford = hanford.update(f_range=[20.*u.Hz, 2048.*u.Hz])
    assert updated_hanford == Detector('H1', psd_o3_h1, f_range=[20.*u.Hz, 2048.*u.Hz])

    hanford = Detector('H1', psd_o3_h1)
    updated_hanford = hanford.update(new_name='H1-updated', f_range=[20.*u.Hz, 2048.*u.Hz])
    assert updated_hanford == Detector('H1-updated', psd_o3_h1, f_range=[20.*u.Hz, 2048.*u.Hz])

def test_copy():
    hanford = Detector('H1', psd_o3_h1, f_range=[10.*u.Hz, 1024.*u.Hz])
    copied_hanford = hanford.__copy__()

    assert copied_hanford == hanford
