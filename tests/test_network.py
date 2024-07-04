# ----- Third Party Imports -----
import astropy.units as u
import pytest

# ----- Local Package Imports -----
from gw_signal_tools.fisher import FisherMatrixNetwork
from gw_signal_tools.waveform_utils import get_wf_generator
from gw_signal_tools.types import Detector
from gw_signal_tools.PSDs import psd_no_noise


#%% Initializing commonly used variables
f_min = 20.*u.Hz
f_max = 1024.*u.Hz

wf_params = {
    'total_mass': 100.*u.solMass,
    'mass_ratio': 0.42*u.dimensionless_unscaled,
    'deltaT': 1./2048.*u.s,
    'f22_start': f_min,
    'f_max': f_max,
    'f22_ref': 20.*u.Hz,
    'phi_ref': 0.*u.rad,
    'distance': 1.*u.Mpc,
    'inclination': 0.0*u.rad,
    'eccentricity': 0.*u.dimensionless_unscaled,
    'longAscNodes': 0.*u.rad,
    'meanPerAno': 0.*u.rad,
    # det is left out in purpose, shows that this is automatically set
    'ra': 0.2*u.rad,
    'dec': 0.2*u.rad,
    'psi': 0.5*u.rad,
    'tgps': 1126259462,
    'condition': 0
}

approximant = 'IMRPhenomXPHM'
phenomx_generator = get_wf_generator(approximant)

# Make sure mass1 and mass2 are not in default_dict (makes messy behaviour)
import lalsimulation.gwsignal.core.parameter_conventions as pc
pc.default_dict.pop('mass1', None);
pc.default_dict.pop('mass2', None);


hanford = Detector('H1', psd_no_noise)
livingston = Detector('L1', psd_no_noise)


fisher_tot_mass = FisherMatrixNetwork(
    wf_params,
    'total_mass',
    phenomx_generator,
    [hanford, livingston]
)

@pytest.mark.parametrize('params', [
    'total_mass',
    ['total_mass'],
    ['total_mass', 'mass_ratio'],
    pytest.param(['chirp_mass'], marks=pytest.mark.xfail(raises=AssertionError,
        strict=True, reason='Invalid parameter')),
])
def test_params(params):
    FisherMatrixNetwork(
        wf_params,
        params,
        phenomx_generator,
        [hanford, livingston]
    )


def test_detector():
    invalid_det = Detector('H96', psd_no_noise)

    with pytest.raises(RuntimeError):
        FisherMatrixNetwork(
            wf_params,
            'total_mass',
            phenomx_generator,
            [invalid_det]
        )
