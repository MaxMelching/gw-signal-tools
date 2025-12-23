# -- Third Party Imports
import astropy.units as u
import numpy as np
from gwpy.timeseries import TimeSeries
import pytest

# -- Local Package Imports
from gw_signal_tools.fisher import FisherMatrixNetwork, FisherMatrix
from gw_signal_tools.waveform import (
    get_wf_generator,
    norm,
    td_to_fd,
    time_phase_wrapper,
)
from gw_signal_tools.types import Detector, HashableDict
from gw_signal_tools.PSDs import psd_no_noise
from gw_signal_tools import enable_caching_locally, disable_caching_locally  # noqa: F401


# %% -- Initializing commonly used variables ----------------------------------
f_min = 20.0 * u.Hz
f_max = 1024.0 * u.Hz

wf_params = HashableDict(
    {
        'total_mass': 100.0 * u.solMass,
        'mass_ratio': 0.42 * u.dimensionless_unscaled,
        'deltaT': 1.0 / 2048.0 * u.s,
        'f22_start': f_min,
        'f_max': f_max,
        'f22_ref': 20.0 * u.Hz,
        'phi_ref': 0.0 * u.rad,
        'distance': 1.0 * u.Mpc,
        'inclination': 0.0 * u.rad,
        'time': 0.0 * u.s,
        'phase': 0.0 * u.rad,
        # -- det is left out in purpose, shows that it is set automatically
        'ra': 0.2 * u.rad,
        'dec': 0.2 * u.rad,
        'psi': 0.5 * u.rad,
        'tgps': 1126259462,
        'condition': 0,
    }
)


with enable_caching_locally():
    # with disable_caching_locally():
    # -- Avoid globally changing caching, messes up test_caching
    phenomx_generator = time_phase_wrapper(get_wf_generator('IMRPhenomXPHM'))
    phenomx_cross_generator = time_phase_wrapper(
        get_wf_generator('IMRPhenomXPHM', mode='cross')
    )
    phenomd_generator = time_phase_wrapper(get_wf_generator('IMRPhenomD'))

# -- Make sure mass1 and mass2 are not in default_dict
import lalsimulation.gwsignal.core.parameter_conventions as pc  # noqa: E402

pc.default_dict.pop('mass1', None)
pc.default_dict.pop('mass2', None)


hanford = Detector('H1', psd_no_noise)
livingston = Detector('L1', psd_no_noise)


fisher_tot_mass = FisherMatrixNetwork(
    point=wf_params,
    params_to_vary=['total_mass', 'time', 'phase'],
    wf_generator=phenomx_generator,
    detectors=[hanford, livingston],
)


# %% -- Simple consistency tests ----------------------------------------------
def test_single_det_consistency():
    fisher_v1 = FisherMatrixNetwork(
        wf_params, 'total_mass', phenomx_generator, [hanford], psd=psd_no_noise
    )  # Passing psd to make sure we catch that in __init__

    fisher_v2 = FisherMatrix(
        wf_params | {'det': 'H1'}, 'total_mass', phenomx_generator, psd=psd_no_noise
    )

    assert fisher_v1.fisher == fisher_v2.fisher
    assert fisher_v1.snr() == fisher_v2.snr()

    for opt in [False, True]:
        sys_bias_1 = fisher_v1.systematic_bias(
            phenomd_generator, optimize=opt, return_opt_info=False
        )

        sys_bias_2 = fisher_v2.systematic_bias(
            phenomd_generator, optimize=opt, return_diagnostics=False
        )

        assert sys_bias_1 == sys_bias_2


# %% -- Feature tests ---------------------------------------------------------
@pytest.mark.parametrize(
    'params',
    [
        'total_mass',
        ['total_mass'],
        ['total_mass', 'mass_ratio'],
        pytest.param(
            ['chirp_mass'],
            marks=pytest.mark.xfail(
                raises=RuntimeError,  # Error depends on routine
                strict=True,
                reason='Error during Fisher matrix calculation',
            ),
        ),
    ],
)
def test_params(params):
    FisherMatrixNetwork(wf_params, params, phenomx_generator, [hanford, livingston])


@pytest.mark.parametrize(
    'det',
    [
        hanford,
        [hanford, livingston],
        pytest.param(
            [Detector('H96', psd_no_noise)],
            marks=pytest.mark.xfail(
                raises=RuntimeError, strict=True, reason='Invalid detector'
            ),
        ),
    ],
)
def test_detector(det):
    FisherMatrixNetwork(wf_params, 'total_mass', phenomx_generator, det)


def test_index_from_det():
    assert fisher_tot_mass._index_from_det(hanford) == 0
    assert fisher_tot_mass._index_from_det(hanford.name) == 0

    assert fisher_tot_mass._index_from_det(livingston) == 1
    assert fisher_tot_mass._index_from_det(livingston.name) == 1


def test_detector_fisher():
    assert (
        fisher_tot_mass.detector_fisher(hanford)
        == fisher_tot_mass.detector_fisher(hanford.name)
    ) and (
        fisher_tot_mass.detector_fisher(hanford) == fisher_tot_mass.detector_fisher(0)
    )

    assert (
        fisher_tot_mass.detector_fisher(livingston)
        == fisher_tot_mass.detector_fisher(livingston.name)
    ) and (
        fisher_tot_mass.detector_fisher(livingston)
        == fisher_tot_mass.detector_fisher(1)
    )


@pytest.mark.parametrize('params', [None, 'total_mass', ['total_mass']])
def test_stat_bias(params):
    noise = dict(
        H1=TimeSeries(
            np.random.normal(scale=0.1, size=1024), sample_rate=1024, unit=u.strain
        ),
        L1=TimeSeries(
            np.random.normal(scale=0.1, size=1024), sample_rate=1024, unit=u.strain
        ),
    )
    fisher_tot_mass.statistical_bias(noise=noise, params=params)

    noise_fd = dict((k, td_to_fd(v)) for k, v in noise.items())
    fisher_tot_mass.statistical_bias(noise=noise_fd, params=params)


@pytest.mark.parametrize('params', [None, 'total_mass', ['total_mass']])
def test_std_dev(params):
    fisher_tot_mass.standard_deviation(params)


@pytest.mark.parametrize(
    'params', [None, 'total_mass', ['total_mass', 'time', 'phase']]
)
def test_sys_bias(params):
    fisher = FisherMatrixNetwork(
        wf_params,
        ['total_mass', 'mass_ratio', 'time', 'phase'],
        phenomx_generator,
        [hanford, livingston],
        direct_computation=False,  # To test this option too
    )

    fisher.systematic_bias(
        phenomd_generator, 'total_mass', optimize=False, return_diagnostics=True
    )

    fisher.systematic_bias(phenomd_generator, params, return_diagnostics='deriv_info')

    fisher.systematic_bias(phenomd_generator, optimize=True)

    fisher.systematic_bias(phenomd_generator, optimize=['time', 'phase'])

    fisher.systematic_bias(phenomd_generator, optimize_fisher=['time', 'phase'])

    fisher.systematic_bias(phenomd_generator, optimize='total_mass')

    fisher = fisher.update_attrs(
        return_info=False,  # Does not make sense to give, is overwritten. This is what we test here
        new_params_to_vary=['total_mass', 'mass_ratio', 'time', 'phase'],
    )

    fisher.systematic_bias(
        phenomd_generator,
        params='mass_ratio',
        optimize='time',
        optimize_fisher='total_mass',
    )

    fisher.systematic_bias(
        phenomd_generator, optimize=False, optimize_fisher='phase', return_opt_info=True
    )

    with pytest.raises(
        RuntimeError, match='Error during systematic bias calculation in detector '
    ):
        fisher.systematic_bias(None)


@pytest.mark.parametrize(
    'inner_prod_kwargs',
    [
        {},
        dict(f_range=[20.0 * u.Hz, 42.0 * u.Hz]),
        dict(df=2**-2, min_dt_prec=1e-5 * u.s),
    ],
)
def test_snr(inner_prod_kwargs):
    snr = 0.0
    for det in [hanford, livingston]:
        snr += (
            norm(
                phenomx_generator(wf_params | {'det': det.name}),
                psd=det.psd,
                **inner_prod_kwargs,
            )
            ** 2
        )
    assert snr**0.5 == fisher_tot_mass.snr(**inner_prod_kwargs)


@pytest.mark.parametrize(
    'new_point', [None, wf_params | {'total_mass': 42.0 * u.solMass}]
)
@pytest.mark.parametrize('new_params_to_vary', [None, ['mass_ratio', 'distance']])
@pytest.mark.parametrize('new_wf_generator', [None, phenomx_cross_generator])
@pytest.mark.parametrize('new_detectors', [None, [hanford]])
@pytest.mark.parametrize(
    'new_metadata',
    [None, {'deriv_routine': 'gw_signal_tools', 'convergence_check': 'mismatch'}],
)
def test_update_attrs(
    new_point, new_params_to_vary, new_wf_generator, new_detectors, new_metadata
):
    if new_metadata is None:
        new_metadata = {}  # Because ** is used below

    fisher_tot_mass_v2 = fisher_tot_mass.update_attrs(
        new_point, new_params_to_vary, new_wf_generator, new_detectors, **new_metadata
    )

    if new_point is not None:
        assert fisher_tot_mass_v2.point == new_point
    if new_params_to_vary is not None:
        assert fisher_tot_mass_v2.params_to_vary == new_params_to_vary
    if new_wf_generator is not None:
        assert fisher_tot_mass_v2.wf_generator == new_wf_generator
    if new_detectors is not None:
        assert fisher_tot_mass_v2.detectors == new_detectors
    assert fisher_tot_mass_v2.metadata == (fisher_tot_mass.metadata | new_metadata)


def test_copy():
    fisher_copy = fisher_tot_mass.copy()

    fisher_copy._fisher = None
    fisher_copy._fisher_inverse = None
    fisher_copy.point = None
    fisher_copy.wf_generator = None
    fisher_copy._detectors = None
    fisher_copy.metadata = None
    fisher_copy._deriv_info = None
    fisher_copy._is_projected = True

    for attr in [
        'fisher',
        'fisher_inverse',
        'point',
        'wf_generator',
        'detectors',
        'metadata',
        'deriv_info',
    ]:
        assert fisher_tot_mass.__getattribute__(attr) is not None

    assert not fisher_tot_mass.is_projected
