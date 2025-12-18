# -- Third Party Imports
import astropy.units as u
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import pytest

# -- Local Package Imports
from gw_signal_tools.waveform import get_wf_generator, norm, td_to_fd
from gw_signal_tools.types import MatrixWithUnits, HashableDict
from gw_signal_tools.fisher import FisherMatrix, fisher_matrix
from gw_signal_tools.test_utils import (
    assert_allclose_MatrixWithUnits,
    assert_allequal_MatrixWithUnits,
)
from gw_signal_tools import enable_caching_locally, disable_caching_locally  # noqa: F401

from gw_signal_tools import PLOT_STYLE_SHEET

plt.style.use(PLOT_STYLE_SHEET)


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
        'deltaF': 2**-5 * u.Hz,
        'f22_ref': 20.0 * u.Hz,
        'phi_ref': 0.0 * u.rad,
        'distance': 1.0 * u.Mpc,
        'inclination': 0.0 * u.rad,
        'eccentricity': 0.0 * u.dimensionless_unscaled,
        'longAscNodes': 0.0 * u.rad,
        'meanPerAno': 0.0 * u.rad,
        'condition': 0,
    }
)


with enable_caching_locally():
    # with disable_caching_locally():
    # -- Avoid globally changing caching, messes up test_caching
    phenomx_generator = get_wf_generator('IMRPhenomXPHM')
    phenomx_cross_generator = get_wf_generator('IMRPhenomXPHM', mode='cross')
    phenomd_generator = get_wf_generator('IMRPhenomD')

# -- Make sure mass1 and mass2 are not in default_dict
import lalsimulation.gwsignal.core.parameter_conventions as pc  # noqa: E402

pc.default_dict.pop('mass1', None)
pc.default_dict.pop('mass2', None)

fisher_tot_mass = FisherMatrix(
    point=wf_params,
    params_to_vary=['total_mass', 'time', 'phase'],
    wf_generator=phenomx_generator,
)


# %% -- Simple consistency tests ----------------------------------------------
def test_unit():
    # -- All ways of accessing must work
    assert fisher_tot_mass.fisher[0, 0].unit == 1.0 / u.solMass**2
    assert fisher_tot_mass.fisher.unit[0, 0] == 1.0 / u.solMass**2
    assert fisher_tot_mass.unit[0, 0] == 1.0 / u.solMass**2


def test_inverse():
    assert np.all(
        np.equal(
            np.linalg.inv(fisher_tot_mass.fisher.value),
            fisher_tot_mass.fisher_inverse.value,
        )
    )

    assert np.all(
        np.equal(fisher_tot_mass.fisher.unit**-1, fisher_tot_mass.fisher_inverse.unit)
    )


def test_fisher_calc():
    fisher_tot_mass_2 = fisher_matrix(
        point=wf_params,
        params_to_vary=['total_mass', 'time', 'phase'],
        wf_generator=phenomx_generator,
        deriv_routine='numdifftools',
    )
    assert_allequal_MatrixWithUnits(
        fisher_tot_mass.fisher,
        fisher_tot_mass_2,
    )


def test_criterion_consistency():
    fisher_tot_mass_1 = fisher_matrix(
        wf_params,
        'total_mass',
        phenomx_generator,
        deriv_routine='gw_signal_tools',
        pass_inn_prod_kwargs_to_deriv=True,
        convergence_check='diff_norm',
    )

    fisher_tot_mass_2 = fisher_matrix(
        wf_params,
        'total_mass',
        phenomx_generator,
        deriv_routine='gw_signal_tools',
        pass_inn_prod_kwargs_to_deriv=True,
        convergence_check='mismatch',
    )

    assert_allclose_MatrixWithUnits(
        fisher_tot_mass_1,
        fisher_tot_mass_2,
        atol=0.0,
        rtol=3e-4,
    )


def test_time_and_phase_shift_consistency():
    # -- Idea: time and phase shift in the waveform should not have
    # -- effect on Fisher matrix entries, cancel out in inner product
    t_shift = 3e-2 * u.s
    phase_shift = 2e-1 * u.rad

    calc_params = ['total_mass', 'mass_ratio', 'distance', 'time', 'phase']

    fisher_v1 = FisherMatrix(wf_params, calc_params, phenomx_generator)

    fisher_v2 = FisherMatrix(
        wf_params | {'time': t_shift, 'phase': phase_shift},
        calc_params,
        phenomx_generator,
    )

    # # print(fisher_v1.fisher - fisher_v2.fisher)
    # print(fisher_v1.fisher.diagonal())
    # print(fisher_v1.fisher.diagonal() - fisher_v2.fisher.diagonal())

    # from gw_signal_tools.waveform import apply_time_phase_shift
    # plt.plot(fisher_v1.deriv_info['total_mass']['deriv'])
    # plt.plot(apply_time_phase_shift(fisher_v1.deriv_info['total_mass']['deriv'], t_shift, phase_shift), '--')
    # plt.plot(fisher_v2.deriv_info['total_mass']['deriv'], '-.')

    # plt.xlim(10, 400)

    # plt.show()

    # plt.plot(fisher_v1.deriv_info['total_mass']['error_estimate'])
    # plt.plot(fisher_v2.deriv_info['total_mass']['error_estimate'], '--')
    # plt.plot((fisher_v2.deriv_info['total_mass']['deriv'] - apply_time_phase_shift(fisher_v1.deriv_info['total_mass']['deriv'], t_shift, phase_shift)).abs(), '-.')

    # plt.xlim(10, 400)

    # plt.show()

    assert_allclose_MatrixWithUnits(
        fisher_v1.fisher,
        fisher_v2.fisher,
        atol=4e-53,
        rtol=4.8e-6,
    )
    # -- Gotta be happy with that. Derivative is accurate to about 0.1%
    # -- (on average), which translates to 1e-6 accuracy in Fisher. atol
    # -- is just one of the off-diagonal values, but super small anyway,
    # -- diagonal values of Fisher are all on scale of 1e-39.


def test_deriv_routine_consistency():
    calc_params = ['total_mass', 'mass_ratio', 'distance', 'time', 'phase']

    global fisher_gw_signal_tools, fisher_numdifftools, fisher_amplitude_phase

    for routine in ['gw_signal_tools', 'numdifftools', 'amplitude_phase']:
        globals()[f'fisher_{routine}'] = FisherMatrix(
            wf_params, calc_params, phenomx_generator, deriv_routine=routine
        )

    # print(fisher_gw_signal_tools.fisher)
    # print(fisher_numdifftools.fisher)
    # print(fisher_amplitude_phase.fisher)

    # -- Ensure mutual consistency
    assert_allclose_MatrixWithUnits(
        fisher_gw_signal_tools.fisher, fisher_numdifftools.fisher, atol=0.0, rtol=9.4e-4
    )
    assert_allclose_MatrixWithUnits(
        fisher_gw_signal_tools.fisher,
        fisher_amplitude_phase.fisher,
        atol=0.0,
        rtol=7e-4,
    )
    assert_allclose_MatrixWithUnits(
        fisher_numdifftools.fisher, fisher_amplitude_phase.fisher, atol=0.0, rtol=2.4e-4
    )
    # -- Comparing values manually, this absolute deviation indicates
    # -- that impact of different routines on is not really siginificant

    # -- Remove variables from global scope
    for routine in ['gw_signal_tools', 'numdifftools', 'amplitude_phase']:
        del globals()[f'fisher_{routine}']


def test_base_step_consistency():
    calc_params = ['total_mass', 'distance', 'time', 'phase']

    fisher_v1 = FisherMatrix(
        wf_params, calc_params, phenomx_generator, deriv_routine='numdifftools'
    )

    fisher_v2 = FisherMatrix(
        wf_params,
        calc_params,
        phenomx_generator,
        deriv_routine='numdifftools',
        base_step=None,  # Enables automatic selection
    )

    assert_allclose_MatrixWithUnits(
        fisher_v1.fisher, fisher_v2.fisher, atol=6.2e-40, rtol=0.0
    )
    # -- Really good agreement upon manual inspection


# %% -- Feature tests ---------------------------------------------------------
def test_covmat():
    fisher_tot_mass.covariance_matrix


def test_get_indices():
    test_params = ['total_mass', 'time', 'phase']

    fisher = FisherMatrix(
        wf_params,
        test_params,
        wf_generator=phenomx_generator,
    )

    indices_1 = fisher.get_param_indices(['time', 'phase'])
    assert np.all(indices_1 == np.array([1, 2]))

    indices_2 = fisher.get_param_indices(['phase', 'time'])
    assert np.all(indices_2 == np.array([2, 1]))

    indices_2 = fisher.get_param_indices()
    # assert np.all(indices_2 == np.array([True, True, True]))
    assert np.all(indices_2 == np.array([0, 1, 2]))

    grid_1 = fisher.get_sub_matrix_indices(['time', 'phase'])
    sub_matr_1 = [
        [fisher.fisher[1, 1], fisher.fisher[1, 2]],
        [fisher.fisher[2, 1], fisher.fisher[2, 2]],
    ]

    grid_2 = fisher.get_sub_matrix_indices(['phase', 'time'])
    sub_matr_2 = [
        [fisher.fisher[2, 2], fisher.fisher[2, 1]],
        [fisher.fisher[1, 2], fisher.fisher[1, 1]],
    ]

    for index in np.ndindex((2, 2)):
        i, j = index
        assert fisher.fisher[grid_1][i, j] == sub_matr_1[i][j]
        assert fisher.fisher[grid_2][i, j] == sub_matr_2[i][j]

    with pytest.raises(ValueError):
        fisher.get_param_indices('mass_ratio')


@pytest.mark.parametrize(
    'inner_prod_kwargs',
    [dict(f_range=[f_min, f_max]), dict(df=2**-2, min_dt_prec=1e-5)],
)
def test_inner_prod_kwargs(inner_prod_kwargs):
    fisher = FisherMatrix(
        wf_params,
        'total_mass',
        wf_generator=phenomx_generator,
        return_info=True,
        direct_computation=False,
        **inner_prod_kwargs,
    )
    assert fisher._inner_prod_kwargs == inner_prod_kwargs


def test_attribute_getting():
    fisher = FisherMatrix(
        wf_params,
        'total_mass',
        wf_generator=phenomx_generator,
        return_info=True,
        direct_computation=False,
    )
    fisher.project_fisher('total_mass')

    fisher = FisherMatrix(
        wf_params,
        'total_mass',
        wf_generator=phenomx_generator,
        return_info=True,
        direct_computation=False,
    )
    fisher.cond()

    fisher = FisherMatrix(
        wf_params,
        'total_mass',
        wf_generator=phenomx_generator,
        return_info=True,
        direct_computation=False,
    )
    fisher.inv(fisher_tot_mass.fisher)

    fisher = FisherMatrix(
        wf_params,
        'total_mass',
        wf_generator=phenomx_generator,
        return_info=False,  # Provoke call in __getattr__
        direct_computation=False,
    )
    fisher._deriv_info


def test_repr():
    print(fisher_tot_mass)


def test_project():
    test_params = ['total_mass', 'mass_ratio', 'time', 'phase', 'distance']

    fisher = FisherMatrix(
        wf_params,
        test_params,
        wf_generator=phenomx_generator,
    )

    project_params = ['time']
    fisher_projected = fisher.project_fisher(project_params)

    assert fisher_projected.shape == 2 * (len(test_params) - len(project_params),)
    assert np.all(
        np.equal(
            fisher_projected.params_to_vary,
            ['total_mass', 'mass_ratio', 'phase', 'distance'],
        )
    )


@pytest.mark.parametrize('params', [None, 'total_mass', ['total_mass']])
def test_stat_bias(params):
    noise = TimeSeries(
        np.random.normal(scale=0.1, size=1024), sample_rate=1024, unit=u.strain
    )
    fisher_tot_mass.statistical_bias(noise=noise, params=params)

    noise_fd = td_to_fd(noise)
    fisher_tot_mass.statistical_bias(noise=noise_fd, params=params)


@pytest.mark.parametrize('params', [None, 'total_mass', ['total_mass']])
def test_std_dev(params):
    fisher_tot_mass.standard_deviation(params)


@pytest.mark.parametrize(
    'params', [None, 'total_mass', ['total_mass', 'time', 'phase']]
)
def test_sys_bias(params):
    fisher = FisherMatrix(
        wf_params,
        ['total_mass', 'time', 'phase'],
        wf_generator=phenomx_generator,
    )

    fisher.systematic_bias(
        phenomd_generator, 'total_mass', optimize=False, return_diagnostics=True
    )

    fisher.systematic_bias(phenomd_generator, params, return_diagnostics='deriv_info')

    fisher.systematic_bias(
        phenomd_generator, optimize=True, return_diagnostics=True, is_true_point=True
    )

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
        phenomd_generator,
        optimize=False,
        optimize_fisher='phase',
        return_diagnostics=True,
    )

    fisher = fisher.update_attrs(deriv_routine='numdifftools')

    fisher.systematic_bias(phenomd_generator, optimize='phase')


def test_return_diagnostic():
    fisher = FisherMatrix(
        wf_params,
        ['total_mass', 'time', 'phase'],
        wf_generator=phenomx_generator,
    )

    for bool1 in [True, False]:
        for bool2 in [True, False]:
            fisher.systematic_bias(
                phenomd_generator,
                return_diagnostics=bool1,
                is_true_point=bool2,
            )


@pytest.mark.parametrize(
    'inner_prod_kwargs',
    [
        {},
        dict(f_range=[20.0 * u.Hz, 42.0 * u.Hz]),
        dict(df=2**-2, min_dt_prec=1e-5 * u.s),
    ],
)
def test_snr(inner_prod_kwargs):
    snr = norm(phenomx_generator(wf_params), **inner_prod_kwargs)
    assert snr == fisher_tot_mass.snr(**inner_prod_kwargs)


def test_plot():
    MatrixWithUnits.plot(fisher_tot_mass.fisher)
    plt.close()

    fisher_tot_mass.plot_matrix(fisher_tot_mass.fisher)
    plt.close()

    fisher_tot_mass.plot_matrix(fisher_tot_mass.fisher, xticks=False, yticks=False)
    plt.close()

    fisher_tot_mass.plot()
    plt.close()

    fisher_tot_mass.plot(only_fisher=True)
    plt.close()

    fisher_tot_mass.plot(only_fisher_inverse=True)
    plt.close()


def test_get_wf_generator():
    fisher_tot_mass.get_wf_generator('IMRPhenomXPHM')


@pytest.mark.parametrize(
    'new_point', [None, wf_params | {'total_mass': 42.0 * u.solMass}]
)
@pytest.mark.parametrize(
    'new_params_to_vary', [None, 'mass_ratio', ['mass_ratio', 'distance']]
)
@pytest.mark.parametrize('new_wf_generator', [None, phenomx_cross_generator])
@pytest.mark.parametrize(
    'new_metadata',
    [None, {'deriv_routine': 'gw_signal_tools', 'convergence_check': 'mismatch'}],
)
def test_update_attrs(new_point, new_params_to_vary, new_wf_generator, new_metadata):
    if new_metadata is None:
        new_metadata = {}  # Because ** is used below

    fisher_tot_mass_v2 = fisher_tot_mass.update_attrs(
        new_point, new_params_to_vary, new_wf_generator, **new_metadata
    )

    if new_point is not None:
        assert fisher_tot_mass_v2.point == new_point
    if new_params_to_vary is not None:
        if isinstance(new_params_to_vary, str):
            assert fisher_tot_mass_v2.params_to_vary == [new_params_to_vary]
        else:
            assert fisher_tot_mass_v2.params_to_vary == new_params_to_vary
    if new_wf_generator is not None:
        assert fisher_tot_mass_v2.wf_generator == new_wf_generator
    assert fisher_tot_mass_v2.metadata == (fisher_tot_mass.metadata | new_metadata)


def test_copy():
    fisher_copy = fisher_tot_mass.copy()

    fisher_copy._fisher = None
    fisher_copy._fisher_inverse = None
    fisher_copy.point = None
    fisher_copy.wf_generator = None
    fisher_copy.metadata = None
    fisher_copy._deriv_info = None
    fisher_copy._is_projected = True

    for attr in [
        'fisher',
        'fisher_inverse',
        'point',
        'wf_generator',
        'metadata',
        'deriv_info',
    ]:
        assert fisher_tot_mass.__getattribute__(attr) is not None

    assert not fisher_tot_mass.is_projected


# %% -- Confirm that certain errors are raised --------------------------------
def test_immutable():
    # -- Setting Fisher matrix related attributes should throw error
    with pytest.raises(AttributeError):
        fisher_tot_mass.fisher = 42

    with pytest.raises(AttributeError):
        fisher_tot_mass.fisher_inverse = 42
