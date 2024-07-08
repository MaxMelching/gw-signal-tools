# ----- Third Party Imports -----
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pytest

# ----- Local Package Imports -----
from gw_signal_tools.waveform_utils import get_wf_generator
from gw_signal_tools.types import MatrixWithUnits
from gw_signal_tools.fisher import (
    fisher_matrix, FisherMatrix
)
from gw_signal_tools.test_utils import (
    assert_allclose_MatrixWithUnits, assert_allequal_MatrixWithUnits
)

from gw_signal_tools import PLOT_STYLE_SHEET
plt.style.use(PLOT_STYLE_SHEET)


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
    'condition': 0
}

approximant = 'IMRPhenomXPHM'
phenomx_generator = get_wf_generator(approximant)
phenomx_cross_generator = get_wf_generator(approximant, mode='cross')

# Make sure mass1 and mass2 are not in default_dict (makes messy behaviour)
import lalsimulation.gwsignal.core.parameter_conventions as pc
pc.default_dict.pop('mass1', None);
pc.default_dict.pop('mass2', None);

fisher_tot_mass = FisherMatrix(
    wf_params,
    'total_mass',
    wf_generator=phenomx_generator,
    return_info=True
)


#%% ----- Simple consistency tests -----
def test_unit():
    # All ways of accessing must work
    assert fisher_tot_mass.fisher[0, 0].unit == 1/u.solMass**2
    assert fisher_tot_mass.fisher.unit[0, 0] == 1/u.solMass**2
    assert fisher_tot_mass.unit[0, 0] == 1/u.solMass**2

def test_inverse():
    assert np.all(np.equal(np.linalg.inv(fisher_tot_mass.fisher.value),
                           fisher_tot_mass.fisher_inverse.value))
    
    assert np.all(np.equal(fisher_tot_mass.fisher.unit**-1,
                           fisher_tot_mass.fisher_inverse.unit))
    
def test_fisher_calc():
    fisher_tot_mass_2 = fisher_matrix(wf_params, 'total_mass',
                                      phenomx_generator)
    assert fisher_tot_mass.fisher == fisher_tot_mass_2

def test_criterion_consistency():
    fisher_tot_mass_2 = fisher_matrix(wf_params, 'total_mass',
                                      phenomx_generator,
                                      convergence_check='mismatch')
    
    assert_allclose_MatrixWithUnits(fisher_tot_mass.fisher, fisher_tot_mass_2,
                                    atol=0.0, rtol=5e-5)

def test_time_and_phase_shift_consistency():
    # Idea: time and phase shift in the waveform should not have effect
    # on Fisher matrix entries, cancel out in inner product
    t_shift = 1e-3 * u.s
    phase_shift = 1e-2 * u.rad

    def shifted_wf_gen(wf_params):
        wf = phenomx_generator(wf_params)
        return wf * np.exp(-2.j*np.pi*wf.frequencies*t_shift + 1.j*phase_shift)
    
    calc_params = ['total_mass', 'mass_ratio', 'distance', 'time', 'phase']

    fisher_v1 = FisherMatrix(
        wf_params,
        calc_params,
        phenomx_generator
    )

    fisher_v2 = FisherMatrix(
        wf_params,
        calc_params,
        shifted_wf_gen
    )

    assert_allclose_MatrixWithUnits(fisher_v1.fisher, fisher_v2.fisher,
                                    atol=3.1e-50, rtol=0.)
    # This is great accuracy. This deviation is for two uncorrelated
    # parameters, where the value itself is ten (!) orders of magnitude
    # below all diagonal values (which means it is of the order of
    # 1e-50). This deviation can certainly be tolerated.


#%% ----- Feature tests -----
def test_get_indices():
    test_params = ['total_mass', 'time', 'phase']

    fisher = FisherMatrix(
        wf_params,
        test_params,
        wf_generator=phenomx_generator,
    )

    indices_1 = fisher.get_param_indices(['time', 'phase'])
    indices_2 = fisher.get_param_indices(['phase', 'time'])

    assert np.all(indices_1 == np.array([1, 2]))
    assert np.all(indices_2 == np.array([2, 1]))


    grid_1 = fisher.get_sub_matrix_indices(['time', 'phase'])
    sub_matr_1 = [[fisher.fisher[1, 1], fisher.fisher[1, 2]],
                  [fisher.fisher[2, 1], fisher.fisher[2, 2]]]

    grid_2 = fisher.get_sub_matrix_indices(['phase', 'time'])
    sub_matr_2 = [[fisher.fisher[2, 2], fisher.fisher[2, 1]],
                  [fisher.fisher[1, 2], fisher.fisher[1, 1]]]


    # assert np.all(fisher.fisher[grid_1] == np.array(sub_matr_1))
    # assert np.all(fisher.fisher[grid_2] == np.array(sub_matr_2))
    for index in np.ndindex((2, 2)):
        i, j = index
        assert fisher.fisher[grid_1][i, j] == sub_matr_1[i][j]
        assert fisher.fisher[grid_2][i, j] == sub_matr_2[i][j]

# Fancy version does not work, unfortunately
# @pytest.mark.parametrize('attr', ['project_fisher(\'total_mass\')', 'cond()',
#                                   'inv(fisher_tot_mass.fisher)'])
# def test_getattr(attr):
#     fisher = FisherMatrix(
#         wf_params,
#         'total_mass',
#         wf_generator=phenomx_generator,
#         return_info=True,
#         direct_computation=False
#     )
#     fisher.__getattribute__(attr)
def test_getattr():
    fisher = FisherMatrix(
        wf_params,
        'total_mass',
        wf_generator=phenomx_generator,
        return_info=True,
        direct_computation=False
    )
    fisher.project_fisher('total_mass')

    fisher = FisherMatrix(
        wf_params,
        'total_mass',
        wf_generator=phenomx_generator,
        return_info=True,
        direct_computation=False
    )
    fisher.cond()

    fisher = FisherMatrix(
        wf_params,
        'total_mass',
        wf_generator=phenomx_generator,
        return_info=True,
        direct_computation=False
    )
    fisher.inv(fisher_tot_mass.fisher)

def test_project():
    test_params = ['total_mass', 'mass_ratio', 'time', 'phase', 'distance']

    fisher = FisherMatrix(
        wf_params,
        test_params,
        wf_generator=phenomx_generator,
    )

    project_params = ['time']
    fisher_projected = fisher.project_fisher(project_params)

    assert fisher_projected.shape == 2*(len(test_params)-len(project_params), )
    assert np.all(np.equal(fisher_projected.params_to_vary,
                           ['total_mass', 'mass_ratio', 'phase', 'distance']))

@pytest.mark.parametrize('params', [None, 'total_mass', ['total_mass']])
def test_stat_error(params):
    fisher_tot_mass.statistical_error(params)

@pytest.mark.parametrize('params', [None, 'total_mass', ['total_mass', 'time', 'phase']])
def test_sys_error(params):
    phenomd_generator = FisherMatrix.get_wf_generator('IMRPhenomD')
    fisher = FisherMatrix(
        wf_params,
        ['total_mass', 'time', 'phase'],
        wf_generator=phenomx_generator,
        return_info=True
    )

    fisher.systematic_error(phenomd_generator, 'total_mass', optimize=False)

    fisher.systematic_error(phenomd_generator, params)
    
    fisher.systematic_error(phenomd_generator, optimize=True)
    
    fisher.systematic_error(phenomd_generator,
                            optimize=['time', 'phase'])
    
    fisher.systematic_error(phenomd_generator,
                            optimize_fisher=['time', 'phase'])
    
    fisher.systematic_error(phenomd_generator, optimize=True,
                            optimize_fisher=['time', 'phase'])

def test_plot():
    fisher = FisherMatrix(
        wf_params,
        ['total_mass', 'mass_ratio', 'distance'],
        wf_generator=phenomx_generator,
        return_info=False
    )
    plt.close()

    print(fisher)  # For verification

    MatrixWithUnits.plot(fisher.fisher)
    # plt.show()
    plt.close()

    fisher.plot()
    # plt.show()
    plt.close()

    fisher.plot(only_fisher=True)
    # plt.show()
    plt.close()

    fisher.plot(only_fisher_inverse=True)
    # plt.show()
    plt.close()

def test_cond():
    assert fisher_tot_mass.cond() == fisher_tot_mass.fisher.cond()

def test_array():
    assert np.all(np.array(fisher_tot_mass) == np.array(fisher_tot_mass.fisher))

# @pytest.mark.skip  # Note finished yet
@pytest.mark.parametrize('new_wf_params_at_point', [None, wf_params | {'total_mass': 42.*u.solMass}])
@pytest.mark.parametrize('new_params_to_vary', [None, ['mass_ratio', 'distance']])
@pytest.mark.parametrize('new_wf_generator', [None, phenomx_cross_generator])
@pytest.mark.parametrize('new_metadata', [None, {'return_info': False, 'convergence_check': 'mismatch'}])
def test_update_attrs(new_wf_params_at_point, new_params_to_vary,
                      new_wf_generator, new_metadata):
    if new_metadata is None:
        new_metadata = {}

    fisher_tot_mass_v2 = fisher_tot_mass.update_attrs(
        new_wf_params_at_point,
        new_params_to_vary,
        new_wf_generator,
        **new_metadata
    )

@pytest.mark.skip  # Note finished yet
def test_copy():
    ...

#%% Confirm that certain errors are raised
def test_immutable():
    # Setting Fisher matrix related attributes should throw error
    with pytest.raises(AttributeError):
        fisher_tot_mass.fisher = 42

    with pytest.raises(AttributeError):
        fisher_tot_mass.fisher_inverse = 42
