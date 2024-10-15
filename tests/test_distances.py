# -- Standard Lib Imports
import unittest

# -- Third Party Imports
import matplotlib.pyplot as plt
import astropy.units as u
from gwpy.testing.utils import assert_quantity_equal
import pytest

# -- Local Package Imports
from gw_signal_tools.waveform import get_wf_generator
from gw_signal_tools.fisher import (
    distance, linearized_distance
)
from gw_signal_tools.test_utils import (
    assert_allclose_quantity, assert_allclose_series
)
from gw_signal_tools.types import HashableDict
from gw_signal_tools import PLOT_STYLE_SHEET
plt.style.use(PLOT_STYLE_SHEET)


#%% -- Initializing commonly used variables -----------------------------------
f_min = 20.*u.Hz
f_max = 1024.*u.Hz

wf_params = HashableDict({
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
})

wf_gen = get_wf_generator('IMRPhenomXPHM', cache=True)

# -- Make sure mass1 and mass2 are not in default_dict
import lalsimulation.gwsignal.core.parameter_conventions as pc
pc.default_dict.pop('mass1', None);
pc.default_dict.pop('mass2', None);


@pytest.mark.parametrize('param_to_vary', ['total_mass', 'mass_ratio', 'distance'])
@pytest.mark.parametrize('optimize', [False, True])
@pytest.mark.parametrize('dist_kind', ['diff_norm', 'mismatch_norm'])
def test_distance(param_to_vary, optimize, dist_kind):
    center_val = wf_params[param_to_vary]
    param_range = u.Quantity([0.9*center_val, 1.1*center_val])
    step_size = 5e-2*center_val

    dist = distance(
        param_to_vary=param_to_vary,
        param_vals=param_range,
        wf_params=wf_params,
        param_step_size=step_size,
        distance_kind=dist_kind,
        wf_generator=wf_gen,
        optimize_time_and_phase=optimize
    )

    assert_allclose_quantity(dist.dx, step_size, atol=0.0, rtol=1e-15)
    assert_allclose_quantity(u.Quantity([dist.xindex[0], dist.xindex[-1]]),
                             param_range, atol=step_size.value, rtol=0.0)
    
    dist_no_step_size = distance(
        param_to_vary=param_to_vary,
        param_vals=param_range,
        wf_params=wf_params,
        distance_kind=dist_kind,
        wf_generator=wf_gen,
        optimize_time_and_phase=optimize
    )

    assert_quantity_equal(param_range, dist_no_step_size.xindex)


@pytest.mark.parametrize('param_to_vary', ['total_mass', 'mass_ratio', 'distance'])
def test_linearized_distance(param_to_vary):
    center_val = wf_params[param_to_vary]
    param_range = u.Quantity([0.9*center_val, 1.1*center_val])
    step_size = 5e-2*center_val

    dist = linearized_distance(
        param_to_vary=param_to_vary,
        param_vals=param_range,
        wf_params=wf_params,
        param_step_size=step_size,
        wf_generator=wf_gen
    )

    assert_allclose_quantity(dist.dx, step_size, atol=0.0, rtol=1e-15)
    assert_allclose_quantity(u.Quantity([dist.xindex[0], dist.xindex[-1]]),
                             param_range, atol=step_size.value, rtol=0.0)


@pytest.mark.parametrize('param_to_vary', ['total_mass', 'mass_ratio', 'distance'])
def test_projected_linearized_distance(param_to_vary):
    center_val = wf_params[param_to_vary]
    param_range = u.Quantity([0.9*center_val, 1.1*center_val])
    step_size = 5e-2*center_val

    dist1 = linearized_distance(
        param_to_vary=[param_to_vary, 'time', 'phase'],
        param_vals=param_range,
        wf_params=wf_params,
        param_step_size=step_size,
        params_to_project=['time', 'phase'],
        wf_generator=wf_gen
    )

    assert_allclose_quantity(dist1.dx, step_size, atol=0.0, rtol=1e-15)
    assert_allclose_quantity(u.Quantity([dist1.xindex[0], dist1.xindex[-1]]),
                             param_range, atol=step_size.value, rtol=0.0)

    dist2 = linearized_distance(
        param_to_vary=[param_to_vary],  # Testing equivalent input
        param_vals=param_range,
        wf_params=wf_params,
        param_step_size=step_size,
        params_to_project=['time', 'phase'],
        wf_generator=wf_gen
    )

    assert_allclose_quantity(dist2.dx, step_size, atol=0.0, rtol=1e-15)
    assert_allclose_quantity(u.Quantity([dist2.xindex[0], dist2.xindex[-1]]),
                             param_range, atol=step_size.value, rtol=0.0)

    assert_allclose_series(dist1, dist2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize('params_to_project', [['time', 'phase'], 'time'])
def test_params_to_project(params_to_project):
    param_to_vary = 'total_mass'
    center_val = wf_params[param_to_vary]
    param_range = u.Quantity([0.9*center_val, 1.1*center_val])
    step_size = 5e-2*center_val

    params_to_vary = [param_to_vary]
    if isinstance(params_to_project, str):
        params_to_vary.append(params_to_project)
    else:
        params_to_vary += params_to_project

    dist1 = linearized_distance(
        param_to_vary=params_to_vary,
        param_vals=param_range,
        wf_params=wf_params,
        params_to_project=params_to_project,
        wf_generator=wf_gen
    )

    dist2 = linearized_distance(
        param_to_vary=params_to_vary,
        param_vals=param_range,
        wf_params=wf_params,
        params_to_project=params_to_project,
        wf_generator=wf_gen
    )

    assert_allclose_series(dist1, dist2, atol=0.0, rtol=0.0)


#%% -- Confirm that certain errors are raised ---------------------------------
class ErrorRaising(unittest.TestCase):
    param_to_vary = 'total_mass'
    center_val = wf_params[param_to_vary]
    param_range = u.Quantity([0.9*center_val, 1.1*center_val])
    step_size = 5e-2*center_val

    def test_invalid_dist_kind(self):
        with self.assertRaises(ValueError):
            distance(
                param_to_vary=self.param_to_vary,
                param_vals=self.param_range,
                wf_params=wf_params,
                param_step_size=self.step_size,
                distance_kind='',
                wf_generator=wf_gen
            )

    def test_invalid_center_val(self):
        wrong_param_range = u.Quantity([0.5*self.center_val,
                                        0.9*self.center_val])
        
        with self.assertRaises(AssertionError):
            distance(
                param_to_vary=self.param_to_vary,
                param_vals=wrong_param_range,
                wf_params=wf_params,
                param_step_size=self.step_size,
                distance_kind='diff_norm',
                wf_generator=wf_gen
            )
        
        with self.assertRaises(AssertionError):
            linearized_distance(
                param_to_vary=self.param_to_vary,
                param_vals=wrong_param_range,
                wf_params=wf_params,
                param_step_size=self.step_size,
                wf_generator=wf_gen
            )

    def test_invalid_param_to_vary(self):
        with self.assertRaises(ValueError):
            linearized_distance(
                param_to_vary=[self.param_to_vary, self.param_to_vary],
                param_vals=self.param_range,
                wf_params=wf_params,
                param_step_size=self.step_size,
                distance_kind='diff_norm',
                wf_generator=wf_gen
            )

        with self.assertRaises(ValueError):
            linearized_distance(
                param_to_vary=[self.param_to_vary, self.param_to_vary,
                               'time', 'phase'],
                param_vals=self.param_range,
                wf_params=wf_params,
                param_step_size=self.step_size,
                distance_kind='diff_norm',
                params_to_project=['time', 'phase'],
                wf_generator=wf_gen
            )
    
    def test_invalid_input_unit(self):
        with self.assertRaises(u.UnitConversionError):
            distance(
                param_to_vary=self.param_to_vary,
                param_vals=self.param_range*u.s,
                wf_params=wf_params,
                param_step_size=self.step_size,
                distance_kind='',
                wf_generator=wf_gen
            )

        with self.assertRaises(u.UnitConversionError):
            distance(
                param_to_vary=self.param_to_vary,
                param_vals=self.param_range,
                wf_params=wf_params,
                param_step_size=self.step_size*u.s,
                distance_kind='',
                wf_generator=wf_gen
            )
