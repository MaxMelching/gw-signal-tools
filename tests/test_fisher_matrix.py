import numpy as np

from gw_signal_tools.fisher_matrix import num_diff, fisher_val_at_point


def test_num_diff():
    step_size = 0.01
    x_vals = np.arange(0.0, 2.0, step=step_size)

    derivative_vals = num_diff(x_vals, h=step_size)

    assert np.all(np.isclose(derivative_vals, np.ones(derivative_vals.size), atol=0.0, rtol=0.01))

    func_vals = 0.5 * x_vals**2
    derivative_vals = num_diff(func_vals, h=step_size)

    assert np.all(np.isclose(derivative_vals[2 : -2], x_vals[2 : -2], atol=0.0, rtol=0.01))
    assert np.all(np.isclose(derivative_vals[1:2], x_vals[1:2], atol=0.0, rtol=0.01))  # First correct value is zero, thus relative deviation is always 1
    assert np.all(np.isclose(derivative_vals[-2:], x_vals[-2:], atol=0.0, rtol=0.01))
    # NOTE: for values at border of interval, rule is not applicable.
    # Thus we make separate checks, methods could be less accurate there

    func_vals = np.sin(x_vals)

    derivative_vals = num_diff(func_vals, h=step_size)

    assert np.all(np.isclose(derivative_vals[2 : -2], np.cos(x_vals)[2 : -2], atol=0.0, rtol=0.01))
    assert np.all(np.isclose(derivative_vals[:2], np.cos(x_vals)[:2], atol=0.0, rtol=0.01))
    assert np.all(np.isclose(derivative_vals[-2:], np.cos(x_vals)[-2:], atol=0.0, rtol=0.02))