# ----- Third party imports -----
import pytest
import astropy.units as u

# ----- Local package imports -----
import gw_signal_tools.units as gw_signal_tools_units
# Do NOT import preferred_unit_system here, want to test the one from package


# Could also test with units=[u.astrophys, gw_signal_tools_units]
@pytest.mark.parametrize('unit', [u.pc, u.s, u.Msun, u.A, u.cd, u.rad, u.K, u.mol])
def test_pure_base_units(unit):
    assert unit.to_system(gw_signal_tools_units)[0] == unit
    assert unit.compose(units=gw_signal_tools_units)[0] == unit

def test_composite_units():
    test_unit = u.Hz**2 * u.s

    test_unit_converted1 = test_unit.to_system(gw_signal_tools_units)[0]
    test_unit_converted2 = test_unit.compose(units=gw_signal_tools_units)[0]

    assert test_unit_converted1 == test_unit_converted2
    assert test_unit_converted1 == (1/u.s)


    test_unit = u.pc.si * u.Msun.si

    test_unit_converted1 = test_unit.to_system(gw_signal_tools_units)[0]
    test_unit_converted2 = test_unit.compose(units=gw_signal_tools_units)[0]

    assert test_unit_converted1 == test_unit_converted2
    assert test_unit_converted1 == (u.pc * u.Msun)

    # With prefactor
    test_unit = u.Unit(u.Mpc.si * u.Msun.si)

    test_unit_converted1 = test_unit.to_system(gw_signal_tools_units)[0]
    test_unit_converted2 = test_unit.compose(units=gw_signal_tools_units)[0]

    assert test_unit_converted1 == test_unit_converted1
    assert test_unit_converted1 == (u.Mpc * u.Msun)


    test_unit = u.Unit(4e6 * u.pc.si * u.Msun.si)

    test_unit_converted1 = test_unit.to_system(gw_signal_tools_units)[0]
    test_unit_converted2 = test_unit.compose(units=gw_signal_tools_units)[0]

    assert test_unit_converted1 == test_unit_converted1
    assert test_unit_converted1 == (4 * u.Mpc * u.Msun)


    test_unit = u.Unit(42e9 * u.pc.si * u.Msun.si)

    test_unit_converted1 = test_unit.to_system(gw_signal_tools_units)[0]
    test_unit_converted2 = test_unit.compose(units=gw_signal_tools_units)[0]

    assert test_unit_converted1 == test_unit_converted1
    assert test_unit_converted1 == (42 * u.Gpc * u.Msun)

def test_strain_definition():
    strain1 = u.Unit('strain')

    strain2 = gw_signal_tools_units.strain

    assert strain1 == strain2

    assert strain1 != u.dimensionless_unscaled

    assert strain2.compose(units=gw_signal_tools_units)[0] != u.dimensionless_unscaled

    assert u.dimensionless_unscaled.compose(units=gw_signal_tools_units)[0] == u.dimensionless_unscaled

def test_docstring():
    assert gw_signal_tools_units.__doc__ is not None
