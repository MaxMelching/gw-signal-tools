# NOTE: the code in this file is essentially copy and pasted from
# https://github.com/astropy/astropy/blob/main/astropy/units/si.py,
# https://github.com/astropy/astropy/blob/main/astropy/units/astrophys.py,
# https://github.com/astropy/astropy/blob/main/astropy/units/cgs.py
# (parts of each file are here). In particular, virtually all credit
# belongs to the astropy developers

import astropy.units as u
from astropy.units.core import Unit, UnitBase, def_unit
from astropy.constants import si as _const_si
import astropy.units.si as _si
import astropy.units.astrophys as _astrophys


__all__: list[str] = []  #  Units are added at the end

_ns = globals()

# u.astrophys uses pc, we would like to use Mpc, thus redefine here
def_unit(
    ['Mpc'],
    _const_si.pc * 1e6,
    namespace=_ns,
    prefixes=False,
    doc='megaparsec: approximately 3.26 * 10^6 light-years.',
)

# # Define strain. We use same one as GWPy to make sure they show same
# # behaviour whether GWPy is loaded or not
# def_unit(
#     ['strain'],
#     # Unit(1),
#     namespace=_ns,
#     # prefixes=False,
#     # doc='strain: relative amplitude common in GW astronomy'
# )


# Mpc = astrophys.Mpc  # Does not produce desired results because base unit is still pc
s = _si.s
Msun = _astrophys.Msun
rad = _si.rad


# All that is left to do is defining a basis that can be converted to

# bases = {Mpc, s, Msun, rad, strain}  # Minimal version. Might cause issues with certain other quantities, though, so just copy other units as well?

A = _si.A
cd = _si.cd
K = _si.K
mol = _si.mol

bases = {Mpc, s, Msun, A, cd, rad, K, mol}


# V2 -> does not produce same result as copying like above, thus do not use
# from astropy.units.si import bases as si_bases

# bases = si_bases.copy()

# m = _si.m
# kg = _si.kg

# bases.remove(m)
# bases.remove(kg)

# bases.add(Mpc)

# Msun = astrophys.Msun
# bases.add(Msun)


import gwpy.detector.units  # Adds strain to astropy units -> custom adding would cause problems when GWPy and this package are loaded
strain = u.Unit('strain')

# try:
#     # If GWPy was loaded, strain already exists as a unit
#     strain = u.Unit('strain')
# except (AttributeError, ValueError):
#     # Define strain. We use same one as GWPy to make sure they show same
#     # behaviour whether GWPy is loaded or not
#     def_unit(
#         ['strain'],
#         # Unit(1),
#         namespace=_ns,
#         # prefixes=False,
#         # doc='strain: relative amplitude common in GW astronomy'
#     )

#     # Add definition from this file to make strain available. Can be accessed
#     # using u.Unit('strain') or cosmo.strain
#     u.add_enabled_units(strain)


__all__ += [n for n, v in _ns.items() if isinstance(v, UnitBase)]

if __doc__ is not None:
    # This generates a docstring for this module that describes all of the
    # standard units defined here.
    from astropy.utils import generate_unit_summary as _generate_unit_summary

    __doc__ += _generate_unit_summary(globals())
