# -- NOTE: the code in this file is essentially copy and pasted from
# -- https://github.com/astropy/astropy/blob/main/astropy/units/si.py,
# -- https://github.com/astropy/astropy/blob/main/astropy/units/astrophys.py,
# -- https://github.com/astropy/astropy/blob/main/astropy/units/cgs.py
# -- (parts of each file are here). In particular, virtually all credit
# -- belongs to the astropy developers

# -- Third Party Imports
import astropy.units as u
from astropy.units.core import def_unit
from astropy.constants import si as _const_si
import astropy.units.si as _si
import astropy.units.astrophys as _astrophys


__doc__: str = """
Custom unit module for the gw-signal-tools package. In here, we define
certain base units that are used most commonly for our calculations.

Moreover, we enable the equivalency between `u.rad` and no unit, i.e. we
run `u.set_enabled_equivalencies(u.dimensionless_angles())`. This is
done because of inconsistencies between numerical and analytical
derivatives with respect to angles that would arise otherwise (also to
enable exponentiation with angles).
"""


__all__: list[str] = []  # Units are added at the end

_ns = globals()

def_unit(
    ['Mpc'],
    _const_si.pc * 1e6,
    namespace=_ns,
    prefixes=True,
    doc='megaparsec: approximately 3.26 * 10^6 light-years.',
)

def_unit(
    ['kpc'],
    _const_si.pc * 1e3,
    namespace=_ns,
    prefixes=True,
    doc='kiloparsec: approximately 3.26 * 10^3 light-years.',
)

def_unit(
    ['Gpc'],
    _const_si.pc * 1e9,
    namespace=_ns,
    prefixes=True,
    doc='gigaparsec: approximately 3.26 * 10^9 light-years.',
)


# -- All that is left to do is defining a basis that can be converted to
s = _si.s
Msun = _astrophys.Msun
rad = _si.rad
pc = _astrophys.pc

# -- Add other units that we won't use, but for completeness (copied from u.si)
A = _si.A
cd = _si.cd
K = _si.K
mol = _si.mol


# -- Handle strain unit. Is a little complicated, e.g. because custom
# -- adding would cause problems when GWPy and this package are loaded
import gwpy.detector.units  # noqa: F401 - Adds strain to astropy units

u.get_current_unit_registry()._registry.pop('strain', None)
# -- Potential problem is that strain was equivalent to astropy unit
# -- dimensionless_unscaled in some versions. Avoid by custom definition.
u.def_unit('strain', namespace=_ns)
u.add_enabled_units(_ns)

strain = u.strain = u.Unit('strain')  # For convenient access
strain.si = strain


bases = {pc, s, Msun, A, cd, rad, K, mol, strain}


u.set_enabled_equivalencies(u.dimensionless_angles())


__all__ += [n for n, v in _ns.items() if isinstance(v, u.UnitBase)]


# -- This generates a docstring for this module that describes all of
# -- the standard units defined here. Also copied from astropy.units.si.
from astropy.units.utils import generate_unit_summary as _generate_unit_summary

__doc__ += '\n\n'
__doc__ += _generate_unit_summary(globals())


# -- Cleanup
del u
del def_unit
del _si
del _const_si
del _astrophys
del gwpy.detector.units
