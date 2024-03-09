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
# def_unit(
#     ['Mpc'],
#     _const_si.pc * 1e6,
#     namespace=_ns,
#     prefixes=True,
#     doc='megaparsec: approximately 3.26 * 10^6 light-years.',
# )


# All that is left to do is defining a basis that can be converted to
s = _si.s
Msun = _astrophys.Msun
rad = _si.rad
pc = _astrophys.pc

# Add other units that we won't use, but for completeness (copied from u.si)
A = _si.A
cd = _si.cd
K = _si.K
mol = _si.mol

bases = {pc, s, Msun, A, cd, rad, K, mol}


import gwpy.detector.units  # Adds strain to astropy units
# -> custom adding would cause problems when GWPy and this package are loaded
strain = u.Unit('strain')


__all__ += [n for n, v in _ns.items() if isinstance(v, UnitBase)]

if __doc__ is not None:
    # This generates a docstring for this module that describes all of the
    # standard units defined here.
    from astropy.utils import generate_unit_summary as _generate_unit_summary

    __doc__ += _generate_unit_summary(globals())
