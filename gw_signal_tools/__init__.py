# Make attribute available, not created automatically -> now handled below
try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = ''

__doc__ = """
Repository with files surrounding computations with waveforms from lal.
"""

# ---------- Make certain paths available for easy access ----------
from os.path import dirname as _path_dirname

PACKAGE_PATH: str = _path_dirname(__file__)

from os.path import join as _path_join

PLOT_STYLE_SHEET: str = _path_join(PACKAGE_PATH, 'plot_stylesheet.sty')

# ---------- Set preferred unit system here (can be changed) ----------
from .units import preferred_unit_system  # noqa: F401

# ---------- Initialize Logging ----------
from .logging import logger  # noqa: F401

# ---------- Initialize Caching ----------
from .caching import use_caching, cache_func, disable_caching, enable_caching  # noqa: F401

# ---------- Dictionary to get nicer display of parameters ----------
from .plotting import latexparams  # noqa: F401
