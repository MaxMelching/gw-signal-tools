# -- Make __version__ attribute available, not created automatically
try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = ''


__doc__ = """
Repository with files surrounding computations with waveforms from lal.
"""

# -- Make certain paths available for easy access
from os.path import dirname as _path_dirname, join as _path_join

PACKAGE_PATH: str = _path_dirname(__file__)
PLOT_STYLE_SHEET: str = _path_join(PACKAGE_PATH, 'plot_stylesheet.sty')

# -- Set preferred unit system here (can be changed)
from .units import *  # noqa: F401

# -- Initialize Logging
from .logging import *  # noqa: F401

# -- Initialize Caching
from .caching import *  # noqa: F401

# -- Dictionary to get nicer display of parameters
from .plotting import *  # noqa: F401
