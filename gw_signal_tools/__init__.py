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

PACKAGE_PATH = _path_dirname(__file__)

from os.path import join as _path_join

PLOT_STYLE_SHEET = _path_join(PACKAGE_PATH, 'plot_stylesheet.sty')

# ---------- Set preferred unit system here (can be changed) ----------
import gw_signal_tools.units as _gw_signal_tools_units
preferred_unit_system = _gw_signal_tools_units

# ---------- Initialize Logging ----------
import logging as _log

logger = _log.getLogger(__name__)

logger.propagate = False  # Otherwise root also prints them
logger.setLevel(_log.INFO)

formatter = _log.Formatter(
    fmt='%(asctime)s  %(levelname)s (%(filename)s: %(lineno)d): %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S'
)

from sys import stderr as _stderr
ch = _log.StreamHandler(stream=_stderr)  # More explicit
ch.setLevel(_log.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)
