# Make attribute available, not created automatically -> now handled below
try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = ''


from os.path import dirname as _path_dirname

PACKAGE_PATH = _path_dirname(__file__)

from os.path import join as _path_join

PLOT_STYLE_SHEET = _path_join(PACKAGE_PATH, 'plot_stylesheet.sty')



# TODO: decide if functions shall be imported here. Otherwise one has to
# import from each module (preferred solution at the moment)


# TODO: decide if equivalency between strain and dimensionless can be enabled
# here by setting some variable to true or so


# TODO: set global preferred_unit_system here (default to our custom) and use this in inner_product
# conversions etc., so that it can be changed to SI from astropy for example
# -> maybe leave possible conversion in Fisher, but default
# is global one given by gw_signal_tools
import gw_signal_tools.units as _gw_signal_tools_units
preferred_unit_system = _gw_signal_tools_units


# ---------- Initialize Logging ----------
import logging as _log

# logger = _log.getLogger()
# _log.basicConfig(
#     level=_log.INFO,
#     # level=_log.DEBUG,
#     format='%(asctime)s  %(levelname)s (%(filename)s: %(lineno)d): %(message)s',
#     datefmt='%Y-%m-%d  %H:%M:%S'
# )
# logger = _log.getLogger()

# Neither of this works in all cases

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


# _log.captureWarnings(True)  # Testing
# warnings_logger = _log.getLogger('py.warnings')
# warnings_logger.propagate = False
# # warnings_logger.addHandler(logger)

# # import warnings
# # warnings.formatwarning()


# warning_formatter = _log.Formatter(
#     fmt='%(message)s',
#     datefmt='%Y-%m-%d  %H:%M:%S'
# )

# warning_ch = _log.StreamHandler(stream=_stderr)  # More explicit
# # warning_ch.setLevel(_log.INFO)
# warning_ch.setFormatter(warning_formatter)

# warnings_logger.addHandler(warning_ch)


# # Could set logger with output to file here (can have different level, this
# # may be desirable at some point). See https://docs.python.org/3/howto/logging-cookbook.html

# # TODO: handle errors via logging? On the other hand, we log to command line anyway...

# # TODO: make function set_logger, where custom logger can be specified?
# # Would be called as gw_signal_tools.set_logger(), i.e. it has to live in this file

# # _log.captureWarnings(True)  # Makes formatting a bit worse, can this be changed?


# # I think the following can be used to catch certain error messages, potentially useful
# # -> adapted from https://github.com/pyro-ppl/numpyro/blob/master/numpyro/__init__.py, line 26
# def _filter_unreviewed_warning(record):
#     # return not record.getMessage().startswith("UserWarning: This code is currently UNREVIEWED, use with caution!")
#     # return not record.getMessage().contains("This code is currently UNREVIEWED, use with caution!")
#     return not "This code is currently UNREVIEWED, use with caution!" in record.getMessage()

# class NoReviewFilter(_log.Filter):
#     def filter(self, record):
#         # return not record.getMessage().startswith("UserWarning: This code is currently UNREVIEWED, use with caution!")
#         # return not record.getMessage().contains("This code is currently UNREVIEWED, use with caution!")
#         return not "This code is currently UNREVIEWED, use with caution!" in record.getMessage()


# warnings_logger.addFilter(_filter_unreviewed_warning)
# warnings_logger.addFilter(NoReviewFilter())

# # _log.captureWarnings(False)  # Do not do, then filters have no effect


# # # Maybe we have to add to logger of lalsuite, not ours?
# # _log.getLogger('lalsuite').addFilter(_filter_unreviewed_warning)
# # _log.getLogger('lalsuite').addFilter(NoReviewFilter())

# # _log.getLogger('lalsuite').info('HELLO?')  # Is printed, but filter still does not work

# # from warnings import filterwarnings as _filterwarnings
# # _filterwarnings('ignore', message=".*This code is currently UNREVIEWED, use with caution!.*")
# # This works. But really do it? Definitely remove in releases, maybe keep privately

# lalsim_logger = _log.getLogger('lalsimulation')
# lalsim_logger.addFilter(_filter_unreviewed_warning)
# lalsim_logger.addFilter(NoReviewFilter())
