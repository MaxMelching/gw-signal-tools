from ._version import __version__  # Make attribute available, not created automatically


from os.path import dirname as _path_dirname

PACKAGE_PATH = _path_dirname(__file__)

from os.path import join as _path_join

PLOT_STYLE_SHEET = _path_join(PACKAGE_PATH, 'plot_stylesheet.sty')



# TODO: decide if functions shall be imported here. Otherwise one has to
# import from each module (preferred solution at the moment)


# ---------- Initialize Logging ----------
import logging

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format='%(asctime)s  %(levelname)s (%(filename)s: %(lineno)d): %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S'
)

# TODO: handle errors via logging? On the other hand, we log to command line anyway...

# TODO: make function set_logger, where custom logger can be specified?
# Would be called as gw_signal_tools.set_logger(), i.e. it has to live in this file

# logging.captureWarnings(True)  # Makes formatting a bit worse, can this be changed?


# I think the following can be used to catch certain error messages, potentially useful
# -> adapted from https://github.com/pyro-ppl/numpyro/blob/master/numpyro/__init__.py, line 26
# def _filter_unreviewed_warning(record):
#     # return not record.getMessage().startswith("UserWarning: This code is currently UNREVIEWED, use with caution!")
#     # return not record.getMessage().contains("This code is currently UNREVIEWED, use with caution!")
#     return not "This code is currently UNREVIEWED, use with caution!" in record.getMessage()

# class NoReviewFilter(logging.Filter):
#     def filter(self, record):
#         # return not record.getMessage().startswith("UserWarning: This code is currently UNREVIEWED, use with caution!")
#         # return not record.getMessage().contains("This code is currently UNREVIEWED, use with caution!")
#         return not "This code is currently UNREVIEWED, use with caution!" in record.getMessage()


# logging.getLogger(__name__).addFilter(_filter_unreviewed_warning)
# logging.getLogger(__name__).addFilter(NoReviewFilter())
# # Does not work...


# # Maybe we have to add to logger of lalsuite, not ours?
# logging.getLogger('lalsuite').addFilter(_filter_unreviewed_warning)
# logging.getLogger('lalsuite').addFilter(NoReviewFilter())

# logging.getLogger('lalsuite').info('HELLO?')  # Is printed, but filter still does not work

# from warnings import filterwarnings as _filterwarnings
# _filterwarnings('ignore', message=".*This code is currently UNREVIEWED, use with caution!.*")
# This works. But really do it? Definitely remove in releases, maybe keep privately