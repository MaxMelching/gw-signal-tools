# -- Standard Lib Imports
import logging as _log


__doc__ = """Module in which the gw-signal-tools logger is defined."""

__all__ = ('logger',)


logger = _log.getLogger(__name__)

logger.propagate = False  # Otherwise root also prints them
logger.setLevel(_log.INFO)

formatter = _log.Formatter(
    fmt='%(asctime)s  %(levelname)s (%(filename)s: %(lineno)d): %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S',
)

from sys import stderr as _stderr

ch = _log.StreamHandler(stream=_stderr)  # More explicit
ch.setLevel(_log.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)
