# -- Third Party Imports
from functools import cache#, lru_cache
from typing import Callable

# -- Third Party Imports
from gw_signal_tools import logger


__all__ = ('disable_caching', 'enable_caching', 'cache')


def _dummy_cache(func):
    """Wrapper returning the input function."""
    return func


# -- Default setting: caching disabled
cache_func: Callable = _dummy_cache
use_caching: bool = False

def disable_caching():
    logger.info('Disabling caching')

    global cache_func
    cache_func = _dummy_cache

    global use_caching
    use_caching = False

def enable_caching():
    logger.info('Enabling caching')

    global cache_func
    cache_func = cache

    global use_caching
    use_caching = True

# TODO: maybe make custom cacher that tries to, but fails if input is
# usual dictionary? Then we could control whether or not something is
# cached via argument that is passed
# -> hmmm, but when we e.g. convert to HashableDict in FisherMatrix
#    automatically, this is hidden behaviour and potentially not wanted
