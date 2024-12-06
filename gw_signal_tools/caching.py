# -- Standard Lib Imports
from functools import cache, lru_cache
from typing import Callable
import contextlib
import logging

# -- Third Party Imports
from gw_signal_tools import logger


__all__ = (
    'disable_caching',
    'enable_caching',
    'cache_func',
    'use_caching',
    'enable_caching_locally',
    'disable_caching_locally',
)


# _cache = cache
_cache = lru_cache(maxsize=128, typed=True)
# TODO: should we use lru_cache with typed=True? Right now, I do not
# see why, but I might be missing something


def _dummy_cache(func):
    """Wrapper returning the input function."""
    return func


# -- Default setting: caching disabled
cache_func: Callable = _dummy_cache
use_caching: bool = False
_calling_from_context: bool = False
# TODO: think about whether this makes sense or not. If we take logging
# level approach, then we also omit all other info messages emitted by
# logger, not sure if I like that...


def disable_caching():
    global _calling_from_context
    if not _calling_from_context:
        logger.info('Disabling caching')

    global cache_func
    cache_func = _dummy_cache

    global use_caching
    use_caching = False


def enable_caching():
    global _calling_from_context
    if not _calling_from_context:
        logger.info('Enabling caching')

    global cache_func
    cache_func = _cache

    global use_caching
    use_caching = True


@contextlib.contextmanager
def enable_caching_locally():
    # -- Save state of things that we change in context
    _switch_off_caching = not use_caching
    # original_level = logger.getEffectiveLevel()
    # logger.setLevel(logging.WARNING)
    global _calling_from_context
    _calling_from_context = True

    try:
        enable_caching()
        yield
    finally:
        # -- Restore state from before this function was called
        if _switch_off_caching:
            disable_caching()

        # logger.setLevel(original_level)
        _calling_from_context = False


@contextlib.contextmanager
def disable_caching_locally():
    # -- Save state of things that we change in context
    _turn_on_caching = use_caching
    # original_level = logger.getEffectiveLevel()
    # logger.setLevel(logging.WARNING)
    global _calling_from_context
    _calling_from_context = True

    try:
        disable_caching()
        yield
    finally:
        # -- Restore state from before this function was called
        if _turn_on_caching:
            enable_caching()

        # logger.setLevel(original_level)
        _calling_from_context = False


# TODO: maybe make custom cacher that tries to, but fails if input is
# usual dictionary? Then we could control whether or not something is
# cached via argument that is passed
# -> hmmm, but when we e.g. convert to HashableDict in FisherMatrix
#    automatically, this is hidden behaviour and potentially not wanted
