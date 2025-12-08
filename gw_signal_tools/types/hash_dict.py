# -- Standard Lib Imports
from __future__ import annotations
from typing import Any


__all__ = ('HashableDict',)


class HashableDict(dict):
    """
    A class based on ``dict`` that can be hashed.

    Note that hashing is based on the items of the dictionary, so the
    hash value associated with an instance is unsafe in the sense that
    it will change if the contents of the dictionary change. However,
    this is fine for use within the scope of `gw_signal_tools`.
    """

    def __hash__(self):
        return hash(frozenset(self.items()))

    # -- For convenience we overwrite merge with dictionaries so that a
    # -- HashableDict is returned too
    def __or__(self, value: Any) -> HashableDict:
        return HashableDict(super().__or__(value))

    def __ror__(self, value: Any) -> HashableDict:
        return HashableDict(super().__ror__(value))

    def copy(self):
        return HashableDict(dict.copy(self))

    def __copy__(self):
        return self.copy()
