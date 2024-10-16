# -- Standard Lib Imports
from types import UnionType
from typing import Any


__all__ = ('HashableDict', )


class HashableDict(dict):
    """A class based on ``dict`` that can be hashed."""
    def __hash__(self):
        return hash(frozenset(self.items()))
        # return hash(frozenset(self.copy().items()))
        # -- Could think about doing this, now that copy is available
    
    # -- For convenience we overwrite merge with dictionaries so that a
    # -- HashableDict is returned too
    def __or__(self, value: Any) -> UnionType:
        return HashableDict(super().__or__(value))
    
    def __ror__(self, value: Any) -> UnionType:
        return HashableDict(super().__ror__(value))
    
    def copy(self):
        return HashableDict(dict.copy(self))
    
    def __copy__(self):
        return self.copy()
