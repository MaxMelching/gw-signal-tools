# -- Standard Lib Imports
from types import UnionType
from typing import Any


__all__ = ('HashableDict', )


class HashableDict(dict):
    """A class based on ``dict`` that can be hashed."""
    def __hash__(self):
        # return hash((frozenset(self.keys()), frozenset(self.values())))
        return hash(frozenset(self.items()))
        # Should work similarly to above one, no? But perhaps a little
        # faster, only one iterator call
    
    # For convenience we overwrite merge with dictionaries so that
    # a HashableDict is returned too
    def __or__(self, value: Any) -> UnionType:
        return HashableDict(super().__or__(value))
    
    def __ror__(self, value: Any) -> UnionType:
        return HashableDict(super().__ror__(value))
