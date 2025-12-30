from typing import Protocol  # , TypeVar


__doc__ = """File for custom type definitions."""

__all__ = ('DerivInfoBase',)


# # DerivInfoBase = TypeVar('DerivInfoBase', bound=tuple)
# # DerivInfoBase = TypeVar('DerivInfoBase', bound=WaveformDerivativeBase.DerivInfo)
# DerivInfoBase = TypeVar('DerivInfoBase', bound=NamedTuple)


# class DerivInfoBase(NamedTuple):
class DerivInfoBase(Protocol):
    """Type stub for derivative information."""

    # pass

    # TODO: or are there some fields that we absolutely want to have here?
    is_exact_deriv: bool = False
    """Whether derivative is exact (analytical) or not."""
