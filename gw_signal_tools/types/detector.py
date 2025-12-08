# -- Standard Lib Imports
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# -- Third Party Imports
from gwpy.frequencyseries import FrequencySeries
import numpy as np


__doc__ = """
Module for the ``Detector`` class that is intended to provide a simple
representation of gravitational wave detectors, with all information
needed by functions in `gw_signal_tools`.
"""

__all__ = ('Detector',)


@dataclass(frozen=True)
class Detector:
    name: str
    psd: FrequencySeries
    inner_prod_kwargs: dict[str, Any] = field(
        default_factory=dict, repr=False, init=False
    )
    """
    Basic representation of a gravitational wave (GW) detector for use
    in the context of waveforms.

    Parameters
    ----------
    name : str
        Name of the detector. Is used during waveform generation as
        input to the `'det'` parameter in LAL dictionaries.
    psd : ~gwpy.frequencyseries.FrequencySeries
        Power spectral density of the detector.
    kw_args :
        All other keyword arguments will be interpreted as arguments
        that are supposed to be used in inner product calculations. This
        allows to specify certain properties that are distinct for
        detector that this instance represents, e.g. a certain starting
        frequency (note: this is not required for the `'det'` parameter,
        which is set automatically based on `name`).

        Note: since the PSD is already an attribute of this class, it
        does not need to be given here (only relevant is `kw_args` is
        passed as a dictionary and not via keyword arguments).
    """

    def __init__(self, name: str, psd: FrequencySeries, **kw_args):
        # -- Use object.__setattr__ for frozen dataclassto circumvent missing setters
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'psd', psd)
        object.__setattr__(self, 'inner_prod_kwargs', kw_args | {'psd': psd})

        # -- Validate types
        if not isinstance(name, str):
            raise TypeError('`name` must be a string.')
        if not isinstance(psd, FrequencySeries):
            raise TypeError('`psd` must be a FrequencySeries.')

    def update(
        self,
        new_name: str | None = None,
        new_psd: FrequencySeries | None = None,
        **kw_args,
    ) -> Detector:
        """
        Create a copy of this ``Detector`` with updated properties.
        The recommended way to replace attributes of ``Detector`` is to
        use this method ``Detector`` is a frozen dataclass.

        Parameters
        ----------
        new_name : str, optional, default = None
            New name of the detector. If `None`, the current name is
            kept.
        new_psd : ~gwpy.frequencyseries.FrequencySeries, optional, default = None
            New power spectral density of the detector. If `None`,
            the current PSD is kept.
        kw_args :
            Additional keyword arguments that will be used to update
            the inner product keyword arguments. If an argument with
            the same name already exists, it will be overwritten.
        """
        new_kw_args = (
            self.inner_prod_kwargs
            | kw_args
            | dict(psd=new_psd if new_psd is not None else self.psd)
        )
        return Detector(
            name=new_name if new_name is not None else self.name,
            **new_kw_args,
        )

    def copy(self) -> Detector:
        return self.update()

    def __copy__(self) -> Detector:
        return self.copy()

    def __repr__(self) -> str:
        # -- Basically copying what GWpy Array does
        prefix = f'<{self.__class__.__name__}('
        indent = ' ' * len(prefix)
        attr_str = ''
        for attr in ['name', 'psd']:
            attr_str += f'\n{indent}{attr}: {repr(getattr(self, attr))}'
        attr_str += f'\n{indent}inner_prod_kwargs: ' + repr(
            {k: v for k, v in self.inner_prod_kwargs.items() if k != 'psd'}
        )
        return prefix + attr_str.lstrip('\n').lstrip(' ') + ')>'

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Detector):
            return NotImplemented
        return (
            self.name == other.name
            and np.all(np.equal(self.psd, other.psd))
            and self.inner_prod_kwargs == other.inner_prod_kwargs
        )
