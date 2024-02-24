# ----- Standard Lib Imports -----
import logging
import warnings
from typing import Optional, Any, Literal, Self, Callable
# from __future__ import annotations  # Hack for type hinting inside a class
                                    # -> https://stackoverflow.com/a/42845998

# ----- Third Party Imports -----
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries
import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm

# ----- Local Package Imports -----
from .inner_product import inner_product, inner_product_computation, norm
from .waveform_utils import get_strain
from .matrix_with_units import MatrixWithUnits
from .fisher_utils import (
    fisher_matrix, get_waveform_derivative_1D,
    get_waveform_derivative_1D_with_convergence
)


class FisherMatrix:
    """
    A data type tailored to Fisher matrices. It stores the Fisher matrix
    itself, along with its inverse.

    Parameters
    ----------
    wf_params_at_point : dict[str, u.Quantity]
        Point in parameter space at which the Fisher matrix is
        evaluated, encoded as key-value-pairs. Input to `wf_generator`.
    params_to_vary : str or list[str]
        Parameter(s) with respect to which the derivatives will be
        computed, the norms of which constitute the Fisher matrix.
        Must be keys in `wf_params_at_point`.
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]],
    FrequencySeries or ArrayLike]
        Arbitrary function that is used for waveform generation. The
        required signature means that it has one non-optional argument,
        which is expected to accept the input provided in
        `wf_params_at_point`, while the output is either a ``~gwpy.
        frequencyseries.FrequencySeries`` or of type ``ArrayLike``, so
        that its subtraction is carried out element-wise. The preferred
        type is ``FrequencySeries`` because it supports astropy units
        (and it is the standard output of LAL gwsignal generators).

        A convenient option is to use the method `FisherMatrix.
        get_wf_generator`, which generates a suitable function from
        a few arguments.

    See also
    --------
    ~gwsignal.fisher_utils.fisher_matrix :
        Routine used for calculation of the Fisher matrix.
    ~gwsignal.fisher_utils.get_waveform_derivative_1D_with_convergence :
        Routine used for calculation of involved derivatives. Used by
        ~gwsignal.fisher_utils.fisher_matrix.
    numpy.linalg.inv : Routine used for inversion of the Fisher matrix.

    Notes
    -----
    In principle, instances of this class constain much more information
    than "just" the Fisher matrix ``FisherMatrix.fisher``, for example
    its inverse ``FisherMatrix.fisher_inverse``. However, to provide an
    intuitive behaviour, remembering the class name, certain operations
    return attributes related only to the Fisher matrix. Array-
    conversion, for example, returns the array representation of
    ``FisherMatrix.fisher``.
    """

    default_metadata = {
        'convergence_check': 'diff_norm',
        'break_upon_convergence': True,
        'convergence_threshold': 0.01
    }

    def __init__(self,
            wf_params_at_point: dict[str, u.Quantity],
            params_to_vary: str | list[str],
            wf_generator: Callable[[dict[str, u.Quantity]],
                                   FrequencySeries | ArrayLike],
            **metadata
        ) -> None:
        """
        Initialize a ``FisherMatrix``.
        """
        self.wf_params_at_point = wf_params_at_point
        self.params_to_vary = params_to_vary
        self.wf_generator = wf_generator
        self.metadata = self.default_metadata | metadata

        if ('return_info' in metadata.keys()) and metadata['return_info']:
            logging.info(
                'The `return_info` key is set to true to collect information '
                'for further use in the class.'
            )

            self.metadata['return_info'] = True
    

        # Dummy Fisher, speed up when testing
        # self._fisher = MatrixWithUnits(
        #     np.full((2, 2), 42),
        #     np.full((2, 2), u.s)
        # )
        
        self._calc_fisher()
    

    def _calc_fisher(self):
        """
        Call `~gw_signal_tools.fisher_utils.fisher_matrix` to calculate
        the Fisher matrix.
        """
        self._fisher, self._deriv_info = fisher_matrix(
            self.wf_params_at_point,
            self.params_to_vary,
            self.wf_generator,
            **self.metadata
        )

        # NOTE: although it may not be good practice to set private attributes
        # like self._fisher, this is our workaround to make self.fisher
        # immutable (has no setter). If we were to set self.fisher here,
        # a setter would be required
        # -> _fisher being set is inevitable, some property has to be settable

        # TODO: decide if this is good idea... Can also be called by user and
        # thus goes against philosophy of immutable... So maybe remove, put
        # back into __init__ and return None upon attribute error in fisher?
        # Perhaps along with message

    # TODO: decide if properties value and unit might make sense, which return
    # the corresponding property of the Fisher matrix... (seems to be most
    # intuitive thing to do)
    
    @property
    def fisher(self):
        """
        Actual Fisher matrix associated with this class.

        :type:`~gw_signal_tools.matrix_with_units.MatrixWithUnits`
        """
        try:
            return self._fisher
        except AttributeError:
            # This case should never be reached, except the matrix was deleted.
            # In that case, although it must be the users fault, recompute
            self._calc_fisher()

            return self._fisher


    @property
    def fisher_inverse(self):
        """
        Inverse of Fisher matrix associated with this class.

        :type:`~gw_signal_tools.matrix_with_units.MatrixWithUnits`
        """
        # TODO: decide if it shall be computed upon call or upon calculation of Fisher
        try:
            return self._fisher_inverse
        except AttributeError:
            # Inverse is called for the first time or has been deleted
            self._fisher_inverse = MatrixWithUnits(
                np.linalg.inv(self.fisher.value),
                self.fisher.unit**-1,
            )

            return self._fisher_inverse
        
    
    @property
    def deriv_info(self):
        # self._deriv_info is available... Soooo, shall we something with it?
        raise NotImplementedError


    def update_metadata(self,
            new_wf_params_at_point: Optional[dict[str, u.Quantity]] = None,
            new_params_to_vary: Optional[str | list[str]] = None,
            new_wf_generator: Optional[Any] = None,
            **new_metadata
        ) -> Self:
        """
        Generate a Fisher matrix with properties like the current
        instance has, but selected updates.
        Note that this creates a new instance of ``FisherMatrix`` since
        updating properties would require new calculation anyway.

        Parameters
        ----------
        new_wf_params_at_point : dict[str, u.Quantity]
            Point in parameter space at which the Fisher matrix is
            evaluated, encoded as key-value-pairs. Given as input to
            `wf_generator`.
        new_params_to_vary : str or list[str]
            Parameter(s) with respect to which the derivatives will be
            computed, the norms of which constitute the Fisher matrix.
            can in principle also be any, but param_to_vary has to be
            accessible as a key and value has to be value of point that
            we want to compute derivative around. Must be keys in
            `new_wf_params_at_point`.

            Note that for this function, it is not required to specify a
            completely novel set. Updating only selected parameters is
            suppported
        new_wf_generator : Callable[[dict[str, ~astropy.units.
        Quantity]], FrequencySeries or ArrayLike]
            Arbitrary function that is used for waveform generation. The
            required signature means that it has one non-optional
            argument, which is expected to accept the input provided in
            `wf_params_at_point`, while the output is either a ``~gwpy.
            frequencyseries.FrequencySeries`` or of type ``ArrayLike``,
            so that its subtraction is carried out element-wise. The
            preferred type is ``FrequencySeries`` because it supports
            astropy units (and it is the standard output of LAL gwsignal
            generators).

            A convenient option is to use the method `FisherMatrix.
            get_wf_generator`, which generates a suitable function from
            a few arguments.

        Returns
        -------
        ~gw_signal_tools.fisher_matrix.FisherMatrix
            New Fisher matrix, calculated with updated metadata.
        """
        if new_wf_params_at_point is None:
            new_wf_params_at_point = self.wf_params_at_point

        if new_params_to_vary is None:
            new_params_to_vary = self.params_to_vary

        if new_wf_generator is None:
            new_wf_generator = self.wf_generator
        
        if len(new_metadata) > 0:
            new_metadata = self.metadata | new_metadata


        return FisherMatrix(new_wf_params_at_point, new_params_to_vary,
                            **new_metadata)

    
    def condition_number(self,
            matrix_norm: float | Literal['fro', 'nuc'] = 'fro'
        ) -> float:
        """
        Condition number of the Fisher matrix.

        Parameters
        ----------
        matrix_norm : float | Literal['fro', 'nuc'], optional,
        default = 'fro'
            Matrix norm that shall be used for the calculation. Must be
            compatible with argument `p` of `~numpy.linalg.cond`.

        Returns
        -------
        float
            Condition number of `self.fisher`.

        See also
        --------
        numpy.linalg.cond : Routine used for calculation.
        """
        return np.linalg.cond(self.fisher.value, p=matrix_norm)


    def project_fisher(self, projection_params):
        # Idea: pply projection onto certain parameters
        # -> can this be applied to multiple ones?
        raise NotImplementedError
    

    def plot(self):
        # plot with color=uncertainty
        raise NotImplementedError
    

    @staticmethod
    def get_wf_generator(
        approximant: str,
        domain: str = 'frequency',
        *args, **kwargs
    ) -> Callable[[dict[str, u.Quantity]],
                  FrequencySeries | ArrayLike]:
        """
        Generates a function that fulfils the requirements of the
        `wf_generator` argument of a ``FisherMatrix``.

        Parameters
        ----------
        approximant : _type_
            _description_
        domain : _type_
            Default is 'frequency', the domain where the Fisher matrix
            is computed.

        Returns
        -------
        _type_
            _description_

        See also
        --------
        gw_signal_tools.waveform_utils.get_strain :
            Function that is wrapped here. All arguments provided in
            addition to the mandatory ones are passed to this function
            (just like `domain` is as well).
        lalsimulation.gwsignal.core.waveform.
        LALCompactBinaryCoalescenceGenerator : 
            Function used to get a generator from `approximant`. This is
            passed to the `generator` argument of `get_strain`.
        """
        generator = wfm.LALCompactBinaryCoalescenceGenerator(approximant)

        def wf_generator(wf_params):
            return get_strain(wf_params, domain, generator, *args, **kwargs)

        return wf_generator
    

    # ----- Set some Python class related goodies -----
    def __repr__(self) -> str:
        return self.fisher.__repr__()
        # TODO: make custom one with more information
        # return '''
        # Fisher Matrix

        # MORE DESCRIPTION TO COME
        # '''
    

    def __array__(self):
        # Most intuitive behaviour: indeed return a Fisher matrix as array
        return self.fisher.__array__()
