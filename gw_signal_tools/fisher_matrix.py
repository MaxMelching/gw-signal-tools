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
    its inverse

    Parameters
    ----------
    wf_params_at_point : _type_
        _description_
    params_to_vary : _type_
        _description_
    wf_generator : _type_
        _description_

        A convenient option to use the method `FisherMatrix.
        get_wf_generator`, which generates a suitable function from
        a few arguments.

    See also
    --------
    ~gwsignal.fisher_utils.fisher_matrix : Routine used for calculation
    of the Fisher matrix.
    numpy.linalg.inv : Routine used for inversion of the Fisher matrix.
    """

    default_metadata = {
        'convergence_check': 'diff_norm',
        'break_upon_convergence': True,
        'convergence_threshold': 0.01
    }

    def __init__(self,
            wf_params_at_point: dict[str, u.Quantity],
            params_to_vary: str,
            wf_generator: Callable[[dict[str, u.Quantity]],
                                   FrequencySeries | ArrayLike],
            **metadata
        ):
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

    
    @property
    def fisher(self):
        try:
            if self._fisher is not None:
                return self._fisher
        except AttributeError:
            pass
            # return None here?

        # Either it was None or not set. Either way, recompute
        # self._fisher = fisher_matrix(
        #     wf_params_at_point=wf_params_at_point,
        #     params_to_vary=params_to_vary,
        # )

        # With new structure, just create new class?
        # -> on the other hand, this should never happen...


        return self._fisher
    

    # @fisher.setter
    # def fisher(self, value):
    #     try:
    #         logging.info('Here we are')
    #         attr_test = self._fisher

    #         raise AttributeError(
    #             'Attribute `fisher` may only be set upon initialization of an '
    #             'instance of ``FisherMatrix``.'
    #         )
    #     except (AttributeError) as err:
    #         err_msg = str(err)
    #         if not ('upon initialization') in err_msg:
    #             self._fisher = value
    #         else:
    #             raise AssertionError(err_msg)
            
    
    # _fisher = property(self.fisher.__get__)
    

    @property
    def fisher_inverse(self):
        # TODO: decide if it shall be computed upon call or upon calculation of Fisher
        try:
            return self._fisher_inverse
        except AttributeError:
            self._fisher_inverse = MatrixWithUnits(
                np.linalg.inv(self.fisher.value),
                self.fisher.unit**-1,
            )

            return self._fisher_inverse


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
        new_wf_params_at_point : _type_, optional
            _description_, by default None
        new_params_to_vary : _type_, optional
            _description_, by default None
        new_wf_generator : _type_, optional
            _description_, by default None

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
                         matrix_norm: float | Literal['fro', 'nuc'] = 'fro'):
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
    def get_wf_generator(approximant: str, domain: str, *args, **kwargs
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
            _description_

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
        print(approximant)
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
        # return self.fisher
        # return self.fisher.value
        return self.fisher.__array__()


    # def __float__():
    #     # Idea: return only matrix with values, not what is usually
    #     # returned (product of values and matrix with units)
    #     return 
    # Does not work, float() wants real float returned, not array with this dtype


    # def __getattr__(self, attr):
    #     # return super().__getattribute__(attr)
    #     return self._fisher.__getattribute__(attr)
    # Thought might be a good idea to get stuff like array or repr
    # from .fisher, which seemed like very intuitive way to do stuff
