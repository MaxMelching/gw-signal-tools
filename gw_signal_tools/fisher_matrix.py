# ----- Standard Lib Imports -----
from __future__ import annotations  # Enables type hinting own type in a class
import logging
import warnings
from typing import Optional, Any, Literal, Self, Callable

# ----- Third Party Imports -----
import numpy as np
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
        Must be `'time'`, `'phase'` or keys in `wf_params_at_point`.
    wf_generator : Callable[[dict[str, ~astropy.units.Quantity]],
    ~gwpy.frequencyseries.FrequencySeries]
        Arbitrary function that is used for waveform generation. The
        required signature means that it has one non-optional argument,
        which is expected to accept the input provided in
        `wf_params_at_point`, while the output must be a ``~gwpy.
        frequencyseries.FrequencySeries`` (the standard output of
        LAL gwsignal generators) because it carries information about
        value, frequencies and units, which are all required for the
        calculations that are carried out.

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
            wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
            _copy: bool = False,
            **metadata
        ) -> None:
        """
        Initialize a ``FisherMatrix``.
        """
        self.wf_params_at_point = wf_params_at_point
        self.params_to_vary = params_to_vary
        self.wf_generator = wf_generator
        self.metadata = self.default_metadata | metadata

        if ('return_info' in metadata.keys()) and not metadata['return_info']:
            logging.info(
                'The `return_info` key is set to true to collect information '
                'for further use in the class.'
            )

        self.metadata['return_info'] = True
    

        if not _copy:
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
    def fisher(self) -> MatrixWithUnits:
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
    def fisher_inverse(self) -> MatrixWithUnits:
        """
        Inverse of Fisher matrix associated with this class.

        :type:`~gw_signal_tools.matrix_with_units.MatrixWithUnits`
        """
        # TODO: decide if it shall be computed upon call or upon calculation of Fisher
        try:
            return self._fisher_inverse  # type: ignore
            # Explanation of ignore: type cannot be inferred and not hinted
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
        ) -> FisherMatrix:
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
            we want to compute derivative around. Must be `'time'`,
            `'phase'` or keys in `wf_params_at_point`.

            Note that for this function, it is not required to specify a
            completely novel set. Updating only selected parameters is
            suppported
        new_wf_generator : Callable[[dict[str, ~astropy.units.
        Quantity]], ~gwpy.frequencyseries.FrequencySeries]
            Arbitrary function that is used for waveform generation. The
            required signature means that it has one non-optional
            argument, which is expected to accept the input provided in
            `wf_params_at_point`, while the output must be a ``~gwpy.
            frequencyseries.FrequencySeries`` (the standard output of
            LAL gwsignal generators) because it carries information
            about value, frequencies and units, which are all required
            for the calculations that are carried out.

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


    # def project_fisher(self, projection_params):
        # Idea: apply projection onto certain parameters
        # -> can this be applied to multiple ones?
        # raise NotImplementedError
    @property
    def projected_fisher(self) -> MatrixWithUnits:
        params_array = np.array(self.params_to_vary)
        # time_index = np.where(params_array == 'time')#[0][0]
        # phase_index = np.where(params_array == 'phase')#[0][0]
        # print(time_index, phase_index)

        # if len(time_index) != 1 or len(phase_index) != 1:
        #     raise ValueError(
        #         'Need keys `time` and `phase` in `self.params_to_vary`.'
        #     )
        # else:
        #     time_index = time_index[0][0]
        #     phase_index = phase_index[0][0]

        if 'time' not in params_array or 'phase' not in params_array:
            raise ValueError(
                'Need keys `time` and `phase` in `self.params_to_vary`.'
            )
        else:
            time_index = np.where(params_array == 'time')[0][0]
            phase_index = np.where(params_array == 'phase')[0][0]
            # print(time_index, phase_index)
        
        # The following is just testing, should not actually be neededs
        # time_index, phase_index = min(time_index, phase_index), max(time_index, phase_index)
        # time_index, phase_index = phase_index, time_index

        n = len(self.params_to_vary)
        gamma = self.fisher.value
        # assert that gamma is (n x n) matrix?
        submatr = np.array([[gamma[time_index, time_index],
                             gamma[time_index, phase_index]],
                            [gamma[phase_index, time_index],
                             gamma[phase_index, phase_index]]])
        sub_matr_inv = np.linalg.inv(submatr)
        print(submatr, sub_matr_inv)

        self._projected_fisher = MatrixWithUnits(
            # self.fisher.value - np.sum(),
            # gamma - (
            #     gamma[:, 0] * sub_matr_inv[0, 0] * gamma[0, :]
            # ),
            np.array(
                [[gamma[i, j] - (
                        gamma[i, 0] * sub_matr_inv[0, 0] * gamma[0, j]
                        + gamma[i, 0] * sub_matr_inv[0, 1] * gamma[1, j]
                        + gamma[i, 1] * sub_matr_inv[1, 0] * gamma[0, j]
                        + gamma[i, 1] * sub_matr_inv[1, 1] * gamma[1, j]
                        ) for i in range(n)
                    ] for j in range(n)]
                    #     ) for i in range(n) if i not in [time_index, phase_index]
                    # ] for j in range(n) if j not in [time_index, phase_index]]
            ),
            # self.fisher.unit
            u.dimensionless_unscaled
        )

        return self._projected_fisher
    

    def plot(self):
        # plot with color=uncertainty
        raise NotImplementedError
    

    @staticmethod
    def get_wf_generator(
        approximant: str,
        domain: str = 'frequency',
        *args, **kwargs
    ) -> Callable[[dict[str, u.Quantity]], FrequencySeries]:
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
    

    def __array__(self) -> np.ndarray:
        # Most intuitive behaviour: indeed return a Fisher matrix as array
        return self.fisher.__array__()
    
    def __copy__(self) -> FisherMatrix:
        # Not even easy because we compute directly...
        # -> handling it right now via argument _copy
        new_matrix = FisherMatrix(
            self.wf_params_at_point,
            self.params_to_vary,
            self.wf_generator,
            _copy=True,
            **self.metadata
        )
        
        new_matrix._fisher = self.fisher
        new_matrix._fisher_inverse = self.fisher_inverse

        # TODO: decide if other attributes like condition number or
        # deriv_info shall be copied

        return new_matrix
    
    def copy(self) -> FisherMatrix:
        return self.__copy__()
