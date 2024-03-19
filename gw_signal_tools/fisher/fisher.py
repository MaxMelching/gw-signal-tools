# ----- Standard Lib Imports -----
from __future__ import annotations  # Enables type hinting own type in a class
import logging
import warnings
from typing import Optional, Any, Literal, Self, Callable

# ----- Third Party Imports -----
import numpy as np
import matplotlib.pyplot as plt

from gwpy.frequencyseries import FrequencySeries
import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm

# ----- Local Package Imports -----
from ..waveform_utils import get_strain
from ..matrix_with_units import MatrixWithUnits
from .fisher_utils import fisher_matrix
from gw_signal_tools import preferred_unit_system


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
        'convergence_threshold': 0.01,
        'return_info': True
    }

    _preferred_unit_sys = preferred_unit_system
    # Idea: display stuff in these units, i.e. apply .to_system to each matrix before saving them
    # TODO: make setter etc. for it

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
    

        if not _copy:
            if self.metadata['return_info']:
                self._fisher, self._deriv_info = fisher_matrix(
                    self.wf_params_at_point,
                    self.params_to_vary,
                    self.wf_generator,
                    **self.metadata
                )
            else:
                self._fisher = fisher_matrix(
                    self.wf_params_at_point,
                    self.params_to_vary,
                    self.wf_generator,
                    **self.metadata
                )

            # NOTE: although it may not be good practice to set private
            # attributes like self._fisher, this is our workaround to make
            # self.fisher immutable (has no setter). If we were to set
            # self.fisher here, a setter would be required
            # -> _fisher being set inevitable, some property has to be settable

    
    @property
    def fisher(self) -> MatrixWithUnits:
        """
        Actual Fisher matrix associated with this class.

        :type:`~gw_signal_tools.matrix_with_units.MatrixWithUnits`
        """
        try:
            return self._fisher
        except AttributeError:
            return None  # User has deleted matrix, cannot happen otherwise
    
    @property
    def value(self) -> np.ndarray:
        """
        Value of Fisher matrix associated with this class.

        :type:`~numpy.ndarray`
        """
        return self.fisher.value
    
    @property
    def unit(self) -> np.ndarray:
        """
        Unit of Fisher matrix associated with this class.

        :type:`~numpy.ndarray`
        """
        return self.fisher.unit


    @property
    def fisher_inverse(self) -> MatrixWithUnits:
        """
        Inverse of Fisher matrix associated with this class.

        :type:`~gw_signal_tools.matrix_with_units.MatrixWithUnits`
        """
        # TODO: decide if it shall be computed upon call or upon calculation of Fisher
        try:
            return self._fisher_inverse  # type: ignore
            # Explanation of ignore: neither can type be inferred nor hinted
        except AttributeError:
            # Inverse is called for the first time or has been deleted
            self._fisher_inverse = MatrixWithUnits.inv(self.fisher)

            return self._fisher_inverse
        
    # TODO: make getattr and setattr where we either pass it to self.fisher
    # (debatable if it makes sense; for numerical input it might) or we only
    # accept string input so that parameters can be accessed using strings
    # -> but would also be cool for inverse, right? So think of way to do this
        
    
    @property
    def deriv_info(self):
        # TODO: self._deriv_info is available... Soooo, shall we something with it?
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
        return MatrixWithUnits.condition_number(self.fisher, matrix_norm)
    
    def project_fisher(self, params: str | list[str]) -> MatrixWithUnits:
        """
        Project Fisher matrix so that its components now live in the
        orthogonal subspace to certain parameters (corresponds to
        optimizing with respect to them).

        Parameters
        ----------
        params : str or list[str]
            Parameter or list of parameters to project out. Must have
            been given in `params_to_vary` upon initialization of the
            ``FisherMatrix`` instance.

        Returns
        -------
        ~gw_signal_tools.matrix_with_units.MatrixWithUnits
            Matrix with same shape as initial Fisher matrix, but
            potentially different component values.
        """
        if isinstance(params, str):
            params = [params]
        
        for param in params:
            assert param in self.params_to_vary, \
                (f'Parameter {param} was not used to calculate the Fisher '
                 'matrix, so it cannot be projected out of it.')
        
        param_indices = np.where(np.isin(self.params_to_vary, params))[0]
        index_grid = np.ix_(param_indices, param_indices)

        n = len(self.params_to_vary)  # Equal to self.fisher.shape[0]

        fisher_val = self.fisher.value
        # sub_matrix = fisher[params_indices][:, params_indices]
        sub_matrix = fisher_val[index_grid]
        sub_matrix_inv = np.linalg.inv(sub_matrix)

        full_inv = np.zeros((n, n))
        full_inv[index_grid] = sub_matrix_inv

        # return MatrixWithUnits(
        #     # fisher_val - np.tensordot(np.tensordot(fisher_val, full_inv, axes=(1, 0)), fisher_val, axes=(1, 0)),
        #     # fisher_val - np.tensordot(np.tensordot(fisher_val, full_inv, axes=(0, 1)), fisher_val, axes=(0, 1)),
        #     # fisher_val - fisher_val[:, param_indices]] @ sub_matr_inv @ fisher_val[param_indices, :],
        #     fisher_val - np.einsum('ij, jk, kl', fisher_val, full_inv, fisher),
        #     self.fisher.unit
        # )

        # Testing with new tool of matrix multiplication -> WORKS!!! Noice
        full_inv = MatrixWithUnits(full_inv, self.unit**-1)
        # full_inv = MatrixWithUnits(full_inv.T, self.unit**-1)
        fisher = self.fisher

        return fisher - fisher @ full_inv @ fisher
        # return fisher - MatrixWithUnits(np.einsum('ij, jk, kl', fisher, full_inv, fisher), self.fisher.unit)
    
    # Just toying around with application of matrix product
    # def statistical_error(self, param_diff: MatrixWithUnits) -> u.Quantity:
    #     r"""
    #     Calculates

    #     .. math:: \sum_{\mu \nu} \Gamma_{\mu \nu} \Delta \lambda^\mu
    #     \Delta \lambda^\nu

    #     using the given parameter difference :math:`\Delta \lambda^\mu`.

    #     Parameters
    #     ----------
    #     param_diff : ~gw_signal_tools.matrix_with_units.MatrixWithUnits
    #         Row or column vector of parameter differences, which must be
    #         specified with units (i.e. as a MatrixWithUnits instance).

    #         Order must match the order used in `params_to_vary` argument
    #         that has been used during Fisher matrix calculation.

    #     Returns
    #     -------
    #     ~astropy.units.quantity
    #         Total statistical error.

    #     Notes
    #     -----
    #     Calculating this does not make sense if a PSD for no noise is
    #     used during the Fisher matrix calculations
    #     """
    #     assert len(param_diff.shape) == 1, \
    #         '`param_diffs` must be either a row or a column vector.'
        
    #     return param_diff @ self.fisher_inverse @ param_diff
    # TODO: return **(1/2) of that? -> can now even do sqrt
    # -> also test if Fraction from functools is better for numerical precision

    def _plot_matrix(self, matrix: MatrixWithUnits):
        # Plan: call MatrixWithUnits.plot, but add labels to axes
        # -> for now, we implement everything here
        
        ax = MatrixWithUnits.plot(matrix)

        tick_labels = self.params_to_vary if not isinstance(self.params_to_vary, str) else [self.params_to_vary]
        from ..fisher import latexparams
        tick_labels = [latexparams[param] if param in latexparams else param for param in tick_labels]
        tick_locs = np.arange(0, len(tick_labels)) + 0.5

        ax.set_xticks(tick_locs, tick_labels, rotation=45,
                      horizontalalignment='right', rotation_mode='anchor')
        ax.set_yticks(tick_locs, tick_labels, rotation=45,
                      verticalalignment='baseline', rotation_mode='anchor')
        ax.tick_params(length=0)

        # -> rotation is good idea if param not in displayparams, otherwise looks strange

        return ax

    def plot(self, only_fisher: bool = False, only_fisher_inverse: bool = False) -> None:
        # NOT final version

        if not (only_fisher or only_fisher_inverse):
            self._plot_matrix(self.fisher)

            self._plot_matrix(self.fisher_inverse)
        elif only_fisher:
            self._plot_matrix(self.fisher)
        elif only_fisher_inverse:
            self._plot_matrix(self.fisher_inverse)

    # Plans for plotting: make one function plot_uncertainty where
    # color denotes uncertainty in fisher_inverse. And then one general
    # function plot where output is plot of fisher and fisher_inverse

    # def __plot__(self, *args, **kwargs):
    #     return self.plot(*args, **kwargs)

    def cond(self, matrix_norm: float | str = 'fro'):
        return self.fisher.cond(matrix_norm)

    @staticmethod
    def get_wf_generator(
        approximant: str,
        domain: Literal['frequency', 'time'] = 'frequency',
        *args, **kwargs
    ) -> Callable[[dict[str, u.Quantity]], FrequencySeries]:
        """
        Generates a function that fulfils the requirements of the
        `wf_generator` argument of a ``FisherMatrix``.

        Parameters
        ----------
        approximant : str
            Name of a waveform model that is accepted by the
            ``~lalsimulation.gwsignal.core.waveform.
            LALCompactBinaryCoalescenceGenerator`` class.
        domain : Literal['frequency', 'time'], optional,
        default = 'frequency'
            String representing the domain where the Fisher matrix is
            computed. Accepted values are 'frequency' and 'time'.

        Returns
        -------
        Callable[[dict[str, ~astropy.units.Quantity]], ~gwpy.
        frequencyseries.FrequencySeries]
            Function that takes dicionary of waveform parameters as
            input and produces a waveform (stored in a GWPy
            ``FrequencySeries``). Can, for example, be used as input
            to `wf_generator` argument during initialization of a
            ``FisherMatrix``.

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
