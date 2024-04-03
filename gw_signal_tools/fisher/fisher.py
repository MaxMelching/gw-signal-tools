# ----- Standard Lib Imports -----
from __future__ import annotations  # Enables type hinting own type in a class
import logging
import warnings
from typing import Optional, Any, Literal, Self, Callable

# ----- Third Party Imports -----
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from gwpy.frequencyseries import FrequencySeries
import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm

# ----- Local Package Imports -----
from ..inner_product import inner_product
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
        if isinstance(params_to_vary, str):
            self.params_to_vary = [params_to_vary]
        else:
            self.params_to_vary = params_to_vary.copy()
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

                plt.close('all')  # Avoid too many open axes
            else:
                self._fisher = fisher_matrix(
                    self.wf_params_at_point,
                    self.params_to_vary,
                    self.wf_generator,
                    **self.metadata
                )

                # self._deriv_info = {'general_info': 'There is no info available.'}
                self._deriv_info = {}

            # NOTE: although it may not be good practice to set private
            # attributes like self._fisher, this is our workaround to make
            # self.fisher immutable (has no setter). If we were to set
            # self.fisher here, a setter would be required
            # -> _fisher being set inevitable, some property has to be settable

    
    @property
    def fisher(self) -> MatrixWithUnits:
        """
        Actual Fisher matrix associated with this class.

        :type: `~gw_signal_tools.matrix_with_units.MatrixWithUnits`
        """
        try:
            return self._fisher
        except AttributeError:
            return None  # User has deleted matrix, cannot happen otherwise
    
    @property
    def value(self) -> np.ndarray:
        """
        Value of Fisher matrix associated with this class.

        :type: `~numpy.ndarray`
        """
        return self.fisher.value
    
    @property
    def unit(self) -> np.ndarray:
        """
        Unit of Fisher matrix associated with this class.

        :type: `~numpy.ndarray`
        """
        return self.fisher.unit


    @property
    def fisher_inverse(self) -> MatrixWithUnits:
        """
        Inverse of Fisher matrix associated with this class.

        :type: `~gw_signal_tools.matrix_with_units.MatrixWithUnits`
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
        return self._deriv_info


    def update_attrs(self,
        new_wf_params_at_point: Optional[dict[str, u.Quantity]] = None,
        new_params_to_vary: Optional[str | list[str]] = None,
        new_wf_generator: Optional[Callable[[dict[str, u.Quantity]],
                                            FrequencySeries]] = None,
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
        else:
            new_metadata = self.metadata


        return FisherMatrix(new_wf_params_at_point, new_params_to_vary,
                            new_wf_generator, **new_metadata)
    
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

        # return fisher - fisher @ full_inv @ fisher
        # return fisher - MatrixWithUnits(np.einsum('ij, jk, kl', fisher, full_inv, fisher), self.fisher.unit)

        # Idea: return FisherMatrix instead of MatrixWithUnits? Would
        # enable things like calculation of systematic error using this
        # projected version (not sure if this makes sense, though)
        out = self.copy()
        for param in params:
            out.params_to_vary.remove(param)
            #  Also look at deriv_info, pop params there
            out._deriv_info.pop(param, None)  # Makes None default if key not there
        
        param_indices_2 = [i for i in range(len(self.params_to_vary))]
        for index in param_indices:
            param_indices_2.remove(index)
        index_grid_2 = np.ix_(param_indices_2, param_indices_2)
        out._fisher = (fisher - fisher @ full_inv @ fisher)[index_grid_2]
        # This is the sneaky way of changing out.fisher (which has no setter,
        # so changing it is not permitted otherwise)
        out._fisher_inverse = MatrixWithUnits.inv(out.fisher)

        # out.metadata['general_info'] = 'This is a projected Fisher matrix.'
        # out.metadata['projected'] = True  # Given to fisher_matrix...
        out._is_projected = True

        return out
    
    @property
    def is_projected(self):
        """
        Information on whether or not the matrix was obtained via
        calculation only or via calculation and subsequent projection.
        In the latter case, one cannot reproduce the result by
        calculating the matrix again with the same parameters.

        :type: boolean
        """
        try:
            return self._is_projected
        except AttributeError:
            self._is_projected = False
            return False
    
    def statistical_error(self, params: Optional[str | list[str]] = None) -> MatrixWithUnits:
        r"""
        Calculates the :math:`1-\sigma` statistical error

        .. math:: \Delta \theta^\mu = \sqrt{\Gamma^{-1}_{\mu \mu}}

        for the selected parameters.

        Parameters
        ----------
        params : str | list[str], optional, default = None
            Parameter(s) to calculate error for. In case it is None (the
            default), the error for all parameters from `self.
            params_to_vary` is calculated. Can also be a string or list
            of strings, but these have to match elements of `self.
            params_to_vary`.

        Returns
        -------
        ~gw_signal_tools.matrix_with_units.MatrixWithUnits
            Vector of statistical errors. Indices match indices of
            `params_to_vary` variable that has been used to initialize
            the class.

        Notes
        -----
        Calculating this does not make sense if a PSD for no noise is
        used during the Fisher matrix calculations.
        """
        if params is not None:
            if isinstance(params, str):
                params = [params]
            
            for param in params:
                assert param in self.params_to_vary, \
                    (f'Parameter {param} was not used to calculate the Fisher '
                    'matrix, so it cannot be projected out of it.')
            
            param_indices = np.where(np.isin(self.params_to_vary, params))[0]
        else:
            # Take all parameters
            param_indices = len(self.params_to_vary)*[True]
        
        return MatrixWithUnits.sqrt(self.fisher_inverse.diag()[param_indices])
        # if len(params) != 1:
        #     # TODO: decide if this distinction makes sense... Multiple return types are meh, right?
        #     return MatrixWithUnits.sqrt(self.fisher_inverse.diag()[param_indices])
        # else:
        #     return np.sqrt(self.fisher_inverse.diag()[param_indices])

    def systematic_error(self,
        wf_generator_2: Callable[[dict[str, u.Quantity]], FrequencySeries],
        # TODO: rename to reference_model?
        params: Optional[str | list[str]] = None,
        **inner_prod_kwargs
    ) -> MatrixWithUnits:
        r"""
        Calculates the systematic error

        .. math:: \Delta \theta^\mu = \sum_{\nu} \Gamma^{-1}_{\mu \nu}
        \langle \frac{\partial h}{\partial \theta^\nu}, \delta h \rangle

        where :math:`\delta h = h - h_2`. Here, :math:`h` is the
        waveform model used to calculate the Fisher matrix instance and
        :math:`h_2` is a second model, with respect to which we want to
        compute the systematic error.

        Parameters
        ----------
        wf_generator_2 : Callable[[dict[str, ~astropy.units.Quantity]],
        ~gwpy.frequencyseries.FrequencySeries]
            Waveform generator for other waveform model that the
            systematic error shall be computed with respect to.
        params : str | list[str], optional, default = None
            Parameter(s) to calculate error for. In case it is None (the
            default), the error for all parameters from `self.
            params_to_vary` is calculated. Can also be a string or list
            of strings, but these have to match elements of `self.
            params_to_vary`.

        Returns
        -------
        MatrixWithUnits
            Column vector containing the systematic errors.
        """
        # TODO: require to give new inner_prod_kwargs here?
        if params is not None:
            if isinstance(params, str):
                params = [params]
            
            for param in params:
                assert param in self.params_to_vary, \
                    (f'Parameter {param} was not used to calculate the Fisher '
                    'matrix, so it cannot be projected out of it.')
            
            param_indices = np.where(np.isin(self.params_to_vary, params))[0]
        else:
            # Take all parameters
            param_indices = len(self.params_to_vary)*[True]

        delta_h = self.wf_generator(self.wf_params_at_point) \
                  - wf_generator_2(self.wf_params_at_point)
        # TODO: enable possibility to optimize this over certain parameters,
        # for example f_ref
        
        if self.metadata['return_info']:
            derivs = [
                self.deriv_info[param]['deriv'] for param in self.params_to_vary
            ]
        else:
            # NOTE: I don't think we can/should calculate derivs for the
            # parameters in params only because this argument is meant
            # to determine return. For error, parameters that are not in
            # in params still play a role and have to be accounted for.
            from gw_signal_tools.fisher import get_waveform_derivative_1D_with_convergence
            derivs = [
                get_waveform_derivative_1D_with_convergence(
                    self.wf_params_at_point,
                    param_to_vary,
                    self.wf_generator
                ) for param_to_vary in self.params_to_vary
            ]


        # Note: we want to use same kwargs for inner product as for
        # calculation of Fisher matrix here, have to be extracted from
        # metadata property
        # -> changed this now, we might want to optimize over some stuff here
        # from inspect import signature
        # fisher_args = list(signature(fisher_matrix).parameters)
        # fisher_args.remove('inner_prod_kwargs')
        # inner_prod_kwargs = self.metadata.copy()

        # # Remove all potential arguments for Fisher, leaves inner_prod_kwargs
        # for key in fisher_args:
        #     try:
        #         inner_prod_kwargs.pop(key)
        #     except KeyError:
        #         pass

        vector = MatrixWithUnits.from_numpy_array(np.zeros((len(derivs), 1)))
        for i, deriv in enumerate(derivs):
            vector[i] = inner_product(delta_h, deriv, **inner_prod_kwargs)
        
        return (self.fisher_inverse @ vector)[param_indices]

    def plot_matrix(self, matrix: MatrixWithUnits, xticks: bool = True,
                    yticks: bool = True, *args, **kwargs) -> mpl.axes.Axes:
        """
        Plotting routine specifically for matrices in a ``FisherMatrix``
        instance. Extends `MatrixWithUnits.plot` by adding labels for
        parameters.

        Parameters
        ----------
        matrix : MatrixWithUnits
            Matrix to plot. Is assumed to have entries that correspond
            to the parameters in `params_to_vary` variable of the
            instance that the method is called upon.
        xticks : bool, optional, default = True
            Whether or not ticks on the x-axis shall be drawn.
        yticks : bool, optional, default = True
            Whether or not ticks on the y-axis shall be drawn.

        Returns
        -------
        ~matplotlib.axes.Axes
            A matplotlib axes object with the plot attached to it. Can
            be further edited or simply plotted by calling `plt.show()`
            after calling this function.
        """
        ax = MatrixWithUnits.plot(matrix, *args, **kwargs)

        tick_labels = self.params_to_vary if not isinstance(self.params_to_vary, str) else [self.params_to_vary]
        from ..fisher import latexparams
        tick_labels = [latexparams[param] if param in latexparams else param for param in tick_labels]
        tick_locs = np.arange(0, len(tick_labels)) + 0.5

        if xticks:
            ax.set_xticks(tick_locs, tick_labels, rotation=45,
                        horizontalalignment='right', rotation_mode='anchor')
        if yticks:
            ax.set_yticks(tick_locs, tick_labels, rotation=45,
                        verticalalignment='baseline', rotation_mode='anchor')
        ax.tick_params(length=0)

        # -> rotation is good idea if param not in displayparams, otherwise looks strange

        return ax

    def plot(self, only_fisher: bool = False, only_fisher_inverse: bool = False) -> None:
        # NOT final version

        if not (only_fisher or only_fisher_inverse):
            self.plot_matrix(self.fisher)

            self.plot_matrix(self.fisher_inverse)
        elif only_fisher:
            self.plot_matrix(self.fisher)
        elif only_fisher_inverse:
            self.plot_matrix(self.fisher_inverse)

    # Plans for plotting: make one function plot_uncertainty where
    # color denotes uncertainty in fisher_inverse. And then one general
    # function plot where output is plot of fisher and fisher_inverse

    def cond(self, matrix_norm: float | str = 'fro'):
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
        # return self.fisher.__repr__()
        # TODO: make custom one with more information

        from shutil import get_terminal_size
        terminal_width = get_terminal_size()[0]

        def get_name_header(name: str) -> str:
            return f"{' ' + name + ' ':-^{terminal_width}}"
        
        return f'''
{get_name_header('Generation Parameters')}
{self.params_to_vary.__repr__()}
\n
{get_name_header('Fisher Matrix')}
{self.fisher.__repr__()}
\n
{get_name_header('Fisher Matrix Inverse')}
{self.fisher_inverse.__repr__()}
        '''

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
        
        new_matrix._fisher = self.fisher.copy()
        new_matrix._fisher_inverse = self.fisher_inverse.copy()
        new_matrix._deriv_info = self.deriv_info.copy()
        new_matrix._is_projected = self.is_projected

        return new_matrix
    
    def copy(self) -> FisherMatrix:
        return self.__copy__()
    
    def __hash__(self) -> int:
        return hash(self.fisher) ^ hash(self.fisher_inverse) \
            ^ hash(self.metadata) ^ hash(self.wf_params_at_point) \
            ^ hash(self.params_to_vary) ^ hash(self.is_projected)
