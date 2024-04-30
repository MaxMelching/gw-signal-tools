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
from lalsimulation.gwsignal import gwsignal_get_waveform_generator

# ----- Local Package Imports -----
from ..inner_product import inner_product, optimize_overlap
from ..waveform_utils import get_wf_generator
from ..matrix_with_units import MatrixWithUnits
from .fisher_utils import fisher_matrix, get_waveform_derivative_1D_with_convergence
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
        Must be `'tc'` (equivalent: `'time'`), `'psi'` (equivalent:
        `'phase'`) or a key in `wf_params_at_point`.
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
    direct_computation : boolean, optional, default = True
        Whether to compute the Fisher matrix upon intialization of the
        class. Usually, this should be the preferred behaviour, but in
        certain cases one might want to save the computation time (e.g.
        if a systematic error shall be computed, where the Fisher
        matrix might be computed in some optimized point and not the one
        given by `wf_params_at_point`).

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
        # First three are chosen to match default of derivative function
        'convergence_check': 'diff_norm',
        'break_upon_convergence': True,
        'convergence_threshold': 0.001,
        'return_info': True
    }

    _preferred_unit_sys = preferred_unit_system
    # Idea: display stuff in these units, i.e. apply .to_system to each matrix before saving them
    # TODO: make setter etc. for it

    def __init__(self,
        wf_params_at_point: dict[str, u.Quantity],
        params_to_vary: str | list[str],
        wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
        direct_computation: bool = True,
        **metadata
    ) -> None:
        """Initialize a ``FisherMatrix``."""
        self.wf_params_at_point = wf_params_at_point
        if isinstance(params_to_vary, str):
            self.params_to_vary = [params_to_vary]
        else:
            self.params_to_vary = params_to_vary.copy()
        self.wf_generator = wf_generator
        self.metadata = self.default_metadata | metadata
    
        if direct_computation:
            self._calc_fisher()
            # TODO: decide if this is really needed. Could also just
            # not allow new computation if this param is False
    
    def _calc_fisher(self):
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
            self._calc_fisher()

            return self._fisher
    
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
    
    def get_param_indices(self, params: str | list[str]):
        if isinstance(params, str):
            params = [params]
        
        for param in params:
            assert param in self.params_to_vary, \
                (f'Parameter \'{param}\' was not used to calculate the Fisher '
                'matrix (which can also mean it was projected out).')
        
        # param_indices = np.where(np.isin(self.params_to_vary, params))[0]
        # param_indices = np.nonzero(np.isin(self.params_to_vary, params))[0]  # Recommended by numpy doc

        _params = np.array(params)
        _params_to_vary = np.array(self.params_to_vary)
        # param_indices = [np.argwhere(param == _params_to_vary)[0,0] for param in _params_to_vary[np.isin(_params_to_vary, _params)]]
        param_indices = [np.argwhere(param == _params_to_vary)[0,0] for param in _params[np.isin(_params, _params_to_vary)]]

        return param_indices

    def get_sub_matrix_indices(self, params: str | list[str]):
        param_indices = self.get_param_indices(params)
        index_grid = np.ix_(param_indices, param_indices)

        return index_grid

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
            we want to compute derivative around. Must be `'tc'`
            (equivalent: `'time'`), `'psi'` (equivalent: `'phase'`) or
            a key in `wf_params_at_point`.

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
        
        index_grid = self.get_sub_matrix_indices(params)

        n = len(self.params_to_vary)  # Equal to self.fisher.shape[0]

        fisher_val = self.fisher.value
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
        # for index in param_indices:
        for index in self.get_param_indices(params):
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
            param_indices = self.get_param_indices(params)
        else:
            # Take all parameters
            param_indices = len(self.params_to_vary)*[True]
        
        return MatrixWithUnits.sqrt(self.fisher_inverse.diagonal()[param_indices])
        # if len(params) != 1:
        #     # TODO: decide if this distinction makes sense... Multiple return types are meh, right?
        #     return MatrixWithUnits.sqrt(self.fisher_inverse.diagonal()[param_indices])
        # else:
        #     return np.sqrt(self.fisher_inverse.diagonal()[param_indices])

    def systematic_error(self,
        reference_wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
        params: Optional[str | list[str]] = None,
        optimize: bool | str | list[str] = True,
        optimize_fisher: Optional[str | list[str]] = None,
        return_opt_info: bool = True,
        **inner_prod_kwargs
    ) -> tuple[MatrixWithUnits, dict[str, Any]]:
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
        reference_wf_generator : Callable[[dict[str, ~astropy.units.
        Quantity]], ~gwpy.frequencyseries.FrequencySeries]
            Waveform generator for other waveform model that the
            systematic error shall be computed with respect to.
        params : str | list[str], optional, default = None
            Parameter(s) to calculate error for. In case it is None (the
            default), the error for all parameters from `self.
            params_to_vary` is calculated. Can also be a string or list
            of strings, but these have to match elements of `self.
            params_to_vary`.
        optimize : boolean | str | list[str], optional, default = True
            Option that allows the to control the optimization procedure
            that is used. Can be True or False to switch automatic
            version on and off, but also a list with parameter names
            that the optimization will be carried out for. Using True is
            recommended because it has been shown that estimates become
            more reliable in that case, so this is the default.

            If it is given as a list, the corresponding parameters must
            be in the instances ``wf_params_at_point`` dictionary or
            'time', 'phase'. They do not have to be part of `params`.
        inner_prod_kwargs : 
            Key word arguments used for the calculations here, i.e. for
            waveform difference and more.

        Returns
        -------
        tuple[MatrixWithUnits, dict[str, Any]]
            Column vector containing the systematic errors and
            dictionary with information about the calculation.
        """
        if isinstance(optimize, str):
            optimize = [optimize]
        
        if isinstance(optimize_fisher, str):
            optimize_fisher = [optimize_fisher]
        
        if len(inner_prod_kwargs) > 0:
            # Update keywords from initial input to the instance

            # Note: we want to use same kwargs for inner product as for
            # calculation of Fisher matrix here, have to be extracted from
            # metadata property
            # -> changed this now, we might want to optimize over some stuff here
            from inspect import signature
            fisher_args = list(signature(fisher_matrix).parameters)
            fisher_args.remove('inner_prod_kwargs')
            init_inner_prod_kwargs = self.metadata.copy()

            # Remove all potential arguments for Fisher, leaves inner_prod_kwargs
            for key in fisher_args:
                try:
                    init_inner_prod_kwargs.pop(key)
                except KeyError:
                    pass
            
            inner_prod_kwargs = init_inner_prod_kwargs | inner_prod_kwargs

        optimization_info = {}

        if ((opt_is_bool := isinstance(optimize, bool) and optimize)
            or isinstance(optimize, list)):
            # Order is crucial, opt_is_bool needs to be defined
            if opt_is_bool:
                opt_params = None
                
                optimization_info['general'] = 'Default optimization was carried out.'
            else:
                # Is list
                opt_params = optimize
            
                optimization_info['general'] = 'Custom optimization was carried out.'

            # Do optimization, get optimal parameters
            opt_wf_1, opt_wf_2, opt_params = optimize_overlap(
                wf_params=self.wf_params_at_point,
                fixed_wf_generator=reference_wf_generator,
                vary_wf_generator=self.wf_generator,
                opt_params=opt_params,
                **inner_prod_kwargs
            )
            delta_h = opt_wf_1 - opt_wf_2

            # TODO: remove t_shift and psi at this point (but store)

            opt_wf_params = self.wf_params_at_point | opt_params
            # Remove parameters that are not used in wf generation
            # if 'tc' in opt_params or 'time' in opt_params:
            #     tc = opt_wf_params.pop('tc', 0.*u.s) + opt_wf_params.pop('time', 0.*u.s)
            # else:
            #     tc = 0.*u.s
            # if 'psi' in opt_params or 'phase' in opt_params:
            #     psi = opt_wf_params.pop('psi', 0.*u.rad) + opt_wf_params.pop('phase', 0.*u.rad)
            # else:
            #     psi = 0.*u.rad
            tc = opt_wf_params.pop('tc', 0.*u.s) + opt_wf_params.pop('time', 0.*u.s)
            # psi = opt_wf_params.pop('psi', 0.*u.rad) + opt_wf_params.pop('phase', 0.*u.rad)
            psi = opt_wf_params.pop('psi', 0.*u.rad) + 0.5*opt_wf_params.pop('phase', 0.*u.rad)  # 0.5 for separate interpretation of phase, psi
            # TODO: maybe test if vary_wf_generator accepts the keywords? And only pop them if not

            optimization_info['opt_params'] = opt_wf_params  # Or opt_params?

            opt_fisher = FisherMatrix(
                opt_wf_params,
                self.params_to_vary,
                self.wf_generator,
                return_info=True
            )

            if optimize_fisher is not None:
                opt_fisher = opt_fisher.project_fisher(optimize_fisher)

                optimization_info['general'] += (
                    ' Fisher Matrix optimization was carried out as well.'
                )

                optimization_info['fisher_opt_params'] = optimize_fisher
            
            fisher_inverse = opt_fisher.fisher_inverse


            # logging.info(f'The optimized Fisher matrix is:\n{opt_fisher}')
            optimization_info['opt_fisher'] = opt_fisher
            

            # Need to calculate derivatives at new point
            # derivs = [
            #     get_waveform_derivative_1D_with_convergence(
            #         opt_wf_params,
            #         param_to_vary,
            #         self.wf_generator
            #     ) for param_to_vary in self.params_to_vary
            # ]
            # TODO: don't forget to add phase shifts here, have to evaluate
            # derivatives in the optimized point
            # -> but phase and time shifts would cancel out in every
            #    inner product and thus in whole Fisher, right?
            # -> and even derivatives w.r.t. phase and tc are independent
            #    of phase shifts we evaluate them in, except for dependence
            #    through the waveforms, which cancels out again in inner prod
            derivs = [
                opt_fisher.deriv_info[param]['deriv'] for param in opt_fisher.params_to_vary
            ]

            # TODO: if there was optimization over time/phase, definitely
            # account for that in derivatives!!! This is where the
            # phase term does not cancel out!!!


            for i, deriv in enumerate(derivs):
                derivs[i] = deriv * np.exp(-2.j*np.pi*deriv.frequencies*tc + 2.j*psi)
        elif isinstance(optimize, bool) and not optimize:
            delta_h = reference_wf_generator(self.wf_params_at_point) \
                - self.wf_generator(self.wf_params_at_point)

            # if self.metadata['return_info']:
            #     self.fisher  # Test call in case direct_computation was False

            #     derivs = [
            #         self.deriv_info[param]['deriv'] for param in self.params_to_vary
            #     ]
            # else:
            #     # NOTE: I don't think we can/should calculate derivs for the
            #     # parameters in params only because this argument is meant
            #     # to determine return. For error, parameters that are not in
            #     # in params still play a role and have to be accounted for.
            #     derivs = [
            #         get_waveform_derivative_1D_with_convergence(
            #             self.wf_params_at_point,
            #             param_to_vary,
            #             self.wf_generator
            #         ) for param_to_vary in self.params_to_vary
            #     ]

            # if optimize_fisher is not None:
            #     opt_fisher = self.project_fisher(optimize_fisher)
            #     fisher_inverse = opt_fisher.fisher_inverse
        
            #     optimization_info['general'] = (
            #         'No optimization of the waveform difference was done, '
            #         'but Fisher Matrix optimization was carried out.'
            #     )
            # else:
            #     fisher_inverse = self.fisher_inverse

            #     optimization_info['general'] = 'No optimization was carried out.'
            
            # opt_params = None

            if optimize_fisher is not None:
                opt_fisher = self.project_fisher(optimize_fisher)
        
                optimization_info['general'] = (
                    'No optimization of the waveform difference was done, '
                    'but Fisher Matrix optimization was carried out.'
                )

                optimization_info['fisher_opt_params'] = optimize_fisher
            else:
                opt_fisher = self

                optimization_info['general'] = 'No optimization was carried out.'
            
            fisher_inverse = opt_fisher.fisher_inverse

            if opt_fisher.metadata['return_info']:
                derivs = [
                    opt_fisher.deriv_info[param]['deriv'] for param in opt_fisher.params_to_vary
                ]
            else:
                # NOTE: I don't think we can/should calculate derivs for the
                # parameters in params only because this argument is meant
                # to determine return. For error, parameters that are not in
                # in params still play a role and have to be accounted for.
                derivs = [
                    get_waveform_derivative_1D_with_convergence(
                        opt_fisher.wf_params_at_point,
                        param_to_vary,
                        opt_fisher.wf_generator
                    ) for param_to_vary in opt_fisher.params_to_vary
                ]

            opt_params = None
        else:
            raise ValueError('Given `optimize` input not accepted.')
        
        vector = MatrixWithUnits.from_numpy_array(np.zeros((len(derivs), 1)))
        for i, deriv in enumerate(derivs):
            vector[i] = inner_product(delta_h, deriv, **inner_prod_kwargs)
        
        # Check which params shall be returned
        if params is not None:
            param_indices = opt_fisher.get_param_indices(params)
        else:
            # Take all parameters
            params = opt_fisher.params_to_vary
            param_indices = len(params)*[True]
        
        fisher_bias = (fisher_inverse @ vector)[param_indices]

        # Bias from Fisher calculation might not be the only one we have
        # to account for, some parameters might change in optimization
        # procedure (has to be taken into account as well).
        if opt_params is not None and np.any(np.isin(opt_params, params)):
            opt_bias = MatrixWithUnits.from_numpy_array(vector.shape)
            for index in np.ndindex(opt_bias.shape):
                param = param_indices[index]
                if param in params.keys():
                    # param was changed by optimization procedure
                    opt_bias = opt_params[param] - params[param]
            
            # TODO: account for correlations with parameters that are
            # not from opt_params, they might still change
            # -> but how to verify this? Maybe don't and argue that
            #    optimization is only meant to be carried out for
            #    external parameters, which fulfil this requirement of
            #    independence.

            # TODO: especially check what they mean by summation of errors after eq 13

        #     return fisher_bias + opt_bias, optimization_info
        # else:
        #     return fisher_bias, optimization_info
        # TODO: maybe only return optimization_info in case optimize != False

            fisher_bias += opt_bias
        
        if optimize is False or return_opt_info is False:
            return fisher_bias
        else:
            return fisher_bias, optimization_info

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

        if only_fisher:
            self.plot_matrix(self.fisher)
        elif only_fisher_inverse:
            self.plot_matrix(self.fisher_inverse)
        else:
            self.plot_matrix(self.fisher)
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
        `wf_generator` argument of a ``FisherMatrix`` by calling
        `~gw_signal_tools.waveform_utils.get_wf_generator`.

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
        """
        return get_wf_generator(approximant, domain, *args, **kwargs)
    

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
        new_matrix = FisherMatrix(
            self.wf_params_at_point,
            self.params_to_vary,
            self.wf_generator,
            direct_computation=False,
            **self.metadata
        )
        
        new_matrix._fisher = self.fisher.copy()
        new_matrix._fisher_inverse = self.fisher_inverse.copy()
        new_matrix._deriv_info = self.deriv_info.copy()
        new_matrix._is_projected = self.is_projected

        return new_matrix
    
    def copy(self) -> FisherMatrix:
        return self.__copy__()
