from functools import cached_property
# TODO: use for some stuff?
from typing import Callable, Optional, Literal, Any

import numpy as np
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import matplotlib as mpl

from .inner_product import norm, inner_product
from ..logging import logger


class Derivative():  # TODO: find better name, kind of interferes with numdifftools one
    """
    Calculate the derivative of an arbitrary waveform with respect to
    an arbitrary input parameter (in both frequency and time domain).
    """
    def __init__(
        self,
        wf_params_at_point: dict[str, u.Quantity],
        param_to_vary: str,
        wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries | TimeSeries],
        step_sizes: Optional[list[float] | np.ndarray] = None,
        start_step_size: Optional[float] = 1e-2,
        convergence_check: Optional[Literal['diff_norm', 'mismatch']] = None,
        convergence_threshold: Optional[float] = None,
        break_upon_convergence: bool = True,
        max_refine_numb: Optional[int] = 3,
        double_convergence: bool = True,  # Whether to demand double convergence or not
        deriv_formula: str = 'five_point',
        # return_info: bool = False,
        **inner_prod_kwargs
    ) -> None:
        self.wf_params_at_point = wf_params_at_point
        self.param_to_vary = param_to_vary
        self.wf_generator = wf_generator

        if step_sizes is None:
            self.step_sizes = np.reshape(np.outer([start_step_size/10**i for i in range(5)], [5, 1]), -1)[1:]
            # Indexing makes sure we do not start at 5*start_step_size
        else:
            if isinstance(step_sizes, float):
                step_sizes = [step_sizes]
            
            if len(step_sizes) == 1:
                self.max_refine_numb = max_refine_numb = 1

            self.step_sizes = step_sizes
        
        self.convergence_check = convergence_check
        self.convergence_threshold = convergence_threshold
        
        self.break_upon_convergence = break_upon_convergence
        self.double_convergence = double_convergence

        self.max_refine_numb = max_refine_numb

        self.deriv_formula = deriv_formula

        self.inner_prod_kwargs = inner_prod_kwargs
        self.inner_prod_kwargs['return_opt_info'] = False
        # Ensure float output of inner_product, even if optimization on

    # -- Properties that are set based on input
    @property
    def wf_params_at_point(self) -> dict[str, u.Quantity]:
        return self._wf_params_at_point
    
    @wf_params_at_point.setter
    def wf_params_at_point(self, point: dict[str, u.Quantity]) -> None:
        self._wf_params_at_point = point
        try:
            self.wf = self.wf_generator(self.wf_params_at_point)
        except AttributeError:
            # Class has just been initialized, no wf_generator yet
            pass

        if hasattr(self, '_deriv'):
            # -- Derivative that was calculated is not valid anymore
            del self._deriv

    @property
    def param_to_vary(self):
        return self._param_to_vary
    
    @param_to_vary.setter
    def param_to_vary(self, param: str):
        if (param != 'time' and param != 'tc'
            and param != 'phase' and param != 'psi'):
            # TODO: could also add distance here. Has analytical
            # derivative too, so strictly speaking it is not required
            # to be in wf_params_at_point
            assert param in self.wf_params_at_point

        self._param_to_vary = param

        if hasattr(self, '_deriv'):
            # -- Derivative that was calculated is not valid anymore
            del self._deriv

    @property
    def wf_generator(self):
        return self._wf_generator
    
    @wf_generator.setter
    def wf_generator(self, generator: Callable[[dict[str, u.Quantity]], FrequencySeries | TimeSeries]) -> None:
        self._wf_generator = generator
        self.wf = generator(self.wf_params_at_point)

        if hasattr(self, '_deriv'):
            # -- Derivative that was calculated is not valid anymore
            del self._deriv
    
    # @property
    # def step_size(self):
    #     return self._step_size
    # @property
    # def step_sizes(self):
    #     return self._step_size_collection[-1]
    
    # @step_sizes.setter
    # def step_sizes(self, steps):
    #     # -- Make sure _step_size_collection is already defined
    #     try:
    #         self._step_size_collection
    #     except AttributeError:
    #         self._step_size_collection = []
        
    #     self._step_size_collection += steps
    #     # TODO: do we have to make sure it is list that we append here?

    @property
    def step_sizes(self) -> dict[int, list[int]]:
        return self._step_sizes
    
    @step_sizes.setter
    def step_sizes(self, step_sizes: list[int]):
        self._step_sizes = step_sizes
    
    @property
    def convergence_check(self) -> str:
        return self._convergence_check
    
    @convergence_check.setter
    def convergence_check(self, convergence_check: Optional[str]) -> None:
        if convergence_check is None:
            convergence_check = 'diff_norm'
        else:
            if convergence_check not in ['mismatch', 'diff_norm']:
                raise ValueError(
                        'Invalid value for `convergence_check`.'
                    )
        
        self._convergence_check = convergence_check

        if hasattr(self, '_deriv'):
            # -- Derivative that was calculated is not valid anymore
            del self._deriv
    
    @property
    def convergence_threshold(self) -> float:
        return self._convergence_threshold
    
    @convergence_threshold.setter
    def convergence_threshold(self, convergence_threshold: Optional[float]) -> None:
        if convergence_threshold is None:
            match self.convergence_check:
                case 'diff_norm':
                    convergence_threshold = 0.001
                case 'mismatch':
                    convergence_threshold = 0.001
        
        self._convergence_threshold = convergence_threshold

        if hasattr(self, '_deriv'):
            # -- Derivative that was calculated is not valid anymore
            del self._deriv

    @property
    def max_refine_numb(self) -> int:
        return self._max_refine_numb
    
    @max_refine_numb.setter
    def max_refine_numb(self, num: int) -> None:
        self._max_refine_numb = int(num)

        if hasattr(self, '_deriv'):
            # -- Derivative that was calculated is not valid anymore
            del self._deriv
    
    # -- Internally used properties
    @property
    def param_center_val(self):
        return self.wf_params_at_point[self.param_to_vary]
    # TODO: is this required? -> yup, fairly frequently accessed

    # @property
    # def wf(self) -> FrequencySeries:
        # return self.wf_generator(self.wf_params_at_point)
    # But then we would loose advantage of storing it, right?
    # -> even cached_property would not be useful
    # -> maybe out own custom cacher would help here. But not used for now
    @property
    def wf(self) -> FrequencySeries | TimeSeries:
        return self._wf
    
    @wf.setter
    def wf(self, wf: FrequencySeries | TimeSeries) -> None:
        self._wf = wf

    @property
    def deriv(self):
        if hasattr(self, '_deriv'):
            # -- Derivative was already computed, just return. We make
            # -- sure that all relevant settings remained the same in
            # -- the corresponding setters, otherwise _deriv deleted
            return self._deriv
        
        # -- Check if parameter has analytical derivative
        if (self.param_to_vary == 'time' or self.param_to_vary == 'tc'):
            deriv = self.wf * (-1.j * 2. * np.pi * self.wf.frequencies)

            derivative_norm = norm(deriv, **self.inner_prod_kwargs)**2

            self.deriv_info = {
                'norm_squared': derivative_norm,
                'description': 'This derivative is exact.'
            }
            return deriv
        elif (self.param_to_vary == 'phase' or self.param_to_vary == 'psi'):
            if self.param_to_vary == 'phase':
                deriv = self.wf * 1.j / u.rad
            else:
                deriv = self.wf * 2.j / u.rad

            derivative_norm = norm(deriv, **self.inner_prod_kwargs)**2

            self.deriv_info = {
                'norm_squared': derivative_norm,
                'description': 'This derivative is exact.'
            }
            return deriv
        elif self.param_to_vary == 'distance':
            deriv = (-1./self.wf_params_at_point['distance']) * self.wf

            derivative_norm = norm(deriv, **self.inner_prod_kwargs)**2

            self.deriv_info = {
                'norm_squared': derivative_norm,
                'description': 'This derivative is exact.'
            }
            return deriv
        
        # TODO: automatically adjust deriv_formula here if needed
        # -> of course with logger.info output

        self.is_converged = False
        self.refine_numb = 0

        for self.refine_numb in range(self.max_refine_numb):
            # -- Initialize value storage
            self._derivative_vals = []
            self._deriv_norms = []
            self._convergence_vals = []
            # TODO: then maybe we should reset the step sizes as well?

            self._iterate_through_step_sizes()
            
            # Check if step sizes shall be refined. This is be done if no breaking
            # upon convergence is wanted or if no convergence was reached yet
            if not self.break_upon_convergence or not self.is_converged:
                # TODO: remove break_upon_convergence and just handle that via convergence_threshold?
                # I.e. set to 0.0 if no breaking wanted
                
                if np.all(np.equal(self._convergence_vals, np.inf)):
                    # Only invalid step sizes for this parameter, we
                    # have to decrease further
                    self.min_dev_index = len(self.step_sizes) - 1
                else:
                    self.min_dev_index = np.nanargmin(self._convergence_vals)  # type: ignore
                # Explanation of ignore: it seems like a signedinteger is returned
                # by nanargmin, violates static checking for int. Note that we do
                # use nan-version here just in case something goes wrong in norm or
                # so, making it zero (should not happen, though)

                if self.refine_numb < (self.max_refine_numb - 1):
                    self._update_step_sizes()
            else:
                break

        if not self.is_converged:
            logger.info(
                'Calculations using the selected step sizes did not converge '
                f'for parameter `{self.param_to_vary}` using convergence check method '
                f'`{self.convergence_check}`, even after {self.refine_numb} refinements of '
                'step sizes. The minimal value of the criterion was '
                f'{self._convergence_vals[self.min_dev_index]}, ' + ((f'which is above the '
                f'selected threshold of {self.convergence_threshold}. ')
                if self._convergence_vals[self.min_dev_index] > self.convergence_threshold else (
                f'which is below the selected threshold of {self.convergence_threshold}'
                ', but the previous and following value were not.')) +
                'If you are not satisfied with the result (for an eye test, you '
                'can plot the `convergence_plot` value returned in case '
                '`return_info=True`), consider changing the initial step sizes.'
            )
        
        self.deriv_info = {
            'norm_squared': self._deriv_norms[self.min_dev_index],
            'final_step_size': self.step_sizes[self.min_dev_index],
            'final_convergence_val': self._convergence_vals[self.min_dev_index],
            'number_of_refinements': self.refine_numb,
            'final_set_of_step_sizes': self.step_sizes
        }
        
        self._deriv = self._derivative_vals[self.min_dev_index]
        return self._deriv    
    
    def _check_converged(self):
        """
        Check if derivative has converged, according to the selected
        convergence check (either two or three consecutive values of
        convergence checker must be below `self.convergence_threshold`).
        """
        if self.double_convergence:
            if (len(self._convergence_vals) >= 2
                and (self._convergence_vals[-1] <= self.convergence_threshold)
                and (self._convergence_vals[-2] <= self.convergence_threshold)):
                # Double checking is more robust
                self.is_converged = True  # Remains true, is never set to False again
        else:
            if (self._convergence_vals[-1] <= self.convergence_threshold):
                self.is_converged = True  # Remains true, is never set to False again
                # We use five-point stencil, which converges fast, so
                # that it is justified to interpret two consecutive
                # results being very similar as convergence
                # -> testing revealed that double_convergence leads to more
                #    consistent results, thus we leave for now


    def _update_step_sizes(self):
        """
        Calculate new set of step sizes based on current ones. These
        will be centered aroud the step size where the current minimal
        deviation between two step sizes occurred.
        """
        # Cut steps made around step size with best criterion value in half
        # compared to current steps (we take average step size in case
        # difference to left and right is unequal)
        current_steps = self.step_sizes
        current_center_step = self.step_sizes[self.min_dev_index]

        # TODO: account for new definition of min_dev_index
        if self.min_dev_index < (len(current_steps) - 1):
            left_step = (current_steps[self.min_dev_index - 1]
                         - current_center_step) / 4.
            right_step = (current_steps[self.min_dev_index + 1]
                          - current_center_step) / 4.
            # -- 4. due to factor of two in step_sizes below

            self.step_sizes = current_center_step + np.array(
                [2.*left_step, 1.*left_step, 1.*right_step, 2.*right_step]
            )
            # TODO: also include 0.0 here? I.e. the optimal one, as of now?
        else:
            # Smallest convergence value at smallest step size, so
            # min_dev_index + 1 is invalid index. Instead of zooming in,
            # smaller step sizes are explored

            # Refine in same way that we do with start_step_size
            self.step_sizes = np.reshape(
                np.outer(
                    [current_center_step/10**i for i in range(4)],
                    [5, 1]
                ),
                -1
            )[1:]  # Indexing makes sure we do not start at 5*start_step_size

    
    def _iterate_through_step_sizes(self):
        """
        Calculate derivatives for current `self.step_sizes`, checking
        if values converge in the meantime.
        """
        # for i, step_size in enumerate(self.step_sizes[-1]):
        for i, step_size in enumerate(self.step_sizes):
            try:
                deriv_param = self.deriv_routine(step_size)
            except ValueError as err:
                err_msg = str(err)

                if 'Input domain error' in err_msg:
                    logger.info(
                        f'{step_size} is not a valid step size for a parameter'
                        f' value of {self.param_center_val}. '
                        'Skipping this step size.'
                    )

                    # TODO: call test_point here, where new deriv_routine is
                    # set. Then maybe call deriv_routine again (?)
                    self.test_point(step_size)

                    # Still have to append something to lists, otherwise
                    # indices become inconsistent with step_sizes
                    self._derivative_vals += [0.0]
                    self._deriv_norms += [np.inf]
                    self._convergence_vals += [np.inf]
                    continue
                else:
                    raise ValueError(err_msg)

            derivative_norm = norm(deriv_param, **self.inner_prod_kwargs)**2

            self._derivative_vals += [deriv_param]
            self._deriv_norms += [derivative_norm]

            self._calc_convergence_val()
            self._check_converged()

            if self.is_converged and self.break_upon_convergence:
                self.min_dev_index = i  # Then it can also be used to access step_sizes
                break

    def _calc_convergence_val(self):
        """
        Calculates the value of the criterion `self.convergence_check`
        for the current values in `self._derivative_vals`.
        """
        if len(self._derivative_vals) >= 2:
            match self.convergence_check:
                case 'diff_norm':
                    crit_val = norm(
                        self._derivative_vals[-1] - self._derivative_vals[-2],
                        **self.inner_prod_kwargs
                    ) / np.sqrt(self._deriv_norms[-1])
                case 'mismatch':
                    # Compute mismatch, using that we already know norms
                    crit_val = 1. - inner_product(
                        self._derivative_vals[-1],
                        self._derivative_vals[-2],
                        **self.inner_prod_kwargs
                    ) / np.sqrt(self._deriv_norms[-1] * self._deriv_norms[-2])
        else:
            crit_val = np.inf
        
        self._convergence_vals += [crit_val]
            
    
    # Idea: this is what we call and what actually returns the derivative
    # TODO: maybe we can create fancy __call__ logic? Where params_at_point are passed?
    # -> could make this optional argument in __init__ then? Nah, this
    # is not possible. 
    def __call__(self, new_point: Optional[dict[str, u.Quantity]] = None) -> FrequencySeries | TimeSeries:
        if new_point is not None:
            self.wf_params_at_point = new_point
        return self.deriv
    

    def test_point(self, step_size: float):
        """
        Check if `self.wf_params_at_point` contains potentially tricky
        values, e.g. mass ratios close to 1. If yes, a subsequent
        adjustment takes place.

        Parameters
        ----------
        step_size : float
            Current step size that produced an 'Input domain error'.
        """
        step_size = self.abs_or_rel_step_size(step_size)
        # -- This is important, determines step size that is actually
        # -- used by the routine (also adds proper unit)

        # TODO: maybe store self.param_center_val in param_val? Would
        # make lots of stuff shorter and would make central access to value easier

        # TODO: check if we need checks whether or not certain formula
        # is used at the moment

        # TODO: I think we should rather check for 2.*step_size, right?
        # Because no failure for 1.* could still mean failure for 2.*
        # -> but maybe this would be resolved in next iteration...
        if self.param_to_vary == 'mass_ratio':
            # if self.param_center_val <= 1. and (self.param_center_val + step_size > 1.):
            #     self.deriv_formula = self.backward
            # elif self.param_center_val > 1. and (self.param_center_val - step_size < 1.):
            #     self.deriv_formula = self.forward
            # elif self.param_center_val > 0. and (self.param_center_val - step_size < 0.):
            #     self.deriv_formula = self.forward

            if self.param_center_val <= 1.:
                # -- q <= 1, but adding step_size makes > 1
                if ((self.param_center_val + 2.*step_size > 1.)
                    or (self.param_center_val + step_size > 1.)):
                    self.deriv_formula = self.backward
            else:
                if ((self.param_center_val - 2.*step_size < 1.)
                    or (self.param_center_val - step_size < 1.)):
                    # -- q > 1, but subtracting step_size makes < 1
                    self.deriv_formula = self.forward
                elif ((self.param_center_val - 2.*step_size < 0.)
                      or (self.param_center_val - step_size < 0.)):
                    # -- q close to 0, subtracting step_size makes < 0
                    self.deriv_formula = self.forward
        elif self.param_to_vary == 'sym_mass_ratio':
            # TODO: check unit here!
            if ((self.param_center_val + 2.*step_size > 0.25)
                or (self.param_center_val + step_size > 0.25)):
                # -- nu close to 0.25, adding step_size makes > 0.25
                self.deriv_formula = self.backward
            elif ((self.param_center_val - 2.*step_size < 0.)
                  or (self.param_center_val - step_size < 0.)):
                # -- nu close to 0, subtracting step_size makes < 0
                self.deriv_formula = self.forward

        # TODO: what other parameters are relevant in this regard?
        # Maybe spins? Inclination probably?
    

    @property
    def deriv_formula(self) -> str:
        """
        (Function) name of the derivative formula that is used.

        :type: `str`
        """
        return self._deriv_formula
    
    @deriv_formula.setter
    def deriv_formula(self, formula: str) -> None:
        # -- Check for valid formula, then set it
        # assert formula in self.__dict__, (
        #     f'Invalid formula name {formula} is given. Available options are '
        #     '`forward`, `backward`, `central`, `five_point` or any custom '
        #     'attribute that might be set by you.'
        # )
        # TODO: or rather make class attribute _allowed_deriv_routines
        # where we store these default ones? And people can inherit
        # from class and then add their own names + attribute to that

        # TODO: shit, actually does not work

        self._deriv_formula = formula

        if hasattr(self, '_deriv'):
            # -- Derivative that was calculated is not valid anymore
            del self._deriv
    
    def deriv_routine(self, *args, **kw_args):
        """Caller that allows access to currently set derivative formula."""
        # return self.__getattribute__(self.deriv_formula)(self, *args, **kw_args)
        # return self.__getattribute__(self.deriv_formula)(*args, **kw_args)
        # -- using __call__ would perhaps make more clear what happens
        # return self.__getattribute__(self.deriv_formula).__call__(self, *args, **kw_args)
        return self.__getattribute__(self.deriv_formula).__call__(*args, **kw_args)

    def abs_or_rel_step_size(self, step_size):
        """
        Choose relative or absolute step size, based on
        `self.param_center_val` (the value of `self.param_to_vary` in
        `self.wf_params_at_point`).
        """
        if np.log10(self.param_center_val.value) < 1:
            step_size = np.abs(u.Quantity(step_size,
                                          unit=self.param_center_val.unit))
        else:
            step_size = np.abs(u.Quantity(step_size * self.param_center_val,
                                          unit=self.param_center_val.unit))
        
        return step_size


    # TODO: check again if this implementation is really most efficient.
    # Something like "return (wf2 - wf1) / step_size" seems like nice option too

    def forward(self, step_size: float) -> FrequencySeries | TimeSeries:
        """
        Calculates the forward difference of `self.wf_generator` at
        `self.wf_params_at_point` with respect to `self.param_to_vary`
        using the given `step_size`.
        """
        step_size = self.abs_or_rel_step_size(step_size)
        wf_p1 = self.wf_generator(self.wf_params_at_point | {
            self.param_to_vary: self.param_center_val + step_size
        })

        deriv_series = wf_p1 - self.wf
        deriv_series /= step_size

        return deriv_series
    
    def backward(self, step_size: float) -> FrequencySeries | TimeSeries:
        """
        Calculates the backward difference of `self.wf_generator` at
        `self.wf_params_at_point` with respect to `self.param_to_vary`
        using the given `step_size`.
        """
        step_size = self.abs_or_rel_step_size(step_size)
        wf_m1 = self.wf_generator(self.wf_params_at_point | {
            self.param_to_vary: self.param_center_val - step_size
        })

        deriv_series = self.wf - wf_m1
        deriv_series /= step_size

        return deriv_series
    
    def central(self, step_size: float) -> FrequencySeries | TimeSeries:
        """
        Calculates the central difference of `self.wf_generator` at
        `self.wf_params_at_point` with respect to `self.param_to_vary`
        using the given `step_size`.
        """
        step_size = self.abs_or_rel_step_size(step_size)
        param_vals = self.param_center_val + np.array([-1., 1.])*step_size

        waveforms = [
            self.wf_generator(
                self.wf_params_at_point | {self.param_to_vary: param_val}
            ) for param_val in param_vals
        ]

        deriv_series = waveforms[1] - waveforms[0]
        deriv_series /= 2.*step_size

        return deriv_series
    
    def five_point(self, step_size: float) -> FrequencySeries | TimeSeries:
        """
        Calculates the five point stencil of `self.wf_generator` at
        `self.wf_params_at_point` with respect to `self.param_to_vary`
        using the given `step_size`.
        """
        step_size = self.abs_or_rel_step_size(step_size)
        param_vals = self.param_center_val + np.array([-2., -1., 1., 2.])*step_size

        waveforms = [
            self.wf_generator(
                self.wf_params_at_point | {self.param_to_vary: param_val}
            ) for param_val in param_vals
        ]

        deriv_series = (waveforms[0] - 8.*waveforms[1]
                        + 8.*waveforms[2] - waveforms[3])
        deriv_series /= 12.*step_size

        return deriv_series


    # -- Information related properties
    @property
    def deriv_info(self) -> dict[str, Any]:
        """
        Information about the calculated derivative, given as a
        dictionary. All keys from this dictionary are also accessible
        as a class attribute.
        """
        return self._deriv_info
    
    @deriv_info.setter
    def deriv_info(self, info: dict[str, Any]):
        for key, val in info.items():
            # -- Make each key from deriv_info available as attribute
            setattr(self, key, val)
        
        self._deriv_info = info
    

    def convergence_plot(self) -> mpl.axes.Axes:
        from ..plotting import latexparams
        # TODO: maybe make sure derivative has been calculated? Maybe
        # check length of self._derivative_vals
        fig = plt.figure()
        ax = fig.subplots(nrows=2, sharex=True)

        for i, deriv_val in enumerate(self._derivative_vals):
            ax[0].plot(
                deriv_val.real,
                '--',
                label=f'{self.step_sizes[self.refine_numb][i]:.3e}'
            )
            ax[1].plot(deriv_val.imag, '--')
            # No label for second because otherwise, everything shows up
            # twice in figure legend

        fig.legend(
            title='Step Sizes',
            bbox_to_anchor=(0.96, 0.5),
            loc='center left'
        )
        
        fig.suptitle(f'Parameter: {latexparams.get(self.param_to_vary, self.param_to_vary)}')
        if isinstance(deriv_val, TimeSeries):
            ax[1].set_xlabel(rf'$t$ [{deriv_val.xindex.unit:latex}]')
        else:
            ax[1].set_xlabel(rf'$f$ [{deriv_val.xindex.unit:latex}]')

        ax[0].set_ylabel('Derivative Re')
        ax[1].set_ylabel('Derivative Im')

        return ax
