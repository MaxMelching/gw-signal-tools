from functools import cached_property
# TODO: use for some stuff?
from typing import Callable, Optional, Literal

import numpy as np
import astropy.units as u
from gwpy.frequencyseries import FrequencySeries
import matplotlib.pyplot as plt

from ..inner_product import norm, inner_product
from ..logging import logger


# TODO: decide if we make two classes, one that only calculates for
# a single step size and one with convergence. Or if we keep the
# former as special of the latter (pretty straightforward to get former
# behaviour if we have latter) -> yep, let's make a single one

class Derivative():
    """
    Calculate the derivative of an arbitrary waveform with respect to
    an arbitrary input parameter.
    """
    def __init__(
        self,
        wf_params_at_point: dict[str, u.Quantity],
        param_to_vary: str,
        wf_generator: Callable[[dict[str, u.Quantity]], FrequencySeries],
        step_sizes: Optional[list[float] | np.ndarray] = None,
        start_step_size: Optional[float] = 1e-2,
        convergence_check: Optional[Literal['diff_norm', 'mismatch']] = None,
        convergence_threshold: Optional[float] = None,
        break_upon_convergence: bool = True,
        max_refine_numb: Optional[int] = 3,
        # return_info: bool = False,
        **inner_prod_kwargs
    ) -> None:
        self.wf_params_at_point = wf_params_at_point
        self.param_to_vary = param_to_vary
        self._wf_generator = wf_generator
        # self._wf = wf_generator(wf_params_at_point)
        # self._step_size = step_size

        if step_sizes is None:
            self.step_sizes = np.reshape(np.outer([start_step_size/10**i for i in range(5)], [5, 1]), -1)[1:]
            # Indexing makes sure we do not start at 5*start_step_size
        else:
            self.step_sizes = step_sizes
        
        self.convergence_check = convergence_check
        self.convergence_threshold = convergence_threshold
        
        self.break_upon_convergence = break_upon_convergence

        self.max_refine_numb = max_refine_numb

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
        self.wf = self.wf_generator(self.wf_params_at_point)

    @property
    def param_to_vary(self):
        return self._param_to_vary
    
    @param_to_vary.setter
    def param_to_vary(self, param: str):
        if (param != 'time' and param != 'tc'
            and param != 'phase' and param != 'psi'):
            assert param in self.wf_params_at_point

        self._param_to_vary = param

    @property
    def wf_generator(self):
        return self._wf_generator
    
    @wf_generator.setter
    def wf_generator(self, generator: Callable[[dict[str, u.Quantity]], FrequencySeries]) -> None:
        self._wf_generator = generator
        self.wf = generator(self.wf_params_at_point)
    
    # @property
    # def step_size(self):
    #     return self._step_size
    @property
    def step_sizes(self):
        return self._step_size_collection[-1]
    
    @step_sizes.setter
    def step_sizes(self, steps):
        # -- Make sure _step_size_collection is already defined
        try:
            self._step_size_collection
        except AttributeError:
            self._step_size_collection = []
        
        self._step_size_collection += steps
        # TODO: do we have to make sure it is list that we append here?
    
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

    @property
    def max_refine_numb(self) -> int:
        return self._max_refine_numb
    
    @max_refine_numb.setter
    def max_refine_numb(self, num: int) -> None:
        self._max_refine_numb = int(num)
    
    # -- Internally used properties
    @property
    def param_center_val(self):
        return self.wf_params[self.param_to_vary]
    # TODO: is this required?

    # @property
    # def wf(self) -> FrequencySeries:
        # return self.wf_generator(self.wf_params_at_point)
    # But then we would loose advantage of storing it, right?
    # -> even cached_property would not be useful
    # -> maybe out own custom cacher would help here. But not used for now
    @property
    def wf(self) -> FrequencySeries:
        return self._wf
    
    @wf.setter
    def wf(self, wf: FrequencySeries) -> None:
        self._wf = wf

    @property
    def deriv(self):
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
                deriv = self.wf * 1.j
            else:
                deriv = self.wf * 2.j

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

        for _ in range(self.max_refine_numb):
            # TODO: call this index here j and then make min_dev_index
            # have ndim 2 so that it also includes the j?

            derivative_vals = []
            deriv_norms = []
            convergence_vals = []
            # TODO: make these here properties as well?

            for i, step_size in enumerate(self.step_sizes):
                try:
                    deriv_param = self.deriv_formula(
                        self.wf_params_at_point,
                        self.param_to_vary,
                        self.wf_generator,
                        step_size
                    )
                except ValueError as err:
                    err_msg = str(err)

                    if 'Input domain error' in err_msg:
                        logger.info(
                            f'{step_size} is not a valid step size for a parameter'
                            f' value of {self.param_center_val}. '
                            'Skipping this step size.'
                        )

                        # Still have to append something to lists, otherwise
                        # indices become inconsistent with step_sizes
                        derivative_vals += [0.0]
                        deriv_norms += [np.inf]
                        convergence_vals += [np.inf]
                        continue
                    else:
                        raise ValueError(err_msg)


                derivative_norm = norm(deriv_param, **self.inner_prod_kwargs)**2

                derivative_vals += [deriv_param]
                deriv_norms += [derivative_norm]


                match self.convergence_check:
                    case 'diff_norm':
                        if len(derivative_vals) >= 2:
                            convergence_vals += [
                                norm(deriv_param - derivative_vals[-2],
                                    **self.inner_prod_kwargs)/np.sqrt(derivative_norm)
                            ]
                        else:
                            convergence_vals += [np.inf]
                            continue
                    case 'mismatch':
                        # Compute mismatch, using that we already know norms
                        if len(derivative_vals) >= 2:
                            convergence_vals += [
                                1. - inner_product(
                                deriv_param,
                                derivative_vals[-2],
                                **self.inner_prod_kwargs
                            ) / np.sqrt(derivative_norm * deriv_norms[-2])
                            ]  # Index -1 is deriv_param
                        else:
                            convergence_vals += [np.inf]
                            continue

                # if (convergence_vals[-1] <= convergence_threshold):
                    # We use five-point stencil, which converges fast, so
                    # that it is justified to interpret two consecutive
                    # results being very similar as convergence
                    # -> testing revealed that criterion below leads to more
                    #    consistent results, thus we leave for now
                if (len(convergence_vals) >= 2
                    and (convergence_vals[-1] <= self.convergence_threshold)
                    and (convergence_vals[-2] <= self.convergence_threshold)):
                    # Double checking is more robust
                    self.is_converged = True  # Remains true, is never set to False again

                    if self.break_upon_convergence:
                        min_dev_index = i  # Then it can also be used to access step_sizes
                        break
            
            
            # Check if step sizes shall be refined. This is be done if no breaking
            # upon convergence is wanted or if no convergence was reached yet
            if not self.break_upon_convergence or not self.is_converged:
                # TODO: remove break_upon_convergence and just handle that via convergence_threshold?
                # I.e. set to 0.0 if no breaking wanted
                
                if np.all(np.equal(convergence_vals, np.inf)):
                    # Only invalid step sizes for this parameter, we have to
                    # decrease further
                    min_dev_index = len(step_sizes) - 1
                else:
                    min_dev_index = np.nanargmin(convergence_vals)  # type: ignore
                # Explanation of ignore: it seems like a signedinteger is returned
                # by nanargmin, violates static checking for int. Note that we do
                # use nan-version here just in case something goes wrong in norm or
                # so, making it zero (should not happen, though)

                # Cut steps made around step size with best criterion value in half
                # compared to current steps (we take average step size in case
                # difference to left and right is unequl)
                if min_dev_index < (len(step_sizes) - 1):
                    left_step = (step_sizes[min_dev_index - 1] - step_sizes[min_dev_index]) / 4.0
                    right_step = (step_sizes[min_dev_index + 1] - step_sizes[min_dev_index]) / 4.0
                    # 4.0 due to factor of two in step_sizes below

                    step_sizes = step_sizes[min_dev_index] + np.array(
                        [2.*left_step, 1.*left_step, 1.*right_step, 2.*right_step]
                    )
                    # TODO: also include 0.0 here? I.e. the optimal one, as of now?
                else:
                    # Smallest convergence value at smallest step size, so
                    # min_dev_index + 1 is invalid index. Instead of zooming in,
                    # smaller step sizes are explored

                    # Refine in same way that we do with start_step_size
                    step_sizes = np.reshape(np.outer([step_sizes[min_dev_index]/10**i for i in range(4)], [5, 1]), -1)[1:]  # Indexing makes sure we do not start at 5*start_step_size


                self.refine_numb += 1
            else:
                break

        if not self.is_converged:
            logger.info(
                'Calculations using the selected step sizes did not converge '
                f'for parameter `{self.param_to_vary}` using convergence check method '
                f'`{self.convergence_check}`, even after {self.refine_numb} refinements of '
                'step sizes. The minimal value of the criterion was '
                f'{convergence_vals[min_dev_index]}, ' + ((f'which is above the '
                f'selected threshold of {self.convergence_threshold}. ')
                if convergence_vals[min_dev_index] > self.convergence_threshold else (
                f'which is below the selected threshold of {self.convergence_threshold}'
                ', but the previous and following value were not.')) +
                'If you are not satisfied with the result (for an eye test, you '
                'can plot the `convergence_plot` value returned in case '
                '`return_info=True`), consider changing the initial step sizes.'
            )
        
        return derivative_vals[min_dev_index]
    
    # Idea: this is what we call and what actually returns the derivative
    # TODO: maybe we can create fancy __call__ logic? Where params_at_point are passed?
    # -> could make this optional argument in __init__ then? Nah, this
    # is not possible. 
    def __call__(self, new_point: Optional[dict[str, u.Quantity]] = None) -> np.Any:
        if new_point is not None:
            self.wf_params_at_point = new_point
        return self.deriv
    

    # def test_point(self):
    def test_point(self, steps):
        if self.param_to_vary == 'mass_ratio':
            # Test for possible issues with parameter
            # -> potentially change deriv_formula based on that
            ...
    

    @property
    def deriv_formula(self) -> str:
        return self._deriv_formula
    
    @deriv_formula.setter
    def deriv_formula(self, formula: str) -> None:
        # TODO: check for valid one?
        self._deriv_formula = formula

    def forward_difference(self):
        return NotImplemented
    
    def backward_difference(self):
        return NotImplemented
    
    def central_difference(self):
        return NotImplemented
    
    def five_point_stencil(self):
        return NotImplemented

    # -- Information related properties
    def deriv_info(self):
        # return certain properties. These should also be accessible
        # using self.
        return {
            ...
        }
    
    # TODO: maybe write getattr and there we try to return deriv_info[attr]?
    # Because this function is only called if attribute is not available,
    # 
    
    def convergence_plot(self):
        derivative_vals = []
        last_used_step_sizes = []

        fig = plt.figure()
        ax = fig.subplots(nrows=2, sharex=True)

        for i in range(len(derivative_vals)):
            ax[0].plot(derivative_vals[i].real, '--',
                       label=f'{self.step_sizes[self.refine_numb][i]:.3e}')
            ax[1].plot(derivative_vals[i].imag, '--')
            # No label for second because otherwise, everything shows up twice
            # in figure legend

        fig.legend(
            title='Step Sizes',
            bbox_to_anchor=(0.96, 0.5),
            loc='center left'
        )
        
        fig.suptitle(f'Parameter: {self.param_to_vary}')  # TODO: use latexparams here?
        ax[1].set_xlabel('$f$')
        ax[0].set_ylabel('Derivative Re')
        ax[1].set_ylabel('Derivative Im')

        return ax
