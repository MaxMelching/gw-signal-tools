import logging
import warnings
from typing import Optional, Any, Literal

import numpy as np
import matplotlib.pyplot as plt

from gwpy.types import Series
from gwpy.frequencyseries import FrequencySeries
import astropy.units as u
import lalsimulation.gwsignal.core.waveform as wfm

from .inner_product import inner_product, inner_product_computation, norm
from .waveform_utils import restrict_f_range
from .matrix_with_units import MatrixWithUnits


# class FisherMatrix:
class FisherMatrix(MatrixWithUnits):
    default_metadata = {
        ...
    }

    # def __init__(self, convergence_check: str = None):
    def __init__(self, wf_params_at_point, metadata: dict[str, Any] = None):
        # Hmmm, just collect all kwargs in metadata? Then not check for None,
        # can just do else case all the time
        if metadata is None:
            self.metadata = self.default_metadata
        else:
            self.metadata = self.default_metadata | metadata
        
        self.fisher = self.calc_fisher_matrix_at_point(wf_params_at_point, **metadata)
        ...

    # Have to do it more manually; specify with_unit=True or False in
    # getter of Fisher
    
    @property
    def fisher(self):
        return self._fisher
    
    @fisher.setter
    def fisher(self, value):
        # Definitely think about conditions/consequences that setting has
        ...
        self._fisher = value
    

    @property
    def inverse(self):
        return self.inverse

    """
    Idea: we could store Fisher matrix and inverse in here and also
    diagonal version, from which we would get condition number.
    
    -> maybe just compute Fisher and diagonal; then inform about
    condition number when inverse is called

    -> ah, condition number is norm(A)*norm(A^-1); so computing inverse
    would also be just fine (depends on norm used)


    Make property deriv_info, where starting frequency etc can be set?
    Can also set all step_sizes and resetting would probably be good
    idea (back to defaults)


    Store in pandas frame? Could be useful for units

    
    Store derivatives?
    """


    def derivative_1D():
        # Derivative with convergence

        # Also allow input like convergence_check and if None, take the
        # one defined in class -> gives possibility to play around with
        # settings without changing class attributes
        raise NotImplementedError
    
    def calc_fisher_matrix_at_point(
        wf_params_at_point: dict[str, float | u.Quantity]
    ) -> np.ndarray:
        raise NotImplementedError

    # def fisher_matrix_projected():
    def project_fisher():
        # Apply projection onto certain parameters
        raise NotImplementedError
    

    def eval_at_point():
        # Could then update Fisher etc based on that
        # -> potentially convenient because one can use a class that
        #    has been setup for multiple points
        raise NotImplementedError
    

    # def __float__():
    #     # Idea: return only matrix with values, not what is usually
    #     # returned (product of values and matrix with units)
    #     return 
    # Does not work, float() wants real float returned, not array with this dtype

    @classmethod
    def plot(self):
        ...
        # plot with color=uncertainty


def fisher_inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)

def condition_number(matrix: np.ndarray, inverse: np.ndarray) -> float:
    return np.linalg.norm(matrix) * np.linalg.norm(inverse)
