# ----- Make all important functions/classes from module available here -----
from .fisher_utils import (
    num_diff, get_waveform_derivative_1D, 
    get_waveform_derivative_1D_with_convergence,
    fisher_matrix
)
from .fisher import FisherMatrix
# from .matrix_with_units import MatrixWithUnits  # Do this? 
