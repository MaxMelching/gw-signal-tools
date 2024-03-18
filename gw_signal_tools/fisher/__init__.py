# ----- Make all important functions/classes from module available here -----
from .fisher_utils import (
    num_diff, get_waveform_derivative_1D, 
    get_waveform_derivative_1D_with_convergence,
    fisher_matrix
)
from .fisher import FisherMatrix
# from .matrix_with_units import MatrixWithUnits  # Do this? 

phenomx_generator = FisherMatrix.get_wf_generator(
    'IMRPhenomXPHM'
)

# Dictionary fo get nicer display of parameters in Fisher plotting method
latexparams = {
    # Masses
    'chirp_mass_source': r'$\mathcal{M}$',
    'chirp_mass': r'$\mathcal{M}$',
    'mass_ratio': '$q$',
    'total_mass': '$M$',
    'total_mass_source': '$M$',
    'mass_1_source': '$m_1$',
    'mass_1': '$m_1$',
    'mass_2_source': '$m_2$',
    'mass_2': '$m_2$',
    # Spins
    'chi_eff': r'$\chi_{\mathrm{eff}}$',
    'chi_p': r'$\chi_p$',
    'a_1': r'$\chi_1$',
    'spin1x': r'$\chi_{1, x}$',
    'spin1y': r'$\chi_{1, y}$',
    'spin1z': r'$\chi_{1, z}$',
    'a_2': r'$\chi_2$',
    'spin2x': r'$\chi_{2, x}$',
    'spin2y': r'$\chi_{2, y}$',
    'spin2z': r'$\chi_{2, z}$',
    # External Parameters
    'luminosity_distance': '$D_L$',
    'distance': '$D_L$',
    'theta_jn': r'$\theta_{jn}$',
    'inclination': r'$\theta_{jn}$',
    'iota': r'$\iota$',
    # Detector parameters
    'log_likelihood': r'$\log\mathcal{L}$',
    'network_optimal_snr': r'$\rho_{\mathrm{opt}}$ (network)',
    'network_matched_filter_snr': r'$\rho_{\mathrm{matched filter}}$ (network)'
}
