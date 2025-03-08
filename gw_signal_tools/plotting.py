latexparams: dict[str, str] = {
    # -- Masses
    'chirp_mass_source': r'$\mathcal{M}$',
    'chirp_mass': r'$\mathcal{M}^\mathrm{det}$',
    'mass_ratio': r'$q$',
    'sym_mass_ratio': r'$\eta$',
    'total_mass': r'$M^\mathrm{det}$',
    'total_mass_source': r'$M$',
    'mass_1_source': r'$m_1$',
    'mass_1': r'$m_1^\mathrm{det}$',
    'mass1': r'$m_1^\mathrm{det}$',
    'mass_2_source': r'$m_2$',
    'mass_2': r'$m_2^\mathrm{det}$',
    'mass2': r'$m_2^\mathrm{det}$',
    # -- Spins
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
    'phi_1': r'$\phi_1$',
    'phi_2': r'$\phi_2$',
    'tilt_1': r'$\theta_1$',
    'tilt_2': r'$\theta_2$',
    'theta_1': r'$\theta_1$',
    'theta_2': r'$\theta_2$',
    'phi_jl': r'$\phi_\mathrm{jl}$',
    'phi_12': r'$\phi_{12}$',
    # -- External Parameters
    'luminosity_distance': r'$D_L$',
    'distance': r'$D_L$',
    'theta_jn': r'$\theta_{jn}$',
    # 'inclination': r'$\theta_{jn}$',
    'inclination': r'$\iota$',
    'iota': r'$\iota$',
    'time': '$t_0$',
    'tc': r'$t_c$',  # Basically alias for time
    'tgps': r'$t_\mathrm{gps}$',
    'geocent_time': r'$t_\mathrm{gps}$',
    'phase': r'$\phi_0$',
    'ra': r'$\mathrm{ra}$',
    'dec': r'$\mathrm{dec}$',
    'psi': r'$\psi$',
    'phi_ref': r'$\phi_{\mathrm{ref}}$',
    # -- Detector parameters
    'log_likelihood': r'$\log\mathcal{L}$',
    'network_optimal_snr': r'$\rho_{\mathrm{opt}}$ (network)',
    'network_matched_filter_snr': r'$\rho_{\mathrm{matched filter}}$ (network)',
}


for ang_param in [
    'inclination',
    'iota',
    'theta_jn',
    'phi_jl',
    'phi_12',
    'phi_1',
    'phi_2',
    'tilt_1',
    'tilt_2',
    'ra',
    'dec',
    'psi',
]:
    cos_param = 'cos_' + ang_param
    latexparams[cos_param] = r'$\cos ' + latexparams[ang_param][1:]  # Cut off first $

    sin_param = 'sin_' + ang_param
    latexparams[sin_param] = r'$\sin ' + latexparams[ang_param][1:]  # Cut off first $

for large_val_param in [
    'chirp_mass',
    'chirp_mass_source',
    'total_mass',
    'total_mass_source',
    'mass1',
    'mass_1',
    'mass_1_source',
    'mass2',
    'mass_2',
    'mass_2_source',
    'distance',
    'luminosity_distance',
]:
    log_param = 'log_' + large_val_param
    latexparams[log_param] = r'$\log ' + latexparams[large_val_param][1:]
