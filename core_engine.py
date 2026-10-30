"""

CER MASTER ENGINE: Core Cosmological Fluid Substrate Model

Tracks flat late-time expansion history kernels accurately.

"""

import numpy as np
"""

    def get_compton_rate(self, z, x_e):
        """Calculates the Compton scattering timescale coefficient (gamma)."""
        Tr = self.T_cmb_0 * (1.0 + z)
        
        # Fundamental physical constants in SI
        sigma_T = 6.6524587e-29       # Thomson cross section (m^2)
        m_e = 9.1093837e-31           # Electron mass (kg)
        a_R = 7.565733e-16            # Radiation constant (J/m^3/K^4)
        f_He = 0.0797                 # Standard Helium-to-Hydrogen number ratio
        
        # Energy density of the CMB photon bath
        rho_photon = a_R * (Tr ** 4.0)
        
        # Compton drag factor per electron particle
        base_rate = (8.0 * sigma_T * rho_photon * self.c) / (3.0 * m_e * self.c**2)
        
        # Scale rate factor by the total ionized particle availability
        ionization_weight = x_e / (1.0 + f_He + x_e)
        return base_rate * ionization_weight

    def evaluate_system_derivatives(self, z, state_vector):
        """
        Simultaneously computes [dxe/dz, dTm/dz] for the pipeline.
        Pipes optimized H0 and Omega_vac values from your cosmic chronometer fit.
        """
        x_e, T_m = state_vector
        Tr = self.T_cmb_0 * (1.0 + z)
        
        # Guard rail system states 
        x_e = np.clip(x_e, 1e-6, 1.0)
        
        # 1. Calculate Ionization Population Derivative (dx_e/dz)
        if x_e >= 0.9999:
            dxe_dz = 0.0
        else:
            nH_z = self.nH0_cm3 * (1.0 + z)**3
            x_2s, x_2p, A_2s, A_2p, B_2s, B_2p = self.solve_qss_populations(z, x_e, T_m, Tr)
            dx_e_dt = (x_2s * B_2s - nH_z * (x_e**2) * A_2s) + (x_2p * B_2p - nH_z * (x_e**2) * A_2p)
            H_z = get_flat_cer_hz(z, self.H0_s1, self.Omega_vac_fit)
            dxe_dz = -dx_e_dt / ((1.0 + z) * H_z)
            
        # 2. Calculate Matter Temperature Decoupling Derivative (dTm/dz)
        H_z = get_flat_cer_hz(z, self.H0_s1, self.Omega_vac_fit)
        gamma_c = self.get_compton_rate(z, x_e)
        
        # Thermal evolution combining expansion cooling and Compton heating terms
        dTm_dz = (2.0 * T_m / (1.0 + z)) + (gamma_c * (T_m - Tr) / ((1.0 + z) * H_z))
        
        return np.array([dxe_dz, dTm_dz])
