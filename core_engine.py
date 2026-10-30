"""
core_engine.py
===========================================================================
CER MASTER ENGINE: Core Cosmological Fluid Substrate Model + HyRec Recombination
Tracks flat expansion history kernels, horizons, and recombination variables.
"""
import numpy as np
from scipy.optimize import minimize

def get_flat_cer_hz(z, H0, Omega_vac, alpha=0.0, Omega_r=9.3e-5):
    """
    Evaluates the flat early/late-time CER expansion history kernel.
    alpha = 0 represents your strict baseline model.
    """
    x = 1.0 + z
    Omega_m = 1.0 - Omega_vac
    
    # Core fluid interaction tracking + High-z early radiation contribution
    F = Omega_vac + Omega_m * (x ** (3.0 - alpha)) + Omega_r * (x ** 4.0)
    return H0 * np.sqrt(np.clip(F, 0.0, None))


class PrimordialRecombinationModule:
    def __init__(self, H0_fit, Omega_vac_fit, alpha=0.0, Omega_b=0.0493):
        # SI / MKS Physical Constants
        self.G = 6.6743e-11
        self.c = 299792458.0
        self.m_H = 1.67355758e-27
        self.k_B = 1.380649e-23
        
        self.H0_fit = H0_fit
        self.Omega_vac_fit = Omega_vac_fit
        self.alpha = alpha
        self.Omega_b = Omega_b
        
        # Convert H0 from km/s/Mpc to inverse seconds (1/s)
        self.H0_s1 = (H0_fit * 1000.0) / 3.085677581e22 
        self.T_cmb_0 = 2.7255
        
        # Effective 4-Level Atom Constants
        self.Lambda_2s = 8.224        # Two-photon decay rate (s^-1)
        self.E_21 = 10.202            # eV (Lyman-alpha transition)
        self.E_1s = 13.5984           # eV (Ionization energy)
        self.eV_to_K = 11604.51812

        # Baryonic Hydrogen Number Density
        rho_crit_0 = (3.0 * self.H0_s1**2) / (8.0 * np.pi * self.G)
        rho_b_0 = self.Omega_b * rho_crit_0
        self.nH0_cm3 = (0.75 * rho_b_0 / self.m_H) * 1e-6 

    def get_atomic_rates(self, Tm, Tr):
        """Case-B atomic cross-section configurations."""
        A_2s = 3.75e-14 * (Tm / 10000.0)**(-0.7)
        A_2p = 4.10e-14 * (Tm / 10000.0)**(-0.8)
        
        de_broglie_factor = 2.414e15 * (Tr)**(1.5)
        B_2s = A_2s * de_broglie_factor * np.exp(-(self.E_1s - self.E_21) * self.eV_to_K / Tr)
        B_2p = A_2p * de_broglie_factor * np.exp(-(self.E_1s - self.E_21) * self.eV_to_K / Tr)
        
        R_2p_2s = 1.2e-4 * (Tr / 3000.0)**(0.5)
        return A_2s, A_2p, B_2s, B_2p, R_2p_2s

    def get_lyman_alpha_escape(self, z, x_e):
        """Sobolev optical depth / photon escape rates."""
        nH_z = self.nH0_cm3 * (1.0 + z)**3
        H_z = get_flat_cer_hz(z, self.H0_fit, self.Omega_vac_fit, self.alpha)
        
        lambda_lyalpha = 1.21567e-5  # cm
        tau_sobolev = (3.0 * nH_z * (lambda_lyalpha**3) * 6.25e8) / (8.0 * np.pi * H_z)
        return (1.0 / (tau_sobolev + 1.0)) * (6.25e8)

    def solve_qss_populations(self, z, x_e, Tm, Tr):
        """Matrix inversion for excited atomic states."""
        A_2s, A_2p, B_2s, B_2p, R_2p_2s = self.get_atomic_rates(Tm, Tr)
        R_LyA = self.get_lyman_alpha_escape(z, x_e)
        
        nH_z = self.nH0_cm3 * (1.0 + z)**3
        x_1s = 1.0 - x_e
        thermal_excitation = np.exp(-self.E_21 * self.eV_to_K / Tr)

        M = np.array([
            [B_2s + 9.0*R_2p_2s + self.Lambda_2s, -3.0*R_2p_2s],
            [-3.0*R_2p_2s,                         B_2p + R_2p_2s + R_LyA]
        ])
        
        Y = np.array([
            nH_z * (x_e**2) * A_2s + self.Lambda_2s * x_1s * thermal_excitation,
            nH_z * (x_e**2) * A_2p + 3.0 * R_LyA * x_1s * thermal_excitation
        ])
        
        return np.linalg.solve(M, Y)[0], np.linalg.solve(M, Y)[1], A_2s, A_2p, B_2s, B_2p

    def get_compton_rate(self, z, x_e):
        """Compton heat exchange coefficient between matter and CMB."""
        Tr = self.T_cmb_0 * (1.0 + z)
        sigma_T, m_e, a_R, f_He = 6.65246e-29, 9.10938e-31, 7.56573e-16, 0.0797
        rho_photon = a_R * (Tr ** 4.0)
        base_rate = (8.0 * sigma_T * rho_photon * self.c) / (3.0 * m_e * self.c**2)
        return base_rate * (x_e / (1.0 + f_He + x_e))

    def dx_e_dz(self, z, state_vector):
        """Simultaneous coupled derivatives: [dxe/dz, dTm/dz]."""
        x_e, T_m = state_vector
        Tr = self.T_cmb_0 * (1.0 + z)
        x_e = np.clip(x_e, 1e-6, 1.0)
        
        H_z = get_flat_cer_hz(z, self.H0_fit, self.Omega_vac_fit, self.alpha)
        
        if x_e >= 0.9999:
            dxe_dz = 0.0
        else:
            nH_z = self.nH0_cm3 * (1.0 + z)**3
            x_2s, x_2p, A_2s, A_2p, B_2s, B_2p = self.solve_qss_populations(z, x_e, T_m, Tr)
            dx_e_dt = (x_2s * B_2s - nH_z * (x_e**2) * A_2s) + (x_2p * B_2p - nH_z * (x_e**2) * A_2p)
            dxe_dz = -dx_e_dt / ((1.0 + z) * H_z)
            
        gamma_c = self.get_compton_rate(z, x_e)
        dTm_dz = (2.0 * T_m / (1.0 + z)) + (gamma_c * (T_m - Tr) / ((1.0 + z) * H_z))
        return np.array([dxe_dz, dTm_dz])


# =====================================================================
# 3. BACKGROUND OPTIMIZER (32 COSMIC CHRONOMETERS)
# =====================================================================
CC_DATA = np.array([
    [0.0700, 69.0, 19.60], [0.0900, 69.0, 12.00], [0.1200, 68.6, 26.20],
    [0.1700, 83.0, 8.00],  [0.1791, 75.0, 4.00],  [0.1993, 75.0, 5.00],
    [0.2000, 72.9, 29.60], [0.2700, 77.0, 14.00], [0.2800, 88.8, 36.60],
    [0.3520, 83.0, 14.00], [0.3802, 83.0, 13.50], [0.4000, 95.0, 17.00],
    [0.4004, 77.0, 10.20], [0.4249, 87.1, 11.20], [0.4497, 92.8, 12.90],
    [0.4783, 80.9, 9.00],  [0.4800, 97.0, 62.00], [0.5930, 104.0, 13.00],
    [0.6800, 92.0, 8.00],  [0.7810, 105.0, 12.00], [0.8750, 125.0, 17.00],
    [0.8800, 90.0, 40.00], [0.9000, 117.0, 23.00], [1.0370, 154.0, 20.00],
    [1.3000, 168.0, 17.00], [1.3630, 160.0, 33.60], [1.4300, 177.0, 18.00],
    [1.5300, 140.0, 14.00], [1.7500, 202.0, 40.00], [1.9650, 186.5, 50.43],
    [1.3900, 167.0, 11.40], [0.5100, 90.4, 1.90]
])

def objective_function(params, alpha=0.0):
    H0, Omega_vac = params
    return float(np.sum(((CC_DATA[:, 1] - get_flat_cer_hz(CC_DATA[:, 0], H0, Omega_vac, alpha)) / CC_DATA[:, 2]) ** 2))

def run_background_optimization(alpha=0.0):
    result = minimize(objective_function, [68.5, 0.70], args=(alpha,), method="L-BFGS-B", bounds=[(58.0, 78.0), (0.5, 0.88)])
    if not result.success: raise RuntimeError("Optimization failed.")
    return result.x[0], result.x[1]
