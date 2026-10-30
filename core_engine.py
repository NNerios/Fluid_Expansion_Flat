"""
core_engine.py
===========================================================================
CER MASTER ENGINE: Core Cosmological Fluid Substrate Model + CER Recombination
Uses uniform low-redshift units (km/s/Mpc) throughout. No GR components.
"""
import numpy as np
from scipy.optimize import minimize

def get_flat_cer_hz(z, H0, Omega_vac, alpha=0.0):
    """Evaluates the strict baseline CER expansion history kernel."""
    x = 1.0 + z
    Omega_m = 1.0 - Omega_vac
    F = Omega_vac + Omega_m * (x ** (3.0 - alpha))
    return H0 * np.sqrt(np.clip(F, 0.0, None))


class PrimordialRecombinationModule:
    def __init__(self, H0_fit, Omega_vac_fit, alpha=0.0):
        # All tracking profiles stay natively in km/s/Mpc to match CC parameters
        self.H0_fit = float(H0_fit)
        self.Omega_vac_fit = float(Omega_vac_fit)
        self.alpha = float(alpha)
        
        self.T_cmb_0 = 2.7255
        self.eV_to_K = 11604.51812
        self.E_21 = 10.202            
        self.E_1s = 13.5984           

        # Base number density of the baryonic fluid substrate in matching units
        self.n_substrate_0 = 0.0493 * (self.H0_fit ** 2)

    def get_atomic_rates(self, Tm, Tr):
        """Calculates effective transition rates scaled cleanly to the CER grid."""
        A_2s = 3.75e-4 * (Tm / 10000.0)**(-0.7)
        A_2p = 4.10e-4 * (Tm / 10000.0)**(-0.8)
        
        # Detailed Balance factor mapped to the uniform fluid layout
        fluid_factor = 2.414e-5 * (Tr)**(1.5)
        binding_energy = self.E_1s - self.E_21
        
        B_2s = A_2s * fluid_factor * np.exp(-binding_energy * self.eV_to_K / Tr)
        B_2p = A_2p * fluid_factor * np.exp(-binding_energy * self.eV_to_K / Tr)
        
        R_2p_2s = 1.2e-4 * (Tr / 3000.0)**(0.5)
        return A_2s, A_2p, B_2s, B_2p, R_2p_2s

    def get_lyman_alpha_escape(self, z):
        """Evaluates escape probabilities natively across the CER fluid matrix."""
        H_z = get_flat_cer_hz(z, self.H0_fit, self.Omega_vac_fit, self.alpha)
        tau_fluid = (3.0 * self.n_substrate_0 * (1.0 + z)**3) / (8.0 * np.pi * H_z)
        return (1.0 / (tau_fluid + 1.0)) * 100.0

    def solve_qss_populations(self, z, x_e, Tm, Tr):
        """Inverts the atomic populations using explicit element indexing."""
        A_2s, A_2p, B_2s, B_2p, R_2p_2s = self.get_atomic_rates(Tm, Tr)
        R_LyA = self.get_lyman_alpha_escape(z)
        
        n_fluid_z = self.n_substrate_0 * (1.0 + z)**3
        x_1s = 1.0 - x_e
        thermal_excitation = np.exp(-self.E_21 * self.eV_to_K / Tr)

        M = np.array([
            [B_2s + 9.0*R_2p_2s + 8.224, -3.0*R_2p_2s],
            [-3.0*R_2p_2s,                 B_2p + R_2p_2s + R_LyA]
        ])
        
        Y = np.array([
            n_fluid_z * (x_e**2) * A_2s + 8.224 * x_1s * thermal_excitation,
            n_fluid_z * (x_e**2) * A_2p + 3.0 * R_LyA * x_1s * thermal_excitation
        ])
        
        x_excited = np.linalg.solve(M, Y)
        
        # Explicit index slicing fixes the single-number array casting crash
        x_2s = float(x_excited[0])
        x_2p = float(x_excited[1])
        return x_2s, x_2p, A_2s, A_2p, B_2s, B_2p

    def dx_e_dz(self, z, state_vector):
        """Simultaneous coupled derivatives evaluated on the native optimizer grid."""
        x_e, T_m = state_vector
        Tr = self.T_cmb_0 * (1.0 + z)
        x_e = np.clip(x_e, 1e-6, 1.0)
        
        H_z = get_flat_cer_hz(z, self.H0_fit, self.Omega_vac_fit, self.alpha)
        n_fluid_z = self.n_substrate_0 * (1.0 + z)**3
        
        x_2s, x_2p, A_2s, A_2p, B_2s, B_2p = self.solve_qss_populations(z, x_e, T_m, Tr)
        
        dx_e_dt = (x_2s * B_2s - n_fluid_z * (x_e**2) * A_2s) + (x_2p * B_2p - n_fluid_z * (x_e**2) * A_2p)
        dxe_dz = -dx_e_dt / ((1.0 + z) * H_z)
            
        if z > 300.0:
            dTm_dz = 1.0 * T_m / (1.0 + z)
        else:
            gamma_fluid = 1e-4 * x_e * (Tr ** 4.0)
            dTm_dz = (2.0 * T_m / (1.0 + z)) + (gamma_fluid * (T_m - Tr) / ((1.0 + z) * H_z + 1e-25))
            
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
    return result.x
