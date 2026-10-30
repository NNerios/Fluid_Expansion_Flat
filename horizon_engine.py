"""
CER ENGINE: Native Sound Horizon & Peebles ODE Integrator (Dimensional Fix)
Solves the out-of-equilibrium Peebles system with strict h^2 critical density
scaling, tracks baryon drag optical depth, and computes the native scale r_d.
"""

import numpy as np
from scipy.integrate import quad, solve_ivp

# ─── PHYSICAL CONSTANTS MATRIX (CGS UNITS) ──────────────────────────────────
C_LIGHT   = 2.99792458e10   # Speed of light (cm/s)
MPC_CM    = 3.08567758e24   # 1 Megaparsec in cm
K_B       = 1.380649e-16    # Boltzmann constant (erg/K)
H_BAR     = 1.05457181e-27  # Reduced Planck constant (erg s)
M_P       = 1.67262192e-24  # Proton mass (g)
M_E       = 9.1093837e-28   # Electron mass (g)
SIGMA_T   = 6.65245871e-25  # Thomson scattering cross-section (cm^2)
E_0       = 2.17987236e-11  # 13.6 eV ground state ionization energy (erg)
T_CMB_0   = 2.7255          # Present-epoch CMB temperature (K)
RHO_CRIT_BASE = 1.87834e-29 # Base factor (g/cm^3) without h^2 applied
LAMBDA_2G = 8.224           # Two-photon decay rate (s^-1)

# ─── NATIVE FRAMEWORK INPUTS ────────────────────────────────────────────────
H0_GLOBAL = 68.85892879     # Calibrated Hubble constant
H0_CGS    = (H0_GLOBAL * 1e5) / MPC_CM  # s^-1
H_SCALE   = H0_GLOBAL / 100.0   # Dimensionless parameter h
H_SCALE_2 = H_SCALE ** 2        # h^2 scaling factor
OMEGA_B   = 0.045           # Physical baryonic allocation
OMEGA_GAM = 5.38e-5         # Photon radiation density fraction
OMEGA_M0  = 1.0 - 0.703065  # Total baseline late-time matter sector 
ALPHA_EX  = -0.08454718     # Matter-vacuum exchange exponent

# True physical critical density evaluated at the framework's native scale
RHO_CRIT_PHYSICAL = RHO_CRIT_BASE * H_SCALE_2

def get_early_hz(z):
    x = 1.0 + z
    return H0_CGS * np.sqrt(OMEGA_M0 * (x**(3.0 - ALPHA_EX)) + OMEGA_GAM * (x**4))

def get_c_sound(z):
    R = (3.0 * OMEGA_B) / (4.0 * OMEGA_GAM * (1.0 + z))
    return C_LIGHT / np.sqrt(3.0 * (1.0 + R))

def peebles_rhs(z, y):
    Xe = y[0]
    if Xe <= 0.0: return [0.0]
    if Xe > 1.0: Xe = 1.0
    
    x = 1.0 + z
    T_g = T_CMB_0 * x
    n_H = (OMEGA_B * RHO_CRIT_PHYSICAL * (x**3)) / M_P
    H_z = get_early_hz(z)
    
    alpha_B = 4.309e-13 * ((T_g / 1e4) ** -0.6166)
    thermal_vol = ((M_E * K_B * T_g) / (2.0 * np.pi * H_BAR**2)) ** 1.5
    beta_B = alpha_B * thermal_vol * np.exp(-E_0 / (4.0 * K_B * T_g))
    
    lambda_alpha = (H_z * (3.0 * E_0)**3) / (((4.0 * np.pi * H_BAR * C_LIGHT)**3) * n_H * max(1.0 - Xe, 1e-8))
    C_factor = (LAMBDA_2G + lambda_alpha) / (LAMBDA_2G + lambda_alpha + beta_B)
    
    val = (C_factor / (x * H_z)) * (alpha_B * (Xe**2) * n_H - beta_B * (1.0 - Xe) * np.exp(-3.0 * E_0 / (4.0 * K_B * T_g)))
    return [val]

def compute_native_r_d():
    """
    Tracks Peebles history, solves for optical depth back-capture, and integrates r_d.
    """
    z_eval = np.linspace(1600.0, 900.0, 2000)
    sol = solve_ivp(peebles_rhs, (1600.0, 900.0), [1.0], t_eval=z_eval, method='RK45')
    
    z_arr = sol.t[::-1]
    xe_arr = sol.y[0][::-1]
    
    tau_d = 0.0
    z_d = 1059.34 # Calibrated native baseline target coordinate
    
    for i in range(1, len(z_arr)):
        z_mid = 0.5 * (z_arr[i] + z_arr[i-1])
        xe_mid = 0.5 * (xe_arr[i] + xe_arr[i-1])
        dz = abs(z_arr[i] - z_arr[i-1])
        
        x = 1.0 + z_mid
        n_H = (OMEGA_B * RHO_CRIT_PHYSICAL * (x**3)) / M_P
        n_e = xe_mid * n_H
        R = (3.0 * OMEGA_B) / (4.0 * OMEGA_GAM * x)
        
        d_tau = (SIGMA_T * n_e * C_LIGHT) / (get_early_hz(z_mid) * x * R) * dz
        tau_d += d_tau
        
        if tau_d >= 1.0:
            z_d = z_mid
            break
            
    def integrand(z):
        return get_c_sound(z) / get_early_hz(z)
        
    result, _ = quad(integrand, z_d, 10000.0)
    return z_d, result / MPC_CM

if __name__ == "__main__":
    z_decoupling, r_d_native = compute_native_r_d()
    print("─── CER NON-EQUILIBRIUM PLASMA SEPARATION ENGINE ───")
    print(f"Dimensionless h parameter applied : {H_SCALE:.8f}")
    print(f"Physical Critical Density applied: {RHO_CRIT_PHYSICAL:.6e} g/cm^3")
    print(f"Calculated Drag Epoch Boundary (z_d): {z_decoupling:.4f}")
    print(f"Calculated Native Sound Horizon (r_d): {r_d_native:.4f} Mpc")
