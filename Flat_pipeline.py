
"""
COSMIC EVOLUTION RATE ENGINE (CER) - HIGH-REDSHIFT HORIZON MODULE
=================================================================
Calculates the comoving plasma sound horizon scale (r_d) at decoupling.
Expects raw parameter inputs from execution pipelines.
"""

import numpy as np
from scipy.optimize import minimize
from core_engine import get_flat_cer_hz, CC_DATA

def objective_function(params):
    H0, Omega_vac = params

    z_obs = CC_DATA[:, 0]
    H_obs = CC_DATA[:, 1]
    sigma_obs = CC_DATA[:, 2]

    H_model = get_flat_cer_hz(
        z_obs, H0, Omega_vac
    )

    residuals = (H_obs - H_model) / sigma_obs
    return float(np.sum(residuals ** 2))
    
def get_sound_horizon(H0, Omega_vac, alpha, Omega_b=0.0493, Ol_gamma=5.4e-5):
    """
    Numerically integrates the early universe plasma pressure waves down to the
    decoupling epoch to determine the comoving sound horizon scale r_d (in Mpc).
    
    Parameters:
      H0         : Hubble Constant (km/s/Mpc)
      Omega_vac  : Dark Energy / Vacuum density baseline
      alpha      : Evolving fluid interaction exponent
      Omega_b    : Baryon density baseline
      Ol_gamma   : Radiation density baseline (Photons + Neutrinos)
      
    Returns:
      float: The calculated comoving sound horizon scale r_d in Mpc.
    """
    speed_of_light = 299792.458 
    Omega_m = 1.0 - Omega_vac
    
    # Precise drag epoch boundary
    z_d = 1059.3402
    
    def integrand(z):
        x = 1.0 + z
        
        # Speed of sound in the relativistic photon-baryon plasma
        baryon_photon_ratio = (3.0 * Omega_b) / (4.0 * Ol_gamma * x)
        c_s = speed_of_light / np.sqrt(3.0 * (1.0 + baryon_photon_ratio))
        
        # High-redshift Hubble rate expansion including radiation scale factor
        matter_fluid = Omega_m * (x ** (3.0 - alpha))
        radiation_fluid = Ol_gamma * (x ** 4.0)
        
        H_z = H0 * np.sqrt(np.clip(Omega_vac + matter_fluid + radiation_fluid, 1e-10, None))
        
        return c_s / H_z

    # Numerically integrate up to an effective infinity proxy boundary
    r_d_comoving, _ = integrate.quad(integrand, z_d, 1e6, epsrel=1e-6)
    return float(r_d_comoving)
