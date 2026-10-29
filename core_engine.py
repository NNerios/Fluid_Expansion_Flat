"""
CER MASTER ENGINE: Core Cosmological Fluid Substrate Model
Tracks flat late-time expansion history kernels accurately.
"""
import numpy as np

def get_flat_cer_hz(z, H0, Omega_vac, alpha):
    """
    Evaluates the flat late-time CER-F expansion history kernel.
    Assumes Omega_m = 1.0 - Omega_vac natively.
    """
    x = 1.0 + z
    Omega_m = 1.0 - Omega_vac
    
    # Core fluid interaction tracking
    F = Omega_vac + Omega_m * (x ** (3.0 - alpha))
    
    # Avoid negative roots near extreme parameter exploration
    return H0 * np.sqrt(np.clip(F, 0.0, None))

