"""
horizon_engine.py
===========================================================================
HORIZON CALCULATOR ENGINE: Integrates the expansion kernel to extract
Particle Horizons and Event Horizons across any given redshift boundary.
"""
import numpy as np
from scipy.integrate import quad
from core_engine import get_flat_cer_hz

class CosmologicalHorizonCalculator:
    def __init__(self, H0, Omega_vac, alpha=0.0):
        self.H0 = H0
        self.Omega_vac = Omega_vac
        self.alpha = alpha
        self.c = 299792.458 # Speed of light in km/s (results in Mpc)

    def comoving_distance(self, z):
        """Integrates 1/H(z) from 0 to z to get Comoving Distance."""
        integrand = lambda z_prime: self.c / get_flat_cer_hz(z_prime, self.H0, self.Omega_vac, self.alpha)
        distance, _ = quad(integrand, 0, z)
        return distance

    def particle_horizon(self, z_eval):
        """Integrates from z_eval to infinity to check the maximum observable scale."""
        integrand = lambda z_prime: self.c / get_flat_cer_hz(z_prime, self.H0, self.Omega_vac, self.alpha)
        horizon, _ = quad(integrand, z_eval, np.inf)
        return horizon

    def event_horizon(self, z_eval):
        """Integrates future light travel boundaries from -1 to z_eval."""
        integrand = lambda z_prime: self.c / get_flat_cer_hz(z_prime, self.H0, self.Omega_vac, self.alpha)
        # In terms of redshift, future infinity is z = -1
        horizon, _ = quad(integrand, -0.9999, z_eval)
        return horizon
