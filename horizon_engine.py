"""
horizon_engine.py - Clean Audited Version
"""
import numpy as np
from scipy.integrate import quad
from core_engine import get_flat_cer_hz

class CosmologicalHorizonCalculator:
    def __init__(self, H0, Omega_vac, alpha=0.0):
        self.H0 = float(H0)
        self.Omega_vac = float(Omega_vac)
        self.alpha = float(alpha)
        self.c = 299792.458 

    def comoving_distance(self, z):
        integrand = lambda z_prime: self.c / get_flat_cer_hz(z_prime, self.H0, self.Omega_vac, self.alpha)
        distance, _ = quad(integrand, 0, z)
        return float(distance)

    def particle_horizon(self, z_eval):
        integrand = lambda z_prime: self.c / get_flat_cer_hz(z_prime, self.H0, self.Omega_vac, self.alpha)
        horizon, _ = quad(integrand, z_eval, np.inf)
        return float(horizon)

    def event_horizon(self, z_eval):
        integrand = lambda z_prime: self.c / get_flat_cer_hz(z_prime, self.H0, self.Omega_vac, self.alpha)
        horizon, _ = quad(integrand, -0.9999, z_eval)
        return float(horizon)
