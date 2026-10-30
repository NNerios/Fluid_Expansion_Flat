
"""
COSMIC EVOLUTION RATE ENGINE (CER) - STRICT FLATLINE BASELINE
============================================================
Alpha fixed to 0.
Fits H0 and Omega_vac using 32 cosmic chronometer measurements
with diagonal observational uncertainties.
"""

import numpy as np
from scipy.optimize import minimize


# 1. CORE MATHEMATICAL ENGINE
def get_flat_cer_hz(z, H0, Omega_vac):
    """
    Strict flatline baseline:
    alpha = 0
    Omega_m = 1 - Omega_vac

    H(z) = H0 * sqrt(
        Omega_vac + Omega_m * (1 + z)**3
    )
    """
    x = 1.0 + z
    Omega_m = 1.0 - Omega_vac

    # Strict baseline: exponent is fixed at 3
    F = Omega_vac + Omega_m * (x ** 3.0)

    return H0 * np.sqrt(np.clip(F, 0.0, None))


# 2. MASTER 32-POINT COSMIC CHRONOMETER DATASET
# Columns: redshift, H(z) [km/s/Mpc], sigma_H(z)
CC_DATA = np.array([
    [0.0700, 69.0, 19.60],
    [0.0900, 69.0, 12.00],
    [0.1200, 68.6, 26.20],
    [0.1700, 83.0, 8.00],
    [0.1791, 75.0, 4.00],
    [0.1993, 75.0, 5.00],
    [0.2000, 72.9, 29.60],
    [0.2700, 77.0, 14.00],
    [0.2800, 88.8, 36.60],
    [0.3520, 83.0, 14.00],
    [0.3802, 83.0, 13.50],
    [0.4000, 95.0, 17.00],
    [0.4004, 77.0, 10.20],
    [0.4249, 87.1, 11.20],
    [0.4497, 92.8, 12.90],
    [0.4783, 80.9, 9.00],
    [0.4800, 97.0, 62.00],
    [0.5930, 104.0, 13.00],
    [0.6800, 92.0, 8.00],
    [0.7810, 105.0, 12.00],
    [0.8750, 125.0, 17.00],
    [0.8800, 90.0, 40.00],
    [0.9000, 117.0, 23.00],
    [1.0370, 154.0, 20.00],
    [1.3000, 168.0, 17.00],
    [1.3630, 160.0, 33.60],
    [1.4300, 177.0, 18.00],
    [1.5300, 140.0, 14.00],
    [1.7500, 202.0, 40.00],
    [1.9650, 186.5, 50.43],
    [1.3900, 167.0, 11.40],
    [0.5100, 90.4, 1.90]
])


# 3. DIAGONAL-ERROR OBJECTIVE FUNCTION
def objective_function(params):
    """Calculate chi-squared using the supplied uncertainties."""
    H0, Omega_vac = params

    z_obs = CC_DATA[:, 0]
    H_obs = CC_DATA[:, 1]
    sigma_obs = CC_DATA[:, 2]

    H_model = get_flat_cer_hz(
        z_obs, H0, Omega_vac
    )

    residuals = (H_obs - H_model) / sigma_obs
    chi2 = np.sum(residuals ** 2)

    return float(chi2)


# 4. BOUNDED OPTIMIZATION
if __name__ == "__main__":
    print("CER Strict Flatline Baseline")
    print("Alpha fixed at 0")
    print("Fitting H0 and Omega_vac only")

    initial_guess = [68.5, 0.70]

    param_bounds = [
        (58.0, 78.0),   # H0
        (0.5, 0.88)     # Omega_vac
    ]

    result = minimize(
        objective_function,
        initial_guess,
        method="L-BFGS-B",
        bounds=param_bounds
    )

    if result.success:
        H0_fit, Omega_vac_fit = result.x
        Omega_m_fit = 1.0 - Omega_vac_fit
        dof = len(CC_DATA) - 2

        print("\nOPTIMIZATION CONVERGED")
        print("-" * 45)
        print(f"H0         = {H0_fit:.4f} km/s/Mpc")
        print(f"Omega_vac  = {Omega_vac_fit:.4f}")
        print(f"Omega_m    = {Omega_m_fit:.4f}")
        print("Alpha      = 0 (fixed)")
        print("-" * 45)
        print(f"Total chi2   = {result.fun:.4f}")
        print(f"Degrees of freedom = {dof}")
        print(f"Reduced chi2 = {result.fun / dof:.4f}")
        print(f"Optimizer message: {result.message}")

    else:
        print("\nOPTIMIZATION FAILED")
        print(result.message)
