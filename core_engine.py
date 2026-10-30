"""CER flat core engine.

The CC table is the historical 32-row diagonal-error table previously
committed to this repository. It is not the later correlated Moresco
covariance compilation; results from this table must be labelled accordingly.
The recombination implementation is a Peebles-style effective three-level
atom approximation and is not a substitute for a full multilevel atom code.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

# CGS constants
C_LIGHT = 2.99792458e10       # cm/s
MPC_CM = 3.08567758e24        # cm
K_B = 1.380649e-16            # erg/K
H_BAR = 1.054571817e-27       # erg s
M_P = 1.67262192369e-24       # g
M_E = 9.1093837015e-28        # g
SIGMA_T = 6.6524587158e-25    # cm^2
E_0 = 2.1798723611e-11        # erg
T_CMB_0 = 2.7255              # K
LAMBDA_2G = 8.22458           # s^-1
RHO_CRIT_BASE = 1.87834e-29   # g/cm^3, multiplied by h^2
OMEGA_B = 0.045
OMEGA_GAMMA = 5.38e-5
F_HE = 0.0797
H0_KMS_MPC_TO_S = 1.0e5 / MPC_CM

# Historical repository CC table: z, H(z) [km/s/Mpc], sigma [km/s/Mpc].
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
], dtype=float)


def get_flat_cer_hz(z, H0, Omega_vac):
    """Strict flat late-time CER baseline (alpha=0), H in km/s/Mpc."""
    zz = np.asarray(z, dtype=float)

    if np.any(~np.isfinite(zz)) or np.any(zz <= -1.0):
        raise ValueError("z must be finite and greater than -1")

    if not np.isfinite(H0) or H0 <= 0:
        raise ValueError("H0 must be finite and positive")

    if not np.isfinite(Omega_vac) or not 0.0 < Omega_vac < 1.0:
        raise ValueError("Omega_vac must be between 0 and 1")

    ans = H0 * np.sqrt(
        Omega_vac + (1.0 - Omega_vac) * (1.0 + zz)**3
    )

    return float(ans) if ans.ndim == 0 else ans


def objective_function(params, data=None):
    """Diagonal-error chi-square for an N x 3 [z,H,sigma] table."""
    tab = CC_DATA if data is None else np.asarray(data, dtype=float)

    if tab.ndim != 2 or tab.shape[1] != 3 or len(tab) < 3:
        raise ValueError("data must be an N x 3 [z,H,sigma] array")

    if not np.all(np.isfinite(tab)) or np.any(tab[:, 2] <= 0):
        raise ValueError(
            "data must be finite and all sigma values positive"
        )

    model = get_flat_cer_hz(
        tab[:, 0],
        float(params[0]),
        float(params[1])
    )

    residuals = (tab[:, 1] - model) / tab[:, 2]

    return float(residuals @ residuals)


def run_background_optimization(data=None):
    """Return (H0_fit [km/s/Mpc], Omega_vac_fit) for Flat_pipeline.py."""
    tab = CC_DATA if data is None else np.asarray(data, dtype=float)

    result = minimize(
        objective_function,
        [68.5, 0.70],
        args=(tab,),
        method="L-BFGS-B",
        bounds=[
            (58.0, 78.0),
            (0.50, 0.88)
        ]
    )

    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"CER background optimization failed: {result.message}"
        )

    return float(result.x[0]), float(result.x[1])


class PrimordialRecombinationModule:
    """Peebles-style recombination engine with legacy and two-state APIs."""

    def __init__(
        self,
        H0_fit,
        Omega_vac_fit,
        T_cmb_0=T_CMB_0,
        omega_b=OMEGA_B,
        omega_gamma=OMEGA_GAMMA
    ):
        self.H0_fit = float(H0_fit)  # km/s/Mpc, as returned by optimizer
        self.Omega_vac_fit = float(Omega_vac_fit)

        if not np.isfinite(self.H0_fit) or self.H0_fit <= 0:
            raise ValueError("H0_fit must be finite and positive")

        if not 0.0 < self.Omega_vac_fit < 1.0:
            raise ValueError(
                "Omega_vac_fit must be between 0 and 1"
            )

        self.H0_s1 = self.H0_fit * H0_KMS_MPC_TO_S

        self.T_cmb_0 = float(T_cmb_0)
        self.omega_b = float(omega_b)
        self.omega_gamma = float(omega_gamma)

        if min(
            self.T_cmb_0,
            self.omega_b,
            self.omega_gamma
        ) <= 0:
            raise ValueError(
                "T_cmb_0, omega_b and omega_gamma must be positive"
            )

        h = self.H0_fit / 100.0

        self.nH0_cm3 = (
            self.omega_b * RHO_CRIT_BASE * h * h / M_P
        )

        self.c = C_LIGHT

    def get_early_hz(z):
    """
    Shared CER expansion history for early-universe calculations.
    Returns H(z) in s^-1.

    Uses the same fitted H0 and Omega_vac as core_engine.py.
    Radiation is included and the matter exponent remains 3 (alpha=0).
    """
    z = float(z)

    if not np.isfinite(z) or z <= -1.0:
        raise ValueError("z must be finite and greater than -1.")

    x = 1.0 + z

    return H0_CGS * np.sqrt(
        OMEGA_VAC
        + OMEGA_M0 * x**3
        + OMEGA_GAM * x**4
    )

    def _peebles_terms(self, z, x_e):
        z = float(z)

        if z <= -1:
            raise ValueError("z must be greater than -1")

        x = 1.0 + z
        Tg = self.T_cmb_0 * x
        xe = float(np.clip(x_e, 1e-8, 1.0))

        nH = self.nH0_cm3 * x**3
        Hz = self.get_expansion_rate_s1(z)

        alpha_b = (
            4.309e-13 * (Tg / 1e4)**-0.6166
        )

        thermal_volume = (
            (M_E * K_B * Tg) /
            (2.0 * np.pi * H_BAR**2)
        )**1.5

        beta_b = (
            alpha_b
            * thermal_volume
            * np.exp(-E_0 / (4.0 * K_B * Tg))
        )

        lambda_alpha = (
            Hz * (3.0 * E_0)**3
            /
            (
                (4.0 * np.pi * H_BAR * C_LIGHT)**3
                * nH
                * max(1.0 - xe, 1e-8)
            )
        )

        c_factor = (
            (LAMBDA_2G + lambda_alpha)
            /
            (LAMBDA_2G + lambda_alpha + beta_b)
        )

        photo = (
            beta_b
            * (1.0 - xe)
            * np.exp(-3.0 * E_0 / (4.0 * K_B * Tg))
        )

        recomb = alpha_b * xe * xe * nH

        return c_factor, recomb, photo, Hz, Tg

    def dx_e_dz(self, z, x_e):
        """Legacy scalar derivative required by Flat_pipeline.py."""
        cfac, recomb, photo, hz, _ = self._peebles_terms(
            z,
            x_e
        )

        return float(
            cfac
            * (recomb - photo)
            / ((1.0 + float(z)) * hz)
        )

    def get_compton_rate(self, z, x_e):
        """Compton thermal coupling coefficient in s^-1."""
        tg = self.T_cmb_0 * (1.0 + float(z))

        rho_gamma = 7.565733e-15 * tg**4  # erg cm^-3

        base = (
            8.0 * SIGMA_T * rho_gamma
            / (3.0 * M_E * C_LIGHT)
        )

        xe = float(np.clip(x_e, 0.0, 1.0))

        return float(
            base * xe / (1.0 + F_HE + xe)
        )

    def evaluate_system_derivatives(self, z, state_vector):
        """Return [dx_e/dz,dT_m/dz] for state [x_e,T_m]."""
        state = np.asarray(state_vector, dtype=float)

        if state.shape != (2,) or not np.all(np.isfinite(state)):
            raise ValueError(
                "state_vector must be finite [x_e, T_m]"
            )

        xe = float(np.clip(state[0], 1e-8, 1.0))
        tm = float(state[1])

        if tm <= 0:
            raise ValueError("T_m must be positive")

        dxe = self.dx_e_dz(z, xe)

        tg = self.T_cmb_0 * (1.0 + float(z))
        hz = self.get_expansion_rate_s1(z)
        gamma = self.get_compton_rate(z, xe)

        dtm = (
            2.0 * tm / (1.0 + float(z))
            + gamma * (tm - tg) /
            ((1.0 + float(z)) * hz)
        )

        return np.array([dxe, dtm], dtype=float)
