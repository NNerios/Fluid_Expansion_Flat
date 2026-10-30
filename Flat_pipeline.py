"""
Flat_pipeline.py
===========================================================================
FLAT DATA PIPELINE RUNNER: Runs optimizations and tracks variables.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core_engine import run_background_optimization, PrimordialRecombinationModule
from horizon_engine import CosmologicalHorizonCalculator

def execute_complete_pipeline(z_start=1500.0, z_end=800.0, step_size=-1.0, alpha=0.0):
    print("================================================================")
    print(f"STEP 1: BACKGROUND MINIMIZATION (Alpha Fixed to {alpha})")
    print("================================================================")
    H0_opt, Omega_vac_opt = run_background_optimization(alpha=alpha)
    print(f"Best-Fit Params -> H0: {H0_opt:.4f} km/s/Mpc | Omega_vac: {Omega_vac_opt:.4f}")
    
    print("\n================================================================")
    print("STEP 2: RUNNING HORIZON CALCULATIONS")
    print("================================================================")
    horizon_calc = CosmologicalHorizonCalculator(H0_opt, Omega_vac_opt, alpha=alpha)
    print(f"Particle Horizon at z=0   : {horizon_calc.particle_horizon(0.0):.2f} Mpc")
    print(f"Cosmic Event Horizon at z=0: {horizon_calc.event_horizon(0.0):.2f} Mpc")
    print(f"Comoving Distance to CMB (z=1100): {horizon_calc.comoving_distance(1100.0):.2f} Mpc")

    print("\n================================================================")
    print(f"STEP 3: HYREC RECOMBINATION EVOLUTION (z={z_start} -> {z_end})")
    print("================================================================")
    rec_engine = PrimordialRecombinationModule(H0_fit=H0_opt, Omega_vac_fit=Omega_vac_opt, alpha=alpha)
    
    current_z = z_start
    state = np.array([0.9999, rec_engine.T_cmb_0 * (1.0 + z_start)]) 
    
    z_history, xe_history, tm_history = [], [], []
    
    while current_z >= z_end:
        z_history.append(current_z)
        
        xe_snapshot = float(state[0])
        tm_snapshot = float(state[1])
        
        xe_history.append(xe_snapshot)
        tm_history.append(tm_snapshot)
        
        k1 = step_size * rec_engine.dx_e_dz(current_z, state)
        k2 = step_size * rec_engine.dx_e_dz(current_z + 0.5 * step_size, state + 0.5 * k1)
        k3 = step_size * rec_engine.dx_e_dz(current_z + 0.5 * step_size, state + 0.5 * k2)
        k4 = step_size * rec_engine.dx_e_dz(current_z + step_size, state + k3)
        
        state += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        current_z += step_size
        
        state[0] = np.clip(state[0], 1e-6, 1.0)
        state[1] = np.maximum(state[1], 0.1)
        
    print("-> Pipeline integration tasks finished completely.")
    return np.array(z_history), np.array(xe_history), np.array(tm_history)

if __name__ == "__main__":
    z, xe, tm = execute_complete_pipeline()
    print("\n" + "-"*64)
    print(f"Final Free Electron Fraction x_e at z={z[-1]}: {xe[-1]:.6e}")
    print(f"Final Matter Temperature T_m at z={z[-1]}      : {tm[-1]:.2f} K")
    print("-"*64)
