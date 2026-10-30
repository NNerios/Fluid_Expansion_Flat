"""
Flat_pipeline.py
===========================================================================
Data pipeline execution hub. Optimizes cosmological data profiles, 
extracts parameter vectors, and integrates the ionization timeline.
"""

import numpy as np
from core_engine import run_background_optimization, PrimordialRecombinationModule

def execute_recombination_pipeline(z_start=1500.0, z_end=800.0, step_size=-0.5):
    """
    Finds cosmological best-fit criteria via cosmic chronometers, 
    then projects early universe free electron fractions forward.
    """
    print("================================================================")
    print("STEP 1: CALCULATING OPTIMAL BACKGROUND BACKGROUND CONFIGURATION")
    print("================================================================")
    
    # 1. Run optimization directly out of engine data metrics
    H0_opt, Omega_vac_opt = run_background_optimization()
    print(f"-> Optimization Complete: H0 = {H0_opt:.4f} | Omega_vac = {Omega_vac_opt:.4f}")
    
    # 2. Wire optimized baseline coordinates into the recombination tracker
    rec_engine = PrimordialRecombinationModule(H0_fit=H0_opt, Omega_vac_fit=Omega_vac_opt)
    
    current_z = z_start
    current_xe = 0.9999
    
    z_stream = []
    xe_stream = []
    
    print("\n================================================================")
    print(f"STEP 2: RUNNING IONIZATION VECTOR TRACKING: z=[{z_start} -> {z_end}]")
    print("================================================================")
    
    while current_z >= z_end:
        z_stream.append(current_z)
        xe_stream.append(current_xe)
        
        # RK4 Integration Loops Slices
        k1 = step_size * rec_engine.dx_e_dz(current_z, current_xe)
        k2 = step_size * rec_engine.dx_e_dz(current_z + 0.5 * step_size, current_xe + 0.5 * k1)
        k3 = step_size * rec_engine.dx_e_dz(current_z + 0.5 * step_size, current_xe + 0.5 * k2)
        k4 = step_size * rec_engine.dx_e_dz(current_z + step_size, current_xe + k3)
        
        current_xe += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        current_z += step_size
        
        # Handle analytical floating point boundaries
        current_xe = np.clip(current_xe, 1e-6, 1.0)
        
    print("-> Pipeline Recombination Sequence Finalized.")
    return np.array(z_stream), np.array(xe_stream)

if __name__ == "__main__":
    redshifts, electron_fractions = execute_recombination_pipeline()
    print("-" * 64)
    print(f"Processed Epochs Count : {len(redshifts)} intervals.")
    print(f"Residual Free Electron Fraction (x_e) at z={redshifts[-1]}: {electron_fractions[-1]:.6e}")
    print("-" * 64)
