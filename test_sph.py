import sys
import numpy as np
import time
from sec_01.benchmarks.B02_dambreak_column_dynamic.B02_SPH_solver import run

config = {
    "seed": 42,
    "sph_particles": 1200,
    "target_sim_time_s": 0.60,
    "max_steps": 500,  # Just run 500 steps
    "dam_width_fraction": 0.25,
    "sph_initial_height": 0.8,
}

t0 = time.time()
res = run(config, prefer_gpu=False)
t1 = time.time()

print("Time taken:", t1 - t0)
print("Metrics:")
print(f"  simulated_time_end: {res.metrics['simulated_time_end']}")
print(f"  peak_particle_speed: {res.metrics['peak_particle_speed']}")
