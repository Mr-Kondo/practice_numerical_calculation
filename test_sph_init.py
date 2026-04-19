import numpy as np

particles = 1200
dam_fraction = 0.25
dam_height = 0.8
fluid_area = dam_fraction * dam_height

spacing = np.sqrt(fluid_area / particles)
nx_part = max(1, int(round(dam_fraction / spacing)))
ny_part = max(1, int(round(dam_height / spacing)))
particles = nx_part * ny_part

x_part = np.linspace(spacing / 2, dam_fraction - spacing / 2, nx_part)
y_part = np.linspace(spacing / 2, dam_height - spacing / 2, ny_part)
xv, yv = np.meshgrid(x_part, y_part)
positions = np.column_stack([xv.ravel(), yv.ravel()])

print(f"Original particles: 1200, New particles: {particles}")
print(f"nx: {nx_part}, ny: {ny_part}")
print(f"positions shape: {positions.shape}")
