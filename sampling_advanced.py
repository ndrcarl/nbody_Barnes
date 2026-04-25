import random
import math

# ============================================================
#  sampling_advanced.py
#  Generates initial conditions for Barnes treecode v1.4.
# ============================================================

G_code = 1.0

def generate_sphere(N, R):
    particles = []
    for _ in range(N):
        u     = random.random()
        v     = random.random()
        w     = random.random()
        r     = R * (u ** (1.0 / 3.0))
        theta = math.acos(1.0 - 2.0 * v)
        phi   = 2.0 * math.pi * w
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        particles.append((x, y, z))
    return particles

# ------------------------------------------------------------
#  Configuration
# ------------------------------------------------------------
N_particles       = 10000
R_pc              = 1.0
mass_per_particle = 0.0001
filename          = "gmc_internal.txt"

# Derived
M_total = N_particles * mass_per_particle
rho_0   = M_total / ((4.0 / 3.0) * math.pi * R_pc**3)
t_ff    = math.sqrt(3.0 * math.pi / (32.0 * G_code * rho_0))

# Generate positions
positions = generate_sphere(N_particles, R_pc)

# Write treecode input file
with open(filename, "w") as f:
    f.write(f"{N_particles}\n")
    f.write(f"3\n")
    f.write(f"0\n")
    for _ in range(N_particles):
        f.write(f"{mass_per_particle}\n")
    for (x, y, z) in positions:
        f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
    for _ in range(N_particles):
        f.write(f"0.0 0.0 0.0\n")

print(f"--- GMC Simulation (Internal Units) ---")
print(f"Radius:         {R_pc} pc")
print(f"Total Mass:     {M_total} M_sun")
print(f"Free-fall time: {t_ff:.4f} code time units")
print(f"File saved as:  {filename}")
