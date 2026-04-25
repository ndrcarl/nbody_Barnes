import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# ============================================================
#  plot_collapse.py
#  3‑D scatter snapshots at selected times.
# ============================================================

N    = 10000
R    = 1.0
MASS = 0.0001
G    = 1.0

rho_0 = (N * MASS) / ((4.0 / 3.0) * math.pi * R**3)
t_ff  = math.sqrt(3.0 * math.pi / (32.0 * G * rho_0))

TARGET_FRACTIONS = [0.0, 0.3, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 4.5]
TARGET_TIMES     = [f * t_ff for f in TARGET_FRACTIONS]
MATCH_TOL        = 0.06
PLOT_N    = 2000
RNG_SEED  = 42

filename = sys.argv[1] if len(sys.argv) > 1 else "gmc_internal.out"

# ============================================================
#  Parser — streaming, matches raggio.py style
# ============================================================
def load_snapshots_at_times(filepath, N, target_times, tol):
    snapshots = {}

    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            try:
                line_n = int(line.strip())
            except ValueError:
                continue

            if line_n != N:
                continue

            f.readline()                     # NDIM
            t = float(f.readline().strip())  # time

            # skip masses
            for _ in range(N):
                f.readline()

            # read positions
            pos = np.empty((N, 3))
            for j in range(N):
                pos[j] = np.fromstring(f.readline(), sep=' ')

            # skip velocities
            for _ in range(N):
                f.readline()

            # skip phi
            for _ in range(N):
                f.readline()

            for t_target in target_times:
                if abs(t - t_target) < tol and t_target not in snapshots:
                    snapshots[t_target] = pos.copy()

    return snapshots

# ============================================================
#  Load
# ============================================================
print(f"t_ff = {t_ff:.4f} code units")
print(f"Reading: {filename}")

data = load_snapshots_at_times(filename, N, TARGET_TIMES, MATCH_TOL)

found_targets = sorted(t for t in TARGET_TIMES if t in data)
print(f"Found {len(found_targets)} / {len(TARGET_TIMES)} requested snapshots")
if not found_targets:
    print("Error: no snapshots matched.")
    raise SystemExit(1)

print("\n--- Centre-of-mass evolution ---")
print(f"{'Time':>7}  {'X_cm':>10}  {'Y_cm':>10}  {'Z_cm':>10}")
for t in found_targets:
    cm = data[t].mean(axis=0)
    print(f"{t:7.4f}  {cm[0]:10.5f}  {cm[1]:10.5f}  {cm[2]:10.5f}")

rng      = np.random.default_rng(RNG_SEED)
plot_idx = rng.choice(N, size=min(PLOT_N, N), replace=False)

all_pos  = np.vstack([data[t][plot_idx] for t in found_targets])
limit    = max(np.abs(all_pos).max() * 1.05, R * 1.1)

N_PANELS  = len(found_targets)
NCOLS     = min(N_PANELS, 5)
NROWS     = math.ceil(N_PANELS / NCOLS)
FIG_W     = 3.8 * NCOLS
FIG_H     = 4.0 * NROWS

fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.suptitle(
    f'GMC cold collapse  (N = {N}, treecode)\n'
    f'$t_{{ff}}$ = {t_ff:.3f} code units',
    fontsize=12, y=1.01
)

for panel_idx, t in enumerate(found_targets):
    ax = fig.add_subplot(NROWS, NCOLS, panel_idx + 1, projection='3d')
    pos_sub = data[t][plot_idx]
    ax.scatter(pos_sub[:, 0], pos_sub[:, 1], pos_sub[:, 2],
               s=4, c='steelblue', alpha=0.5, edgecolors='none')
    cm = data[t].mean(axis=0)
    ax.scatter(*cm, s=60, c='red', marker='+', zorder=5)
    frac = t / t_ff
    ax.set_title(f't = {t:.3f}\n({frac:.2f} $t_{{ff}}$)', fontsize=9)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X', fontsize=7, labelpad=2)
    ax.set_ylabel('Y', fontsize=7, labelpad=2)
    ax.set_zlabel('Z', fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.grid(True, lw=0.3, alpha=0.4)

plt.tight_layout()
plt.savefig("gmc_collapse_3d.pdf", dpi=150, bbox_inches='tight')
print("\nSaved: gmc_collapse_3d.pdf")
