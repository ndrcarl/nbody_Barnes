import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
#  raggio.py
#  Plots r(t) for all N particles from a treecode output file.
#  Updated for options=phi: skips the extra phi block.
# ============================================================

N    = 10000
R    = 1.0
mass = 0.0001
G    = 1.0

rho_0 = (N * mass) / ((4.0 / 3.0) * math.pi * R**3)
t_ff  = math.sqrt(3.0 * math.pi / (32.0 * G * rho_0))

# ============================================================
#  Parser (skip phi)
# ============================================================
def read_treecode(filepath, N):
    times = []
    radii = []
    
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            
            try:
                line_n = int(line.strip())
                if line_n != N: continue 
            except ValueError:
                continue

            f.readline()  # skip NDIM
            t = float(f.readline().strip())
            
            # skip masses
            for _ in range(N): f.readline()
            
            # read positions
            pos = np.empty((N, 3))
            for j in range(N):
                pos[j] = np.fromstring(f.readline(), sep=' ')
            
            # skip velocities
            for _ in range(N): f.readline()
            
            # skip phi (added when options=phi)
            for _ in range(N): f.readline()

            times.append(t)
            radii.append(np.linalg.norm(pos, axis=1))

    return np.array(times), np.array(radii)

# ============================================================
#  Load data
# ============================================================
times, radii = read_treecode("gmc_internal.out", N)
if len(times) == 0:
    print("Error: no snapshots read from gmc_internal.out")
    raise SystemExit(1)

print(f"Loaded {len(times)} snapshots  (t: {times[0]:.3f} → {times[-1]:.3f})")

# ============================================================
#  Plot
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

med = np.median(radii, axis=1)
lo  = np.min(radii,    axis=1)
hi  = np.max(radii,    axis=1)

ax.fill_between(times, lo, hi, color='steelblue', alpha=0.15,
                label='min–max range')
ax.plot(times, np.percentile(radii, 25, axis=1),
        color='steelblue', lw=0.8, ls='--', alpha=0.6)
ax.plot(times, np.percentile(radii, 75, axis=1),
        color='steelblue', lw=0.8, ls='--', alpha=0.6,
        label='25th–75th percentile')
ax.plot(times, med, color='steelblue', lw=2.2, label='median $r$')

rng       = np.random.default_rng(42)
plot_idx  = rng.choice(N, size=min(300, N), replace=False)
for i in plot_idx:
    ax.plot(times, radii[:, i], color='k', lw=0.3, alpha=0.12)

ax.axvline(t_ff, color='red', lw=1.5, ls='--',
           label=f'$t_{{ff}}$ = {t_ff:.3f}')
ax.set_xlabel('Time [code units]')
ax.set_ylabel('Radius $r$ [code units]')
ax.set_title(f'Homogeneous sphere collapse: $r(t)$  (N = {N}, treecode)')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.grid(True, lw=0.4, alpha=0.4)

plt.tight_layout()
plt.savefig("r_vs_t.pdf", dpi=200)
print("Saved: r_vs_t.pdf")
