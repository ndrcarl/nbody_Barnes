import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
#  raggio_min.py
#  For each particle, finds its first bounce (dr/dt sign change)
#  inside a time window, then plots t_collapse vs r_min.
#
#  Also prints the half-mass radius ratio (final/initial) so
#  that summary_runs.py can parse it.
#
#  treecode output format per snapshot (options=out-phi):
#    N                    line 0
#    3                    line 1  (NDIM, skipped)
#    t                    line 2
#    mass[0..N-1]         N lines
#    x y z [0..N-1]       N lines
#    vx vy vz [0..N-1]    N lines
#    phi[0..N-1]          N lines  ← skipped (not needed here)
#  Total: 3 + 4*N lines per snapshot
#
#  FIXES:
#  1. OFF-BY-ONE FIXED: t_collapse was previously set to times_win[k]
#     (the first *outward* step, one snapshot AFTER the minimum).
#     It is now correctly set to times_win[k-1], which is the actual
#     time of the minimum radius.  With dtout=0.05 this removed a
#     systematic +0.05 bias from every run.
#  2. GLOBAL ESTIMATOR added: t_ff_sim (global min of median r) is
#     computed as a noise-resistant cross-check on the per-particle
#     bounce detection.
#  3. Bounce detection still requires MIN_RISING_STEPS consecutive
#     outward steps before confirming.
#  4. Explicit phi skip added for robustness (treecode run with
#     options=out-phi appends N phi lines per snapshot).
# ============================================================

N    = 10000
R    = 1.0
mass = 0.0001
G    = 1.0

rho_0 = (N * mass) / ((4.0 / 3.0) * math.pi * R**3)
t_ff  = math.sqrt(3.0 * math.pi / (32.0 * G * rho_0))

# Time window to search for first bounce
t1 = 0.8 * t_ff
t2 = 1.8 * t_ff

# Minimum number of consecutive outward steps required to confirm a bounce.
# With dtout = 1/20 = 0.05 code units this is 2 × 0.05 = 0.10 code units
# of sustained outward motion, which filters single-step noise while still
# resolving the sharp collapse.
MIN_RISING_STEPS = 2


# ============================================================
#  Parser  (phi block is present but not needed — skip it)
# ============================================================
def read_treecode(filepath, N):
    times = []
    radii = []

    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            try:
                line_n = int(line.strip())
                if line_n != N:
                    continue
            except ValueError:
                continue

            f.readline()                          # skip NDIM
            t = float(f.readline().strip())       # time

            for _ in range(N): f.readline()       # skip masses

            pos = np.empty((N, 3))
            for j in range(N):
                pos[j] = np.fromstring(f.readline(), sep=' ')

            for _ in range(N): f.readline()       # skip velocities
            for _ in range(N): f.readline()       # skip phi

            times.append(t)
            radii.append(np.linalg.norm(pos, axis=1))

    return np.array(times), np.array(radii)


# ============================================================
#  Load
# ============================================================
times, radii = read_treecode("gmc_internal.out", N)
if len(times) == 0:
    print("Error: no snapshots read from gmc_internal.out")
    raise SystemExit(1)

print(f"Loaded {len(times)} snapshots  "
      f"(t: {times[0]:.3f} → {times[-1]:.3f})")


# ============================================================
#  Global collapse time estimator
#
#  Track the median radius of ALL particles over time.  The time
#  of its global minimum is a single, noise-resistant estimate of
#  t_collapse that does not require per-particle bounce detection.
# ============================================================
med_r_all = np.median(radii, axis=1)
t_ff_sim  = times[np.argmin(med_r_all)]
print(f"\nt_ff (analytical)              = {t_ff:.4f}")
print(f"t_ff_sim (global min median r) = {t_ff_sim:.4f}  "
      f"(bias = {t_ff_sim - t_ff:+.4f})")


# ============================================================
#  Half-mass radius ratio  (printed for summary_runs.py)
#
#  NOTE: this uses the median over ALL particles (bound + unbound).
#  The bound-only ratio is computed in summary_runs.py.
# ============================================================
r_hm_initial = np.median(radii[0,  :])
r_hm_final   = np.median(radii[-1, :])
ratio         = r_hm_final / r_hm_initial
print(f"Final/initial ratio: {ratio:.4f}  (all particles; see summary_runs.py for bound-only)")


# ============================================================
#  Restrict to collapse time window
# ============================================================
mask = (times >= t1) & (times <= t2)
if not np.any(mask):
    print(f"Error: no snapshots in window [{t1:.3f}, {t2:.3f}]")
    raise SystemExit(1)

times_win = times[mask]
radii_win = radii[mask]

print(f"\nCollapse-search window: [{t1:.3f}, {t2:.3f}]  "
      f"({len(times_win)} snapshots)")


# ============================================================
#  Collapse time per particle: first confirmed bounce
#
#  BUG FIX (off-by-one): t_collapse is set to times_win[k-1]
#  (the snapshot of the actual minimum radius), not times_win[k]
#  (the first outward step, which is one dtout=0.05 later).
#
#  A bounce at step k is accepted only if the radius increases
#  for at least MIN_RISING_STEPS consecutive steps after k-1.
# ============================================================
r0         = radii[0, :]
t_collapse = np.full(N, np.nan)
r_min      = np.full(N, np.nan)

n_win = len(times_win)

for p in range(N):
    r_p = radii_win[:, p]
    for k in range(1, n_win - MIN_RISING_STEPS + 1):
        if r_p[k] > r_p[k - 1]:
            confirmed = all(
                r_p[k + s] > r_p[k + s - 1]
                for s in range(1, MIN_RISING_STEPS)
            )
            if confirmed:
                t_collapse[p] = times_win[k - 1]
                r_min[p]      = r_p[k - 1]
                break


# ============================================================
#  Print statistics  (parsed by summary_runs.py)
# ============================================================
n_found = np.sum(~np.isnan(t_collapse))
print(f"\nt_ff  (analytical)        = {t_ff:.4f}")
print(f"Snapshots with a bounce:  {n_found} / {N}")
print(f"median(t_collapse)        = {np.nanmedian(t_collapse):.4f}")
print(f"median(r_at_collapse)     = {np.nanmedian(r_min):.4f}")
print(f"median(t_collapse - t_ff) = {np.nanmedian(t_collapse - t_ff):+.4f}")

n_out = int(np.sum(radii[1, :] > radii[0, :]))
print(f"Particles moving out at t0: {n_out} / {N}")


# ============================================================
#  Plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

sc = ax.scatter(t_collapse, r_min, c=r0, cmap='viridis',
                s=8, alpha=0.6, zorder=3)
ax.axvline(t_ff, color='r', lw=1.5, ls='--',
           label=f'$t_{{ff}}$ (analytical) = {t_ff:.3f}')
ax.axvline(t_ff_sim, color='green', lw=1.5, ls='-.',
           label=f'$t_{{ff,sim}}$ (global) = {t_ff_sim:.3f}')
ax.axvline(np.nanmedian(t_collapse), color='blue', lw=1.2, ls=':',
           label=f'median $t_{{collapse}}$ = {np.nanmedian(t_collapse):.3f}')

cb = plt.colorbar(sc, ax=ax)
cb.set_label('$r_0$ (initial radius)')

ax.set_xlabel('$t_{\\mathrm{collapse}}$ (time of radius minimum)')
ax.set_ylabel('$r$ at bounce minimum')
ax.set_title(f'Collapse time per particle  '
             f'(treecode, N = {N},  min_rising_steps = {MIN_RISING_STEPS})\n'
             f'[off-by-one fixed: t_collapse = time of minimum, not first outward step]')
ax.legend(fontsize=9)
ax.grid(True, lw=0.4, alpha=0.4)

plt.tight_layout()
plt.savefig("t_collapse_scatter.pdf", dpi=200)
print("\nSaved: t_collapse_scatter.pdf")
