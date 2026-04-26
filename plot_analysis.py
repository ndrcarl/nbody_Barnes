import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
#  plot_analysis.py
#  Energy analysis of the final treecode snapshot:
#    - per-particle energy E_i = ½v²_i + φ_i
#    - φ_i is the treecode's own N-body potential when options=phi
#      is used (HAS_PHI=True), otherwise falls back to the mean-field
#      approximation Φ_mf(r) = -G*M_tot / sqrt(r² + ε²)
#    - scatter: initial radius vs final energy, coloured bound/unbound
#    - prints "Unbound particles: X / N"  ← parsed by summary_runs.py
#
#  treecode output format per snapshot:
#    N                    line 0
#    3                    line 1  (NDIM, skipped)
#    t                    line 2
#    mass[0..N-1]         N lines
#    x y z [0..N-1]       N lines
#    vx vy vz [0..N-1]    N lines
#    phi[0..N-1]          N lines  ← only present when HAS_PHI = True
#  Total: 3 + 3*N lines (no phi) or 3 + 4*N lines (with phi)
#
#  FIXES vs original:
#  1. HAS_PHI flag: when True (requires options=phi in run.sh),
#     the treecode's own self-consistent N-body potential is used
#     instead of the mean-field approximation.  The mean-field
#     over-estimates |Φ| for core particles (treating all mass as
#     a point at r=0), causing them to be misclassified as unbound
#     — which is why the old code gave ~40% escape vs. ~20% from
#     summary_runs.py.
#  2. Parser updated to read the phi block when present.
#  3. Energy formula and labels updated accordingly.
# ============================================================

# ---- Configuration ----
# Set HAS_PHI = True if run.sh used options=phi in the treecode call.
# When True, the exact N-body potential is used for bound/unbound;
# when False, the mean-field approximation is used as a fallback.
HAS_PHI     = True   # ← change to True once you re-run with options=phi

N_PARTICLES = 10000
MASS        = 0.0001
G           = 1.0
import sys
EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001    # softening (same as treecode run parameter)

M_tot = N_PARTICLES * MASS
rho_0 = M_tot / ((4.0 / 3.0) * math.pi * 1.0**3)
t_ff  = math.sqrt(3.0 * math.pi / (32.0 * G * rho_0))


# ============================================================
#  Parser
# ============================================================
def read_treecode(filepath, N, has_phi=False):
    """
    Read all snapshots.  Returns list of dicts with keys:
        time, pos (N,3), vel (N,3), mass (N,), phi (N,) or None
    has_phi: set True when treecode was run with options=phi.
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Lines per snapshot depend on whether phi is present
    lines_per_snap = 3 + 3 * N + (N if has_phi else 0)
    snapshots = []
    i = 0

    while i + lines_per_snap <= len(lines):
        t = float(lines[i + 2].strip())
        i += 3

        masses = np.array([float(lines[i + j]) for j in range(N)])
        i += N

        pos = np.empty((N, 3))
        for j in range(N):
            pos[j] = lines[i + j].split()
        i += N

        vel = np.empty((N, 3))
        for j in range(N):
            vel[j] = lines[i + j].split()
        i += N

        # Read gravitational potential per particle (specific, i.e. per unit mass)
        phi = None
        if has_phi:
            phi = np.array([float(lines[i + j]) for j in range(N)])
            i += N

        snapshots.append({'time': t, 'pos': pos, 'vel': vel,
                          'mass': masses, 'phi': phi})

    return snapshots


# ============================================================
#  Load snapshots
# ============================================================
snapshots = read_treecode("gmc_internal.out", N_PARTICLES, has_phi=HAS_PHI)
if not snapshots:
    print("Error: no data read from gmc_internal.out")
    raise SystemExit(1)

initial = snapshots[0]
final   = snapshots[-1]
print(f"Loaded {len(snapshots)} snapshots")
print(f"  t_initial = {initial['time']:.4f}")
print(f"  t_final   = {final['time']:.4f}")
print(f"  HAS_PHI   = {HAS_PHI}")


# ============================================================
#  Per-particle specific energy at final snapshot
#
#  CASE 1 — HAS_PHI = True (preferred, requires options=phi in run.sh):
#    E_i = ½ v²_i  +  φ_i
#    φ_i is the treecode's own softened N-body potential, computed
#    self-consistently during the integration.  This is the exact
#    criterion: E_i < 0 ↔ bound.
#
#  CASE 2 — HAS_PHI = False (fallback, less accurate):
#    E_i = ½ v²_i  +  Φ_mf(r_i)
#    Φ_mf(r) = -G M_tot / sqrt(r² + ε²)
#    This is the Plummer sphere potential with all mass at the origin.
#    It is a reasonable approximation for escaped particles (r >> core),
#    but OVER-ESTIMATES |Φ| for core particles, making them appear
#    unbound when they are actually deeply bound.  Use only as a
#    fallback when phi is not available.
# ============================================================
def specific_energy_meanfield(pos, vel, M_tot, G, eps):
    """Mean-field (Plummer) approximation — fallback only."""
    r   = np.linalg.norm(pos, axis=1)
    v2  = np.sum(vel**2, axis=1)
    phi = -G * M_tot / np.sqrt(r**2 + eps**2)
    return 0.5 * v2 + phi


if HAS_PHI and final['phi'] is not None:
    # Exact: treecode's own self-consistent N-body potential
    # phi is already the specific potential (energy per unit mass)
    e_final = 0.5 * np.sum(final['vel']**2, axis=1) + final['phi']
    method_label = "N-body φ (treecode)"
else:
    # Fallback: mean-field Plummer approximation
    e_final = specific_energy_meanfield(
        final['pos'], final['vel'], M_tot, G, EPS)
    method_label = "mean-field Φ_mf (fallback — less accurate)"

e_initial = specific_energy_meanfield(
    initial['pos'], initial['vel'], M_tot, G, EPS)

unbound   = e_final > 0
n_unbound = int(np.sum(unbound))
n_bound   = N_PARTICLES - n_unbound

# ----  output lines parsed by summary_runs.py  ----
print(f"Energy method: {method_label}")
print(f"Unbound particles: {n_unbound} / {N_PARTICLES}")
print(f"Unbound fraction:  {100.0 * n_unbound / N_PARTICLES:.2f}%")
# ---------------------------------------------------


# ============================================================
#  Plot: initial radius vs final energy
# ============================================================
r_initial = np.linalg.norm(initial['pos'], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f'Energy analysis  —  t_final = {final["time"]:.2f}  (t_ff = {t_ff:.3f})\n'
    f'Energy method: {method_label}',
    fontsize=11)

# --- Left panel: scatter by bound/unbound ---
ax = axes[0]
ax.scatter(r_initial[~unbound], e_final[~unbound],
           s=2, color='steelblue', alpha=0.4, label=f'Bound  ({n_bound})')
ax.scatter(r_initial[unbound],  e_final[unbound],
           s=2, color='crimson',   alpha=0.4, label=f'Unbound ({n_unbound})')
ax.axhline(0, color='k', lw=1.2, ls='--', label='$E = 0$')
ax.set_xlabel('Initial radius  $r_0$  [code units]')
ax.set_ylabel('Final specific energy  $E$  [code units]')
ax.set_title('Bound / Unbound by initial position')
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, lw=0.4, alpha=0.4)

# --- Right panel: energy distribution histogram ---
ax = axes[1]
bins = np.linspace(np.percentile(e_final, 1),
                   np.percentile(e_final, 99), 80)
ax.hist(e_final[~unbound], bins=bins, color='steelblue',
        alpha=0.7, label='Bound')
ax.hist(e_final[unbound],  bins=bins, color='crimson',
        alpha=0.7, label='Unbound')
ax.axvline(0, color='k', lw=1.5, ls='--', label='$E = 0$')
ax.set_xlabel('Final specific energy  $E$')
ax.set_ylabel('Number of particles')
ax.set_title('Energy distribution at final snapshot')
ax.legend(fontsize=9)
ax.grid(True, lw=0.4, alpha=0.4)

plt.tight_layout()
plt.savefig("escape_analysis.pdf", dpi=200)
print("Saved: escape_analysis.pdf")
