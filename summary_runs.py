# ============================================================
#  summary_runs.py — treecode version, lazy reading
#
#  TWO MODES:
#
#  PER-EPS MODE  (called from run.sh as):
#    python3 summary_runs.py <eps_dir>
#    - chdirs into eps_dir
#    - parses the per-eps log, reads snapshot files lazily
#    - saves per-eps PDFs inside eps_dir
#    - serialises all aggregated stats to eps_dir/summary_stats.npz
#      (tiny file: ~few MB — just time-series arrays, no raw particles)
#
#  COMBINED MODE  (called once after all eps are done):
#    python3 summary_runs.py --combined
#    - scans cwd for eps_*/summary_stats.npz
#    - loads only those small npz files (no snapshot re-reading)
#    - produces combined/ plots with one colour per eps value
#    - RAM usage: negligible (aggregated arrays only)
#
#  LAZY READING (per-eps mode):
#    Snapshot files are streamed one snapshot at a time.
#    Only one run's data is in RAM at a time.
#    Raw arrays (pos, vel, phi) are deleted immediately after use.
#
#  QUANTITIES COMPUTED:
#  - t_collapse vs t_ff analytical
#  - energy conservation error |ΔE/E| — quality control
#  - N_unbound (exact phi criterion when HAS_PHI=True)
#  - R_hm,f/R_hm,i bound only (theoretical ≈ 0.5)
#  - 2K/|W| bound only at t_final (theoretical = 1)
#  - sigma_v(t), mean_vr(t), sigma_vr/sigma_vt(t)
#  - r(t) for all / bound / unbound particles
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
import glob
import math
import warnings

# ============================================================
#  MODE DETECTION
# ============================================================

COMBINED_MODE = (len(sys.argv) > 1 and sys.argv[1] == "--combined")

if not COMBINED_MODE:
    # Per-eps mode: chdir into the given eps directory
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    BASE_DIR = os.getcwd()   # now inside eps_dir


# ============================================================
#  COMBINED MODE — load npz files and plot, then exit
# ============================================================

if COMBINED_MODE:
    BASE_DIR = os.getcwd()
    npz_files = sorted(glob.glob(os.path.join(BASE_DIR, "eps_*", "summary_stats.npz")))
    if not npz_files:
        print("Combined mode: no eps_*/summary_stats.npz files found.")
        print("Run per-eps mode first (summary_stats.npz is written automatically).")
        sys.exit(1)

    print(f"Combined mode: loading {len(npz_files)} eps groups.")
    _CMAP = plt.get_cmap("tab10")

    # ---- load all npz ----
    groups = []
    for fpath in npz_files:
        d   = np.load(fpath, allow_pickle=False)
        tag = os.path.basename(os.path.dirname(fpath))   # e.g. eps_0p007
        groups.append({"tag": tag, "path": fpath, "d": d})
        print(f"  Loaded: {tag}")

    n_eps  = len(groups)
    colors = [_CMAP(i % 10) for i in range(n_eps)]
    tags   = [g["tag"] for g in groups]

    # ---- helper: plot_band for combined ----
    def plot_band_c(ax, t, mat, color, label, alpha=0.15, lw=2.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(mat, axis=0)
            lo  = np.nanmin(mat,   axis=0)
            hi  = np.nanmax(mat,   axis=0)
        valid = np.isfinite(med)
        if not np.any(valid):
            return
        ax.fill_between(t[valid], lo[valid], hi[valid], color=color, alpha=alpha)
        ax.plot(t[valid], med[valid], color=color, lw=lw, label=label)

    outdir = os.path.join(BASE_DIR, "combined")
    os.makedirs(outdir, exist_ok=True)

    # recover t_ff from first group (same for all)
    t_ff_c = float(groups[0]["d"]["t_ff"])
    N_c    = int(groups[0]["d"]["N"])

    # ----------------------------------------------------------------
    # FIG C1: scalar statistics vs eps — one point per run, one
    #         colour per eps.  Shows individual scatter + median line.
    # ----------------------------------------------------------------
    figC1, axC = plt.subplots(2, 2, figsize=(13, 8))
    figC1.suptitle(f"Collapse statistics — all eps values  (N={N_c})\n"
                   "each point = one run,  dotted line = median per eps", fontsize=12)

    axC[0,0].axhline(t_ff_c, color='red', lw=1.5, ls='--',
                     label=f'$t_{{ff}}={t_ff_c:.3f}$', zorder=1)
    axC[0,1].axhline(0.5, color='red', lw=1.5, ls='--',
                     label='theoretical = 0.5', zorder=1)
    axC[1,0].axhline(1.0, color='red', lw=1.5, ls='--',
                     label='virial equilibrium', zorder=1)

    for g, c in zip(groups, colors):
        d   = g["d"]
        tag = g["tag"]
        good = d["good"].astype(bool)
        rn   = d["run_numbers"]
        tc   = d["t_collapse"]
        rhm  = d["r_hm_ratio_bound"]
        vir  = d["virial_bound_final"]
        dt   = d["delta_t"]
        nu   = d["n_unbound"]

        # t_collapse
        ax = axC[0,0]
        ax.scatter(rn[good], tc[good], color=c, s=55, zorder=3, label=tag, alpha=0.85)
        ax.axhline(np.nanmedian(tc[good]), color=c, lw=1.2, ls=':', alpha=0.8)

        # R_hm ratio
        ax = axC[0,1]
        ax.scatter(rn[good], rhm, color=c, s=55, zorder=3, label=tag, alpha=0.85)
        ax.axhline(np.nanmedian(rhm), color=c, lw=1.2, ls=':', alpha=0.8)

        # virial
        ax = axC[1,0]
        ax.scatter(rn[good], vir, color=c, s=60, marker='^',
                   zorder=3, label=tag, alpha=0.85)
        ax.axhline(np.nanmedian(vir), color=c, lw=1.2, ls=':', alpha=0.8)

        # R_hm vs delta_t (coloured scatter)
        ax = axC[1,1]
        ax.scatter(dt[good], rhm, color=c, s=60, zorder=3, label=tag,
                   alpha=0.85, edgecolors='k', lw=0.4)

    for ax, yl, tl in [
        (axC[0,0], '$t_{collapse}$',       'Collapse time per run'),
        (axC[0,1], '$R_{hm,f}/R_{hm,i}$',  'Half-mass radius ratio (bound)'),
        (axC[1,0], '$2K/|W|$',             'Virial ratio at $t_{final}$ (bound)'),
        (axC[1,1], '$R_{hm,f}/R_{hm,i}$',  'Collapse depth vs timing offset'),
    ]:
        ax.set_ylabel(yl); ax.set_title(tl)
        ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.4)
    axC[0,0].set_xlabel('Run'); axC[0,1].set_xlabel('Run')
    axC[1,0].set_xlabel('Run'); axC[1,1].set_xlabel('$\\Delta t = t_{coll} - t_{ff}$')

    plt.tight_layout()
    out = os.path.join(outdir, "combined_collapse.pdf")
    plt.savefig(out, dpi=300); plt.close(figC1)
    print(f"Saved: {out}")

    # ----------------------------------------------------------------
    # FIG C2: median scalar vs eps — the key physics summary
    # ----------------------------------------------------------------
    figC2, axS = plt.subplots(1, 3, figsize=(15, 5))
    figC2.suptitle(f"Median statistics vs softening $\\varepsilon$  (N={N_c})", fontsize=12)

    eps_vals   = []
    med_tc     = []
    med_rhm    = []
    med_vir    = []
    med_unb    = []

    for g in groups:
        d    = g["d"]
        good = d["good"].astype(bool)
        eps_vals.append(float(d["eps_val"]))
        med_tc.append(np.nanmedian(d["t_collapse"][good]))
        med_rhm.append(np.nanmedian(d["r_hm_ratio_bound"]))
        med_vir.append(np.nanmedian(d["virial_bound_final"]))
        med_unb.append(100.0 * np.nanmedian(d["n_unbound"][good]) / N_c)

    eps_arr = np.array(eps_vals)
    axS[0].plot(eps_arr, med_tc,  'o-', lw=2, ms=8, color='steelblue')
    axS[0].axhline(t_ff_c, color='red', lw=1.5, ls='--', label=f'$t_{{ff}}$')
    axS[0].set_xscale('log'); axS[0].set_xlabel('$\\varepsilon$')
    axS[0].set_ylabel('median $t_{collapse}$')
    axS[0].set_title('Collapse time'); axS[0].legend(); axS[0].grid(True, lw=0.4, alpha=0.4)

    axS[1].plot(eps_arr, med_unb, 's-', lw=2, ms=8, color='crimson')
    axS[1].set_xscale('log'); axS[1].set_xlabel('$\\varepsilon$')
    axS[1].set_ylabel('median unbound fraction [%]')
    axS[1].set_title('Unbound fraction'); axS[1].grid(True, lw=0.4, alpha=0.4)

    axS[2].plot(eps_arr, med_rhm, 'D-', lw=2, ms=8, color='teal',  label='$R_{hm}$ ratio')
    axS[2].plot(eps_arr, med_vir, '^-', lw=2, ms=8, color='green', label='$2K/|W|$')
    axS[2].axhline(0.5, color='teal',  lw=1.2, ls=':', alpha=0.7)
    axS[2].axhline(1.0, color='green', lw=1.2, ls=':', alpha=0.7)
    axS[2].set_xscale('log'); axS[2].set_xlabel('$\\varepsilon$')
    axS[2].set_title('$R_{hm}$ ratio  &  virial ratio')
    axS[2].legend(); axS[2].grid(True, lw=0.4, alpha=0.4)

    for ax in axS:
        for e, c in zip(eps_arr, colors):
            ax.axvline(e, color=c, lw=0.6, ls='--', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(outdir, "combined_scalars_vs_eps.pdf")
    plt.savefig(out, dpi=300); plt.close(figC2)
    print(f"Saved: {out}")

    # ----------------------------------------------------------------
    # FIG C3: r(t) — one band per eps
    # ----------------------------------------------------------------
    figC3, axR = plt.subplots(1, 3, figsize=(18, 5))
    figC3.suptitle(f"$r(t)$ across all eps values  (N={N_c})\n"
                   "(line = median of runs, shading = run-to-run scatter)", fontsize=12)
    for g, c in zip(groups, colors):
        d = g["d"]; t = d["times_ref"]
        plot_band_c(axR[0], t, d["r_med_mat"],   c, g["tag"])
        plot_band_c(axR[1], t, d["r_med_b_mat"], c, g["tag"])
        plot_band_c(axR[2], t, d["r_med_u_mat"], c, g["tag"])
    for ax, tl in zip(axR, ['All particles', 'Bound particles', 'Unbound particles']):
        ax.axvline(t_ff_c, color='red', lw=1.5, ls='--', alpha=0.7, label='$t_{ff}$')
        ax.set_xlabel('t'); ax.set_ylabel('r'); ax.set_title(tl)
        ax.set_ylim(bottom=0); ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_r_vs_t.pdf")
    plt.savefig(out, dpi=300); plt.close(figC3)
    print(f"Saved: {out}")

    # ----------------------------------------------------------------
    # FIG C4: velocity time series — one band per eps
    # ----------------------------------------------------------------
    figC4, axV = plt.subplots(1, 3, figsize=(18, 5))
    figC4.suptitle(f"Velocity evolution across all eps values  (N={N_c})\n"
                   "(line = median, shading = run envelope)", fontsize=12)
    for g, c in zip(groups, colors):
        d = g["d"]; t = d["times_ref"]
        plot_band_c(axV[0], t, d["sv_mat"],    c, g["tag"])
        plot_band_c(axV[1], t, d["mvr_mat"],   c, g["tag"])
        plot_band_c(axV[2], t, d["ratio_mat"], c, g["tag"])
    for ax in axV:
        ax.axvline(t_ff_c, color='red', lw=1.5, ls='--', alpha=0.8, label='$t_{ff}$')
        ax.set_xlabel('Time [code units]')
        ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.4)
    axV[0].set_ylabel('$\\sigma_v$');          axV[0].set_title('Total velocity dispersion')
    axV[1].axhline(0, color='k', lw=0.8, ls=':')
    axV[1].set_ylabel('$\\langle v_r \\rangle$'); axV[1].set_title('Mean radial velocity')
    axV[2].axhline(1.0, color='k', lw=0.8, ls=':')
    axV[2].set_ylabel('$\\sigma_{v_r}/\\sigma_{v_t}$')
    axV[2].set_title('Radial/tangential dispersion ratio')
    plt.tight_layout()
    out = os.path.join(outdir, "combined_velocities.pdf")
    plt.savefig(out, dpi=300); plt.close(figC4)
    print(f"Saved: {out}")

    # ----------------------------------------------------------------
    # FIG C5: bound vs unbound velocity — one band per eps, 2 rows
    # ----------------------------------------------------------------
    figC5, axBU = plt.subplots(2, n_eps, figsize=(5*n_eps, 8), squeeze=False)
    figC5.suptitle(f"Bound vs Unbound velocity  (N={N_c})\n"
                   "top = $\\sigma_v$,  bottom = $\\langle v_r \\rangle$  |  each column = one eps",
                   fontsize=12)
    for col, (g, c) in enumerate(zip(groups, colors)):
        d = g["d"]; t = d["times_ref"]
        ax = axBU[0, col]
        plot_band_c(ax, t, d["sv_b_mat"],  'steelblue', 'Bound',   alpha=0.15)
        plot_band_c(ax, t, d["sv_u_mat"],  'crimson',   'Unbound', alpha=0.15)
        ax.axvline(t_ff_c, color='k', lw=1.0, ls='--', alpha=0.5)
        ax.set_title(g["tag"]); ax.set_xlabel('t'); ax.set_ylabel('$\\sigma_v$')
        ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.4)

        ax = axBU[1, col]
        plot_band_c(ax, t, d["mvr_b_mat"], 'steelblue', 'Bound',   alpha=0.15)
        plot_band_c(ax, t, d["mvr_u_mat"], 'crimson',   'Unbound', alpha=0.15)
        ax.axhline(0, color='k', lw=0.8, ls=':')
        ax.axvline(t_ff_c, color='k', lw=1.0, ls='--', alpha=0.5)
        ax.set_xlabel('t'); ax.set_ylabel('$\\langle v_r \\rangle$')
        ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.4)
    plt.tight_layout()
    out = os.path.join(outdir, "combined_bound_vs_unbound.pdf")
    plt.savefig(out, dpi=300); plt.close(figC5)
    print(f"Saved: {out}")

    print(f"\nAll combined plots written to: {outdir}/")
    plt.show()
    sys.exit(0)


# ============================================================
#  PER-EPS MODE — everything below is unchanged except:
#  1. log regex fixed to match run.sh echo format
#  2. npz save added at the very end
# ============================================================

# ============================================================
#  SECTION 0 — CONFIGURATION
# ============================================================

# Set HAS_PHI = True if run.sh used options phi in the treecode call.
# When True, the exact N-body potential is used for the bound/unbound
# criterion and the bound mask computation is instantaneous.
# When False, the APPROX sampling fallback is used instead.
HAS_PHI = True 


# ============================================================
#  SECTION 1 — PHYSICAL PARAMETERS
# ============================================================

N    = 10000
mass = 0.0001
G    = 1.0
R    = 1.0

M_tot = N * mass
rho_0 = M_tot / ((4.0/3.0) * math.pi * R**3)
t_ff  = math.sqrt(3.0 * math.pi / (32.0 * G * rho_0))

E_ERR_THRESHOLD = 1.0   # reject runs with > 1% energy drift

print("=" * 60)
print("PHYSICAL PARAMETERS")
print(f"  N = {N},  mass per particle = {mass} M_sun")
print(f"  R = {R} pc,  rho_0 = {rho_0:.6f} M_sun/pc^3")
print(f"  t_ff (analytical) = {t_ff:.4f} code units")
print(f"  HAS_PHI = {HAS_PHI}")
print("=" * 60)


# ============================================================
#  SECTION 2 — PARSE THE LOG FILE
# ============================================================

# Find the per-eps log (e.g. master_run_0p007.log).
# Falls back to master_run.log for backwards compatibility.
_logs    = sorted(glob.glob("master_run_*.log"))
LOG_FILE = _logs[0] if _logs else "master_run.log"
print(f"Reading log: {LOG_FILE}")
with open(LOG_FILE) as f:
    log = f.read()

blocks     = re.split(r"(=== eps=[\d.]+ +run \d+ at )", log)
run_blocks = [blocks[i] + blocks[i+1] for i in range(1, len(blocks), 2)]
print(f"\nFound {len(run_blocks)} run blocks in {LOG_FILE}")

# Extract eps value from the log header line (e.g. "########## eps = 0.007 ...")
_eps_match = re.search(r"eps\s*=\s*([\d.eE+\-]+)", log)
EPS_VAL = float(_eps_match.group(1)) if _eps_match else np.nan
print(f"eps value: {EPS_VAL}")
eps_label = os.path.basename(os.getcwd())   # e.g. eps_0p007


def extract_float(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else np.nan

def extract_int(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else -1


# Pre-compile the treecode energy diagnostic pattern.
# Each integration step outputs a header line:
#   "        time   |T+U|       T      -U    -T/U  |Vcom|  |Jtot|  CPUtot"
# followed immediately by a data line:
#   "       0.000 0.59517 0.00000 0.59517 0.00000 0.00000 0.00000   0.001"
# We capture the first 5 columns: time, |T+U|, T, |W|, T/|W|.
# The virial ratio 2T/|W| = 2 × col[4].
ENERGY_PAT = re.compile(
    r"time\s+\|T\+U\|\s+T\s+-U\s+-T/U[^\n]*\n"
    r"([^\n]+)"
)
_ECOLS = [(0, 12), (13, 20), (21, 28), (29, 36), (37, 44)]
def _parse_diag_line(line):
    return [float(line[a:b]) for a, b in _ECOLS]

run_numbers = []
t_collapse  = []
n_unbound   = []
n_outward   = []
energy_err  = []

for block in run_blocks:
    run_numbers.append(extract_int(r"run (\d+)", block))
    t_collapse.append(extract_float(r"median\(t_collapse\)\s*=\s*([\d.]+)", block))
    n_unbound.append(extract_int(r"Unbound particles:\s*(\d+)\s*/", block))
    n_outward.append(extract_int(r"Particles moving out at t0:\s*(\d+)", block))

    # ---- Parse treecode energy diagnostics ----
    # Only |T+U| is used — for the energy conservation error (quality control).
    # The all-particle virial ratio from the log is NOT used: it is polluted
    # by the kinetic energy of escaping particles and has no physical meaning
    # for the virialisation of the bound remnant.  The bound-only virial ratio
    # is computed in Section 8 directly from the snapshot files.
    diag_matches = ENERGY_PAT.findall(block)

    if diag_matches:
        arr   = np.array([_parse_diag_line(line) for line in diag_matches])
        e0    = arr[0, 1]
        e_err = float(np.max(np.abs(arr[:, 1] - e0)) / e0) if e0 > 0 else np.nan
        energy_err.append(e_err)
    else:
        energy_err.append(np.nan)

run_numbers = np.array(run_numbers)
t_collapse  = np.array(t_collapse)
n_unbound   = np.array(n_unbound,  dtype=float)
n_outward   = np.array(n_outward,  dtype=float)
energy_err  = np.array(energy_err)
delta_t     = t_collapse - t_ff


# ============================================================
#  SECTION 3 — QUALITY CONTROL
# ============================================================

bad  = np.abs(energy_err) > E_ERR_THRESHOLD
good = ~bad

print(f"\nQUALITY CONTROL")
print(f"  Bad runs  (excluded): {run_numbers[bad].tolist()}")
print(f"  Good runs (kept):     {run_numbers[good].tolist()}")
print(f"  {good.sum()} / {len(run_numbers)} runs pass quality cut (|ΔE/E| < {E_ERR_THRESHOLD:.0%})\n")


# ============================================================
#  SECTION 4 — SUMMARY TABLE (from log; bound half-mass added in Section 8)
# ============================================================

print("=" * 80)
print(f"{'Run':<6} {'t_coll':>8} {'Δt':>8} {'Unbound':>8} {'E_err':>10}")
print("-" * 80)
for i in range(len(run_numbers)):
    flag = "  X BAD" if bad[i] else ""
    print(f"{run_numbers[i]:<6} {t_collapse[i]:>8.3f} {delta_t[i]:>+8.3f} "
          f"{n_unbound[i]:>8.0f} "
          f"{energy_err[i]:>10.2e}{flag}")
print("-" * 80)
print(f"{'median':<6} {np.nanmedian(t_collapse[good]):>8.3f} "
      f"{np.nanmedian(delta_t[good]):>+8.3f} "
      f"{np.nanmedian(n_unbound[good]):>8.0f}  (good runs only)")
print("=" * 80)

print(f"\nPHYSICAL SUMMARY — from log (good runs):")
print(f"  t_ff analytical:        {t_ff:.4f}")
print(f"  Median t_collapse (pp): {np.nanmedian(t_collapse[good]):.4f}  "
      f"(bias Δt = {np.nanmedian(delta_t[good]):+.4f} = "
      f"{100*np.nanmedian(delta_t[good])/t_ff:.1f}% of t_ff)")
print(f"  Unbound fraction:       {100*np.nanmedian(n_unbound[good])/N:.1f}%  "
      f"({'exact N-body phi' if HAS_PHI else 'mean-field approx'})")


# ============================================================
#  SECTION 5 — FILE READER (treecode format, one snapshot at a time)
#
#  treecode output format per snapshot:
#    N
#    3
#    t
#    mass[1..N]       one per line
#    x y z [1..N]     one per line
#    vx vy vz [1..N]  one per line
#    phi[1..N]        one per line  (only when HAS_PHI = True)
# ============================================================

def iter_snapshots(filepath, N, has_phi=False):
    """
    Generator: yields (t, pos, vel, phi) one snapshot at a time.
    pos and vel are shape (N, 3).
    phi is shape (N,) when has_phi=True, else None.

    Uses sentinel-based parsing (searches for the N sentinel line rather
    than assuming a fixed stride), so it is robust against any alignment
    drift or unexpected content in the file.
    """
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                return
            # Match the N sentinel: must be exactly N as an integer
            try:
                if int(line.strip()) != N:
                    continue
            except ValueError:
                continue

            # Verify NDIM=3 on next line to avoid false sentinel matches
            ndim_line = f.readline()
            try:
                if int(ndim_line.strip()) != 3:
                    continue
            except ValueError:
                continue

            t_line = f.readline()
            if not t_line:
                return
            try:
                t = float(t_line.strip())
            except ValueError:
                continue

            for _ in range(N): f.readline()   # skip masses

            pos = np.empty((N, 3))
            for j in range(N):
                pos[j] = np.fromstring(f.readline(), sep=' ')

            vel = np.empty((N, 3))
            for j in range(N):
                vel[j] = np.fromstring(f.readline(), sep=' ')

            phi = None
            if has_phi:
                phi = np.empty(N)
                for j in range(N):
                    raw = f.readline()
                    if not raw:
                        return   # truncated file — stop gracefully
                    phi[j] = float(raw)

            yield t, pos, vel, phi


# ============================================================
#  SECTION 6 — BOUND/UNBOUND MASK
#
#  When HAS_PHI = True (requires options=phi in run.sh):
#    E_i = ½|v_i|^2 + φ_i < 0  ← exact, uses treecode's own N-body potential
#    This is instant (O(N)) and the most accurate possible criterion.
#
#  When HAS_PHI = False (fallback):
#    APPROX pairwise sum over a random sample of SAMPLE_SIZE particles.
#    Scaled by N/SAMPLE_SIZE to approximate the full N-body potential.
#    Accuracy ~10% for N=10000, SAMPLE_SIZE=1000.  Slow (~20s per run).
# ============================================================

def compute_bound_mask(vel_last, phi_last=None, pos_last=None):
    """
    Returns boolean array shape (N,): True = gravitationally bound.

    If phi_last is provided (HAS_PHI=True), uses the exact criterion:
        E_i = ½|v_i|^2 + phi_i < 0
    Otherwise falls back to the APPROX pairwise sampling approach.
    """
    ke = 0.5 * np.sum(vel_last**2, axis=1)   # specific kinetic energy

    if phi_last is not None:
        # Exact: treecode's own softened N-body specific potential
        return (ke + phi_last) < 0

    # ---- APPROX fallback (HAS_PHI = False) ----
    # Estimate specific potential energy via random sample of neighbours.
    # U_i ≈ -G * mass * sum_{j in sample} m / |r_i - r_j| * (N / SAMPLE_SIZE)
    SAMPLE_SIZE = 1000
    rng   = np.random.default_rng(42)
    idx   = rng.choice(N, size=SAMPLE_SIZE, replace=False)
    pos_s = pos_last[idx]   # (SAMPLE_SIZE, 3)

    U = np.zeros(N)
    for i in range(N):
        dr     = pos_s - pos_last[i]
        r_dist = np.sqrt(np.sum(dr**2, axis=1))
        r_dist = np.where(r_dist == 0, np.inf, r_dist)
        U[i]   = -G * mass * np.sum(1.0 / r_dist) * (N / SAMPLE_SIZE)  # mass once: gives specific potential

    # NOTE: U above is potential ENERGY (not specific); ke is specific KE.
    # To compare consistently: bound iff KE_specific + U_specific < 0
    # where U_specific = U / mass.
    return (ke + U / mass) < 0


# ============================================================
#  SECTION 6b — DENSITY PROFILE HELPER
# ============================================================

def bin_density_profile(pos, cm, mass_per_particle, r_inner=0.0, n_bins=40):
    """
    Compute radial density profile rho(r) for a set of particles.
    pos:  (M, 3) positions
    cm:   (3,) centre of mass to measure distances from
    Returns: r_mid (n_bins,), rho (n_bins,)  — only non-empty bins
    """
    r = np.linalg.norm(pos - cm, axis=1)
    r_fit = r[r > r_inner]
    if len(r_fit) < 3:
        return None, None
    bins   = np.logspace(np.log10(max(r_fit.min(), 1e-5)),
                         np.log10(r_fit.max()), n_bins)
    counts, edges = np.histogram(r_fit, bins=bins)
    r_mid   = 0.5 * (edges[:-1] + edges[1:])
    V_shell = (4.0/3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    rho     = counts * mass_per_particle / V_shell
    ok      = counts > 0
    return r_mid[ok], rho[ok]


def read_mid_snapshot(fpath, N, t_mid_target, has_phi=HAS_PHI):
    """
    Stream through file and return pos of snapshot closest to t_mid_target.
    Memory: only one snapshot in RAM at a time.
    """
    best_pos  = None
    best_diff = np.inf
    for t, pos, vel, phi in iter_snapshots(fpath, N, has_phi=has_phi):
        diff = abs(t - t_mid_target)
        if diff < best_diff:
            best_diff = diff
            best_pos  = pos.copy()
            best_t    = t
    return best_pos, best_t if best_pos is not None else (None, None)


# ============================================================
#  SECTION 7 — VELOCITY STATISTICS (computed snapshot by snapshot)
# ============================================================

def velocity_stats_from_snapshot(pos, vel):
    """
    Compute velocity statistics from a single snapshot.
    pos, vel: shape (M, 3).
    Returns: sigma_v, sigma_vr, sigma_vt, mean_vr  (all scalars)

    All quantities are computed in the CM frame of the passed particles:
    - pos is shifted by the CM position so r_hat points from CM, not origin
    - vel is shifted by the CM velocity so mean_vr is not contaminated
      by bulk CM drift (which causes a spurious positive vr offset)
    """
    # Shift to CM frame of this particle subset
    pos_cm = np.mean(pos, axis=0)   # CM position
    vel_cm = np.mean(vel, axis=0)   # CM velocity
    pos = pos - pos_cm
    vel = vel - vel_cm

    r_mag = np.linalg.norm(pos, axis=1, keepdims=True)
    r_mag = np.where(r_mag == 0, 1e-10, r_mag)
    r_hat = pos / r_mag

    vr = np.sum(vel * r_hat, axis=1)
    v2 = np.sum(vel**2,      axis=1)
    vt = np.sqrt(np.maximum(v2 - vr**2, 0))

    return (np.sqrt(np.mean(v2)),
            np.std(vr),
            np.std(vt),
            np.mean(vr))


# ============================================================
#  SECTION 8 — LAZY LOOP OVER ALL GOOD RUNS
#
#  For each good run we:
#   a) Stream snapshots one at a time to build velocity time series
#   b) Collect last snapshot (pos, vel, phi) for bound mask
#   c) Compute bound mask (exact if HAS_PHI, APPROX otherwise)
#   d) Re-stream to split stats by bound/unbound and compute r(t)
#   e) Delete all large arrays immediately after use
#
#  Accumulators hold only scalar arrays (one value per snapshot per run).
# ============================================================

run_dirs    = sorted([d for d in os.listdir('.') if re.match(r'run_\d+', d)])
good_ids           = []
nfw_rs_list        = []   # NFW scale radius per good run
nfw_rhos_list      = []   # NFW scale density per good run
nfw_ok_list        = []   # fit converged?
nfw_profiles       = []   # full binned profile dicts for plotting
density_run_data   = []   # per-run positions + bound mask for density evolution
r_hm_ratio_bound   = []   # bound-only half-mass radius ratio
N_bound_list       = []
N_unbound_list     = []
virial_bound_final = []   # 2K/|W| bound-only at t=t_final — virialisation test

# Velocity accumulators — shape will be (n_good_runs, n_snaps)
times_ref   = None   # snapshot times (same for all runs)
sv_mat      = []     # total sigma_v
mvr_mat     = []     # mean radial velocity
ratio_mat   = []     # sigma_vr / sigma_vt

# Bound/unbound velocity accumulators
sv_b_mat    = []; sv_u_mat  = []
mvr_b_mat   = []; mvr_u_mat = []

# r(t) accumulators
r_med_mat   = []   # median |r| all particles
r_med_b_mat = []   # median |r| bound
r_med_u_mat = []   # median |r| unbound
r_cm_mat    = []   # |CM| all
r_cm_b_mat  = []   # |CM| bound
r_cm_u_mat  = []   # |CM| unbound
# Vector CM offsets for momentum conservation panel (shape: n_runs x n_snaps x 3)
r_cm_b_vec_mat = []  # cm_b - cm_all  (3D vector)
r_cm_u_vec_mat = []  # cm_u - cm_all  (3D vector)

for idx, d in enumerate(run_dirs):
    fpath = os.path.join(d, "gmc_internal.out")
    if not os.path.exists(fpath):
        print(f"WARNING: {fpath} not found, skipping.")
        continue

    is_good = bool(good[idx]) if idx < len(good) else True
    if not is_good:
        print(f"Skipping bad run {d}")
        continue

    run_id = int(run_numbers[idx]) if idx < len(run_numbers) else idx + 1
    print(f"Processing run {run_id} from {fpath} ...")

    # ---- Pass 1: full velocity stats (all particles) ----
    sv_run    = []; mvr_run  = []; ratio_run = []
    times_run = []
    pos_last  = None
    vel_last  = None
    phi_last  = None   # will hold phi from the last snapshot (or None)
    pos_initial = None  # first snapshot positions for density evolution

    for t, pos, vel, phi in iter_snapshots(fpath, N, has_phi=HAS_PHI):
        sv, svr, svt, mvr = velocity_stats_from_snapshot(pos, vel)
        ratio = svr / svt if svt > 0 else np.nan

        times_run.append(t)
        sv_run.append(sv)
        mvr_run.append(mvr)
        ratio_run.append(ratio)

        # Save first snapshot for density evolution
        if pos_initial is None:
            pos_initial = pos.copy()

        # Keep the last snapshot for bound mask computation
        pos_last = pos.copy()
        vel_last = vel.copy()
        phi_last = phi.copy() if phi is not None else None

    if times_ref is None:
        times_ref = np.array(times_run)

    sv_mat.append(np.array(sv_run))
    mvr_mat.append(np.array(mvr_run))
    ratio_mat.append(np.array(ratio_run))

    # ---- Compute bound mask from last snapshot ----
    method = "exact N-body phi" if (HAS_PHI and phi_last is not None) else "APPROX sampling"
    print(f"  Computing bound mask (N={N}, method={method})...")
    bound = compute_bound_mask(vel_last, phi_last=phi_last, pos_last=pos_last)
    print(f"  Bound: {bound.sum()},  Unbound: {(~bound).sum()}")
    N_bound_list.append(int(bound.sum()))
    N_unbound_list.append(int((~bound).sum()))

    # ---- Bound-only virial ratio at t=t_final (t >> t_ff) ----
    # 2K/|W| → 1 if the bound remnant has reached virial equilibrium.
    # vel_last and phi_last are already in memory from the end of Pass 1.
    # KE = ½ Σ_{bound} |v_i|² × mass,  PE = ½ Σ_{bound} φ_i × mass
    if phi_last is not None and bound.sum() > 0:
        ke_b = 0.5 * np.sum(vel_last[bound] ** 2) * mass
        pe_b = 0.5 * np.sum(phi_last[bound]) * mass
        vir  = 2.0 * ke_b / abs(pe_b) if pe_b != 0 else np.nan
        virial_bound_final.append(vir)
        print(f"  2K/|W| (bound, t=t_final={times_run[-1]:.2f} = {times_run[-1]/t_ff:.1f}*t_ff) = {vir:.4f}")
    else:
        virial_bound_final.append(np.nan)

    # ---- NFW profile fit of bound remnant ----
    # Uses pos_last (last snapshot positions) and bound mask.
    # Bins bound particles radially (log-spaced), fits log(rho) vs log(r)
    # with NFW: rho(r) = rho_s / [(r/rs)*(1+r/rs)^2]
    # Inner cutoff: r > 2*EPS_VAL to avoid softened force region.
    # Outer cutoff: r < r_bound_max (natural edge of bound system).
    # Fit parameters saved for later plotting.
    from scipy.optimize import curve_fit

    _nfw_rs   = np.nan
    _nfw_rhos = np.nan
    _nfw_ok   = False

    if bound.sum() > 10 and not np.isnan(EPS_VAL):
        _cm_b   = np.mean(pos_last[bound], axis=0)
        _r_b    = np.linalg.norm(pos_last[bound] - _cm_b, axis=1)
        _r_cut  = 2.0 * EPS_VAL                      # inner softening cutoff
        _r_b_fit = _r_b[_r_b > _r_cut]

        if len(_r_b_fit) > 10:
            _r_min = _r_b_fit.min()
            _r_max = _r_b_fit.max()
            _bins  = np.logspace(np.log10(_r_min), np.log10(_r_max), 40)
            _counts, _edges = np.histogram(_r_b_fit, bins=_bins)
            _r_mid  = 0.5 * (_edges[:-1] + _edges[1:])
            _V_shell = (4.0/3.0) * np.pi * (_edges[1:]**3 - _edges[:-1]**3)
            _rho    = _counts * mass / _V_shell

            # only fit bins with at least 1 particle
            _ok = _counts > 0
            if _ok.sum() > 4:
                def _nfw_log(log_r, log_rhos, log_rs):
                    r  = np.exp(log_r)
                    rs = np.exp(log_rs)
                    rhos = np.exp(log_rhos)
                    return np.log(rhos / ((r/rs) * (1.0 + r/rs)**2))

                try:
                    _p0 = [np.log(_rho[_ok].max()), np.log(_r_mid[_ok].mean())]
                    _popt, _ = curve_fit(
                        _nfw_log,
                        np.log(_r_mid[_ok]), np.log(_rho[_ok]),
                        p0=_p0, maxfev=5000)
                    _nfw_rhos = np.exp(_popt[0])
                    _nfw_rs   = np.exp(_popt[1])
                    _nfw_ok   = True
                    print(f"  NFW fit: rho_s={_nfw_rhos:.4e}  r_s={_nfw_rs:.4f}")
                except Exception as _e:
                    print(f"  NFW fit failed: {_e}")

    nfw_rs_list.append(_nfw_rs)
    nfw_rhos_list.append(_nfw_rhos)
    nfw_ok_list.append(_nfw_ok)
    # store binned profile for plotting (last run overwrites — we plot all runs)
    nfw_profiles.append({
        "run_id": run_id,
        "r_mid":  _r_mid  if _nfw_ok else None,
        "rho":    _rho    if _nfw_ok else None,
        "rs":     _nfw_rs,
        "rhos":   _nfw_rhos,
        "ok":     _nfw_ok,
        "r_cut":  _r_cut  if not np.isnan(EPS_VAL) else 0,
    })

    # store data needed for density evolution figure (Section 13c)
    density_run_data.append({
        "run_id":      run_id,
        "fpath":       fpath,
        "bound":       bound.copy(),
        "pos_initial": pos_initial.copy() if pos_initial is not None else None,
        "pos_last":    pos_last.copy(),
        "t_final":     times_run[-1],
    })

    del pos_last, vel_last, phi_last

    # ---- Pass 2: velocity stats split by bound/unbound ----
    # and r(t) statistics
    sv_b_run  = []; sv_u_run  = []
    mvr_b_run = []; mvr_u_run = []
    r_med_run = []; r_med_b_run = []; r_med_u_run = []
    r_cm_run  = []; r_cm_b_run  = []; r_cm_u_run  = []
    r_cm_b_vec_run = []; r_cm_u_vec_run = []  # 3D vector offsets
    r_at_initial_b = None   # bound particles' radii at first snapshot

    for snap_idx, (t, pos, vel, phi) in enumerate(
            iter_snapshots(fpath, N, has_phi=HAS_PHI)):

        # --- velocity split ---
        # FIX: compute bound/unbound velocities in the ALL-PARTICLE CM frame,
        # not each subset's own CM frame.  Using subset-CM frames makes the
        # mean radial velocity of the unbound population look like internal
        # spread rather than bulk escape, making the two panels incomparable.
        # The all-particle CM velocity is subtracted once, then r_hat is
        # computed from the all-particle CM position for both subsets.
        vel_cm_all = np.mean(vel, axis=0)
        pos_cm_all = np.mean(pos, axis=0)
        vel_inertial = vel - vel_cm_all
        pos_inertial = pos - pos_cm_all

        if bound.sum() > 0:
            sv_b, _, _, mvr_b = velocity_stats_from_snapshot(
                pos_inertial[bound, :], vel_inertial[bound, :])
        else:
            sv_b, mvr_b = np.nan, np.nan

        if (~bound).sum() > 0:
            sv_u, _, _, mvr_u = velocity_stats_from_snapshot(
                pos_inertial[~bound, :], vel_inertial[~bound, :])
        else:
            sv_u, mvr_u = np.nan, np.nan

        sv_b_run.append(sv_b);   sv_u_run.append(sv_u)
        mvr_b_run.append(mvr_b); mvr_u_run.append(mvr_u)

        # --- r statistics ---
        # Each subset's radii are measured from its OWN CM:
        #   - bound   r from bound CM   → true size of the gravitational remnant
        #   - unbound r from unbound CM → true spread of the escaping shell
        #   - all     r from all-particle CM → overall system size
        # The dashed "CM" line in each panel shows how far that subset's CM
        # has moved relative to the all-particle CM (i.e. the separation).
        cm_all = np.mean(pos, axis=0)   # all-particle CM (reference for CM tracks)

        # All particles: distance from all-particle CM
        r_all = np.linalg.norm(pos - cm_all, axis=1)
        r_med_run.append(np.median(r_all))
        r_cm_run.append(np.linalg.norm(cm_all))   # drift of system from origin

        # Bound: distance from bound CM
        if bound.sum() > 0:
            cm_b = np.mean(pos[bound, :], axis=0)
            r_b  = np.linalg.norm(pos[bound, :] - cm_b, axis=1)
            r_med_b_run.append(np.median(r_b))
            r_cm_b_run.append(np.linalg.norm(cm_b - cm_all))  # bound CM offset from all-CM
            # FIX: store ABSOLUTE CM position (not offset from cm_all).
            # The old code stored (cm_b - cm_all) and then computed
            #   M_b*(cm_b-cm_all) + M_u*(cm_u-cm_all)
            # which equals M_b*cm_b + M_u*cm_u - M_tot*cm_all = 0 by definition
            # of cm_all — a pure tautology that can never show a non-zero residual
            # (the non-zero result seen before was only due to using median N_b/N_u
            # weights instead of the per-run values, which broke the identity).
            # Correct check: the all-particle CM = (sum m_i * r_i)/M_tot
            # should be conserved (= constant in time). We store absolute positions
            # so the plot can show |cm_all(t) - cm_all(0)| as the real residual.
            r_cm_b_vec_run.append(cm_b.copy())   # absolute position of bound CM
        else:
            r_b = np.array([np.nan])
            r_med_b_run.append(np.nan)
            r_cm_b_run.append(np.nan)
            r_cm_b_vec_run.append(np.full(3, np.nan))

        # Unbound: distance from unbound CM
        if (~bound).sum() > 0:
            cm_u = np.mean(pos[~bound, :], axis=0)
            r_u  = np.linalg.norm(pos[~bound, :] - cm_u, axis=1)
            r_med_u_run.append(np.median(r_u))
            r_cm_u_run.append(np.linalg.norm(cm_u - cm_all))  # unbound CM offset from all-CM
            r_cm_u_vec_run.append(cm_u.copy())   # absolute position of unbound CM
        else:
            r_med_u_run.append(np.nan)
            r_cm_u_run.append(np.nan)
            r_cm_u_vec_run.append(np.full(3, np.nan))

        # Store initial radii of bound particles for half-mass ratio
        if snap_idx == 0:
            r_at_initial_b = r_b.copy() if bound.sum() > 0 else np.array([np.nan])

        del pos, vel, phi

    # ---- Bound-only half-mass radius ratio ----
    # R_hm(bound) = median(|r| of bound particles)
    # Initial: where bound particles *were* at t=0
    # Final:   where they *are* at t=t_final
    # Converges to ~0.5 for a relaxed virialized remnant (virial theorem).
    r_hm_b_initial = np.median(r_at_initial_b) if r_at_initial_b is not None else np.nan
    r_hm_b_final   = r_med_b_run[-1] if r_med_b_run else np.nan
    r_hm_ratio_bound.append(
        r_hm_b_final / r_hm_b_initial
        if (np.isfinite(r_hm_b_initial) and r_hm_b_initial > 0) else np.nan)

    sv_b_mat.append(np.array(sv_b_run));   sv_u_mat.append(np.array(sv_u_run))
    mvr_b_mat.append(np.array(mvr_b_run)); mvr_u_mat.append(np.array(mvr_u_run))
    r_med_mat.append(np.array(r_med_run))
    r_med_b_mat.append(np.array(r_med_b_run))
    r_med_u_mat.append(np.array(r_med_u_run))
    r_cm_mat.append(np.array(r_cm_run))
    r_cm_b_mat.append(np.array(r_cm_b_run))
    r_cm_u_mat.append(np.array(r_cm_u_run))
    r_cm_b_vec_mat.append(np.array(r_cm_b_vec_run))  # shape (n_snaps, 3)
    r_cm_u_vec_mat.append(np.array(r_cm_u_vec_run))  # shape (n_snaps, 3)

    del bound
    good_ids.append(run_id)
    print(f"  Done. ({len(good_ids)} good runs processed so far)")

# Convert to 2D arrays: shape (n_good_runs, n_snaps)
# Runs may differ by ±1 snapshot due to the leapfrog not landing exactly
# on tstop.  Truncate every per-run array to the minimum length so that
# np.array() can stack them into a regular 2D matrix.  times_ref is also
# trimmed to match, so all time axes stay consistent.
_all_scalar = [sv_mat, mvr_mat, ratio_mat,
               sv_b_mat, sv_u_mat, mvr_b_mat, mvr_u_mat,
               r_med_mat, r_med_b_mat, r_med_u_mat,
               r_cm_mat, r_cm_b_mat, r_cm_u_mat]
_min_len = min(len(a) for lst in _all_scalar for a in lst) if sv_mat else 0
if times_ref is not None:
    _min_len = min(_min_len, len(times_ref))
    times_ref = times_ref[:_min_len]

sv_mat      = np.array([a[:_min_len] for a in sv_mat])
mvr_mat     = np.array([a[:_min_len] for a in mvr_mat])
ratio_mat   = np.array([a[:_min_len] for a in ratio_mat])
sv_b_mat    = np.array([a[:_min_len] for a in sv_b_mat])
sv_u_mat    = np.array([a[:_min_len] for a in sv_u_mat])
mvr_b_mat   = np.array([a[:_min_len] for a in mvr_b_mat])
mvr_u_mat   = np.array([a[:_min_len] for a in mvr_u_mat])
r_med_mat   = np.array([a[:_min_len] for a in r_med_mat])
r_med_b_mat = np.array([a[:_min_len] for a in r_med_b_mat])
r_med_u_mat = np.array([a[:_min_len] for a in r_med_u_mat])
r_cm_mat    = np.array([a[:_min_len] for a in r_cm_mat])
r_cm_b_mat  = np.array([a[:_min_len] for a in r_cm_b_mat])
r_cm_u_mat  = np.array([a[:_min_len] for a in r_cm_u_mat])
# Vector arrays: shape (n_runs, n_snaps, 3)
r_cm_b_vec_mat = np.array([a[:_min_len] for a in r_cm_b_vec_mat])
r_cm_u_vec_mat = np.array([a[:_min_len] for a in r_cm_u_vec_mat])
r_hm_ratio_bound   = np.array(r_hm_ratio_bound)
virial_bound_final = np.array(virial_bound_final)

print(f"\nProcessed {len(good_ids)} good runs.")

# ---- Extended physical summary (after file processing) ----
if len(r_hm_ratio_bound) > 0:
    t_final = times_ref[-1] if times_ref is not None else np.nan
    print(f"\nPHYSICAL SUMMARY — from file data (good runs):")
    print(f"  R_hm ratio (bound only):    {np.nanmedian(r_hm_ratio_bound):.3f}  "
          f"(theoretical ≈ 0.5 for virialized remnant; "
          f"deviation = {np.nanmedian(r_hm_ratio_bound) - 0.5:+.3f})")
    if np.any(np.isfinite(virial_bound_final)):
        print(f"  2K/|W| bound at t_final:    {np.nanmedian(virial_bound_final):.4f}  "
              f"(t_final = {t_final:.2f} = {t_final/t_ff:.1f} t_ff; "
              f"theoretical = 1.0 for virialized remnant)")


# ============================================================
#  SECTION 9 — PLOT HELPER
# ============================================================

def plot_band(ax, t, mat, color, label):
    """Median (thick line) + full min/max envelope (shading).
    Columns that are all-NaN (e.g. snapshot count mismatch between runs)
    are silently ignored — nanmedian/nanmin/nanmax would warn about them."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(mat, axis=0)
        lo  = np.nanmin(mat,    axis=0)
        hi  = np.nanmax(mat,    axis=0)
    # Only plot where at least one run has valid data
    valid = np.isfinite(med)
    if not np.any(valid):
        return
    ax.fill_between(t[valid], lo[valid], hi[valid], color=color, alpha=0.20)
    ax.plot(t[valid], med[valid], color=color, lw=2.2, label=label)


# ============================================================
#  SECTION 10 — FIGURE 1: COLLAPSE STATISTICS
# ============================================================

fig1, axes = plt.subplots(2, 2, figsize=(11, 7))
fig1.suptitle(f'Summary of {len(run_numbers)} runs — collapse statistics '
              f'({bad.sum()} bad excluded) — treecode N={N}', fontsize=12)

ax = axes[0, 0]
ax.scatter(run_numbers[good], t_collapse[good], color='blue', s=60, zorder=3,
           label='per-particle bounce')
if bad.sum():
    ax.scatter(run_numbers[bad], t_collapse[bad], color='red', s=80, zorder=3,
               marker='x', linewidths=2, label='bad (excluded)')
ax.axhline(t_ff, color='red', lw=1.5, ls='--', label=f'$t_{{ff}}$ = {t_ff:.3f}')
ax.axhline(np.nanmedian(t_collapse[good]), color='blue', lw=1.2, ls=':',
           label=f'median = {np.nanmedian(t_collapse[good]):.3f}')
ax.set_xlabel('Run'); ax.set_ylabel('$t_{collapse}$')
ax.set_title('Collapse time (per-particle bounce)'); ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.4)

ax = axes[0, 1]
ax.scatter(run_numbers[good], r_hm_ratio_bound, color='teal', s=60, zorder=3,
           label='bound only')
if bad.sum():
    ax.scatter(run_numbers[bad], np.full(bad.sum(), np.nan), color='red', s=80, zorder=3,
               marker='x', linewidths=2, label='bad (excluded)')
ax.axhline(0.5, color='red', lw=1.5, ls='--', label='theoretical = 0.5')
ax.axhline(np.nanmedian(r_hm_ratio_bound), color='teal', lw=1.2, ls=':',
           label=f'median = {np.nanmedian(r_hm_ratio_bound):.3f}')
ax.set_xlabel('Run'); ax.set_ylabel('$R_{hm,f}/R_{hm,i}$')
ax.set_title('Half-mass radius ratio (bound only)'); ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.4)

ax = axes[1, 0]
ax.scatter(run_numbers[good], virial_bound_final, color='green', s=80, zorder=3,
           marker='^', label='$2K/|W|$ bound only at $t_{final}$')
ax.axhline(np.nanmedian(virial_bound_final), color='green', lw=1.2, ls=':',
           label=f'median = {np.nanmedian(virial_bound_final):.3f}')
ax.axhline(1.0, color='red', lw=1.5, ls='--', label='virial equilibrium = 1')
ax.set_xlabel('Run'); ax.set_ylabel('$2K/|W|$')
ax.set_title('Virial ratio — bound only at $t_{final}$'); ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.4)

ax = axes[1, 1]
sc = ax.scatter(delta_t[good], r_hm_ratio_bound, c=n_unbound[good],
                cmap='viridis', s=80, zorder=3, edgecolors='k', lw=0.5)
for i in np.where(good)[0]:
    ax.annotate(str(run_numbers[i]), (delta_t[i], r_hm_ratio_bound[i] if i < len(r_hm_ratio_bound) else np.nan),
                textcoords='offset points', xytext=(5, 4), fontsize=7)
plt.colorbar(sc, ax=ax).set_label('N unbound')
ax.set_xlabel('$\\Delta t$'); ax.set_ylabel('$R_{hm,f}/R_{hm,i}$ (bound)')
ax.set_title('Collapse depth vs. timing offset'); ax.grid(True, lw=0.4, alpha=0.4)

plt.tight_layout()
plt.savefig("summary_collapse.pdf", dpi=300)
print("\nSaved: summary_collapse.pdf")


# ============================================================
#  SECTION 11 — FIGURE 2: VELOCITY EVOLUTION (ALL PARTICLES)
# ============================================================

if len(sv_mat) > 0:
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    fig2.suptitle(f'Velocity analysis — {len(good_ids)} good runs (treecode N={N})\n'
                  '(thick = median, shading = full min/max envelope)', fontsize=12)

    plot_band(axes2[0], times_ref, sv_mat,    'blue', '$\\sigma_v$')
    plot_band(axes2[1], times_ref, mvr_mat,   'blue', '$\\langle v_r \\rangle$')
    plot_band(axes2[2], times_ref, ratio_mat, 'blue', '$\\sigma_{v_r}/\\sigma_{v_t}$')

    for ax in axes2:
        ax.axvline(t_ff, color='red', lw=1.5, ls='--', alpha=0.8, label=f'$t_{{ff}}$')
        ax.set_xlabel('Time [code units]')
        ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.4)

    axes2[0].set_ylabel('$\\sigma_v$');          axes2[0].set_title('Total velocity dispersion')
    axes2[1].axhline(0, color='k', lw=0.8, ls=':')
    axes2[1].set_ylabel('$\\langle v_r \\rangle$'); axes2[1].set_title('Mean radial velocity')
    axes2[2].axhline(1.0, color='k', lw=0.8, ls=':', label='isotropic')
    axes2[2].set_ylabel('$\\sigma_{v_r}/\\sigma_{v_t}$')
    axes2[2].set_title('Radial/tangential dispersion ratio')

    plt.tight_layout()
    plt.savefig("summary_velocities.pdf", dpi=300)
    print("Saved: summary_velocities.pdf")


# ============================================================
#  SECTION 12 — FIGURE 3: BOUND vs UNBOUND VELOCITY
# ============================================================

if len(sv_b_mat) > 0:
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle(f'Bound vs Unbound — velocity evolution (treecode N={N})\n'
                  '(thick = median, shading = full range)', fontsize=11)

    ax = axes3[0]
    plot_band(ax, times_ref, sv_b_mat,  'steelblue', 'Bound')
    plot_band(ax, times_ref, sv_u_mat,  'crimson',   'Unbound')
    ax.axvline(t_ff, color='k', lw=1.2, ls='--', alpha=0.6, label=f'$t_{{ff}}$')
    ax.set_xlabel('Time [code units]'); ax.set_ylabel('$\\sigma_v$')
    ax.set_title('Velocity dispersion'); ax.legend(fontsize=9); ax.grid(True, lw=0.4, alpha=0.4)

    ax = axes3[1]
    plot_band(ax, times_ref, mvr_b_mat, 'steelblue', 'Bound')
    plot_band(ax, times_ref, mvr_u_mat, 'crimson',   'Unbound')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.axvline(t_ff, color='k', lw=1.2, ls='--', alpha=0.6, label=f'$t_{{ff}}$')
    ax.set_xlabel('Time [code units]'); ax.set_ylabel('$\\langle v_r \\rangle$')
    ax.set_title('Mean radial velocity\n(negative=infalling, positive=escaping)')
    ax.legend(fontsize=9); ax.grid(True, lw=0.4, alpha=0.4)

    plt.tight_layout()
    plt.savefig("summary_bound_vs_unbound.pdf", dpi=300)
    print("Saved: summary_bound_vs_unbound.pdf")


# ============================================================
#  SECTION 13 — FIGURE 4: r(t) WITH CENTER OF MASS
# ============================================================

if len(r_med_mat) > 0:
    fig4, axes4 = plt.subplots(1, 4, figsize=(20, 5))
    fig4.suptitle(f'$r(t)$ across {len(good_ids)} good runs (treecode N={N})\n'
                  '(thick = median of median-r, shading = full range; dashed = CM offset from all-particle CM)',
                  fontsize=11)

    ax = axes4[0]
    plot_band(ax, times_ref, r_med_mat, 'grey', 'median $|r|$')
    ax.plot(times_ref, np.nanmedian(r_cm_mat, axis=0),
            color='black', lw=2.0, ls='--', label='CM drift from origin')
    ax.axvline(t_ff, color='red', lw=1.2, ls='--', alpha=0.7, label=f'$t_{{ff}}$')
    ax.set_xlabel('t'); ax.set_ylabel('r')
    ax.set_title('All particles'); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.4)

    ax = axes4[1]
    plot_band(ax, times_ref, r_med_b_mat, 'steelblue', 'median $|r_{bound} - CM_{bound}|$')
    ax.plot(times_ref, np.nanmedian(r_cm_b_mat, axis=0),
            color='steelblue', lw=2.0, ls='--', label='bound CM offset from all-CM')
    ax.axvline(t_ff, color='red', lw=1.2, ls='--', alpha=0.7, label=f'$t_{{ff}}$')
    ax.set_xlabel('t'); ax.set_ylabel('r')
    ax.set_title('Bound particles\n(r from bound CM)'); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.4)

    ax = axes4[2]
    plot_band(ax, times_ref, r_med_u_mat, 'crimson', 'median $|r_{unbound} - CM_{unbound}|$')
    ax.plot(times_ref, np.nanmedian(r_cm_u_mat, axis=0),
            color='crimson', lw=2.0, ls='--', label='unbound CM offset from all-CM')
    ax.axvline(t_ff, color='red', lw=1.2, ls='--', alpha=0.7, label=f'$t_{{ff}}$')
    ax.set_xlabel('t'); ax.set_ylabel('r')
    ax.set_title('Unbound particles\n(r from unbound CM)'); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8); ax.grid(True, lw=0.4, alpha=0.4)

    ax = axes4[3]
    # ----------------------------------------------------------------
    # Correct momentum-conservation check:
    #   The all-particle CM = (Σ m_i r_i) / M_tot must be constant in
    #   time (no external forces).  We plot |CM_all(t) - CM_all(t=0)|
    #   for each run as the true residual, plus the individual motions
    #   of the bound and unbound sub-CMs for physical context.
    #
    # Previous (wrong) approach: stored (cm_b - cm_all) and (cm_u - cm_all)
    # and computed M_b*(cm_b-cm_all) + M_u*(cm_u-cm_all).  This is
    # identically zero by definition of cm_all — a tautology that told
    # us nothing.  The non-zero result seen before was purely an artefact
    # of using global-median N_b/N_u weights instead of per-run values,
    # which inconsistently broke the algebraic identity.
    # ----------------------------------------------------------------
    # r_cm_b_vec_mat and r_cm_u_vec_mat now hold absolute CM positions.
    # Reconstruct all-particle CM per snapshot per run:
    #   cm_all = (N_b * cm_b + N_u * cm_u) / N
    # using the per-run N_b and N_u (not a global median).
    cm_all_mat = np.full_like(r_cm_b_vec_mat, np.nan)  # (n_runs, n_snaps, 3)
    for ri, (nb, nu) in enumerate(zip(N_bound_list, N_unbound_list)):
        cm_all_mat[ri] = (nb * r_cm_b_vec_mat[ri] + nu * r_cm_u_vec_mat[ri]) / N

    # Drift of all-particle CM from its t=0 position
    cm_all_0   = cm_all_mat[:, 0:1, :]                      # (n_runs, 1, 3)
    cm_drift   = np.linalg.norm(cm_all_mat - cm_all_0, axis=2)  # (n_runs, n_snaps)

    # Distance of bound/unbound sub-CMs from all-particle CM (scalar, per run)
    d_b_mat = np.linalg.norm(r_cm_b_vec_mat - cm_all_mat, axis=2)  # (n_runs, n_snaps)
    d_u_mat = np.linalg.norm(r_cm_u_vec_mat - cm_all_mat, axis=2)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med_drift = np.nanmedian(cm_drift,  axis=0)
        med_d_b   = np.nanmedian(d_b_mat,   axis=0)
        med_d_u   = np.nanmedian(d_u_mat,   axis=0)

    ax.plot(times_ref, med_d_b, color='steelblue', lw=2.0,
            label=rf'$|\mathrm{{CM}}_b - \mathrm{{CM}}_{{all}}|$  ($N_b$≈{int(np.median(N_bound_list))})')
    ax.plot(times_ref, med_d_u, color='crimson',   lw=2.0,
            label=rf'$|\mathrm{{CM}}_u - \mathrm{{CM}}_{{all}}|$  ($N_u$≈{int(np.median(N_unbound_list))})')
    ax.plot(times_ref, med_drift, color='k', lw=1.8, ls='--',
            label=r'$|\mathrm{CM}_{all}(t) - \mathrm{CM}_{all}(0)|$ (momentum drift)')
    ax.axvline(t_ff, color='red', lw=1.2, ls='--', alpha=0.7, label=f'$t_{{ff}}$')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('t')
    ax.set_ylabel('Distance [code units]')
    ax.set_title('Center-of-mass tracks\n'
                 r'dashed = $|$CM$_{all}$ drift$|$ (→ 0 if momentum conserved)')
    ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.4)

    plt.tight_layout()
    plt.savefig("summary_r_vs_t_cm.pdf", dpi=300)
    print("Saved: summary_r_vs_t_cm.pdf")


# ============================================================
#  SECTION 13b — NFW DENSITY PROFILE OF BOUND REMNANT
#
#  One panel per good run, all on the same figure.
#  Shows: binned rho(r) as points, NFW best-fit as line.
#  Vertical dashed line at r = 2*eps (softening cutoff —
#  data below this was excluded from the fit).
#  Log-log axes to see the power-law behaviour clearly.
# ============================================================

if any(p["ok"] for p in nfw_profiles):
    n_prof  = len(nfw_profiles)
    ncols   = min(n_prof, 5)
    nrows   = math.ceil(n_prof / ncols)
    fig_nfw, axes_nfw = plt.subplots(nrows, ncols,
                                     figsize=(4*ncols, 4*nrows),
                                     squeeze=False)
    fig_nfw.suptitle(f'NFW density profile of bound remnant — {eps_label}\n'
                     r'$\rho(r) = \rho_s\,/\,[(r/r_s)(1+r/r_s)^2]$  '
                     '— fit excludes $r < 2\varepsilon$',
                     fontsize=12)

    for pi, prof in enumerate(nfw_profiles):
        row, col = divmod(pi, ncols)
        ax = axes_nfw[row, col]

        if prof["ok"]:
            r_data = prof["r_mid"]
            rho_data = prof["rho"]
            rs   = prof["rs"]
            rhos = prof["rhos"]
            r_cut = prof["r_cut"]

            # data range for axis limits and fit line
            r_data_min = r_data[rho_data > 0].min() if (rho_data > 0).any() else r_cut
            r_data_max = r_data[rho_data > 0].max() if (rho_data > 0).any() else 1.0

            # binned data — only plot bins with actual particles
            _ok_bins = rho_data > 0
            ax.scatter(r_data[_ok_bins], rho_data[_ok_bins],
                       s=18, color='steelblue', zorder=3,
                       label='binned $\\rho$ (bound)')

            # NFW fit line — drawn only over actual data range
            _r_fit = np.logspace(np.log10(max(r_cut, r_data_min)),
                                 np.log10(r_data_max), 200)
            _rho_fit = rhos / ((_r_fit/rs) * (1.0 + _r_fit/rs)**2)
            ax.plot(_r_fit, _rho_fit, color='crimson', lw=2,
                    label=f'NFW  $r_s$={rs:.3f}\n$\\rho_s$={rhos:.2e}')

            # softening cutoff
            ax.axvline(r_cut, color='orange', lw=1.5,
                       ls='--', label=f'$2\\varepsilon$={r_cut:.4f}')

            # set axis limits to actual data range with small margin
            ax.set_xlim(r_data_min * 0.5, r_data_max * 2.0)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('$r$ [code units]')
            ax.set_ylabel('$\\rho$ [code units]')
            ax.set_title(f'Run {prof["run_id"]}')
            ax.legend(fontsize=7); ax.grid(True, lw=0.4, alpha=0.3,
                                           which='both')
        else:
            ax.text(0.5, 0.5, f'Run {prof["run_id"]}\nfit failed',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(True)

    # hide unused panels
    for pi in range(n_prof, nrows * ncols):
        row, col = divmod(pi, ncols)
        axes_nfw[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig("summary_nfw_profiles.pdf", dpi=300)
    plt.close(fig_nfw)
    print("Saved: summary_nfw_profiles.pdf")

    # print NFW parameter table
    print(f"\nNFW FIT SUMMARY — {eps_label}")
    print(f"{'Run':<6} {'r_s':>10} {'rho_s':>14} {'fit_ok':>8}")
    print("-" * 42)
    for pi, prof in enumerate(nfw_profiles):
        print(f"{prof['run_id']:<6} {prof['rs']:>10.4f} "
              f"{prof['rhos']:>14.4e} {'yes' if prof['ok'] else 'NO':>8}")


# ============================================================
#  SECTION 13c — DENSITY PROFILE EVOLUTION
#
#  For each good run: 3 panels — initial (t=0), mid (t≈t_final/2),
#  final (t=t_final) — split into ALL / BOUND / UNBOUND.
#
#  WHY THIS MATTERS:
#  The "cuspy" appearance when ALL particles are used at mid/final
#  time is because unbound particles have escaped to r >> 1.
#  Showing BOUND-only at each epoch reveals the true collapse.
#
#  Data sources:
#    initial : pos_initial stored during Pass 1 (t=0)
#    mid     : read_mid_snapshot() — one extra file read per run
#    final   : pos_last stored during Pass 1 (t=t_final)
#    bound   : computed from final snapshot phi criterion
# ============================================================

if density_run_data and times_ref is not None:
    t_final_val = times_ref[-1]
    _r_cut_dens = 2.0 * EPS_VAL if not np.isnan(EPS_VAL) else 0.0

    n_dens = len(density_run_data)
    fig_dens, axes_dens = plt.subplots(
        n_dens, 3,
        figsize=(15, 4 * n_dens),
        squeeze=False)
    fig_dens.suptitle(
        f'Density profile evolution — {eps_label}\n'
        f'blue = all  |  green = bound  |  red = unbound\n'
        f'dashed = $2\\varepsilon = {_r_cut_dens:.4f}$ (softening floor, fit cutoff)',
        fontsize=12)

    for col, ttl in enumerate([
            'Initial  (t = 0)',
            f'Mid  (t ≈ {t_final_val/2:.2f} = {t_final_val/2/t_ff:.1f}$t_{{ff}}$)',
            f'Final  (t = {t_final_val:.2f} = {t_final_val/t_ff:.1f}$t_{{ff}}$)']):
        axes_dens[0, col].set_title(ttl, fontsize=10)

    for row_idx, dr in enumerate(density_run_data):
        run_id  = dr["run_id"]
        fpath_d = dr["fpath"]
        bound_d = dr["bound"]
        pos_fin = dr["pos_last"]
        t_fin   = dr["t_final"]

        # one extra file stream to get the mid snapshot
        pos_mid_r, t_mid_r = read_mid_snapshot(fpath_d, N, t_fin / 2.0)
        print(f"  Density profile run {run_id}: mid snapshot at t={t_mid_r:.3f}")

        for col_idx, pos_snap in enumerate(
                [dr["pos_initial"], pos_mid_r, pos_fin]):
            if pos_snap is None:
                continue
            ax = axes_dens[row_idx, col_idx]

            cm_all = np.mean(pos_snap, axis=0)

            # ALL particles
            r_m, rho_v = bin_density_profile(
                pos_snap, cm_all, mass, r_inner=_r_cut_dens)
            if r_m is not None:
                ax.scatter(r_m, rho_v, s=12, color='steelblue',
                           alpha=0.7, label='all', zorder=2)

            # BOUND particles — centred on bound CM
            if bound_d.sum() > 5:
                cm_b = np.mean(pos_snap[bound_d], axis=0)
                r_m_b, rho_b = bin_density_profile(
                    pos_snap[bound_d], cm_b, mass, r_inner=_r_cut_dens)
                if r_m_b is not None:
                    ax.scatter(r_m_b, rho_b, s=12, color='green',
                               alpha=0.8, label='bound', zorder=3)

            # UNBOUND particles — centred on unbound CM
            if (~bound_d).sum() > 5:
                cm_u = np.mean(pos_snap[~bound_d], axis=0)
                r_m_u, rho_u = bin_density_profile(
                    pos_snap[~bound_d], cm_u, mass, r_inner=_r_cut_dens)
                if r_m_u is not None:
                    ax.scatter(r_m_u, rho_u, s=12, color='crimson',
                               alpha=0.8, label='unbound', zorder=3)

            if _r_cut_dens > 0:
                ax.axvline(_r_cut_dens, color='orange', lw=1.2,
                           ls='--', alpha=0.7)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('$r$ [code units]')
            if col_idx == 0:
                ax.set_ylabel(f'Run {run_id}\n$\\rho$ [code units]', fontsize=9)
            else:
                ax.set_ylabel('$\\rho$ [code units]')
            if row_idx == 0 and col_idx == 2:
                ax.legend(fontsize=8, markerscale=2)
            ax.grid(True, lw=0.3, alpha=0.3, which='both')

        del pos_mid_r

    plt.tight_layout()
    plt.savefig("summary_density_evolution.pdf", dpi=200)
    plt.close(fig_dens)
    print("Saved: summary_density_evolution.pdf")

# ============================================================
#  SECTION 14 — SERIALISE AGGREGATED STATS TO NPZ
#  Saves only time-series arrays and scalar summaries — NOT
#  raw particle data.  File size: a few MB at most.
#  Combined mode loads these without touching snapshot files.
# ============================================================

_npz_path = "summary_stats.npz"
np.savez(
    _npz_path,
    eps_val            = np.array(EPS_VAL),
    N                  = np.array(N),
    t_ff               = np.array(t_ff),
    run_numbers        = run_numbers,
    t_collapse         = t_collapse,
    delta_t            = delta_t,
    n_unbound          = n_unbound,
    energy_err         = energy_err,
    good               = good,
    bad                = bad,
    r_hm_ratio_bound   = r_hm_ratio_bound,
    virial_bound_final = virial_bound_final,
    N_bound_list       = np.array(N_bound_list),
    N_unbound_list     = np.array(N_unbound_list),
    times_ref          = times_ref if times_ref is not None else np.array([]),
    sv_mat             = sv_mat,
    mvr_mat            = mvr_mat,
    ratio_mat          = ratio_mat,
    sv_b_mat           = sv_b_mat,
    sv_u_mat           = sv_u_mat,
    mvr_b_mat          = mvr_b_mat,
    mvr_u_mat          = mvr_u_mat,
    r_med_mat          = r_med_mat,
    r_med_b_mat        = r_med_b_mat,
    r_med_u_mat        = r_med_u_mat,
)
print(f"\nSaved aggregated stats to: {_npz_path}")
print("Run 'python3 summary_runs.py --combined' from BASE_DIR for cross-eps plots.")

plt.show()
