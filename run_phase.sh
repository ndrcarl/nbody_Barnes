#!/bin/bash
# ============================================================
#  run_phase.sh — run all realisations for ONE eps value
#
#  Usage:
#    bash run_phase.sh <eps>
#
#  Examples:
#    bash run_phase.sh 0.0001
#    bash run_phase.sh 0.1
#
#  This script is called by the individual phase_*.sh wrappers,
#  but can also be invoked directly.
# ============================================================

set -euo pipefail

# ---- argument check ----
if [ $# -ne 1 ]; then
    echo "Usage: bash run_phase.sh <eps>"
    echo "  e.g. bash run_phase.sh 0.001"
    exit 1
fi

eps="$1"
NUM_RUNS=5
BASE_DIR=$(pwd)

# ---- sanity check ----
if [ ! -x "$BASE_DIR/treecode" ]; then
    echo "Error: treecode not found or not executable in $BASE_DIR"
    exit 1
fi

EPS_TAG=$(echo "$eps" | tr '.' 'p')          # e.g. 0.001 -> 0p001
EPS_DIR="$BASE_DIR/eps_${EPS_TAG}"
EPS_LOG="$EPS_DIR/master_run_${EPS_TAG}.log"

mkdir -p "$EPS_DIR"
> "$EPS_LOG"   # truncate / create

echo "" | tee -a "$EPS_LOG"
echo "########## eps = $eps  (tag: $EPS_TAG) ##########" | tee -a "$EPS_LOG"
echo "Phase started at $(date)" | tee -a "$EPS_LOG"

for i in $(seq 1 $NUM_RUNS); do
    RUN_NUM=$(printf "%03d" $i)
    RUN_DIR="$EPS_DIR/run_${RUN_NUM}"

    echo "" | tee -a "$EPS_LOG"
    echo "=== eps=$eps  run $i at $(date) ===" | tee -a "$EPS_LOG"

    mkdir -p "$RUN_DIR"
    cd "$RUN_DIR" || exit 1

    # ----------------------------------------------------------
    # 1) Generate initial conditions (cold uniform sphere)
    # ----------------------------------------------------------
    echo "Run eps=$eps $i: sampling_advanced.py" >> "$EPS_LOG"
    python3 "$BASE_DIR/sampling_advanced.py" >> "$EPS_LOG" 2>&1

    # ----------------------------------------------------------
    # 2) Run treecode
    # ----------------------------------------------------------
    echo "Run eps=$eps $i: treecode" >> "$EPS_LOG"
    "$BASE_DIR/treecode" \
        in=gmc_internal.txt  \
        out=gmc_internal.out \
        dtime=1/8192         \
        eps=$eps             \
        theta=0.50           \
        usequad=false        \
        tstop=5.6            \
        dtout=1/120          \
        options=out-phi      \
        >> "$EPS_LOG" 2>&1

    # ----------------------------------------------------------
    # 3) Analysis scripts
    # ----------------------------------------------------------
    echo "Run eps=$eps $i: raggio.py" >> "$EPS_LOG"
    python3 "$BASE_DIR/raggio.py" >> "$EPS_LOG" 2>&1

    echo "Run eps=$eps $i: raggio_min.py" >> "$EPS_LOG"
    python3 "$BASE_DIR/raggio_min.py" >> "$EPS_LOG" 2>&1

    echo "Run eps=$eps $i: plot_analysis.py" >> "$EPS_LOG"
    python3 "$BASE_DIR/plot_analysis.py" "$eps" >> "$EPS_LOG" 2>&1

    echo "Run eps=$eps $i: plot_collapse.py" >> "$EPS_LOG"
    python3 "$BASE_DIR/plot_collapse.py" gmc_internal.out >> "$EPS_LOG" 2>&1

    echo "=== eps=$eps run $i completed at $(date) ===" | tee -a "$EPS_LOG"

    cd "$BASE_DIR" || exit 1
done

# ----------------------------------------------------------
# 4) Per-eps summary
# ----------------------------------------------------------
echo "" | tee -a "$EPS_LOG"
echo "Running summary_runs.py for eps=$eps ..." | tee -a "$EPS_LOG"
python3 "$BASE_DIR/summary_runs.py" "$EPS_DIR" >> "$EPS_LOG" 2>&1
echo "Summary done for eps=$eps." | tee -a "$EPS_LOG"

echo "Phase eps=$eps finished at $(date)" | tee -a "$EPS_LOG"
