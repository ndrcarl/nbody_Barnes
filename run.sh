#!/bin/bash
# ============================================================
#  run.sh — automated multi-run script for Barnes treecode v1.4
#
#  Pipeline per run:
#    1. sampling_advanced.py  → gmc_internal.txt   (IC file)
#    2. treecode               → gmc_internal.out   (N-body snapshots)
#    3. raggio.py             → r_vs_t.pdf
#    4. raggio_min.py         → t_collapse_scatter.pdf
#                               + prints  "median(t_collapse) = ..."
#                               + prints  "t_ff_sim (global): ..."
#                               + prints  "Final/initial ratio: ..."
#    5. plot_analysis.py      → escape_analysis.pdf
#                               + prints  "Unbound particles: X / N"
#  After all runs:
#    6. summary_runs.py       → summary_*.pdf  + table on stdout
#
#  NOTE on options=phi:
#    The treecode is now run with options=phi, which appends the
#    self-consistent N-body potential phi_i per particle to each
#    snapshot block.  This changes the output format from:
#      3 + 3*N lines/snapshot  →  3 + 4*N lines/snapshot
#    The Python scripts use HAS_PHI=True to read the extra block.
#    Make sure HAS_PHI = True is set at the top of:
#      plot_analysis.py, summary_runs.py
#
#  Usage:
#    bash run.sh
# ============================================================

#!/bin/bash
# ============================================================
#  run.sh — automated multi-run script for Barnes treecode v1.4
#
#  Outer loop: eps values
#  Inner loop: NUM_RUNS realisations per eps
#
#  Directory structure:
#    eps_0p007/
#      master_run_0p007.log
#      run_001/  run_002/  ...
#    eps_0p015/
#      master_run_0p015.log
#      run_001/  ...
#
#  After all eps groups, logs are concatenated into master_run.log
#
#  Usage:
#    bash run.sh
# ============================================================

NUM_RUNS=5
BASE_DIR=$(pwd)

# ---- sanity check ----
if [ ! -x "$BASE_DIR/treecode" ]; then
    echo "Error: treecode not found or not executable in $BASE_DIR"
    exit 1
fi

# ---- eps values to sweep ----
EPS_VALUES=(0.0001 0.001 0.01 0.1 1.0)

for eps in "${EPS_VALUES[@]}"; do

    EPS_TAG=$(echo "$eps" | tr '.' 'p')          # e.g. 0.007 -> 0p007
    EPS_DIR="$BASE_DIR/eps_${EPS_TAG}"
    EPS_LOG="$EPS_DIR/master_run_${EPS_TAG}.log"

    mkdir -p "$EPS_DIR"
    > "$EPS_LOG"   # truncate / create

    echo "" | tee -a "$EPS_LOG"
    echo "########## eps = $eps  (tag: $EPS_TAG) ##########" | tee -a "$EPS_LOG"

    for i in $(seq 1 $NUM_RUNS); do
        RUN_NUM=$(printf "%03d" $i)
        RUN_DIR="$EPS_DIR/run_${RUN_NUM}"

        echo "" | tee -a "$EPS_LOG"
        echo "=== eps=$eps  run $i at $(date) ===" | tee -a "$EPS_LOG"

        mkdir -p "$RUN_DIR"
        cd "$RUN_DIR" || exit 1

        # ------------------------------------------------------
        # 1) Generate initial conditions (cold uniform sphere)
        # ------------------------------------------------------
        echo "Run eps=$eps $i: sampling_advanced.py" >> "$EPS_LOG"
        python3 "$BASE_DIR/sampling_advanced.py" >> "$EPS_LOG" 2>&1

        # ------------------------------------------------------
        # 2) Run treecode
        # ------------------------------------------------------
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

        # ------------------------------------------------------
        # 3) Analysis scripts
        # ------------------------------------------------------
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
    # 4) Per-eps summary (reads its own clean log)
    # ----------------------------------------------------------
    echo "" | tee -a "$EPS_LOG"
    echo "Running summary_runs.py for eps=$eps ..." | tee -a "$EPS_LOG"
    python3 "$BASE_DIR/summary_runs.py" "$EPS_DIR" >> "$EPS_LOG" 2>&1
    echo "Summary done for eps=$eps." | tee -a "$EPS_LOG"

done

# ---- concatenate all per-eps logs into one global log ----
echo ""
echo "Concatenating logs into master_run.log ..."
cat "$BASE_DIR"/eps_*/master_run_*.log > "$BASE_DIR/master_run.log"
echo "Done."
