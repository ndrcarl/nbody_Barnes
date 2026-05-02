#!/bin/bash
# ============================================================
#  finalize.sh — concatenate all per-eps logs into master_run.log
#
#  Run this after all phases (or all desired phases) are done.
#
#  Usage:
#    bash finalize.sh
# ============================================================

BASE_DIR=$(pwd)
MASTER_LOG="$BASE_DIR/master_run.log"

echo "Concatenating per-eps logs into master_run.log ..."

# Check at least one eps dir exists
shopt -s nullglob
LOG_FILES=("$BASE_DIR"/eps_*/master_run_*.log)

if [ ${#LOG_FILES[@]} -eq 0 ]; then
    echo "Error: no per-eps log files found under $BASE_DIR/eps_*/"
    exit 1
fi

cat "${LOG_FILES[@]}" > "$MASTER_LOG"

echo "Done. Combined ${#LOG_FILES[@]} log file(s) → $MASTER_LOG"
echo ""
echo "Phases included:"
for f in "${LOG_FILES[@]}"; do
    echo "  $f"
done

# ----------------------------------------------------------
# Run combined mode — cross-eps comparison plots
# Requires all eps phases to have completed so that
# eps_*/summary_stats.npz files exist.
# ----------------------------------------------------------
echo ""
echo "Running summary_runs.py --combined ..."
python3 "$BASE_DIR/summary_runs.py" --combined
echo "Combined plots written to combined/"
