#!/usr/bin/env bash
# =============================================================================
# SimpleX Baseline Evaluation Script
# =============================================================================
# Run from: moe_early_fusion/baseline/
#
# Prerequisites:
#   1. Train embeddings first:
#      cd ../plugin/simplex && bash train.sh
#
# Usage:
#   bash run_simplex.sh [task_set] [scenario]
#
#   Examples:
#     bash run_simplex.sh yelp classic
#     bash run_simplex.sh amazon classic
#     bash run_simplex.sh amazon user_cold_start
#     bash run_simplex.sh goodreads classic
#
#   Or run all scenarios at once (no arguments):
#     bash run_simplex.sh
# =============================================================================

set -e
cd "$(dirname "$0")"   # Ensure we're in baseline/

# Use project venv
PYTHON="/home/research/nghialt/.venv/bin/python"

EMB_DIR="../plugin/simplex/embeddings"
DATA_DIR="../dataset/output_data_all"
TASKS_DIR="../dataset/tasks5"
NUM_TASKS=None
MAX_WORKERS=10

run_eval() {
    local TASK_SET=$1
    local SCENARIO=$2
    echo "------------------------------------------------------------"
    echo "  SimpleX: ${TASK_SET} / ${SCENARIO}"
    echo "------------------------------------------------------------"
    $PYTHON SimpleX_baseline.py \
        --task_set    "$TASK_SET" \
        --scenario    "$SCENARIO" \
        --data_dir    "$DATA_DIR" \
        --emb_dir     "$EMB_DIR" \
        --tasks_dir   "$TASKS_DIR" \
        --num_tasks   $NUM_TASKS \
        --max_workers $MAX_WORKERS
}

# If arguments are provided, run a single evaluation
if [ "$#" -eq 2 ]; then
    run_eval "$1" "$2"
    exit 0
fi

# Otherwise, run all scenarios
echo "============================================================"
echo "  SimpleX Baseline — Running all datasets & scenarios"
echo "============================================================"

run_eval yelp      classic
run_eval amazon    classic
run_eval amazon    user_cold_start
run_eval goodreads classic

echo ""
echo "============================================================"
echo "  All done! Results saved to baseline/results/{scenario}/"
echo "============================================================"
