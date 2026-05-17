#!/bin/bash
# Run Mode 6 (active_main.py) in parallel across 8 workers.
# Each worker connects to its own FP16 server (port 8000-8007) and has a unique worker_id.
# Output files are per-worker (checkpoint_w{id}.jsonl, active_dataset_w{id}.jsonl).
#
# Prerequisites:
#   - 8 FP16 servers must already be running (via step_8_gpu.sh):
#       GPU 0 -> port 8000
#       GPU 1 -> port 8001
#       ...
#       GPU 7 -> port 8007
#
# Usage: bash run_active_parallel.sh

set -e

# ── Paths ─────────────────────────────────────────────────────────────────────
COMBINED_DIR="/home/chengyuxuan/vla/openpi/examples/quant_experiment/data_new/quant_10/quant_w4a4"
OUTPUT_DIR="/home/chengyuxuan/vla/openpi/examples/active_experiment/data_new"
LOG_DIR="logs/active_3"
PYTHON_SCRIPT="examples/active_experiment/active_main.py"

# ── Parallelism ───────────────────────────────────────────────────────────────
NUM_WORKERS=8

# ── Server base ────────────────────────────────────────────────────────────────
# Worker w (0-indexed) connects to port (FP16_PORT_BASE + w)
FP16_HOST="0.0.0.0"
FP16_PORT_BASE=8000

# ── Task suite ────────────────────────────────────────────────────────────────
TASK_SUITE="libero_10"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ── Collect and sort _combined.json files ──────────────────────────────────────
mapfile -t COMBINED_FILES < <(ls "$COMBINED_DIR"/*_combined.json | sort -V)

if [ ${#COMBINED_FILES[@]} -eq 0 ]; then
    echo "No _combined.json files found in $COMBINED_DIR"
    exit 1
fi

echo "Found ${#COMBINED_FILES[@]} _combined.json files"
echo "Splitting into $NUM_WORKERS workers"

# ── Round-robin split ─────────────────────────────────────────────────────────
# File i goes to worker (i % NUM_WORKERS)
declare -a WORKER_FILES=()
for ((w = 0; w < NUM_WORKERS; w++)); do
    WORKER_FILES[$w]=""
done

for ((i = 0; i < ${#COMBINED_FILES[@]}; i++)); do
    w=$((i % NUM_WORKERS))
    if [ -z "${WORKER_FILES[$w]}" ]; then
        WORKER_FILES[$w]="${COMBINED_FILES[$i]}"
    else
        WORKER_FILES[$w]="${WORKER_FILES[$w]},${COMBINED_FILES[$i]}"
    fi
done

# ── Strip trailing commas ────────────────────────────────────────────────────
for ((w = 0; w < NUM_WORKERS; w++)); do
    WORKER_FILES[$w]="${WORKER_FILES[$w]%,}"
done

# ── Per-worker file counts ────────────────────────────────────────────────────
for ((w = 0; w < NUM_WORKERS; w++)); do
    count=$(echo -n "${WORKER_FILES[$w]}" | tr ',' '\n' | grep -c . || true)
    port=$((FP16_PORT_BASE + w))
    echo "  Worker $w: $count files  |  port=$port  |  server=$FP16_HOST:$port"
done

# ── Launch workers ─────────────────────────────────────────────────────────────
pids=()

for ((w = 0; w < NUM_WORKERS; w++)); do
    if [ -z "${WORKER_FILES[$w]}" ]; then
        echo "  Worker $w: no files assigned, skipping"
        continue
    fi

    port=$((FP16_PORT_BASE + w))
    log_file="$LOG_DIR/worker_${w}.log"

    echo "Launching Worker $w -> port=$port, worker_id=$w, files=$(
        echo -n "${WORKER_FILES[$w]}" | tr ',' '\n' | grep -c .
    ), log=$log_file"

    uv run python "$PYTHON_SCRIPT" \
        --args.host "$FP16_HOST" \
        --args.port "$port" \
        --args.task_suite_name "$TASK_SUITE" \
        --args.combined_dir "$COMBINED_DIR" \
        --args.trajectory_list "${WORKER_FILES[$w]}" \
        --args.worker_id "$w" \
        > "$log_file" 2>&1 &

    pids+=($!)
    echo "  Worker $w: PID=${pids[-1]}"
done

# ── Wait and report ────────────────────────────────────────────────────────────
echo ""
echo "All $NUM_WORKERS workers launched. Waiting for completion..."

failed=0
for ((w = 0; w < ${#pids[@]}; w++)); do
    pid=${pids[$w]}
    if ! wait "$pid"; then
        echo "  Worker $w (PID=$pid) FAILED"
        failed=$((failed + 1))
    else
        echo "  Worker $w (PID=$pid) OK"
    fi
done

echo ""
if [ $failed -eq 0 ]; then
    echo "All workers completed successfully."
else
    echo "WARNING: $failed worker(s) failed. Check $LOG_DIR/worker_*.log"
fi

echo ""
echo "Worker output files in $OUTPUT_DIR:"
echo "  checkpoint_w{id}.jsonl     - per-worker checkpoint (append-safe)"
echo "  active_dataset_w{id}.jsonl - per-worker final dataset"
echo ""
echo "To merge into final dataset:"
echo "  python merge_active_dataset.py --input_dir $OUTPUT_DIR --output_dir $OUTPUT_DIR"
