#!/bin/bash
# Run mode 4 (step-level perturbation) evaluation in parallel across multiple servers.
# Each combined.json gets its own port (8000-8007).
# Usage: bash run_step_perturb_parallel.sh

set -e

DATA_DIR="/home/chengyuxuan/vla/openpi/data/quant_spatial/quant_w4a4"
OUTPUT_DIR="data/quant_spatial/step_perturb_results"
START_PORT=8000
MAX_PARALLEL=8
PYTHON_SCRIPT="examples/libero/quant_main.py"

# Find all combined.json files sorted
mapfile -t JSON_FILES < <(ls "$DATA_DIR"/*_combined.json | sort)

if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo "No combined.json files found in $DATA_DIR"
    exit 1
fi

echo "Found ${#JSON_FILES[@]} trajectory files"
echo "Will run up to $MAX_PARALLEL parallel jobs"

# Kill any existing processes on these ports (optional, be careful)
# for port in $(seq $START_PORT $((START_PORT + MAX_PARALLEL - 1))); do
#     fuser -k ${port}/tcp 2>/dev/null || true
# done

# Function to run a single job
run_job() {
    local idx=$1
    local json_path=$2
    local port=$3
    local log_file="logs/step5_perturb_${idx}.log"
    mkdir -p logs
    echo "[$idx] Starting: $(basename "$json_path") on port $port -> $log_file"
    uv run python "$PYTHON_SCRIPT" \
        --args.mode 3 \
        --args.trajectory-list "$json_path" \
        --args.port "$port" \
        --args.output_dir "$OUTPUT_DIR" \
        > "$log_file" 2>&1
    echo "[$idx] Done: $(basename "$json_path") (exit: $?)"
}

# Launch up to MAX_PARALLEL jobs at a time
pids=()
indices=()

for i in "${!JSON_FILES[@]}"; do
    # Assign port cyclically
    port=$((START_PORT + (i % MAX_PARALLEL)))
    json_path="${JSON_FILES[$i]}"

    run_job "$i" "$json_path" "$port" &
    pids+=($!)
    indices+=("$i")

    # If we've reached MAX_PARALLEL, wait for the oldest to finish
    if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
        # Wait for the first process in the list
        wait ${pids[0]}
        # Remove the first element from both arrays
        pids=("${pids[@]:1}")
        indices=("${indices[@]:1}")
    fi
done

# Wait for all remaining jobs
echo "Waiting for remaining ${#pids[@]} jobs to finish..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All done! Results in $OUTPUT_DIR"
