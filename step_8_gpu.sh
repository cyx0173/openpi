#!/bin/bash
# Launch multiple serve_policy.py instances across GPUs (FP16).
# Run from: /home/chengyuxuan/vla/openpi

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$SCRIPT_DIR"

START_GPU=0
NUM_GPUS=8
START_PORT=8000
POLICY_CONFIG="pi05_libero"
POLICY_DIR="/share/chengyuxuan-local/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
QUANTIZE="0"
QUANTIZE_BITS="4"
CALIBRATION_STEPS=""

usage() {
    echo "Usage: $0"
    echo ""
    echo "Environment variables:"
    echo "  START_GPU       First GPU index  (default: $START_GPU)"
    echo "  NUM_GPUS        Number of GPUs  (default: $NUM_GPUS)"
    echo "  START_PORT      First port       (default: $START_PORT)"
    echo "  POLICY_CONFIG   Config name, e.g. pi05_libero  (default: $POLICY_CONFIG)"
    echo "  POLICY_DIR      Checkpoint dir   (default: $POLICY_DIR)"
    echo "  QUANTIZE        Set to 1 to enable WxAy quantization"
    echo "  QUANTIZE_BITS   Quantization bits (default: 4)"
    echo "  CALIBRATION_STEPS Calibration steps (default: 0 = skip)"
    echo ""
    echo "Examples:"
    echo "  # FP16 on GPUs 0-7, ports 8000-8007"
    echo "  $0"
    echo ""
    echo "  # W4A4 on GPUs 0-3, ports 8000-8003"
    echo "  NUM_GPUS=4 QUANTIZE=1 QUANTIZE_BITS=4 $0"
    echo ""
    echo "  # First 4 GPUs only"
    echo "  START_GPU=0 NUM_GPUS=4 $0"
    exit 1
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

LOG_DIR="$OPENPI_DIR/logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Multi-GPU Policy Server Launcher"
echo "============================================"
echo "  GPUs:        $START_GPU .. $((START_GPU + NUM_GPUS - 1))"
echo "  Ports:       $START_PORT .. $((START_PORT + NUM_GPUS - 1))"
echo "  Policy:      $POLICY_CONFIG @ $POLICY_DIR"
if [[ "$QUANTIZE" == "1" ]]; then
    echo "  Quantize:    W${QUANTIZE_BITS:-4}A${QUANTIZE_BITS:-4}"
    echo "  Calib steps: ${CALIBRATION_STEPS:-0} (skip if dir missing)"
else
    echo "  Quantize:    FP16 (no quantization)"
fi
echo "============================================"

pids=()

for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$((START_GPU + i))
    PORT=$((START_PORT + i))
    LOG_FILE="$LOG_DIR/gpu${GPU_ID}.log"

    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python $OPENPI_DIR/scripts/serve_policy.py \
        --port $PORT \
        policy:checkpoint \
        --policy.config $POLICY_CONFIG \
        --policy.dir $POLICY_DIR"

    if [[ "$QUANTIZE" == "1" ]]; then
        CMD="$CMD --quantize --quantize-bits ${QUANTIZE_BITS:-4} --calibration-steps ${CALIBRATION_STEPS:-0}"
    fi

    echo "  [GPU $GPU_ID] Launching server on port $PORT -> $LOG_FILE"
    echo "  [GPU $GPU_ID] CMD: $CMD"

    bash -c "cd $OPENPI_DIR && $CMD" > "$LOG_FILE" 2>&1 &
    pids+=($!)
done

echo ""
echo "============================================"
echo "  All $NUM_GPUS servers launched!"
echo "  PIDs: ${pids[*]}"
echo "============================================"
echo ""
echo "Logs:   tail -f $LOG_DIR/gpu{0,1,2,3,4,5,6,7}.log"
echo "Check:  curl http://localhost:$START_PORT/healthz"
echo ""
echo "Stop all:  kill ${pids[*]}"
echo ""

echo "${pids[*]}" > "$LOG_DIR/multi_gpu_servers.pids"
echo "PIDs saved to $LOG_DIR/multi_gpu_servers.pids"
