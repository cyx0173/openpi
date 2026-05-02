#!/bin/bash
# Record LIBERO_SPATIAL trajectories (Mode 2: FP16 + W4A4 dual-server).
# Usage: bash run_record_spatial.sh
#
# Prerequisites (run these first, each in its own terminal):
#
#   Terminal 1 - FP16 server (port 8001):
#     uv run python scripts/serve_policy.py --env LIBERO --port 8001
#
#   Terminal 2 - W4A4 server (port 8000):
#     uv run python scripts/serve_policy.py --env LIBERO --port 8000 --quantize --quantize-bits 4
#
#   Terminal 3 - this script:
#     bash run_record_spatial.sh
#
# Output: data/libero/videos/quant_w4a4/rollout_*_combined.json
# These _combined.json files are required by quant_main.py Mode 3/4.

set -e

OPENPI_DIR="/home/chengyuxuan/vla/openpi"
LOG_FILE="$OPENPI_DIR/logs/record_spatial.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "Recording LIBERO_SPATIAL trajectories (Mode 2: FP16 + W4A4)"
echo "  FP16 server:  0.0.0.0:8001  (full precision, for recording + perturbation inference)"
echo "  W4A4 server:  0.0.0.0:8000  (quantized, for recording)"
echo "  Log: $LOG_FILE"
echo ""
echo "Make sure BOTH servers are running before proceeding!"
echo ""

cd "$OPENPI_DIR"
uv run python examples/libero/quant_main.py \
    --mode 2 \
    --task_suite_name libero_spatial \
    --port 8001 \
    --w4a4_port 8000 \
    --output_dir data/libero/videos/quant_w4a4 \
    --video_dir data/libero/videos \
    --replan_steps 5 \
    --num_steps_wait 10 \
    --seed 7 \
    --host 0.0.0.0 \
    > "$LOG_FILE" 2>&1

echo "Done. Results in data/libero/videos/quant_w4a4/"
echo "Log: $LOG_FILE"
