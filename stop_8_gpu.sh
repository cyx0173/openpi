#!/bin/bash
# Stop all serve_policy.py servers and quant_main.py instances.
# Usage: bash run_step_perturb_stop.sh

set -e

OPENPI_DIR="/home/chengyuxuan/vla/openpi"
LOG_DIR="$OPENPI_DIR/logs"
START_PORT=8000
MAX_PARALLEL=8

echo "Killing quant_main.py processes..."
pkill -f "quant_main.py" 2>/dev/null || true

echo "Killing serve_policy.py processes..."
pkill -f "serve_policy.py" 2>/dev/null || true

echo "Checking ports $START_PORT-$((START_PORT + MAX_PARALLEL - 1))..."
for port in $(seq $START_PORT $((START_PORT + MAX_PARALLEL - 1))); do
    fuser -k ${port}/tcp 2>/dev/null && echo "  Killed process on port $port" || echo "  Port $port already free"
done

if [ -f "$LOG_DIR/multi_gpu_servers.pids" ]; then
    PIDS=$(cat "$LOG_DIR/multi_gpu_servers.pids")
    echo "Killing servers by PID: $PIDS"
    kill $PIDS 2>/dev/null || true
    rm -f "$LOG_DIR/multi_gpu_servers.pids"
fi

echo "All done."
