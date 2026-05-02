#!/bin/bash
# Stop all servers and eval processes launched by run_multi_gpu_servers.sh / run_multi_gpu_eval.sh
# Run from: /home/chengyuxuan/vla/openpi

LOG_DIR="/home/chengyuxuan/vla/openpi/logs"

echo "Stopping multi-GPU servers..."
if [[ -f "$LOG_DIR/multi_gpu_servers.pids" ]]; then
    PIDS=$(cat "$LOG_DIR/multi_gpu_servers.pids")
    kill $PIDS 2>/dev/null && echo "  Servers killed (PIDs: $PIDS)" || echo "  No server processes found"
else
    echo "  No PID file found. Trying to kill by port range..."
    for PORT in $(seq 8000 8007); do
        PID=$(lsof -ti:$PORT 2>/dev/null || true)
        if [[ -n "$PID" ]]; then
            kill $PID 2>/dev/null && echo "  Killed process on port $PORT (PID: $PID)" || echo "  Failed to kill port $PORT"
        fi
    done
fi

echo ""
echo "Stopping eval processes..."
if [[ -f "$LOG_DIR/multi_gpu_eval.pids" ]]; then
    PIDS=$(cat "$LOG_DIR/multi_gpu_eval.pids")
    kill $PIDS 2>/dev/null && echo "  Eval processes killed (PIDs: $PIDS)" || echo "  No eval processes found"
else
    echo "  No PID file found."
fi

echo ""
echo "Done."
