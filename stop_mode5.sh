#!/bin/bash
# Stop all Mode 5 servers (ports 8000-8003)

for PORT in 8000 8001 8002 8003; do
    PID=$(lsof -ti :$PORT 2>/dev/null)
    if [ -n "$PID" ]; then
        echo "Killing PID $PID on port $PORT"
        kill -9 $PID 2>/dev/null
    else
        echo "No process on port $PORT"
    fi
done

echo "Done."
