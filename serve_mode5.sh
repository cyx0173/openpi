#!/bin/bash
# Mode 5: FP16 + W4A4 + W4A8 + W4A16 Quad-Server Launcher

OPENPI_DIR="/home/chengyuxuan/vla/openpi"
mkdir -p "$OPENPI_DIR/logs"

# FP16 Server - Port 8001 - GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/serve_policy.py \
    --port 8001 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir /share/chengyuxuan-local/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
    > "$OPENPI_DIR/logs/mode5_port8001.log" 2>&1 &

# W4A4 Server - Port 8000 - GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py \
    --port 8000 \
    --quantize \
    --quantize-bits-w 4 \
    --quantize-bits-a 4 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir /share/chengyuxuan-local/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
    > "$OPENPI_DIR/logs/mode5_port8000.log" 2>&1 &

# W4A8 Server - Port 8002 - GPU 0
CUDA_VISIBLE_DEVICES=2 python scripts/serve_policy.py \
    --port 8002 \
    --quantize \
    --quantize-bits-w 4 \
    --quantize-bits-a 8 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir /share/chengyuxuan-local/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
    > "$OPENPI_DIR/logs/mode5_port8002.log" 2>&1 &

# W4A16 Server - Port 8003 - GPU 0
CUDA_VISIBLE_DEVICES=3 python scripts/serve_policy.py \
    --port 8003 \
    --quantize \
    --quantize-bits-w 4 \
    --quantize-bits-a 16 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir /share/chengyuxuan-local/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
    > "$OPENPI_DIR/logs/mode5_port8003.log" 2>&1 &

