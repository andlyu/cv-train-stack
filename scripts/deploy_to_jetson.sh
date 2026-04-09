#!/bin/bash
# Deploy model and inference server to Jetson Orin Nano
#
# Usage: ./scripts/deploy_to_jetson.sh <user@jetson-ip> <model_weights_path>
# Example: ./scripts/deploy_to_jetson.sh user@192.168.1.100 outputs/train/blackberry-seg/weights/best.pt

set -euo pipefail

JETSON_HOST="${1:?Usage: $0 <user@jetson-ip> [model_path]}"
MODEL_PATH="${2:-outputs/train/blackberry-seg/weights/best.pt}"
REMOTE_DIR="/opt/models/blackberry-seg"
REMOTE_SERVER_DIR="/opt/inference-server"

echo "=== Deploying to Jetson: ${JETSON_HOST} ==="

# 1. Create remote directories
echo "[1/5] Creating remote directories..."
ssh "${JETSON_HOST}" "sudo mkdir -p ${REMOTE_DIR} ${REMOTE_SERVER_DIR} && sudo chown \$(whoami) ${REMOTE_DIR} ${REMOTE_SERVER_DIR}"

# 2. Copy model weights
echo "[2/5] Copying model weights..."
scp "${MODEL_PATH}" "${JETSON_HOST}:${REMOTE_DIR}/best.pt"

# 3. Copy inference server
echo "[3/5] Copying inference server..."
scp server/inference_server.py "${JETSON_HOST}:${REMOTE_SERVER_DIR}/inference_server.py"
scp requirements.txt "${JETSON_HOST}:${REMOTE_SERVER_DIR}/requirements.txt"

# 4. Install dependencies on Jetson
echo "[4/5] Installing dependencies on Jetson..."
ssh "${JETSON_HOST}" "cd ${REMOTE_SERVER_DIR} && pip3 install -r requirements.txt"

# 5. Convert to TensorRT on Jetson (FP16 for Orin Nano)
echo "[5/5] Converting to TensorRT on Jetson (this may take a few minutes)..."
ssh "${JETSON_HOST}" "python3 -c \"
from ultralytics import YOLO
model = YOLO('${REMOTE_DIR}/best.pt')
model.export(format='engine', imgsz=640, half=True, device=0)
print('TensorRT engine created successfully')
\""

echo ""
echo "=== Deployment complete ==="
echo "Start the server on the Jetson with:"
echo "  ssh ${JETSON_HOST} 'cd ${REMOTE_SERVER_DIR} && python3 inference_server.py'"
echo ""
echo "Test with:"
echo "  curl -X POST http://<jetson-ip>:8000/predict -F 'file=@test_image.jpg'"
