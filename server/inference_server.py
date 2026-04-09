#!/usr/bin/env python3
"""FastAPI inference server for YOLO11n-seg on Jetson Orin Nano.

Endpoints:
  POST /predict        - Run segmentation on an uploaded image
  GET  /health         - Health check
  GET  /model/info     - Model metadata
"""

import io
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# --- Config ---
MODEL_PATH = Path("/opt/models/blackberry-seg/best.engine")  # TRT on Jetson
FALLBACK_PATH = Path("/opt/models/blackberry-seg/best.pt")    # PyTorch fallback
CONFIDENCE_THRESHOLD = 0.5
INPUT_SIZE = 640

app = FastAPI(title="Blackberry Segmentation API", version="1.0.0")
model = None


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        print(f"Loading TensorRT model from {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
    elif FALLBACK_PATH.exists():
        print(f"Loading PyTorch model from {FALLBACK_PATH}")
        model = YOLO(str(FALLBACK_PATH))
    else:
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH} or {FALLBACK_PATH}. "
            "Deploy model files first."
        )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model/info")
def model_info():
    return {
        "model_path": str(MODEL_PATH if MODEL_PATH.exists() else FALLBACK_PATH),
        "input_size": INPUT_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), confidence: float = CONFIDENCE_THRESHOLD):
    """Run segmentation inference on uploaded image."""
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)

    # Run inference
    start = time.time()
    results = model(img_array, imgsz=INPUT_SIZE, conf=confidence, verbose=False)
    inference_ms = (time.time() - start) * 1000

    # Parse results
    result = results[0]
    detections = []

    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            det = {
                "class_id": int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            }
            # Add segmentation mask if available
            if result.masks is not None and i < len(result.masks):
                mask = result.masks[i].data[0].cpu().numpy()
                # Encode mask as RLE or return polygon points
                contours, _ = cv2.findContours(
                    (mask > 0.5).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    # Return largest contour as polygon
                    largest = max(contours, key=cv2.contourArea)
                    det["polygon"] = largest.squeeze().tolist()

            detections.append(det)

    return JSONResponse({
        "detections": detections,
        "inference_ms": round(inference_ms, 1),
        "image_size": [img_array.shape[1], img_array.shape[0]],
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
