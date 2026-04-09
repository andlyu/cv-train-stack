#!/usr/bin/env python3
"""Export trained YOLO model to ONNX for Jetson Orin Nano deployment.

TensorRT conversion should be done ON the Jetson for optimal compatibility.
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model")
    parser.add_argument("--weights", type=str,
                        default="outputs/train/blackberry-seg/weights/best.pt",
                        help="Path to trained weights")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--half", action="store_true",
                        help="Export with FP16 (recommended for Jetson)")
    args = parser.parse_args()

    model = YOLO(args.weights)

    # Export to ONNX (portable, convert to TRT on Jetson)
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        simplify=True,
        opset=17,
    )
    print(f"ONNX exported to: {onnx_path}")

    # Also export TensorRT engine if on a machine with TRT
    try:
        trt_path = model.export(
            format="engine",
            imgsz=args.imgsz,
            half=True,  # FP16 for Jetson Orin Nano
            device=0,
        )
        print(f"TensorRT exported to: {trt_path}")
    except Exception as e:
        print(f"TensorRT export skipped (do this on the Jetson): {e}")


if __name__ == "__main__":
    main()
