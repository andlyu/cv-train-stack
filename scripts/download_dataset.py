#!/usr/bin/env python3
"""Download dataset from Roboflow for YOLO segmentation training."""

import os
import sys
from roboflow import Roboflow

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    print("Error: Set ROBOFLOW_API_KEY environment variable")
    sys.exit(1)

WORKSPACE = "andrews-workspace-gzmvj"
PROJECT = "blackberry-in-gripper-kckew"
VERSION = 50
FORMAT = "yolov11"  # Roboflow export format for YOLO segmentation

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

dataset = version.download(FORMAT, location="./datasets/blackberry-in-gripper")
print(f"Dataset downloaded to: {dataset.location}")
