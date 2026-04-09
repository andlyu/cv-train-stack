#!/usr/bin/env python3
"""Visualize random dataset samples with segmentation label overlays.

Opens images in the default viewer with polygons and class labels drawn on top.
Used during dataset audit to spot-check label quality.
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Class colors (BGR for OpenCV)
COLORS = [
    (0, 255, 0),    # berry - green
    (255, 0, 0),    # gripper - blue
    (0, 255, 255),  # lense - yellow
    (0, 0, 255),    # red drouplet - red
]


def draw_labels_on_image(img_path, label_path, class_names):
    """Draw segmentation polygons and class labels on an image."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Could not read {img_path}")
        return None

    h, w = img.shape[:2]
    overlay = img.copy()

    if not label_path.exists():
        # No labels — draw "NO LABELS" text
        cv2.putText(img, "NO LABELS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            color = COLORS[cls_id % len(COLORS)]
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"

            if len(coords) > 4:
                # Segmentation polygon (normalized x,y pairs)
                points = []
                for i in range(0, len(coords), 2):
                    px = int(coords[i] * w)
                    py = int(coords[i + 1] * h)
                    points.append([px, py])
                pts = np.array(points, dtype=np.int32)

                # Filled polygon with transparency
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(img, [pts], True, color, 2)

                # Label text at centroid
                cx = int(np.mean([p[0] for p in points]))
                cy = int(np.mean([p[1] for p in points]))
                cv2.putText(img, cls_name, (cx - 20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(img, cls_name, (cx - 20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # Bounding box (cx, cy, w, h normalized)
                cx, cy, bw, bh = coords[:4]
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, cls_name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Blend overlay (30% opacity for filled polygons)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    # Add filename at top
    fname = Path(img_path).name
    cv2.putText(img, fname, (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset samples with labels")
    parser.add_argument("--data-dir", type=str, default="datasets/blackberry-in-gripper",
                        help="Dataset root directory")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to sample from")
    parser.add_argument("--n", type=int, default=6,
                        help="Number of samples to show")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--save", type=str, default=None,
                        help="Save grid to this path instead of opening viewer")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data_dir = Path(args.data_dir)
    img_dir = data_dir / args.split / "images"
    label_dir = data_dir / args.split / "labels"

    if not img_dir.exists():
        print(f"Error: {img_dir} not found")
        sys.exit(1)

    # Read class names from data.yaml
    class_names = []
    yaml_path = data_dir / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            for line in f:
                if line.strip().startswith("names:"):
                    # Parse YAML list: names: ['berry', 'gripper', ...]
                    names_str = line.split(":", 1)[1].strip()
                    if names_str.startswith("["):
                        class_names = [n.strip().strip("'\"") for n in
                                       names_str.strip("[]").split(",")]
                    break
    if not class_names:
        class_names = [f"class_{i}" for i in range(20)]

    # Sample random images
    all_images = sorted(os.listdir(img_dir))
    n = min(args.n, len(all_images))
    samples = random.sample(all_images, n)

    print(f"Visualizing {n} samples from {args.split} split...")
    print(f"Classes: {class_names}")
    print()

    annotated = []
    for fname in samples:
        img_path = img_dir / fname
        label_path = label_dir / (Path(fname).stem + ".txt")

        # Count instances in this image
        instance_counts = {}
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
                    instance_counts[name] = instance_counts.get(name, 0) + 1

        print(f"  {fname}: {dict(instance_counts) if instance_counts else 'no labels'}")

        result = draw_labels_on_image(img_path, label_path, class_names)
        if result is not None:
            annotated.append(result)

    if not annotated:
        print("No images could be loaded.")
        sys.exit(1)

    # Build a grid (2 columns)
    cols = 2
    rows = (len(annotated) + cols - 1) // cols

    # Resize all to same size for grid
    target_h, target_w = 400, 600
    resized = []
    for img in annotated:
        resized.append(cv2.resize(img, (target_w, target_h)))

    # Pad to fill grid
    while len(resized) < rows * cols:
        resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        row_imgs = resized[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_imgs))
    grid = np.vstack(grid_rows)

    # Save and open
    if args.save:
        out_path = args.save
    else:
        out_path = tempfile.mktemp(suffix="_dataset_audit.png")

    cv2.imwrite(out_path, grid)
    print(f"\nGrid saved to: {out_path}")

    # Open in default viewer
    if not args.save:
        if sys.platform == "darwin":
            subprocess.run(["open", out_path])
        elif sys.platform == "linux":
            subprocess.run(["xdg-open", out_path])
        else:
            print(f"Open manually: {out_path}")


if __name__ == "__main__":
    main()
