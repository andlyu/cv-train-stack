#!/usr/bin/env python3
"""Train YOLO11n-seg on blackberry-in-gripper dataset.

Optimized for Jetson Orin Nano deployment.
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


def create_preflight_dataset(data_yaml, n_images=10, seed=42):
    """Create a tiny dataset copy with ~n_images for preflight checks.

    Copies a small random subset of train and val images+labels into a temp
    directory and writes a new data.yaml pointing to them. Returns the path
    to the temp data.yaml.
    """
    random.seed(seed)
    data_dir = Path(data_yaml).parent
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    tmp_dir = Path(tempfile.mkdtemp(prefix="preflight_"))

    for split, key in [("train", "train"), ("valid", "val")]:
        # Resolve image dir — try relative to data.yaml first, fall back to sibling dir
        img_rel = cfg.get(key, f"../{split}/images")
        img_dir = (data_dir / img_rel).resolve()
        if not img_dir.exists():
            # Roboflow relative paths can be off — try common alternatives
            img_dir = (data_dir / split / "images").resolve()
        if not img_dir.exists():
            print(f"WARNING: Could not find image dir for {split}, tried {img_rel}")
            continue
        # Labels sit next to images dir
        label_dir = img_dir.parent / "labels"
        if not label_dir.exists():
            label_dir = img_dir.parent.parent / split / "labels"

        # Pick random subset
        all_imgs = sorted(os.listdir(img_dir))
        n = min(n_images, len(all_imgs))
        subset = random.sample(all_imgs, n)

        # Copy to temp
        tmp_img = tmp_dir / split / "images"
        tmp_lbl = tmp_dir / split / "labels"
        tmp_img.mkdir(parents=True)
        tmp_lbl.mkdir(parents=True)

        for fname in subset:
            shutil.copy2(img_dir / fname, tmp_img / fname)
            lbl_name = Path(fname).stem + ".txt"
            lbl_src = label_dir / lbl_name
            if lbl_src.exists():
                shutil.copy2(lbl_src, tmp_lbl / lbl_name)

    # Write temp data.yaml
    tmp_yaml = tmp_dir / "data.yaml"
    new_cfg = {
        "train": str(tmp_dir / "train" / "images"),
        "val": str(tmp_dir / "valid" / "images"),
        "nc": cfg["nc"],
        "names": cfg["names"],
    }
    with open(tmp_yaml, "w") as f:
        yaml.dump(new_cfg, f)

    print(f"Preflight dataset: {n_images} images per split in {tmp_dir}")
    return str(tmp_yaml), str(tmp_dir)


def save_train_command(args, output_dir):
    """Save the exact training command and metadata for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as a runnable shell script
    cmd = f"python3 {' '.join(sys.argv)}"
    script_path = output_dir / "train_command.sh"
    with open(script_path, "w") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"# Training command — {datetime.now().isoformat()}\n")
        f.write(f"# Copy-paste to reproduce this training run\n\n")
        f.write(f"{cmd}\n")

    # Also save structured args as JSON
    args_path = output_dir / "train_args.json"
    with open(args_path, "w") as f:
        json.dump({
            "command": cmd,
            "args": vars(args),
            "timestamp": datetime.now().isoformat(),
            "python": sys.executable,
        }, f, indent=2)

    print(f"Train command saved to: {script_path}")
    print(f"Train args saved to: {args_path}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n-seg")
    parser.add_argument("--data", type=str, default="datasets/blackberry-in-gripper/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 for auto)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to train on")
    parser.add_argument("--project", type=str, default="outputs/train",
                        help="Output project directory")
    parser.add_argument("--name", type=str, default="blackberry-seg",
                        help="Run name")
    parser.add_argument("--preflight", action="store_true",
                        help="Run preflight check: 2 epochs on ~10 images, then exit")
    args = parser.parse_args()

    # Preflight mode: tiny dataset, 2 epochs, fast sanity check
    preflight_tmp = None
    if args.preflight:
        print("=" * 50)
        print("PREFLIGHT CHECK")
        print("=" * 50)
        args.data, preflight_tmp = create_preflight_dataset(args.data, n_images=10)
        args.epochs = 2
        args.batch = 2
        args.imgsz = 320
        args.patience = 0
        args.project = "outputs/preflight"
        args.name = "check"
        # Always use CPU for preflight — MPS has bf16/f32 broadcast bugs with
        # YOLO that crash even with AMP disabled. CPU is fine for 10 images.
        args.device = "cpu"
        args._disable_amp = True

    # Save exact train command before training starts
    run_dir = Path(args.project) / args.name
    if not args.preflight:
        save_train_command(args, run_dir)

    # Load YOLO11n-seg pretrained on COCO
    model = YOLO("yolo11n-seg.pt")

    # Disable AMP on MPS (bf16/f32 broadcast bug)
    amp = not getattr(args, '_disable_amp', False)

    # Train with recommended settings for edge deployment
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        amp=amp,
        # Optimizer
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        warmup_bias_lr=0.1,
        weight_decay=0.0005,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        # Logging
        save=True,
        save_period=-1,  # Save best only
        plots=True,
        seed=42,
        deterministic=True,
        exist_ok=True,
    )

    if args.preflight:
        # Clean up temp dataset and preflight outputs
        if preflight_tmp:
            shutil.rmtree(preflight_tmp, ignore_errors=True)
        shutil.rmtree("outputs/preflight", ignore_errors=True)
        print("\n" + "=" * 50)
        print("PREFLIGHT PASSED")
        print("=" * 50)
    else:
        print(f"\nTraining complete. Best model: {results.save_dir}/weights/best.pt")

    return results


if __name__ == "__main__":
    main()
