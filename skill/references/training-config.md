# Training Configuration

## Hyperparameters

| Parameter | Classifier (MobileNet) | YOLO Detect/Segment | Notes |
|---|---|---|---|
| **Optimizer** | AdamW, lr=1e-3 | auto (MuSGD >10K iter) | SGD for batch <=512 |
| **Learning rate** | 1e-4 to 1e-3 | 0.01 (SGD) | |
| **LR schedule** | Cosine annealing or StepLR | Linear with warmup | Cosine often better |
| **Warmup** | 1-3 epochs | 3 epochs | warmup_bias_lr=0.1 |
| **Weight decay** | 0.0005 | 0.0005 | L2 regularization |
| **Batch size** | 16-64 | Largest GPU allows | Small batch = bad batchnorm |
| **Epochs** | 50-100 with early stopping | 300 with patience=100 | |
| **Early stopping** | patience=5-30 | patience=50-100 | |
| **Input size** | Must match inference size | Match training imgsz | **Check constants** |
| **Pretrained** | Yes (ImageNet) | Yes (COCO) | Always for small datasets |

## FPS / Input Size Tradeoff

**Ask the user about FPS requirements if targeting edge hardware.**

| Target Device | imgsz=320 | imgsz=640 | imgsz=1280 |
|--------------|-----------|-----------|------------|
| Jetson Orin Nano (FP16) | ~60-80 FPS | ~15-25 FPS | ~4-8 FPS |
| Jetson Orin NX (FP16) | ~100+ FPS | ~30-50 FPS | ~10-15 FPS |
| Jetson AGX Orin (FP16) | ~150+ FPS | ~60-80 FPS | ~20-30 FPS |
| RTX 3060 (FP16) | ~200+ FPS | ~80-120 FPS | ~25-40 FPS |

**Decision guide:**
- Need >30 FPS real-time: imgsz=320 or 640 depending on device
- Need >60 FPS (robotics, real-time control): likely imgsz=320
- Accuracy-first (offline processing): imgsz=640 or larger
- If unsure, train at 640 — best default balance

## Transfer Learning (Classifier)

1. Freeze base, train head only (5-10 epochs)
2. Unfreeze top layers, fine-tune with 10x lower LR
3. Use ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Augmentation

Compare current augmentation against recommended ranges:

| Augmentation | Recommended |
|---|---|
| Resize + Crop | 1.06-1.15x then crop to input_size |
| Horizontal flip | p=0.5 |
| Rotation | 10-30 degrees |
| Color jitter (brightness) | 0.2-0.4 |
| Color jitter (contrast) | 0.2-0.4 |
| Color jitter (saturation) | 0.2-0.3 |
| Color jitter (hue) | 0.02-0.1 |
| Gaussian blur | kernel=3-5, sigma=0.1-1.5 |
| Gaussian noise | std=0.02-0.1, p=0.3 |
| Random erasing | p=0.1-0.3, scale=0.02-0.08 |
| **JPEG compression** | **quality 70-95, p=0.3** |

**Key insight:** JPEG compression artifacts can shift classifier confidence by several
points. If inference touches JPEG data, add JPEG quality randomization.

Also check:
- Does augmentation reflect real-world variation?
- Are known failure modes represented?
- Is mosaic disabled in final epochs? (close_mosaic=10 for YOLO)

## Logging & Reproducibility

Before training starts, verify:

- [ ] **Exact train command saved** to `train_command.sh` in output dir (copy-pasteable)
- [ ] Training args/hyperparameters saved
- [ ] Random seed set
- [ ] Dataset version/composition recorded
- [ ] **Metrics logged to `output/training_log.csv`** every epoch (append-mode, flushed)
  - Columns: `epoch,train_loss,train_acc,val_loss,val_acc,lr`
- [ ] If W&B selected, `wandb.log()` called each epoch
- [ ] Best model checkpoint saved with validation metrics

## Logging Backend

| Option | Notes |
|--------|-------|
| **Custom CSV (default)** | Zero deps. One row per epoch. Append-mode. |
| Weights & Biases | Richer dashboards. Requires `wandb` installed + authenticated. |

CSV log is **required** regardless of backend choice. W&B is additive.

## Preflight Check (before remote training)

Run 2 epochs on ~10 images locally. Goal: verify pipeline runs end-to-end.

```bash
# 1. Create temp dataset (~10 images)
mkdir -p /tmp/preflight/{train,valid}/{images,labels}
ls datasets/<name>/train/images | shuf -n 10 | while read f; do
  cp datasets/<name>/train/images/$f /tmp/preflight/train/images/
  cp datasets/<name>/train/labels/${f%.*}.txt /tmp/preflight/train/labels/ 2>/dev/null
done
# (same for valid)

# 2. Run 2 epochs
python3 scripts/train.py --data /tmp/preflight/data.yaml --epochs 2 --batch 2 --imgsz 320 --device cpu

# 3. Clean up
rm -rf /tmp/preflight
```

**Catches:** missing paths, label format errors, corrupt images, class count mismatches,
import errors, augmentation crashes, shape mismatches.

**If preflight fails:** fix locally. Do NOT launch remote until it passes.
