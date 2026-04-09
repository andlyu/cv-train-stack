---
name: cv-training
description: |
  CV model training review and execution. Covers dataset validation, preprocessing
  consistency, augmentation strategy, training configuration, export/quantization,
  and deployment validation. Use when training classifiers, YOLO detection/segmentation,
  or any CV model.
  Use when: "train model", "retrain", "training review", "check training config",
  "prepare training", "cv training", "model training".
  Proactively suggest when the user is about to train a model or has just finished training.
---

# CV Model Training Skill

Structured CV model training with built-in best practices. Catches the mistakes
that silently kill accuracy in production: preprocessing mismatches, augmentation
gaps, quantization drift, and training-serving skew.

## Configuration

Look for `.cv-training.json` in the project root. If it exists, use it for model
definitions, paths, and normalization values. If not, auto-detect by scanning the
project for training scripts, model files, and preprocessing code.

## Subcommands

Parse the user's input to determine which subcommand to run:

- `/cv-training review` → **Review** an existing or planned training run
- `/cv-training run <type>` → **Execute** a training run (type depends on project)
- `/cv-training validate <model>` → **Validate** a trained model before deployment
- `/cv-training audit` → **Audit** all training/inference paths for consistency

Default to `review` if no subcommand given.

---

## Review

Walk through the training configuration and flag issues before training starts.

### Step 0: Read Current State

```
1. Read .cv-training.json if it exists, otherwise scan the project
2. Identify the model type (classifier, YOLO, segmentation, etc.)
3. Find the training script and current hyperparameters
4. Find ALL inference paths for this model type
```

### Step 1: Dataset Audit

Check the training data quality. For each item, report PASS/FAIL/WARN:

| Check | Target | How to verify |
|-------|--------|---------------|
| Images per class | >=1500 (detection), >=500 (classifier) | Count files in dataset dirs |
| Class balance | No class >3x another | Count per-class, compute ratio |
| Label completeness | All instances labeled | Spot-check 10 random images |
| Background images | 0-10% of dataset | Count unlabeled images |
| Environment diversity | Multiple lighting/angles | Check metadata or visually sample |
| Train/val leak | Zero overlap | Compare file lists |
| Data format | Consistent resolution/format | Check a sample |

For classifier specifically, also check:
- Crop source diversity (different frames, positions, angles)
- Whether crops include known failure modes from production

Report findings. If any FAIL, stop and address before proceeding.

### Step 2: Preprocessing Consistency Audit

**This is the #1 source of silent accuracy loss.**

Find every place preprocessing is defined and verify they match:

```
Locations to check:
1. Training script transforms (train_transform, val_transform)
2. Pipeline inference path (both TRT and non-TRT paths)
3. Evaluation/extract scripts
4. Accuracy test scripts
5. Export scripts
```

For each location, extract and compare:

| Parameter | Training | Inference TRT | Inference PyTorch | Test | Export |
|-----------|----------|--------------|------------------|------|--------|
| Input size | ? | ? | ? | ? | ? |
| Resize method | ? | ? | ? | ? | ? |
| Normalization mean | ? | ? | ? | ? | ? |
| Normalization std | ? | ? | ? | ? | ? |
| Channel order | ? | ? | ? | ? | ? |
| Value range | ? | ? | ? | ? | ? |

**ANY mismatch is a critical bug.** Flag it and fix before training.

Reference: Google's Rules of ML — "training-serving skew is one of the most common production ML bugs."

### Step 3: Augmentation Review

Compare current augmentation against recommended ranges:

| Augmentation | Current | Recommended | Status |
|---|---|---|---|
| Resize + Crop | ? | 1.06-1.15x then crop to input_size | ? |
| Horizontal flip | ? | p=0.5 | ? |
| Rotation | ? | 10-30 degrees | ? |
| Color jitter (brightness) | ? | 0.2-0.4 | ? |
| Color jitter (contrast) | ? | 0.2-0.4 | ? |
| Color jitter (saturation) | ? | 0.2-0.3 | ? |
| Color jitter (hue) | ? | 0.02-0.1 | ? |
| Gaussian blur | ? | kernel=3-5, sigma=0.1-1.5 | ? |
| Gaussian noise | ? | std=0.02-0.1, p=0.3 | ? |
| Random erasing | ? | p=0.1-0.3, scale=0.02-0.08 | ? |
| **JPEG compression** | ? | **quality 70-95, p=0.3** | ? |

**Key insight:** JPEG compression artifacts can shift classifier confidence by several
points on sensitive images. If inference ever touches JPEG-compressed data (saved
frames, transferred images), add JPEG quality randomization to training augmentation.

Also check:
- Does augmentation reflect real-world variation? (lighting, angles, motion blur)
- Are known failure modes represented in augmentation?
- Is mosaic augmentation disabled in final epochs? (close_mosaic=10 for YOLO)

### Step 4: Training Configuration Review

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

For transfer learning (classifier):
1. Freeze base, train head only (5-10 epochs)
2. Unfreeze top layers, fine-tune with 10x lower LR
3. Use ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Step 5: Logging and Reproducibility

Before training starts, verify:

- [ ] Training command will be saved (args file or config json)
- [ ] Random seed is set for reproducibility
- [ ] Dataset version/composition is recorded
- [ ] Training metrics (loss, accuracy) are logged per epoch
- [ ] Best model checkpoint is saved with validation metrics

Present findings as a summary table and ask user to confirm before proceeding to training.

---

## Run

Execute a training run with all checks passed.

### Steps

1. Run the **Review** subcommand first (all 5 steps). Do not skip.
2. If any critical issues found, fix them before proceeding.
3. Execute training using the project's training script.
4. During training, monitor for:
   - Training loss decreasing while val loss increases (overfitting)
   - Val accuracy plateauing (need more data or augmentation)
   - NaN losses (LR too high, data corruption)
5. After training completes, automatically run **Validate**.

---

## Validate

Validate a trained model before deployment. Run after every training and before every deploy.

### Step 1: Numerical Equivalence Test

Compare PyTorch model output vs exported model (ONNX/TRT) on 10+ test images:

```
For each test image:
  1. Load raw image (PNG, not JPEG)
  2. Preprocess with the SHARED preprocessing function
  3. Run through PyTorch model -> get logits/probs
  4. Run through ONNX model -> get logits/probs
  5. Run through TRT engine -> get logits/probs (if applicable)
  6. Compare: max absolute difference should be <0.01 for ONNX, <0.05 for TRT FP16
```

If TRT FP16 divergence >0.05 on any test image, flag it. Consider:
- Quantization-Aware Training (QAT)
- FP32 fallback for sensitive layers
- Adjusting classification threshold to account for FP16 shift

### Step 2: Accuracy Test

Run the project's accuracy test suite on the exported model. Look for test files
matching patterns like `test_*accuracy*`, `test_*classification*`, `test_*model*`.

Common thresholds:
- Overall accuracy >= 90%
- Per-class recall >= 85%
- Per-class precision >= 85%
- Inference speed within target

### Step 3: Batch Sensitivity Test

Test that classification results are consistent across batch sizes:

```
For 5 test images:
  Run at batch=1, batch=4 (with zeros), batch=4 (with real images)
  All should produce identical results (within FP16 tolerance)
```

FP16 TensorRT engines can give different results at different batch sizes.
This test catches cases where dynamic batching causes issues.

### Step 4: Golden Fixture Test

If a golden test fixture exists (directory of labeled test crops with expected scores),
run all crops through the full pipeline and compare against expected scores:

```
For each golden crop:
  1. Load PNG
  2. Preprocess (pipeline-identical)
  3. Run through exported model
  4. Compare against expected score (tolerance +/-0.02)
```

If no golden fixture exists, recommend creating one from this training run's outputs.

### Step 5: Save Model Metadata

Write a metadata file alongside the model:

```yaml
date: YYYY-MM-DD
command: <exact training command>
dataset: <dataset path and version>
epochs_trained: N
best_epoch: N
final_metrics:
  train_loss: X.XXX
  val_loss: X.XXX
  val_accuracy: XX.X%
preprocessing:
  input_size: 128
  normalization_mean: [0.485, 0.456, 0.406]
  normalization_std: [0.229, 0.224, 0.225]
  channel_order: RGB
```

---

## Audit

Full audit of all training/inference paths for preprocessing consistency.
Run this periodically or after any code changes to inference paths.

### Steps

1. Find ALL files that preprocess images for any model:
   ```bash
   grep -rn "resize\|Resize\|normalize\|Normalize\|INPUT_SIZE\|input_size\|IMGSZ" \
     --include="*.py" | grep -v __pycache__
   ```

2. For each file found, extract:
   - What model it feeds
   - Resize target size
   - Resize method (cv2 vs PIL, interpolation mode)
   - Normalization values
   - Channel ordering (RGB vs BGR)

3. Build the consistency table (see Review Step 2)

4. Flag any mismatches

5. Report:
   ```
   PREPROCESSING AUDIT
   ===================
   Files checked:     N
   Models covered:    <list>
   Mismatches found:  N

   [details per mismatch]

   Status: PASS | FAIL
   ```

---

## Common Mistakes

These are real production bugs. The skill exists to catch them before they ship.

1. **Resize size mismatch** — training at 224x224 but inference at 128x128 (40pt score divergence)
2. **JPEG compression artifacts** — re-extracting crops from JPEG frames vs raw PNG shifts scores 7+ points
3. **Missing training args** — can't reproduce how a model was trained
4. **Channel order mismatch** — training in RGB but inference in BGR (OpenCV default)
5. **Normalization skew** — ImageNet normalization in training but raw [0,1] in inference
6. **Batch size sensitivity** — FP16 TRT giving different results at batch=1 vs batch=4
7. **No golden test fixture** — can't validate model export against known-good scores

## References

- Google Rules of ML: https://developers.google.com/machine-learning/guides/rules-of-ml
- Google ML Test Score: https://research.google/pubs/pub46555/
- Ultralytics Training Tips: https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/
- MIT Foundations of CV: https://visionbook.mit.edu/
- Practical ML for CV (Google): https://github.com/GoogleCloudPlatform/practical-ml-vision-book
