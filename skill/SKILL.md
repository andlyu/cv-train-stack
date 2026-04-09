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

Auto-detect by scanning the project. Look for training scripts, model files,
dataset directories, and preprocessing code. No config file needed — just drop
the skill in and go.

If a `.cv-training.json` exists, use it as hints, but never require it.

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

### Step 0a: GPU Detection (ALWAYS run first)

Before anything else, detect available compute. This determines the entire training path.

**Detection sequence:**

```bash
# 1. Check for NVIDIA GPU locally
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}, Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# 2. Check VRAM if GPU found
python3 -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB') if torch.cuda.is_available() else None"

# 3. Check for Apple Silicon MPS
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

**Decision tree:**

| Result | Action |
|--------|--------|
| NVIDIA GPU with >=6GB VRAM | **Use local GPU.** Proceed to Step 0b. |
| NVIDIA GPU with <6GB VRAM | WARN: may OOM on large models. Suggest reducing batch/imgsz, or use VAST. |
| Apple Silicon (MPS) | WARN: MPS works for small models but is slower than CUDA and has known issues with some ops. Acceptable for quick iterations, recommend VAST for final training runs. |
| **No GPU (CPU only)** | **STOP. Do not train on CPU.** Inform the user that CPU training is impractical (10-50x slower, hours instead of minutes). Guide them to set up a VAST.ai account for cloud GPU access. |

**If no GPU is available, present this to the user:**

> Training on CPU is not recommended — a 300-epoch YOLO run that takes 30 minutes on GPU
> will take 10+ hours on CPU. VAST.ai offers GPUs starting at ~$0.15/hr.
>
> To get started:
> 1. Create an account at https://vast.ai
> 2. Add your API key: `pip install vastai && vastai set api-key <KEY>`
> 3. Re-run this skill and we'll launch a cloud GPU automatically.

Do NOT proceed with training until a GPU path is confirmed.

### Step 0b: Compute Selection

After GPU detection, present the user with their options. Always show estimated cost.

**Ask the user which compute to use:**

| Option | When to show | Est. cost |
|--------|-------------|-----------|
| Local GPU | NVIDIA GPU detected | Free |
| Local MPS | Apple Silicon detected | Free (with caveats) |
| VAST.ai | Always (if no local NVIDIA GPU, make this the recommended option) | Show live pricing |
| Other cloud (RunPod, Lambda, etc.) | Only if user mentions it | Varies |

**For VAST.ai, show live pricing before the user commits:**

```bash
# Search for suitable instances and show top 3 cheapest
vastai search offers 'gpu_ram >= 8 reliability > 0.95 num_gpus == 1 cuda_vers >= 12.0' \
  --order 'dph' --limit 3 --output 'id gpu_name gpu_ram disk_space dph_total dlperf'
```

Present a table like:

```
Available VAST.ai GPUs:
┌────────┬──────────────┬──────────┬───────────┬────────────────────┐
│ ID     │ GPU          │ VRAM     │ $/hr      │ Est. training cost │
├────────┼──────────────┼──────────┼───────────┼────────────────────┤
│ 123456 │ RTX 3090     │ 24 GB    │ $0.18/hr  │ ~$0.09 (30 min)   │
│ 234567 │ RTX 4090     │ 24 GB    │ $0.35/hr  │ ~$0.18 (30 min)   │
│ 345678 │ A100 40GB    │ 40 GB    │ $0.80/hr  │ ~$0.40 (30 min)   │
└────────┴──────────────┴──────────┴───────────┴────────────────────┘
```

**Estimate training time** based on dataset size and model:
- YOLO nano/small + <5K images: ~20-40 min on consumer GPU
- YOLO medium/large + <5K images: ~40-90 min
- YOLO nano/small + 5-20K images: ~1-2 hrs
- Classifier (MobileNet) + <5K images: ~10-20 min

Multiply time estimate by $/hr to get estimated cost. Always show this before the user
commits to a VAST instance.

### Step 0c: Read Current State

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

### Step 1b: Visual Spot-Check

After the quantitative audit, **always show sample images with label overlays** so the
user can visually verify label quality. Run the visualization script:

```bash
python3 scripts/visualize_samples.py --data-dir <dataset_path> --split train --n 6 --seed 42
```

This opens a grid of 6 random training images with segmentation polygons and class labels
drawn on top. Check for:
- Labels covering the correct objects (no offset or misaligned polygons)
- Missing labels (unlabeled objects visible in the image)
- Wrong class assignments
- Label quality at edges (tight vs loose polygon fits)

If the project has no `scripts/visualize_samples.py`, create one that:
1. Picks N random images from the requested split
2. Draws segmentation polygons (filled with 30% opacity) and class names
3. Arranges them in a 2-column grid
4. Opens the grid in the system image viewer (macOS `open`, Linux `xdg-open`)

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

- [ ] **Exact train command is saved** — write the full CLI invocation (with all args) to a `train_command.sh` file in the output directory. This must be copy-pasteable to reproduce the run. For Python API calls, save the equivalent CLI command or a JSON of all parameters.
- [ ] Training args/hyperparameters are saved (args file or config json)
- [ ] Random seed is set for reproducibility
- [ ] Dataset version/composition is recorded (source, version, class counts, split sizes)
- [ ] Training metrics (loss, accuracy) are logged per epoch
- [ ] Best model checkpoint is saved with validation metrics

**Why save the exact command:** "How was this model trained?" is the most common question
when debugging production issues months later. Framework-generated args files miss CLI
flags, environment variables, and dataset paths. A runnable shell script is the gold standard.

Present findings as a summary table and ask user to confirm before proceeding to training.

---

## Run

Execute a training run with all checks passed.

### Steps

1. Run the **Review** subcommand first (all steps including 0a/0b GPU detection). Do not skip.
2. If any critical issues found, fix them before proceeding.
3. **If training will run on a remote machine (VAST, RunPod, SSH, etc.), run a local
   preflight check FIRST.** This catches data loading errors, config typos, shape
   mismatches, and import issues before spending money on a cloud GPU.

#### Preflight Check (required before any remote training)

Run 2 epochs on ~10 images locally. The goal is NOT to train — it's to verify the
full pipeline runs end-to-end without errors. This should finish in seconds.

**How it works:** Copy ~10 random images + labels from train and val into a temp
directory, write a temporary data.yaml pointing to them, and run 2 epochs. Use the
training script's `--preflight` flag which handles this automatically:

```bash
python3 scripts/train.py --preflight --data <data.yaml>
```

This will:
1. Create a temp dataset with ~10 images per split (copied from the real dataset)
2. Run 2 epochs, batch=2, imgsz=320, on CPU or MPS
3. Report PASS/FAIL
4. Clean up the temp dataset

If the training script doesn't have a `--preflight` flag, do it manually:

```bash
# 1. Create temp dataset (~10 images)
mkdir -p /tmp/preflight/{train,valid}/{images,labels}
ls datasets/<name>/train/images | shuf -n 10 | while read f; do
  cp datasets/<name>/train/images/$f /tmp/preflight/train/images/
  cp datasets/<name>/train/labels/${f%.*}.txt /tmp/preflight/train/labels/ 2>/dev/null
done
ls datasets/<name>/valid/images | shuf -n 10 | while read f; do
  cp datasets/<name>/valid/images/$f /tmp/preflight/valid/images/
  cp datasets/<name>/valid/labels/${f%.*}.txt /tmp/preflight/valid/labels/ 2>/dev/null
done

# 2. Write temp data.yaml pointing to /tmp/preflight

# 3. Run 2 epochs
python3 scripts/train.py \
  --data /tmp/preflight/data.yaml \
  --epochs 2 --batch 2 --imgsz 320 --device cpu \
  --project outputs/preflight --name check

# 4. Clean up
rm -rf /tmp/preflight outputs/preflight
```

**What this catches:**
- Missing or malformed data.yaml paths
- Label format errors (wrong number of columns, bad class IDs)
- Image loading failures (corrupt files, unsupported formats)
- Model architecture mismatches (wrong number of classes vs data.yaml)
- Import errors or missing dependencies
- Augmentation pipeline crashes on real data
- Shape mismatches between model input and preprocessing

**Pass criteria:** Both epochs complete without errors. Ignore metrics — they're
meaningless on 10 images.

**If preflight fails:** Fix the error locally. Do NOT launch the remote instance until
preflight passes. Every preflight failure caught here saves 5-10 minutes of cloud billing
for an instance that would have failed the same way.

**If preflight passes:** Clean up temp files and proceed to the selected training path.

---

4. **Route to the correct training path based on Step 0b compute selection:**

#### Path A: Local GPU Training

4a. Execute training using the project's training script.

#### Path B: VAST.ai Remote Training

4b. Launch and manage a VAST instance:

**GPU compatibility:** The Ultralytics Docker image requires compute capability >= 7.5
(Turing or newer). Always filter for `compute_cap >= 750` when searching. Older GPUs
(Pascal: GTX 1080, 1080 Ti, Titan Xp) will fail with `CUDA error: no kernel image
is available for execution on the device`.

```
1. Search for a suitable instance (see Step 0b pricing query)
   IMPORTANT: filter for compute_cap >= 750 to avoid Pascal GPU incompatibility
2. Create the instance with the ultralytics Docker image:
   vastai create instance <ID> --image ultralytics/ultralytics:latest --disk 20
3. Wait for instance to start, get SSH details:
   vastai show instances --raw
4. Get dataset onto the instance (prefer remote pull over local upload):
   a. If dataset has a remote source (Roboflow, HuggingFace, S3, etc.):
      → Pull directly on the instance. This is faster and avoids
        uploading gigabytes over your local connection.
      → Example: SSH in and run the Roboflow download script on the instance.
   b. If dataset is local-only (no remote source):
      → Upload via SCP as a fallback.
5. Upload training script via SCP (small file, always fast)
6. SSH in and run training
7. Monitor training progress (tail logs via SSH)
8. Download trained weights when complete
9. DESTROY the instance immediately to stop billing:
   vastai destroy instance <INSTANCE_ID>
```

**Principle: pull, don't push.** Remote training machines have fast datacenter internet.
If the dataset exists on any remote source, always pull it directly on the training
machine rather than uploading from your local machine. SCP upload is a last resort
for datasets that only exist locally.

**Cost tracking:** Record the instance start time. When training finishes, calculate
and report the actual cost:

```
Actual cost: <hours> hrs × $<rate>/hr = $<total>
```

Always destroy the instance after downloading weights. Remind the user if they
haven't destroyed it.

#### Path C: Other Remote (SSH to user's machine)

4c. Upload dataset, run training via SSH, download weights.

### Background Operations

**Run GPU launches, dataset downloads, and training in the background** so the user
can continue working (reviewing config, updating the skill, chatting, etc.). Don't
block the conversation waiting for:
- VAST instance to start (can take 30-120 seconds)
- Dataset upload/download
- Training to complete (can take 30+ minutes)

Use background execution for these long-running operations and check on them
periodically or when the user asks.

### During Training (all paths)

5. Monitor for:
   - Training loss decreasing while val loss increases (overfitting)
   - Val accuracy plateauing (need more data or augmentation)
   - NaN losses (LR too high, data corruption)
6. After training completes, automatically run **Validate**.

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
