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

### Step 0d: Logging Selection

**Ask the user which logging backend to use.** Default to custom CSV.

| Option | When to show | Notes |
|--------|-------------|-------|
| **Custom CSV (default)** | Always | Zero dependencies. Writes `training_log.csv` to output dir with one row per epoch. Append-mode so metrics are visible mid-training. |
| Weights & Biases (W&B) | Always | Richer dashboards, experiment comparison. Requires `wandb` installed and authenticated. |

**Regardless of choice, the training script MUST:**
1. Write metrics to `output/training_log.csv` every epoch (append-mode, never buffered)
2. CSV columns: `epoch,train_loss,train_acc,val_loss,val_acc,lr`
3. Write the CSV header before training starts
4. Open the file in append mode and flush/close after each epoch write — metrics must
   be readable mid-training (no buffering, no writing only at the end)

This is non-negotiable — the CSV log is the minimum. W&B is additive on top.

**If W&B is selected**, also:
- `wandb.init(project=<project_name>, config=<all hyperparams>)`
- `wandb.log()` each epoch with train/val loss, accuracy, and LR
- Log the best model as a W&B artifact

### Step 0e: Colab Notebook Detection

**Check if the user has an existing Colab/Jupyter notebook for training.**

```bash
# Find all notebooks in the project
find . -name "*.ipynb" -not -path "./.git/*" -not -path "*/.ipynb_checkpoints/*" | head -20
```

If notebooks are found, read them and determine if they contain training logic:

```bash
# Quick scan for training-related content in notebooks
python3 -c "
import json, sys, glob
for nb_path in glob.glob('**/*.ipynb', recursive=True):
    with open(nb_path) as f:
        nb = json.load(f)
    sources = ' '.join(c.get('source',['']) if isinstance(c.get('source',''), str)
                       else ' '.join(c.get('source',[])) for c in nb.get('cells',[]))
    has_training = any(k in sources.lower() for k in ['model.train', '.fit(', 'trainer.', 'yolo', 'epochs', 'train_loader'])
    has_gpu = any(k in sources for k in ['cuda', 'gpu', 'runtime'])
    if has_training:
        print(f'TRAINING NOTEBOOK: {nb_path} (GPU refs: {has_gpu})')
"
```

**If a training notebook is found:**

1. **Ask the user:** "Found `<notebook.ipynb>` — would you like to use this notebook
   as your training script? I can convert it to a runnable Python script, wire up
   GPU compute, and integrate its visualizations into the verification steps."

2. **Convert to executable script:**

   ```bash
   # Convert notebook to Python script (strips outputs and magic commands)
   jupyter nbconvert --to script <notebook.ipynb> --output training_from_notebook

   # Or if jupyter is not installed:
   python3 -c "
   import json
   with open('<notebook.ipynb>') as f:
       nb = json.load(f)
   with open('training_from_notebook.py', 'w') as out:
       for cell in nb['cells']:
           if cell['cell_type'] == 'code':
               source = ''.join(cell['source'])
               # Skip Colab-specific magic commands
               lines = []
               for line in source.split('\n'):
                   stripped = line.strip()
                   if stripped.startswith(('!pip ', '!apt ', '%', 'from google.colab',
                                          'import google.colab', 'drive.mount',
                                          'files.download', 'files.upload')):
                       lines.append(f'# [COLAB-SKIP] {line}')
                   elif stripped.startswith('!'):
                       # Convert shell commands to subprocess calls
                       cmd = stripped[1:]
                       lines.append(f'import subprocess; subprocess.run({cmd!r}, shell=True, check=True)')
                   else:
                       lines.append(line)
               out.write('\n'.join(lines))
               out.write('\n\n')
   print('Converted to training_from_notebook.py')
   "
   ```

3. **Review the converted script** for issues:
   - **Hardcoded Colab paths** (`/content/drive/`, `/content/`): replace with local project paths
   - **Google Drive mounts**: remove or replace with local paths
   - **Colab-specific installs** (`!pip install`): extract into `requirements.txt`
   - **Inline `%matplotlib` magic**: replace with `matplotlib.use('Agg')` for headless rendering
   - **Missing `plt.savefig()`**: add saves for any `plt.show()` calls so visualizations are captured
   - **Hardcoded dataset URLs**: check they're still accessible, or point to local data

4. **Extract dependencies** the notebook assumes:

   ```bash
   # Pull all imports from the converted script
   python3 -c "
   import ast, sys
   with open('training_from_notebook.py') as f:
       tree = ast.parse(f.read())
   imports = set()
   for node in ast.walk(tree):
       if isinstance(node, ast.Import):
           for alias in node.names:
               imports.add(alias.name.split('.')[0])
       elif isinstance(node, ast.ImportFrom) and node.module:
           imports.add(node.module.split('.')[0])
   # Filter out stdlib
   stdlib = {'os','sys','json','math','time','glob','pathlib','shutil','collections',
             'itertools','functools','random','copy','csv','datetime','re','io','argparse'}
   print('\n'.join(sorted(imports - stdlib)))
   "
   ```

   Install any missing packages before training.

5. **Patch visualization cells** to save outputs to disk:

   Any `plt.show()` in the notebook should become `plt.savefig()` followed by `plt.show()`.
   The converted script should save all plots to an `outputs/plots/` directory with
   descriptive filenames (e.g., `training_curves.png`, `confusion_matrix.png`,
   `sample_predictions.png`). These plots will be used in verification steps.

   ```python
   # Template patch: replace bare plt.show() calls
   import os
   os.makedirs('outputs/plots', exist_ok=True)
   PLOT_COUNTER = [0]

   _original_show = plt.show
   def _saving_show(*args, **kwargs):
       PLOT_COUNTER[0] += 1
       plt.savefig(f'outputs/plots/notebook_plot_{PLOT_COUNTER[0]:03d}.png',
                   dpi=150, bbox_inches='tight')
       _original_show(*args, **kwargs)
   plt.show = _saving_show
   ```

   Add this patch near the top of the converted script, after matplotlib is imported.

**If NO notebook found but user mentions they have one:**

Ask them to place it in the project directory or provide the path/URL. If it's a
Colab URL, download it:

```bash
# Download notebook from Colab share link
# Convert share URL to download URL: replace /edit with /export?format=ipynb
COLAB_URL="<user's share link>"
DOWNLOAD_URL=$(echo "$COLAB_URL" | sed 's|/edit.*|/export?format=ipynb|')
curl -L "$DOWNLOAD_URL" -o notebook.ipynb
```

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

**If a Colab notebook was detected (Step 0e)** and it contains visualization cells
(e.g., displaying sample images, augmentation previews, or label overlays), extract
those cells into the visualization script or reuse their output images from
`outputs/plots/`. Notebook authors often include good visual QA — don't duplicate it,
integrate it.

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

### Step 4a: FPS / Input Size Tradeoff

**If the model will run on edge hardware, ask the user about FPS requirements.**

Input size is the biggest lever for inference speed. Larger images = better accuracy
but slower inference. This tradeoff must be decided before training, not after.

**Ask:** "What FPS do you need on your target device?"

| Target Device | imgsz=320 | imgsz=640 | imgsz=1280 |
|--------------|-----------|-----------|------------|
| Jetson Orin Nano (FP16) | ~60-80 FPS | ~15-25 FPS | ~4-8 FPS |
| Jetson Orin NX (FP16) | ~100+ FPS | ~30-50 FPS | ~10-15 FPS |
| Jetson AGX Orin (FP16) | ~150+ FPS | ~60-80 FPS | ~20-30 FPS |
| RTX 3060 (FP16) | ~200+ FPS | ~80-120 FPS | ~25-40 FPS |

*Estimates for YOLO11n. Actual FPS depends on model size, batch size, and TensorRT optimization.*

**Decision guide:**
- Need >30 FPS real-time: use imgsz=320 or 640 depending on device
- Need >60 FPS (robotics, real-time control): likely imgsz=320
- Accuracy-first (offline processing): use imgsz=640 or larger
- If unsure, train at 640 — it's the best default balance

**After training, always benchmark actual FPS on the target device** (see Validate
Step 6). Training-time estimates are rough — TensorRT optimization and device-specific
factors can shift FPS significantly.

For transfer learning (classifier):
1. Freeze base, train head only (5-10 epochs)
2. Unfreeze top layers, fine-tune with 10x lower LR
3. Use ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Step 4b: GPU Utilization Check

**If training on a paid GPU, offer to check utilization during the first few epochs.**
You're paying by the hour — an underutilized GPU is wasted money.

After training starts, sample GPU stats:

```bash
# On the training machine (via SSH if remote)
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
```

| Metric | Target | If below target |
|--------|--------|----------------|
| GPU utilization | >80% | Data loading bottleneck — increase `workers`, enable `cache=True`, or use faster storage |
| Memory utilization | >50% | Batch size too small — increase `batch` until memory is 60-80% used |
| Memory used vs total | 60-80% of total | If <40%, double the batch size. If >90%, reduce batch to avoid OOM |

**Common fixes for low GPU utilization:**

1. **Increase batch size** — the single biggest lever. Use `batch=-1` (auto) for YOLO,
   or manually set to fill 60-80% of VRAM.
2. **Increase dataloader workers** — `workers=8` or higher. CPU preprocessing can
   starve the GPU if workers is too low.
3. **Enable caching** — `cache=True` or `cache='disk'` loads all images into RAM/disk
   once, eliminating repeated I/O.
4. **Increase image size** — if GPU memory allows, larger `imgsz` uses more compute
   per batch and can improve accuracy.

**When to skip:** If GPU util is already >80% and memory is 60-80% used, the config
is already well-tuned. Don't over-optimize.

### Step 5: Logging and Reproducibility

Before training starts, verify:

- [ ] **Exact train command is saved** — write the full CLI invocation (with all args) to a `train_command.sh` file in the output directory. This must be copy-pasteable to reproduce the run. For Python API calls, save the equivalent CLI command or a JSON of all parameters.
- [ ] Training args/hyperparameters are saved (args file or config json)
- [ ] Random seed is set for reproducibility
- [ ] Dataset version/composition is recorded (source, version, class counts, split sizes)
- [ ] **Training metrics are continuously logged to `output/training_log.csv`** — CSV with columns `epoch,train_loss,train_acc,val_loss,val_acc,lr`. Must be append-mode and flushed after each epoch so metrics are readable mid-training. This is REQUIRED regardless of logging backend choice (see Step 0d).
- [ ] If W&B was selected in Step 0d, `wandb.log()` is called each epoch
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

#### Path D: Colab Notebook as Training Script

If a Colab notebook was detected in Step 0e and the user confirmed its use:

4d. **Prepare and execute the converted notebook script:**

```
1. Verify the converted script exists (training_from_notebook.py)
   - If not, run the conversion from Step 0e now

2. Ensure all Colab-specific paths are patched:
   a. Replace /content/drive/MyDrive/... → local project paths
   b. Replace /content/ → ./
   c. Replace Colab dataset download cells → local dataset path or
      remote pull commands suitable for the target machine

3. Verify the visualization capture patch is in place:
   - All plt.show() calls save to outputs/plots/
   - Training curve callbacks save to outputs/plots/training_curves.png
   - Confusion matrix / PR curves save to outputs/plots/

4. Run on the selected compute (local GPU, VAST, etc.):
   python3 training_from_notebook.py
```

**Common Colab notebook adaptations for local/VAST execution:**

| Colab Pattern | Local/VAST Replacement |
|---|---|
| `from google.colab import drive; drive.mount(...)` | Remove — use local paths |
| `!pip install ultralytics` | Add to requirements.txt, install before run |
| `!gdown <id>` | `gdown <id>` or point to local dataset |
| `%cd /content/` | Remove or `os.chdir(project_root)` |
| `from google.colab import files; files.download(...)` | `shutil.copy(src, 'outputs/')` |
| `from IPython.display import Image, display` | `plt.imshow(plt.imread(path)); plt.savefig(...)` |
| `model.train(data='/content/data.yaml', ...)` | Update `data=` path to local data.yaml |
| GPU runtime selection (Colab menu) | Handled by Step 0a/0b GPU detection |

**If the notebook downloads a dataset from Roboflow, HuggingFace, or a URL:**

Keep that download logic but adapt it:
- On VAST: let it download directly on the instance (fast datacenter internet)
- Locally: download to `datasets/` directory and update paths

**After training completes**, the converted script should have produced:
- Trained model weights in the expected output directory
- Training plots in `outputs/plots/` (from the visualization capture patch)
- CSV training log in `outputs/training_log.csv` (add if notebook doesn't produce one)

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

### Step 5: FPS Benchmark

**If the model targets edge/real-time deployment, benchmark inference speed.**

Run on the target device if accessible, otherwise benchmark locally and note
that actual device FPS will differ.

```bash
# Quick benchmark with Ultralytics (runs 100 inference passes)
yolo benchmark model=best.pt imgsz=640 device=0 half=True

# Or via Python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.benchmark(imgsz=640, half=True, device=0)
```

**If target device is accessible via SSH:**

```bash
# Copy model to device and benchmark there
scp best.pt user@device:/tmp/
ssh user@device 'python3 -c "
from ultralytics import YOLO
model = YOLO(\"/tmp/best.pt\")
results = model.benchmark(imgsz=640, half=True, device=0)
"'
```

**Report:**

```
FPS BENCHMARK
=============
Device:          <device name>
Model:           <model file>
Input size:      <imgsz>
Precision:       FP16 / FP32
Inference FPS:   XX.X
Latency (ms):    XX.X

Target FPS:      <user requirement>
Status:          PASS / FAIL
```

**If FPS is below target:**
1. Reduce `imgsz` (biggest impact — halving size roughly 4x faster)
2. Use a smaller model variant (n → pico, s → n)
3. Ensure TensorRT export with FP16 on the target device
4. Reduce input resolution at the camera/pipeline level

### Step 6: Save Model Metadata

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

### Step 7: Notebook Visualization Review

**If the training run originated from a Colab notebook (Step 0e), review all captured
visualizations as part of validation.**

```bash
# List all captured plots
ls -la outputs/plots/*.png 2>/dev/null
```

**For each visualization found, display it and verify:**

```bash
# Open all plots for review (macOS)
open outputs/plots/*.png

# Or display inline if running in a notebook/terminal with image support
python3 -c "
import glob
for p in sorted(glob.glob('outputs/plots/*.png')):
    print(f'Plot: {p}')
"
```

**Expected visualizations from typical training notebooks and what to check:**

| Plot | What to verify | Red flags |
|------|---------------|-----------|
| **Training curves** (loss vs epoch) | Train loss decreasing, val loss following | Val loss diverging from train = overfitting |
| **Confusion matrix** | Diagonal-dominant, no systematic misclassifications | Off-diagonal clusters = class confusion |
| **PR curve / F1 curve** | Smooth curves, high AUC | Jagged = insufficient val data |
| **Sample predictions** | Correct labels, reasonable confidence | Low confidence on easy examples = problem |
| **Augmentation previews** | Augmentations look realistic | Over-aggressive distortion = hurting accuracy |
| **Class distribution** | Roughly balanced or intentionally weighted | Extreme imbalance not addressed |
| **Learning rate schedule** | Smooth warmup then decay | Spikes or flat regions = misconfigured scheduler |

**If key visualizations are missing**, generate them from training outputs:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os

os.makedirs('outputs/plots', exist_ok=True)

# Generate training curves from CSV log
if os.path.exists('outputs/training_log.csv'):
    with open('outputs/training_log.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if rows:
        epochs = [int(r['epoch']) for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, [float(r['train_loss']) for r in rows], label='train')
        axes[0].plot(epochs, [float(r['val_loss']) for r in rows], label='val')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].set_xlabel('Epoch')
        axes[1].plot(epochs, [float(r['train_acc']) for r in rows], label='train')
        axes[1].plot(epochs, [float(r['val_acc']) for r in rows], label='val')
        axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].set_xlabel('Epoch')
        plt.tight_layout()
        plt.savefig('outputs/plots/training_curves.png', dpi=150, bbox_inches='tight')
        print('Saved: outputs/plots/training_curves.png')

# Generate confusion matrix from YOLO results
results_csv = None
for p in ['runs/segment/train/results.csv', 'runs/detect/train/results.csv']:
    if os.path.exists(p):
        results_csv = p
        break

confusion_png = None
for p in ['runs/segment/train/confusion_matrix.png',
          'runs/detect/train/confusion_matrix.png']:
    if os.path.exists(p):
        confusion_png = p
        break

if confusion_png:
    import shutil
    shutil.copy(confusion_png, 'outputs/plots/confusion_matrix.png')
    print(f'Copied: {confusion_png} → outputs/plots/confusion_matrix.png')
```

**Integrate visualizations into the validation report:**

After running all validation steps (1-6), present a summary that includes
the notebook's visual outputs alongside numerical results:

```
VALIDATION SUMMARY
==================
Numerical equivalence:  PASS (max diff: 0.003)
Accuracy test:          PASS (96.2%)
Batch sensitivity:      PASS
Golden fixture:         PASS (12/12 within tolerance)
FPS benchmark:          PASS (45 FPS on Jetson Orin NX)

TRAINING VISUALIZATIONS
=======================
✓ Training curves:      outputs/plots/training_curves.png
  → Loss converged at epoch 87, no overfitting detected
✓ Confusion matrix:     outputs/plots/confusion_matrix.png
  → Clean diagonal, no systematic misclassifications
✓ Sample predictions:   outputs/plots/sample_predictions.png
  → 20/20 correct on spot-check
✗ PR curve:             NOT FOUND — recommend generating from val results
```

Open all available plots for the user to review visually before signing off
on the model.

---

## Post-Deployment Verification

After exporting and deploying a model (to device, server, or TRT engine), verify it
works correctly in the target environment. This is a separate step from Validate —
Validate checks the model in isolation, this checks it in the real pipeline.

### Path A: Automated Tests Exist

If the project has test files (e.g. `test_*accuracy*`, `test_*pipeline*`, `test_*classification*`):

1. **Run accuracy tests on the deployed model** — these should test the exported format
   (.engine, .onnx) not the .pt file. Verify:
   - Overall accuracy meets threshold (e.g. >=90%)
   - Per-class recall/precision meet thresholds
   - Inference speed meets target
   
2. **Run pipeline/integration tests** — these run the full pipeline end-to-end with
   the deployed model. Verify:
   - Model loads correctly in the pipeline
   - Processing speed is acceptable (no regression)
   - Classification results are consistent with training metrics

3. **Compare against previous version** — if a previous model version exists, run the
   same tests on both and compare. Flag any regressions.

Report results as a comparison table:

```
| Metric           | Previous (vN-1) | Current (vN) | Delta  |
|------------------|-----------------|--------------|--------|
| Accuracy         | 96.7%           | 98.3%        | +1.6%  |
| Bad recall       | 92.0%           | 95.0%        | +3.0%  |
| Good precision   | 97.0%           | 98.0%        | +1.0%  |
| Inference (ms)   | 4.8             | 5.9          | +1.1   |
| Misclassified    | 2               | 1            | -1     |
```

### Path B: No Tests (First Deploy or New Project)

If no automated tests exist yet:

1. **Visual verification** — run the model on 10-20 representative inputs and show
   the results to the user. For classifiers, show the image with the predicted label
   and confidence. For detectors, show bounding boxes overlaid on images.

2. **Spot-check failure modes** — specifically test known hard cases:
   - Edge-of-frame crops
   - Unusual angles or lighting
   - Ambiguous examples near the decision boundary
   - Examples the previous model got wrong (if known)

3. **Live pipeline test** — if the model runs in a real-time pipeline, run it briefly
   and check the output visually. Look for:
   - Are predictions sensible?
   - Any obvious false positives or false negatives?
   - Is latency acceptable?

4. **Recommend creating tests** — after visual verification passes, suggest creating
   a test suite from the verified examples so future deploys can be tested automatically.

### Path C: Both

When automated tests exist but you want extra confidence (e.g. major architecture change,
new training data source, or production-critical deploy):

1. Run Path A (automated tests)
2. Then run a brief visual check from Path B on the hardest examples
3. If the pipeline supports it, run a short live test with real data

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
