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
dataset directories, and preprocessing code. No config file needed.

If a `.cv-training.json` exists, use it as hints, but never require it.

## Subcommands

- `/cv-training review` → **Review** an existing or planned training run
- `/cv-training run <type>` → **Execute** a training run
- `/cv-training validate <model>` → **Validate** a trained model before deployment
- `/cv-training audit` → **Audit** all training/inference paths for consistency

Default to `review` if no subcommand given.

---

## Review

Walk through training config and flag issues before training starts.

### Workflow

1. **GPU detection** → Load `references/gpu-compute.md`
   - Detect GPU, present compute options with cost estimates
   - Do NOT proceed until a GPU path is confirmed

2. **Colab check** → Load `references/colab-notebooks.md` (only if notebooks found)
   - Scan for `.ipynb` files with training logic
   - Convert to runnable script if user confirms

3. **Logging selection** — Ask CSV (default, zero deps) or W&B (additive).
   CSV log to `output/training_log.csv` is **required** regardless.

4. **Read current state** — Scan project for model type, training script, hyperparams,
   all inference paths.

5. **Dataset audit** → Load `references/dataset-audit.md`
   - Quantitative checks (class balance, label completeness, train/val leak)
   - Visual spot-check with label overlays
   - Stop if any FAIL

6. **Preprocessing consistency** → Load `references/preprocessing-audit.md`
   - Build consistency table across all preprocessing locations
   - ANY mismatch is a critical bug

7. **Training config review** → Load `references/training-config.md`
   - Augmentation, hyperparameters, FPS/input size tradeoff
   - Logging and reproducibility checklist

Present findings as summary table. Ask user to confirm before proceeding.

---

## Run

Execute a training run with all checks passed.

1. Run **Review** first (all steps). Do not skip.
2. Fix any critical issues before proceeding.
3. **Preflight check** (required before remote training) → See `references/training-config.md`
   - Run 2 epochs on ~10 images locally to catch errors before spending money
4. Route to compute path:
   - **Local GPU**: run training script directly
   - **VAST.ai**: launch instance, upload, train, download, destroy → See `references/gpu-compute.md`
   - **Other remote**: upload, SSH, train, download
   - **Colab notebook**: use converted script → See `references/colab-notebooks.md`
5. Monitor for overfitting, plateaus, NaN losses
6. After training completes, automatically run **Validate**

---

## Validate

Validate a trained model before deployment. → Load `references/validation.md`

Steps:
1. Numerical equivalence (PyTorch vs ONNX vs TRT)
2. Accuracy test on exported model
3. Batch sensitivity test (FP16 batch=1 vs batch=4)
4. Golden fixture test (if exists)
5. FPS benchmark on target device
6. Save model metadata
7. Notebook visualization review (if applicable)
8. Post-deployment verification (automated tests or visual check)

---

## Audit

Full preprocessing consistency audit across all training/inference paths.
→ Load `references/preprocessing-audit.md`

---

## Common Mistakes

Real production bugs this skill catches:

1. **Resize mismatch** — training at 224x224, inference at 128x128 (40pt divergence)
2. **JPEG artifacts** — re-extracting from JPEG vs raw PNG shifts scores 7+ points
3. **Missing training args** — can't reproduce how model was trained
4. **Channel order** — training in RGB, inference in BGR (OpenCV default)
5. **Normalization skew** — ImageNet normalization in training, raw [0,1] in inference
6. **Batch sensitivity** — FP16 TRT different results at batch=1 vs batch=4
7. **No golden fixture** — can't validate export against known-good scores

## References (load only if needed)

- `references/gpu-compute.md` — GPU detection, VAST.ai, cost tracking, utilization
- `references/dataset-audit.md` — dataset quality checks, visual spot-check
- `references/preprocessing-audit.md` — consistency table, training-serving skew
- `references/training-config.md` — hyperparams, augmentation, logging, preflight
- `references/validation.md` — equivalence tests, accuracy, FPS, deployment verification
- `references/colab-notebooks.md` — notebook detection, conversion, visualization capture
