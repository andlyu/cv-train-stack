# Preprocessing Consistency Audit

**This is the #1 source of silent accuracy loss.**

## Find All Preprocessing Locations

```
1. Training script transforms (train_transform, val_transform)
2. Pipeline inference path (both TRT and non-TRT paths)
3. Evaluation/extract scripts
4. Accuracy test scripts
5. Export scripts
```

## Build the Consistency Table

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

## Full Audit Command

```bash
grep -rn "resize\|Resize\|normalize\|Normalize\|INPUT_SIZE\|input_size\|IMGSZ" \
  --include="*.py" | grep -v __pycache__
```

For each file found, extract:
- What model it feeds
- Resize target size
- Resize method (cv2 vs PIL, interpolation mode)
- Normalization values
- Channel ordering (RGB vs BGR)

## Audit Report Format

```
PREPROCESSING AUDIT
===================
Files checked:     N
Models covered:    <list>
Mismatches found:  N

[details per mismatch]

Status: PASS | FAIL
```

Reference: Google's Rules of ML — "training-serving skew is one of the most common production ML bugs."
