# Model Validation

Run after every training and before every deploy.

## Step 1: Numerical Equivalence Test

Compare PyTorch model vs exported model (ONNX/TRT) on 10+ test images:

```
For each test image:
  1. Load raw image (PNG, not JPEG)
  2. Preprocess with the SHARED preprocessing function
  3. Run through PyTorch model -> get logits/probs
  4. Run through ONNX model -> get logits/probs
  5. Run through TRT engine -> get logits/probs (if applicable)
  6. Compare: max abs diff <0.01 for ONNX, <0.05 for TRT FP16
```

If TRT FP16 divergence >0.05, consider QAT, FP32 fallback, or threshold adjustment.

## Step 2: Accuracy Test

Run project's accuracy test suite on the exported model. Look for `test_*accuracy*`,
`test_*classification*`, `test_*model*`.

Thresholds:
- Overall accuracy >= 90%
- Per-class recall >= 85%
- Per-class precision >= 85%
- Inference speed within target

## Step 3: Batch Sensitivity Test

```
For 5 test images:
  Run at batch=1, batch=4 (with zeros), batch=4 (with real images)
  All should produce identical results (within FP16 tolerance)
```

FP16 TensorRT engines can give different results at different batch sizes.

## Step 4: Golden Fixture Test

If a golden test fixture exists (labeled test crops with expected scores):

```
For each golden crop:
  1. Load PNG
  2. Preprocess (pipeline-identical)
  3. Run through exported model
  4. Compare against expected score (tolerance +/-0.02)
```

If no fixture exists, recommend creating one from this run's outputs.

## Step 5: FPS Benchmark

```bash
# Quick benchmark (100 inference passes)
yolo benchmark model=best.pt imgsz=640 device=0 half=True

# On target device via SSH
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
Device:          <device name>
Model:           <model file>
Input size:      <imgsz>
Precision:       FP16 / FP32
Inference FPS:   XX.X
Target FPS:      <user requirement>
Status:          PASS / FAIL
```

If below target: reduce imgsz, use smaller variant, ensure TRT FP16, reduce camera resolution.

## Step 6: Save Model Metadata

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

## Post-Deployment Verification

After exporting and deploying to target environment:

### If automated tests exist:
1. Run accuracy tests on exported format (.engine, .onnx)
2. Run pipeline/integration tests end-to-end
3. Compare against previous model version — flag regressions

```
| Metric           | Previous (vN-1) | Current (vN) | Delta  |
|------------------|-----------------|--------------|--------|
| Accuracy         | 96.7%           | 98.3%        | +1.6%  |
| Bad recall       | 92.0%           | 95.0%        | +3.0%  |
| Inference (ms)   | 4.8             | 5.9          | +1.1   |
```

### If no tests exist:
1. Visual verification on 10-20 representative inputs
2. Spot-check known failure modes (edge crops, unusual lighting, ambiguous examples)
3. Brief live pipeline test
4. Recommend creating test suite from verified examples
