# CV Training Stack

A Claude Code skill for computer vision model training. Drop it into any project and get a structured training workflow with best-practices checks built in.

## What it does

When you say "train a model" or `/cv-training`, Claude will:

1. **Audit your dataset** — class balance, size, train/val leak, format consistency
2. **Check preprocessing consistency** — finds every place preprocessing is defined and flags mismatches between training and inference
3. **Review augmentation** — compares your augmentation pipeline against recommended ranges
4. **Review training config** — optimizer, LR, batch size, early stopping, etc.
5. **Validate after training** — numerical equivalence between PyTorch/ONNX/TRT, accuracy tests, batch sensitivity

## Install

Copy the skill into your project's `.claude/skills/` directory:

```bash
# From your project root
mkdir -p .claude/skills/cv-training
cp /path/to/cv-training-stack/skill/SKILL.md .claude/skills/cv-training/SKILL.md
```

Or clone and symlink:

```bash
git clone https://github.com/andlyu/cv-training-stack.git
ln -s $(pwd)/cv-training-stack/skill .claude/skills/cv-training
```

## Configuration

Create a `.cv-training.json` in your project root to customize the skill for your project:

```json
{
  "models": {
    "classifier": {
      "architecture": "mobilenetv3_small",
      "input_size": 128,
      "num_classes": 2,
      "class_names": ["bad", "good"],
      "training_script": "scripts/train_classifier.py",
      "inference_paths": [
        "src/pipeline/vision.py",
        "tests/test_accuracy.py"
      ],
      "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
      }
    },
    "detector": {
      "architecture": "yolov8n",
      "input_size": 640,
      "training_script": "scripts/train_detector.py"
    }
  },
  "dataset_dir": "datasets/",
  "output_dir": "outputs/train/",
  "export_formats": ["onnx", "tensorrt"]
}
```

If no config file exists, the skill will auto-detect by scanning your project.

## Usage

```
/cv-training review          # Pre-training checklist
/cv-training run classifier  # Execute training with all checks
/cv-training validate model.pt  # Post-training validation
/cv-training audit           # Full preprocessing consistency audit
```

## What gets checked

### Dataset Audit
- Images per class (min thresholds by model type)
- Class balance (no class >3x another)
- Train/val data leak (zero overlap)
- Format consistency (resolution, file types)
- Environment diversity

### Preprocessing Consistency
The #1 source of silent accuracy loss in production ML. The skill finds every file that preprocesses images and verifies they all match:
- Input size
- Resize method
- Normalization values
- Channel ordering (RGB vs BGR)
- Value range

### Augmentation Review
Compares against recommended ranges for CV training:
- Resize + crop strategy
- Color jitter, blur, noise
- Geometric transforms (rotation, flip, affine)
- JPEG compression artifacts (critical for pipeline-saved images)
- Random erasing

### Training Config
- Optimizer and LR schedule
- Batch size and its effect on batch norm
- Early stopping configuration
- Pretrained weights and transfer learning strategy
- Reproducibility (seeds, logging)

### Post-Training Validation
- PyTorch vs ONNX vs TensorRT numerical equivalence
- Accuracy test suite
- Batch size sensitivity (batch=1 vs batch=4)
- Golden fixture regression test

## References

- [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Google ML Test Score](https://research.google/pubs/pub46555/)
- [Ultralytics Training Tips](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)
- [MIT Foundations of CV](https://visionbook.mit.edu/)

## License

MIT
