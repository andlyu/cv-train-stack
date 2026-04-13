# Dataset Audit

## Quantitative Checks

For each item, report PASS/FAIL/WARN:

| Check | Target | How to verify |
|-------|--------|---------------|
| Images per class | >=1500 (detection), >=500 (classifier) | Count files in dataset dirs |
| Class balance | No class >3x another | Count per-class, compute ratio |
| Label completeness | All instances labeled | Spot-check 10 random images |
| Background images | 0-10% of dataset | Count unlabeled images |
| Environment diversity | Multiple lighting/angles | Check metadata or visually sample |
| Train/val leak | Zero overlap | Compare file lists |
| Data format | Consistent resolution/format | Check a sample |

For classifiers specifically:
- Crop source diversity (different frames, positions, angles)
- Whether crops include known failure modes from production

## Visual Spot-Check

After quantitative audit, **always show sample images with label overlays**:

```bash
python3 scripts/visualize_samples.py --data-dir <dataset_path> --split train --n 6 --seed 42
```

Check for:
- Labels covering the correct objects (no offset or misaligned polygons)
- Missing labels (unlabeled objects visible)
- Wrong class assignments
- Label quality at edges (tight vs loose polygon fits)

If no `scripts/visualize_samples.py` exists, create one that:
1. Picks N random images from the requested split
2. Draws segmentation polygons (filled with 30% opacity) and class names
3. Arranges them in a 2-column grid
4. Opens in system image viewer

Report findings. If any FAIL, stop and address before proceeding.
