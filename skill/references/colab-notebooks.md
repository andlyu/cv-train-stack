# Colab Notebook Integration

## Detection

```bash
# Find all notebooks
find . -name "*.ipynb" -not -path "./.git/*" -not -path "*/.ipynb_checkpoints/*" | head -20

# Scan for training content
python3 -c "
import json, glob
for nb_path in glob.glob('**/*.ipynb', recursive=True):
    with open(nb_path) as f:
        nb = json.load(f)
    sources = ' '.join(c.get('source','') if isinstance(c.get('source',''), str)
                       else ' '.join(c.get('source',[])) for c in nb.get('cells',[]))
    has_training = any(k in sources.lower() for k in ['model.train', '.fit(', 'trainer.', 'yolo', 'epochs', 'train_loader'])
    if has_training:
        print(f'TRAINING NOTEBOOK: {nb_path}')
"
```

## Conversion to Script

```bash
jupyter nbconvert --to script <notebook.ipynb> --output training_from_notebook
```

Or without jupyter:

```python
import json
with open('<notebook.ipynb>') as f:
    nb = json.load(f)
with open('training_from_notebook.py', 'w') as out:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            lines = []
            for line in source.split('\n'):
                stripped = line.strip()
                if stripped.startswith(('!pip ', '!apt ', '%', 'from google.colab',
                                        'import google.colab', 'drive.mount')):
                    lines.append(f'# [COLAB-SKIP] {line}')
                elif stripped.startswith('!'):
                    cmd = stripped[1:]
                    lines.append(f'import subprocess; subprocess.run({cmd!r}, shell=True, check=True)')
                else:
                    lines.append(line)
            out.write('\n'.join(lines))
            out.write('\n\n')
```

## Post-Conversion Review

Check for:
- **Hardcoded Colab paths** (`/content/drive/`, `/content/`): replace with local paths
- **Google Drive mounts**: remove
- **Colab installs** (`!pip install`): extract into `requirements.txt`
- **`%matplotlib` magic**: replace with `matplotlib.use('Agg')`
- **Missing `plt.savefig()`**: add for any `plt.show()` calls

## Common Adaptations

| Colab Pattern | Replacement |
|---|---|
| `drive.mount(...)` | Remove — use local paths |
| `!pip install ultralytics` | Add to requirements.txt |
| `!gdown <id>` | `gdown <id>` or local dataset |
| `%cd /content/` | Remove |
| `files.download(...)` | `shutil.copy(src, 'outputs/')` |
| `from IPython.display import Image` | `plt.imshow(); plt.savefig()` |

## Visualization Capture Patch

Add near top of converted script after matplotlib import:

```python
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

## Visualization Review (post-training)

| Plot | What to verify | Red flags |
|------|---------------|-----------|
| Training curves | Loss decreasing, val following | Val diverging = overfitting |
| Confusion matrix | Diagonal-dominant | Off-diagonal clusters |
| PR / F1 curve | Smooth, high AUC | Jagged = insufficient val data |
| Sample predictions | Correct labels, good confidence | Low confidence on easy examples |
| Augmentation previews | Realistic | Over-aggressive distortion |
