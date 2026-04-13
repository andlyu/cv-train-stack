# GPU Detection & Compute Selection

## Step 1: Detect Available GPU

Run this first. It determines the entire training path.

```bash
# 1. Check for NVIDIA GPU locally
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}, Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# 2. Check VRAM if GPU found
python3 -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB') if torch.cuda.is_available() else None"

# 3. Check for Apple Silicon MPS
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

## Decision Tree

| Result | Action |
|--------|--------|
| NVIDIA GPU with >=6GB VRAM | **Use local GPU.** |
| NVIDIA GPU with <6GB VRAM | WARN: may OOM. Suggest reducing batch/imgsz, or use VAST. |
| Apple Silicon (MPS) | WARN: works for small models, slower than CUDA. OK for quick iterations, recommend VAST for final runs. |
| **No GPU (CPU only)** | **STOP. Do not train on CPU.** Guide user to VAST.ai. |

**If no GPU:**

> Training on CPU is not recommended — a 300-epoch YOLO run that takes 30 minutes on GPU
> will take 10+ hours on CPU. VAST.ai offers GPUs starting at ~$0.15/hr.
>
> To get started:
> 1. Create an account at https://vast.ai
> 2. Add your API key: `pip install vastai && vastai set api-key <KEY>`
> 3. Re-run this skill and we'll launch a cloud GPU automatically.

## Step 2: Present Compute Options

| Option | When to show | Est. cost |
|--------|-------------|-----------|
| Local GPU | NVIDIA GPU detected | Free |
| Local MPS | Apple Silicon detected | Free (with caveats) |
| VAST.ai | Always (recommended if no local NVIDIA) | Show live pricing |
| Other cloud | Only if user mentions it | Varies |

**For VAST.ai, show live pricing:**

```bash
vastai search offers 'gpu_ram >= 8 reliability > 0.95 num_gpus == 1 cuda_vers >= 12.0' \
  --order 'dph' --limit 3 --output 'id gpu_name gpu_ram disk_space dph_total dlperf'
```

**Estimate training time** based on dataset size and model:
- YOLO nano/small + <5K images: ~20-40 min on consumer GPU
- YOLO medium/large + <5K images: ~40-90 min
- YOLO nano/small + 5-20K images: ~1-2 hrs
- Classifier (MobileNet) + <5K images: ~10-20 min

Multiply time estimate by $/hr for estimated cost. Always show before committing.

## VAST.ai Remote Training

**GPU compatibility:** Ultralytics Docker requires compute capability >= 7.5 (Turing+).
Always filter for `compute_cap >= 750`. Older GPUs (Pascal: GTX 1080, Titan Xp) will fail.

```
1. Search for instance (see pricing query above)
   IMPORTANT: filter for compute_cap >= 750
2. Create: vastai create instance <ID> --image ultralytics/ultralytics:latest --disk 20
3. Wait, get SSH: vastai show instances --raw
4. Get dataset onto instance:
   a. If dataset has remote source (Roboflow, HuggingFace, S3) → pull on instance (faster)
   b. If local-only → upload via SCP as fallback
5. Upload training script via SCP
6. SSH in and run training
7. Monitor (tail logs via SSH)
8. Download trained weights
9. DESTROY instance: vastai destroy instance <INSTANCE_ID>
```

**Principle: pull, don't push.** Remote machines have fast internet. Pull datasets directly.

**Cost tracking:** Record start time. Report actual cost when done:
```
Actual cost: <hours> hrs x $<rate>/hr = $<total>
```

Always destroy the instance after downloading weights.

## Background Operations

Run GPU launches, dataset downloads, and training in the background so the user can
continue working. Don't block the conversation.

## GPU Utilization Check (paid GPUs)

After training starts, sample GPU stats:

```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
```

| Metric | Target | If below target |
|--------|--------|----------------|
| GPU utilization | >80% | Increase `workers`, enable `cache=True` |
| Memory utilization | >50% | Increase `batch` until 60-80% used |
| Memory used vs total | 60-80% | If <40%, double batch. If >90%, reduce batch. |

**Common fixes:** increase batch size, increase workers, enable caching, increase imgsz.
