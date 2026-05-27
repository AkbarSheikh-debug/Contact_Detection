# RTX 5080 (Blackwell) — GPU Setup Guide for PyTorch Projects

Tested configuration that works:
- **GPU**: NVIDIA GeForce RTX 5080 (Blackwell GB203)
- **Driver**: 591.86
- **Max CUDA**: 13.1
- **OS**: Windows 11 Pro
- **Python**: 3.14.x
- **PyTorch**: 2.11.0+cu128
- **Install index**: `https://download.pytorch.org/whl/cu128`

---

## 1. Check Your GPU and Driver First

Run in PowerShell or CMD:

```powershell
nvidia-smi
```

Look for two numbers:
- **Driver Version** — e.g. `591.86`
- **CUDA Version** — e.g. `13.1` (this is the *maximum* CUDA your driver supports)

You can use any PyTorch CUDA build **at or below** that version.  
RTX 5080 needs **CUDA 12.8 minimum** for full Blackwell support.

---

## 2. The Problem: pip Installs CPU-Only PyTorch by Default

Running `pip install torch` gives you `torch-x.x.x+cpu` — no GPU support.  
Even `pip install torch torchvision` from the default index gives CPU-only.

You **must** point pip at the CUDA wheel index explicitly.

---

## 3. Install PyTorch with CUDA 12.8 (RTX 5080)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If PyTorch is **already installed** (even as CPU-only), pip will say "Requirement already satisfied" and skip it.  
In that case, force reinstall:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

The `--no-deps` flag prevents pip from also reinstalling every dependency (numpy, pillow, etc.), which would take much longer.

---

## 4. Verify GPU is Available

```python
import torch

print("PyTorch version  :", torch.__version__)
print("CUDA available   :", torch.cuda.is_available())
print("CUDA version     :", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU              :", torch.cuda.get_device_name(0))
    print("VRAM             :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
```

Expected output:
```
PyTorch version  : 2.11.0+cu128
CUDA available   : True
CUDA version     : 12.8
GPU              : NVIDIA GeForce RTX 5080
VRAM             : 16.3 GB
```

If `CUDA available: False` — see Section 6 (Troubleshooting).

---

## 5. Boilerplate Device Setup for Any Project

Put this near the top of every script:

```python
import torch

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps"  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE.upper()}", end="")
if DEVICE == "cuda":
    print(f"  ({torch.cuda.get_device_name(0)})", end="")
print()
```

Move any model or tensor to GPU:

```python
model = MyModel()
model = model.to(DEVICE)          # move model

tensor = torch.tensor([1.0, 2.0])
tensor = tensor.to(DEVICE)        # move tensor

# Or create directly on GPU:
tensor = torch.zeros(1000, 3, device=DEVICE)
```

---

## 6. Troubleshooting

### `CUDA available: False` after correct install

**Check 1 — Is the installed package actually the CUDA build?**
```bash
pip show torch
```
The version should end in `+cu128`, not `+cpu`.  
If it says `+cpu`, do the force reinstall from Section 3.

**Check 2 — Python environment mismatch**  
You may have multiple Python installs. Confirm pip and python point to the same env:
```bash
python -m pip show torch   # always targets the active python's pip
```

**Check 3 — Driver too old**  
The RTX 5080 needs driver **≥ 561.x** for CUDA 12.8. Update via [NVIDIA's driver page](https://www.nvidia.com/Download/index.aspx).

**Check 4 — CUDA toolkit not needed**  
PyTorch CUDA wheels bundle their own CUDA runtime. You do **not** need to install the CUDA toolkit separately — nvidia-smi and a recent driver are enough.

---

### `torch.version.cuda` is `None` even after cu128 install

The package is CPU-only. The cu128 wheel was not picked up.  
Run:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

### RTX 5080 shows `device capability 12.0` warning

Blackwell (RTX 5080) has compute capability **sm_120**.  
PyTorch 2.6+ compiles kernels for sm_120. If you see warnings about unsupported compute capability, upgrade PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --upgrade
```

---

### Out of memory during inference

RTX 5080 has 16 GB VRAM. If you hit OOM:

```python
# Clear cache between large operations
torch.cuda.empty_cache()

# Check how much is currently used
allocated = torch.cuda.memory_allocated(0) / 1e9
reserved  = torch.cuda.memory_reserved(0)  / 1e9
print(f"Allocated: {allocated:.2f} GB  |  Reserved: {reserved:.2f} GB")
```

---

## 7. Segment Anything (SAM) on GPU

Install:
```bash
pip install segment-anything
```

Load on GPU:
```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(torch.device("cuda"))       # moves all model weights to GPU

predictor = SamPredictor(sam)
```

SAM checkpoints:
| Model   | Size   | Download |
|---------|--------|---------|
| ViT-B   | 375 MB | `sam_vit_b_01ec64.pth` |
| ViT-L   | 1.2 GB | `sam_vit_l_0b3195.pth` |
| ViT-H   | 2.5 GB | `sam_vit_h_4b8939.pth` |

ViT-B fits in VRAM easily alongside other models. ViT-H uses ~5 GB VRAM by itself.

---

## 8. Other Libraries — GPU Build Notes

### OpenCV
Standard `pip install opencv-python` uses CPU only for most ops. For GPU video decode you need OpenCV built with CUDA — not usually needed for standard projects.

### NumPy → Torch for GPU acceleration
Replace NumPy operations with Torch equivalents to keep computation on GPU:

```python
# NumPy (CPU)
import numpy as np
result = np.linalg.norm(np.diff(positions, axis=0), axis=1)

# Torch (GPU)
import torch
positions_t = torch.tensor(positions, dtype=torch.float32, device="cuda")
result = torch.norm(torch.diff(positions_t, dim=0), dim=1)
```

Avoid `.cpu().numpy()` in inner loops — each call copies data across PCIe.

---

## 9. Quick Reference

| Task | Command |
|------|---------|
| Check GPU + driver | `nvidia-smi` |
| Check PyTorch CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |
| Install PyTorch (cu128) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| Force reinstall (if cpu build stuck) | add `--force-reinstall --no-deps` |
| Check installed build | `pip show torch` — version must end in `+cu128` |
| Clear GPU cache | `torch.cuda.empty_cache()` |
| Move model to GPU | `model.to("cuda")` |
| Move tensor to GPU | `tensor.to("cuda")` or `torch.zeros(..., device="cuda")` |

---

## 10. Confirmed Working `requirements.txt` Snippet

```
# Install torch separately with the CUDA index — do NOT put torch in requirements.txt
# Run this first:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

segment-anything
opencv-python
numpy
tqdm
```

> **Why not put torch in requirements.txt?**  
> `pip install -r requirements.txt` uses the default index (PyPI), which gives the CPU-only build.  
> Always install torch manually with the `--index-url` flag, then install everything else from requirements.txt.
