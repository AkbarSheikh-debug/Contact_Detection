# Session Handoff — Impact Detection (Ubuntu Resume)

Dense handoff so Claude on a fresh Ubuntu machine can resume exactly where
this Windows session left off. Read this before touching any code.

---

## What This Project Does

Decide whether each boxing punch/kick **landed** (impact) vs missed.
ASFormer gives **when a strike is thrown** (frame windows); we judge **did it connect**.

---

## Current State (as of 2026-06-25)

### Annotation progress

| Fight | Labeled | Impact | Not-impact | impact_frame marked |
|---|---|---|---|---|
| lillyella_vs_zoe | 496/496 | — | — | partial (27 known GT frames from earlier) |
| cameron_vs_liam | 259/259 | — | — | — |
| jamie_vs_ryan | 234/234 | — | — | — |
| **1st_fight** | **243/258** | **168** | **75** | **79/168** |
| 2nd_fight | 341/446 | — | — | — |
| 3rd_fight | 0/641 | — | — | — |
| 4th_fight | 0/929 | — | — | — |

**Active fight is `1st_fight`.**
- 15 clips still unlabeled
- 89 impact clips still missing an exact `impact_frame` mark
- The `impact_frame` field = the exact global frame number where contact happens
  (used to anchor the model's training window at the right moment)

### Model trained this session

Pipeline: `extract_impact_anchored_dataset.py` → `train_keypoint_model.py` (ImpactTCN)

Results (leave-one-round-out, the honest metric):
- Round 1: AUC 0.518 (random)
- Round 2: AUC 0.466 (worse than random)
- Round 3: AUC 0.831 (18 clips only — too small to trust)
- **Average AUC 0.605 — barely above random**

Verdict: **not good yet**. Single fight, only 79/168 exact anchors, class imbalance (2.24:1).
What's needed: add 2nd_fight data + finish marking impact frames.

---

## Repository Layout (key new files from this session)

```
tools/
  annotate_clips.py               ← annotation GUI (updated this session — see below)
  extract_impact_anchored_dataset.py  ← NEW: impact-frame-anchored feature extraction
  render_anchored_predictions.py      ← NEW: renders model predictions on clip videos
  train_keypoint_model.py         ← unchanged, trains ImpactTCN / ImpactGRU
  keypoint_model.py               ← unchanged, model architectures
  extract_keypoint_dataset.py     ← original extraction (window_start anchor)
  predict_keypoint_model.py       ← predict on a new raw video

dataset/
  fights.py                       ← fight registry (ALL PATHS ARE WINDOWS — update for Ubuntu)
  prepare_fight_dataset.py        ← processes Gladius output into dataset/
  1st_fight/manifest.json         ← annotation labels + impact_frame marks (committed)
  2nd_fight/manifest.json         ← 341 labeled clips
  lillyella_vs_zoe/manifest.json  ← fully annotated
  cameron_vs_liam/manifest.json   ← fully annotated
  jamie_vs_ryan/manifest.json     ← fully annotated

outputs/keypoint_dataset/
  1st_fight_anchored.npz          ← extracted dataset (NOT in git — regenerate)

outputs/keypoint_model/
  tcn_1st_fight_R1_best.pt        ← trained checkpoints (NOT in git — *.pt gitignored)
  tcn_1st_fight_R2_best.pt
  tcn_1st_fight_R3_best.pt
```

---

## Annotation Tool Changes (this session)

`tools/annotate_clips.py` was updated with a proper transport row:
- **◀ ⏸ ▶** buttons — step frame back/forward 1 frame, pause/play toggle
- **Canvas scrub bar** — click or drag anywhere to jump to that frame
- **M key** — marks the currently displayed frame as `impact_frame`
- **B key** — go back to previous clip
- **I / N keys** — label impact / not_impact

The scrub bar is a blue canvas that fills left-to-right as the clip plays,
with a white playhead. Clicking auto-pauses.

---

## Ubuntu Setup

### 1. Clone the repo

```bash
git clone https://github.com/AkbarSheikh-debug/Contact_Detection.git
cd Contact_Detection
```

### 2. Create conda environment

```bash
conda create -n impact python=3.11 -y
conda activate impact
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy scikit-learn xgboost opencv-python pillow
# tkinter is usually bundled with python — if not:
sudo apt-get install python3-tk
```

### 3. Update path config for Ubuntu

`dataset/fights.py` has Windows absolute paths for video_folder and gladius_folder.
Update them to wherever your data lives on the Ubuntu machine. The `out_base` paths
are already repo-relative so those are fine.

Also update `dataset/fights.py` line:
```python
FFMPEG = r"C:\Users\XRIG\Downloads\ffmpeg_extracted\..."
```
to:
```python
FFMPEG = "ffmpeg"   # assuming ffmpeg is on PATH: sudo apt install ffmpeg
```

### 4. Regenerate the dataset and model (outputs/ is gitignored)

```bash
# Extract features (impact-anchored)
python tools/extract_impact_anchored_dataset.py --fights 1st_fight \
    --out outputs/keypoint_dataset/1st_fight_anchored.npz

# Train
cd tools
python train_keypoint_model.py \
    --data ../outputs/keypoint_dataset/1st_fight_anchored.npz \
    --model tcn --epochs 100 --patience 15
cd ..

# Render predictions video
python tools/render_anchored_predictions.py
```

### 5. Resume annotation

```bash
# Continue labeling remaining 15 unlabeled clips in 1st_fight
python tools/annotate_clips.py --fight 1st_fight

# Review all clips from the beginning (current mode when we stopped)
python tools/annotate_clips.py --fight 1st_fight --show-labeled

# Only show impact clips missing an exact impact_frame mark (89 remaining)
python tools/annotate_clips.py --fight 1st_fight --needs-impact-frame
```

---

## What To Do Next (priority order)

1. **Finish marking impact frames in 1st_fight** — 89 impact clips still need M pressed.
   Run `--needs-impact-frame` mode. More exact anchors = better model.

2. **Label remaining 15 unlabeled clips** in 1st_fight (default mode, no flags).

3. **Add 2nd_fight** — 341 clips already labeled, none with impact_frame yet.
   Start labeling impact frames there too, then:
   ```bash
   python tools/extract_impact_anchored_dataset.py \
       --fights 1st_fight 2nd_fight \
       --out outputs/keypoint_dataset/combined_anchored.npz
   python tools/train_keypoint_model.py \
       --data outputs/keypoint_dataset/combined_anchored.npz
   ```

4. **Cross-fight evaluation** — train on 1st+2nd, test on 3rd. This is the first
   honest test of whether the model generalizes across fights.

---

## Hard Findings (do NOT redo these experiments)

- **world_coords is BROKEN** in the SAM3D export — cross-person head-Z gap median 1m,
  p90 7.7m, max 122m. `world_coords_reliable=True` everywhere (meaningless). Skip it.
- **Audio is dead** for landed-vs-missed. Loudest sound = bell/crowd, not punches.
  All audio classifiers (loudness, AudioSet, trained classifier) AUC 0.47–0.62.
- **Every static method ≈ 47% F1 ceiling** — single monocular broadcast can't resolve
  15cm fist extension at 4m range.
- The impact-anchored keypoint model is the right direction but needs more data.
  Current single-fight AUC 0.605 is too low to be useful. Target: 3+ fights.

---

## Data Locations (Windows paths — translate to Ubuntu equivalents)

| Asset | Windows path |
|---|---|
| lillyella_vs_zoe video | `C:\Users\XRIG\Downloads\drive-download-...\(BLUE) Lillyella...` |
| cameron_vs_liam video | same drive-download folder |
| jamie_vs_ryan video | same drive-download folder |
| 1st_fight / 2nd / 3rd / 4th video | `C:\Users\XRIG\Desktop\Gladius_Data\{fight}-...\{fight}\` |
| Gladius output (SAM3D) | `C:\Users\XRIG\Desktop\Gladius_Output\{fight_name}\` |

The videos and SAM3D outputs are NOT in git. You need to copy them to the Ubuntu machine
or update `dataset/fights.py` to point at wherever they live.

---

## Python Environment Notes

- The Windows machine used `C:\Users\XRIG\anaconda3\envs\myenv` for torch/sklearn.
- System Python 3.14 has NO numpy — always activate the conda env.
- `torch` is CPU-only on this project (no GPU used). `device=cpu` everywhere.
- xgboost, sklearn, pillow, opencv-python are all needed.
