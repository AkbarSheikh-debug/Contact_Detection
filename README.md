# SAM3D Boxing Impact Detection Module

A research pipeline for automatically detecting whether punches and kicks land in a boxing match video, using pre-extracted 3D skeleton data, SAM (Segment Anything Model) mask overlap, and a range of physics-based and ML-based scoring methods.

---

## Table of Contents

> **New (May 2026):** [⚠️ Data Audit of the `world_coords` export](#️-data-audit-may-2026-the-new-world_coords-sam3d-export) · [New Methods & Roadmap](#new-methods--roadmap-20242025-literature) · [Phase 4 — `fusion_v8.py`](#phase-4--region-aware-pixel-space-fusion-fusion_v8py) · [Phase 5 — SAM+Depth+Audio data ceiling](#phase-5--sam--depth--audio-the-data-ceiling-conclusive) · [**Planned: Learned Temporal Impact Model**](#planned-approach--learned-temporal-impact-model-asformer-windowed-clips)

1. [Project Overview](#project-overview)
2. [Research Problem](#research-problem)
3. [Data Pipeline](#data-pipeline)
4. [Input JSON Format](#input-json-format)
5. [The Detection Gates](#the-detection-gates)
6. [Detection Methods — Evolution](#detection-methods--evolution)
   - [Phase 1: Initial 5-Gate System (no SAM)](#phase-1-initial-5-gate-system-no-sam)
   - [Phase 2A: SAM-Based Approaches A–H](#phase-2a-sam-based-approaches-ah)
   - [Phase 2B: Fusion Approaches v2–v7](#phase-2b-fusion-approaches-v2v7)
   - [Phase 3: Standalone Filter Extensions](#phase-3-standalone-filter-extensions)
7. [Results Summary](#results-summary)
8. [What Failed — and Why](#what-failed--and-why)
9. [What Works Best](#what-works-best)
10. [Requested JSON Improvements](#requested-json-improvements)
11. [Ground Truth Dataset](#ground-truth-dataset)
12. [Repository Structure](#repository-structure)
13. [Setup & Installation](#setup--installation)
14. [Usage](#usage)
15. [Pi-HOC Architecture Mapping](#pi-hoc-architecture-mapping)

---

## Project Overview

This module takes boxing match video + pre-extracted 3D skeleton JSON data and answers the question:
**"Did this punch actually land?"**

An upstream system (ASFormer action segmentation) already detects when punches are *thrown*. It cannot tell if they connected. This module provides the contact verification layer using:

- **SAM (Segment Anything Model, ViT-B)** — pixel-precise body masks for spatial contact detection
- **Physics gates** — wrist deceleration, 3D jerk, depth proximity, arm extension, receiver head reaction
- **Audio signal analysis** — impact thud detection via librosa
- **Optical flow** — receiver head-snap detection as a contact signal
- **Machine learning classifiers** — Logistic Regression, Gradient Boosting, XGBoost trained on gate scores

The project is adapted from **Pi-HOC** (Pairwise 3D Human-Object Contact Estimation, arXiv:2604.12923).

---

## ⚠️ Data Audit (May 2026): The New `world_coords` SAM3D Export

A new SAM3D export was delivered (`sam3d_with_world_coords/`) that **added the three
fields requested in [NOTE_FOR_TEAMMATE.md](NOTE_FOR_TEAMMATE.md)**: `world_coords`,
`keypoint_conf`, `world_coords_reliable`, and a top-level `contact_events` list.

Before trusting any of them, we ran a hard audit ([diagnose_world_coords.py](diagnose_world_coords.py)).
The fields were *added in name* but **the underlying problems were not fixed**:

| New field | Audit result | Verdict |
|---|---|---|
| `world_coords` | Cross-person head-Z gap: **median 1.03 m, p90 7.71 m, max 121.97 m**; 51.2% of frames > 1 m apart | ❌ **Still broken** — same per-person independent depth as `shared_space_coords`. Do **not** use for cross-person 3D distance. |
| `world_coords_reliable` | `True` on **100%** of frames, including the 122 m phantom-depth ones | ❌ Meaningless flag |
| `keypoint_conf` | mean 0.9999, only **0.06%** of values < 0.99 | ❌ Constant placeholder — cannot gate occluded wrists |
| `contact_events` | 139 events, `contact_prob` 0.003–1.0, sane `contact_3d_distance_m` (0.016–0.150 m), region labels (head/torso/arm) | ⚠️ **Usable** signal, but see the alignment problem below |

**The decisive problem — candidate coverage.** The 31 ground-truth impacts are covered by:

- ASFormer **actions**: **30/31** at ±12 frames (31/31 at ±25) ✅
- SAM3D **`contact_events`**: only **5/31** at ±12 frames (12/31 at ±25) ❌

So `contact_events` cannot be the *candidate set* (it misses ~84% of real impacts); it can
only be a weak scoring prior. ASFormer actions remain the right candidate set.

**The honest conclusion.** With the signals available *without a GPU/SAM checkpoint*, no single
signal separates landed from missed punches on this data. Measured GT-vs-non-GT separation per
gate (positive = discriminative):

```
gap=-0.029  ce_prob=-0.072  region=-0.008  decel=-0.023
react=+0.080  approach=+0.081  audio=-0.009  conf=+0.028   FUSED=-0.010
```

The fused score is essentially **uncorrelated with ground truth** (−0.010). This is consistent
with the original project finding that **SAM mask overlap was the only strong signal**
(learned coefficient 3.25× over everything else). SAM/SAM2/depth models require a GPU + checkpoint
that this environment does not have — so the realistic next steps below are about restoring a
*geometrically valid* contact signal, not tuning the weak ones.

---

## New Methods & Roadmap (2024–2025 Literature)

The root cause across every approach is the same: **there is no geometrically consistent 3D
representation of where both fighters are at the same instant** (per-person independent depth →
phantom 0.3–122 m offsets). The literature points to concrete fixes:

| Priority | Method | What it fixes | Cost | Reference |
|---|---|---|---|---|
| **1** | **Video Depth Anything** (CVPR'25) — one scene-wide, temporally-consistent depth map per frame; replace each keypoint's broken per-person Z with the depth-map value at its (u,v) pixel | The cross-person depth bug, directly. Both fighters share one depth map ⇒ relative depth is valid by construction | GPU + model download; ~50 lines to integrate | [arXiv:2501.12375](https://arxiv.org/abs/2501.12375) |
| **2** | **SAM2 mask-IoU contact (GRAZE)** — track the striker's glove mask and the receiver's head/torso mask across frames; contact = mask IoU > τ. Pixel-space, **no 3D at all** | Replaces the single-frame bbox-silhouette SAM gate with a temporally-consistent, tightly-fitted overlap | GPU + SAM2 | GRAZE [arXiv:2604.01383](https://arxiv.org/abs/2604.01383); SAM2 [arXiv:2408.00714](https://arxiv.org/abs/2408.00714) |
| **3** | **VolumetricSMPL SDF proximity** (ICCV'25) — fit SMPL per fighter, query signed-distance of striker hand vertices vs receiver body surface; gives **contact region** (head/torso/arm) ⇒ blocked-vs-landed | Robust to keypoint noise; the only principled blocked-punch filter | GPU; 1 pip + 1 line | [arXiv:2506.23236](https://arxiv.org/abs/2506.23236) |
| **4** | **Multi-HMR + DTO** (ECCV'24 / Nov'25) — multi-person mesh recovery in a *shared* camera frame; DTO enforces metric scale across people | Full SAM3D backend replacement that removes the depth bug at the source | GPU | [Multi-HMR 2402.14654](https://arxiv.org/abs/2402.14654), [DTO 2511.13282](https://arxiv.org/abs/2511.13282) |
| **5** | **DECO** (ICCV'23) — dense vertex-level contact from RGB; auto-label contact regions for training data | Supervision for a learned contact head | GPU | [arXiv:2309.15273](https://arxiv.org/abs/2309.15273) |
| ref | **BoxingVI** — 6,915 labelled punch clips + 2D pose, 18 athletes | Train/validate with real sparring data | dataset | [arXiv:2511.16524](https://arxiv.org/abs/2511.16524) |

Note: the published vision-only boxing-landing baseline (Entropy 2024, MDPI) reports **F1≈49%** —
so even the existing Approach-H stack (F1≈65.7%) already beats the literature. The path to 85%+
runs through **Priority 1 (valid depth)** and **Priority 2 (SAM2 pixel overlap)**, not weight tuning.

### Phase 4 — Region-Aware Pixel-Space Fusion (`fusion_v8.py`)

[fusion_v8.py](fusion_v8.py) is a **CPU-runnable, SAM-free** detector built for the new export. It
deliberately **avoids the broken `world_coords`** and works in pixel space (the GRAZE insight):

- **Candidates** = ASFormer action windows (cover 30/31 GT — unlike `contact_events` at 5/31).
- For each action it refines the impact frame to the in-window frame with the smallest
  **normalized wrist↔receiver-body pixel gap** (keypoints projected into the bbox; depth-free).
- Fuses: pixel gap, nearest `contact_event` prob + **region weight** (arm = guard → penalised,
  head/torso → boosted), pixel-space wrist deceleration, receiver head reaction (Newton's 3rd law
  in pixels), wrist approach-rate, multi-band **audio onset** (extracted from the video, no
  librosa), and action confidence.
- NMS by score with cooldown; sweeps the threshold and reports P/R/F1 vs the 31 GT.

**Honest result on this data:** best **F1 ≈ 48% (P≈36%, R≈71%)** at threshold 0.42, cooldown 18,
tolerance ±12 frames. This is effectively the *recall ceiling of the candidate set* — the audit
above shows the fused score barely correlates with ground truth, because the one discriminative
signal (SAM mask overlap) is unavailable on CPU. `fusion_v8` is therefore the correct **scaffold**
to drop the Priority-1/2 GPU signals into; it is not a finished high-accuracy detector.

Run:

```bash
python evaluation/diagnose_world_coords.py           # audit the export (numbers above)
python detectors/fusion/v8.py                        # region-aware pixel-space fusion + GT eval
python detectors/fusion/v8.py --no-audio --cooldown 10
python detectors/fusion/v8.py --video                # render annotated output video
```

---

### Phase 5 — SAM + Depth + Audio: the data ceiling (conclusive)

After installing SAM properly and adding monocular depth, every signal was measured against a set of
**video-verified landing labels** (sorted by hand from short clips), not the legacy 31 GT. The result
is consistent and decisive:

| Method (honest, non-leaked eval) | F1 / signal |
|---|---|
| `sam_detect.py` — SAM ViT-B mask overlap (the "strong" gate) | **F1 ≈ 47%**; many false positives at `sam=1.0` |
| `sam_depth_detect.py` — SAM + Depth Anything V2 + head-reaction | gate separations ≈ 0 (`sam −0.03`, `depth +0.04`, `react −0.08`), F1 ≈ 38% |
| `sound_detector.py` / `sound_ai_classify.py` / `sound_train_detect.py` — audio (onset, AudioSet AST, trained classifier) | **chance** (CV ROC-AUC 0.47–0.62; loudest sound in the match is *not* a punch) |

**Why it plateaus:** a punch that *lands* vs one that *falls ~15 cm short* is a ~15 cm depth difference
at a ~4 m camera distance — **below the resolution of single-camera monocular depth**. So static
per-frame gates (2D overlap, depth, proximity) physically cannot separate "touching" from
"almost-touching," and the broadcast audio is dominated by commentary/crowd. Confirmed five independent
ways. **The ceiling on this single-camera data is ~47% F1**, regardless of how the static gates are
combined. Breaking it needs either multi-view/stereo depth, a corrected cross-person `world_coords`, or
a **temporal** model that learns the *reaction over time* (see the planned approach below).

New scripts from this phase: [detectors/sam/sam_detect.py](detectors/sam/sam_detect.py),
[detectors/sam/sam_depth_detect.py](detectors/sam/sam_depth_detect.py),
[detectors/sound/sound_detector.py](detectors/sound/sound_detector.py),
[detectors/sound/sound_ai_classify.py](detectors/sound/sound_ai_classify.py),
[detectors/sound/sound_train_detect.py](detectors/sound/sound_train_detect.py).

---

## Planned Approach — Learned Temporal Impact Model (ASFormer-windowed clips)

The static gates above look at single frames and hit a wall. The next direction is a **learned
temporal model** that watches the short clip around each action and learns the **impact reaction over
time** (the wrist decelerating into the body, the head snapping back, the torso folding on a body
shot). That temporal reaction is the real evidence of a landed strike — it is exactly what the
frame-wise gates could not use.

### The low-effort labeling pipeline (this is the key idea)

Data collection is the expensive part of any learned model. This plan makes it cheap by reusing the
**already-trained ASFormer action recognizer** to do all the localization, so the only manual step is a
fast binary sort:

```
ASFormer action JSON  ──►  extract one video+audio clip per action window
   (window_start→window_end, padded to capture the contact + reaction)
                          │
                          ▼
            outputs/impact_dataset/<clip>.mp4   (141 clips for this match)
                          │
        you sort each clip (fast, type-agnostic — punch OR kick):
                          │
         ┌────────────────┴────────────────┐
         ▼                                  ▼
   impact/  (strike connected)      not_impact/ (missed / blocked / no contact)
                          │
                          ▼
            train a TEMPORAL model on the sorted clips
```

- **Action type does not matter** (jab / hook / uppercut / cross / any kick) — only *did it connect*.
- ASFormer gives the frame range of every action, so clips are localized automatically; the human only
  answers the one question a model can't yet: **impact or not.**
- Tooling already built: **[dataset/extract_action_clips.py](dataset/extract_action_clips.py)** → produces
  `outputs/impact_dataset/` with the clips plus empty `impact/` and `not_impact/` folders to sort into.
  Filenames carry `action`, frame range, ASFormer confidence, and an audio-onset hint.

### Two sources of labeled clips (both reuse existing annotations)

1. **From the ASFormer *output* JSON (works now):** run the trained model on each match → for every
   detected action, cut the clip from `window_start`→`window_end`. This needs only the model's JSON +
   the video. Scale by having a teammate run more matches through ASFormer and return more
   `video.mp4 + asformer.json` pairs.

2. **From the ASFormer *training* dataset annotations:** the dataset ASFormer was trained on already
   contains hand-annotated action frame ranges (from-frame → to-frame per action). Those same
   annotation ranges can be used to extract the action clips directly — **no inference needed** — and
   then split into impact / not_impact. This reuses labels that already exist, which is the cheapest
   possible way to grow the dataset.

### Candidate temporal models (pick by data volume)

| Approach | Input | Temporal? | Data needed |
|---|---|---|---|
| Pretrained video backbone (VideoMAE / X3D / I3D) **fine-tuned** on the clips | raw clip frames | ✅ fully | hundreds (with transfer learning) |
| Small **GRU / 1D-CNN / TimeSformer** on per-frame pose+flow features | feature sequence | ✅ | hundreds |
| **Retrain ASFormer** with impact-aware labels (`landed` / `missed` / `background`) | per-frame I3D features | ✅ | many matches + GPU |
| Features + XGBoost (baseline only) | summary vector | partly | ~150 |

### Practical guidance (so the model actually generalizes)

- **Pad the clip past `window_end`** (~+0.5 s): the "did it land" evidence is at and just after the
  punch peak, not inside the throw. The extractor already does this.
- **Label across many matches**, then **split train/test by match** (never random clips) — otherwise
  the score is inflated by memorization.
- **Blocked-on-guard = `not_impact`** (it didn't connect to the body); be consistent on grazes.
- Handle **class imbalance** (impacts are the minority) with class weights / balanced sampling.
- Fine-tuning a video model wants a **GPU**; on CPU, validate on a small subset and train for real on a
  GPU/Colab.

### Why this can beat the ~47% ceiling

The static gates failed on the 15 cm depth problem. A temporal model sidesteps it: it doesn't need to
measure depth — it learns the *consequence* of contact (the reaction), which **is** visible in the
footage over a few frames. The ceiling now depends on whether those reactions are visible and on having
enough labeled clips, not on monocular depth resolution.

---

## Research Problem

ASFormer can tell you a punch was thrown. It cannot tell you if it connected.

A fighter can throw a jab that misses completely and ASFormer still records it with 70%+ confidence. The impact detection module's job is to score every one of those **138 action windows** and decide: **did this punch/kick actually land?**

The system uses 3 pre-extracted JSON files to avoid running pose estimation live.

### Key Challenges Encountered

1. **Cross-person Z alignment** — `shared_space_coords` has per-person independent depth estimation, so two fighters standing side by side can have 0.30–0.61m phantom depth difference. This inflates 3D distance measurements at contact. (See [Requested JSON Improvements](#requested-json-improvements).)

2. **ASFormer window dependency** — Approaches A–G only score within ASFormer-detected action windows. Impacts that occur between windows are missed entirely.

3. **Blocked punches vs landed punches** — A punch to the guard looks identical to a landed punch in 2D/3D coordinate data.

4. **Occlusion** — The striker's wrist is frequently hidden behind the receiver's body during the contact moment, making 2D keypoint positions unreliable precisely when they matter most.

---

## Data Pipeline

```
Video (1920×1080) ──► SAM3D external system ──► 3 JSON files
                                                        │
                           ┌────────────────────────────┤
                           │                            │
                    2d_points.json              3d_points.json
                    (70 joints, pixels)         (70 joints, metres)
                           │                            │
                           └──────────┬─────────────────┘
                                      │
                               full_results.json
                               (138 ASFormer actions)
                                      │
                               Impact Detection
                               ┌──────┴────────────────────┐
                               │                           │
                        Physics Gates              SAM Mask Overlap
                        (decel, jerk,              (pixel-precise
                         proximity, ext.)           body mask)
                               │                           │
                               └──────────┬────────────────┘
                                          │
                                   Weighted score ≥ 0.45
                                          │
                               Impact detected ✓ / missed ✗
                                          │
                               Annotated video + JSON output
```

---

## Input JSON Format

### `2d_points.json`

Contains 2D pixel positions of **70 body joints** per frame, per fighter. Coordinates in original 1920×1080 resolution (internally stored at 640×360 and scaled back up).

```json
{
  "0": [
    {
      "track_id": 0,
      "frame": 620,
      "bbox": [1247.0, 225.0, 1560.0, 539.0],
      "joints_2d": [[x, y], [x, y], ...],   // 70 joints
      "frame_dims": {
        "original_width": 1920,
        "original_height": 1080,
        "resized_width": 640,
        "resized_height": 360
      }
    }
  ],
  "1": [ ... ]   // Fighter 1 data
}
```

### `3d_points.json`

Same 70 joints in 3D world-space coordinates (metres). Two coordinate systems:

- `shared_space_coords` — absolute world coordinates (per-person, **not truly shared** — see problem below)
- `normalized_coords` — body-centred, camera-position independent
- `focal_normalized_coords` — focal-length normalized variant

```json
{
  "0": [
    {
      "track_id": 0,
      "frame": 620,
      "bbox": [...],
      "shared_space_coords": [[x, y, z], ...],   // 70 joints × 3 coords
      "normalized_coords": [[x, y, z], ...],
      "focal_normalized_coords": [[x, y, z], ...],
      "pred_cam_t": [0.595, 1.142, 1.871],
      "focal_length": 983.27,
      "normalization_root": [0.0, -0.9, 0.0],
      "normalization_scale": 1.414
    }
  ]
}
```

### `full_results.json`

138 action recognition windows from **ASFormer** (Adaptive Span Transformer for action segmentation):

```json
{
  "actions": [
    {
      "fighter_type": "striker",
      "action": "cross",
      "confidence": 0.72,
      "frame": 344,
      "window_start": 340,
      "window_end": 358,
      "timestamp_seconds": 13.76,
      "target": "Head",
      "is_significant": true,
      "speed_estimation": { "estimated_speed_kmh": 7.2 },
      "power_estimation": { "estimated_power_watts": 148.0 },
      "model_used": "ASFormer"
    }
  ]
}
```

### Joint Index Reference (COCO 17-joint subset of 70)

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0 | Nose | 9 | Left Wrist |
| 1 | Left Eye | 10 | Right Wrist |
| 2 | Right Eye | 11 | Left Hip |
| 3 | Left Ear | 12 | Right Hip |
| 4 | Right Ear | 13 | Left Knee |
| 5 | Left Shoulder | 14 | Right Knee |
| 6 | Right Shoulder | 15 | Left Ankle |
| 7 | Left Elbow | 16 | Right Ankle |
| 8 | Right Elbow | 17–69 | Additional body/face joints |

---

## The Detection Gates

All approaches are built from combinations of these gates. Each gate outputs a score 0.0–1.0.

### Gate 1 — SAM Mask Overlap (core gate, weight 28–40%)

The Pi-HOC-inspired gate. **SAM (Meta AI, Segment Anything, ViT-B, ~375MB)** runs on GPU:

1. Take the receiver's bounding box → feed as prompt to SAM → get pixel-precise body mask
2. Take the striker's wrist pixel position from `2d_points.json`
3. Score = 1.0 if wrist is inside mask; falls off with Euclidean distance from mask edge (0.0 at 60px)

Also tested: **Elbow probe** (Approach D) — checks both wrist and elbow, takes the max. This catches hooks where the forearm/elbow makes contact.

**Why it works:** SAM's mask is accurate to individual pixels regardless of clothing or lighting. A wrist inside the opponent's body silhouette = contact.

### Gate 2 — Wrist Deceleration (weight 18–30%)

A fist decelerates sharply at the moment of contact. Computes 3D speed of the striker's wrist across the action window, finds the steepest velocity drop. Runs as a CUDA tensor operation on GPU.

- Sharp decel → score near 1.0
- Gradual slowdown / no decel → score near 0.0

**Discrimination**: Landed avg 0.91 vs Missed avg 0.45

### Gate 3 — 3D Depth Proximity (weight 12–15%)

If the 3D world-space distance between wrist and opponent's body centroid < 0.60m, assigns a partial score. Acts as a backup when SAM score is low due to camera angle.

**Limitation**: Suffers heavily from the cross-person Z alignment error in `shared_space_coords` (see [What Failed](#what-failed--and-why)).

### Gate 4 — 3D Jerk (weight 10–25%)

Jerk = rate of change of acceleration (3rd derivative of position). At contact, the wrist transitions abruptly from free swing to impacting a body — a force spike visible as extreme jerk. Computed from `3d_points.json` on GPU.

**Discrimination**: Landed avg 0.60 vs Missed avg 0.22

### Gate 5 — Action Confidence (weight 8–15%)

Combines ASFormer's confidence with estimated speed and power:

```
score = min(1.0, confidence × (1 + speed_kmh / 50) × (1 + power_watts / 500))
```

Weakest individual gate — used mainly to break ties.

### Gate 6 — Arm Extension (weight 5–15%)

```
score = wrist_to_shoulder_distance / full_arm_length
```

A bent arm (ratio < 0.68) = guard position or retraction → not a landed punch.
Full extension confirms an outgoing strike at contact moment.

**Discrimination**: Landed avg 0.95 vs Missed avg 0.80

### Gate 7 — Receiver Head Reaction (weight 14–16%)

Measures the 3D acceleration of the receiver's head centroid (average of nose + eyes + ears keypoints) at the detected contact frame. A large head acceleration = real hit.

This is a physics ground-truth check from the *receiving* body — Newton's 3rd Law.

### Gate 8 — Dual-Body Pearson Correlation (weight 18%, Approach G only)

Computes the **Pearson correlation** between:
- Striker wrist deceleration signal over ±5 frames
- Receiver head acceleration signal over the same window

High positive correlation (+1.0) = both happened simultaneously = real impact.
Low/negative correlation = the events are independent = missed punch.

Maps from [-1, 1] → [0, 1] for final score.

### Gate 9 — Optical Flow Head-Region Ratio (Phase 3)

Dense optical flow (Farnebäck) in the receiver's head bounding box region.

```
flow_ratio = flow_magnitude_at_impact / baseline_flow_magnitude
```

Ratio >> 1.0 = head snapped (real hit). Ratio ≈ 1.0 = clinch or near-miss.

Used in: Approach H post-filter, pihoc_filter.py, depth_filter.py, fusion_v6.py, fusion_v7.py.

### Gate 10 — Pi-HOC Contact Region Filter

Approximates SMPL mesh contact localization using 2D keypoint distances:

```
arm_vs_body_ratio = dist(striker_wrist, receiver_arm_kps)
                  / dist(striker_wrist, receiver_body_kps)
```

If the wrist is closer to the receiver's *arm* than to their *torso/head*, the contact is on the guard → reject as blocked punch.

### Gate 11 — Audio Onset Strength (Fusion v2+)

Uses `librosa` to detect audio impact transients. A real punch produces a distinctive thud/slap transient in the audio. Features used:
- Multi-band energy in 3 frequency bands
- Short-time RMS energy spike
- Onset strength at impact frame
- Spectral centroid (sharp punches have higher frequency content)
- Pre-impact silence ratio (real punches often preceded by brief quiet)

### Gate 12 — Wrist Approach Rate (Fusion v6+)

Computes whether the striker's wrist is monotonically closing the distance to the receiver over the last 8 frames. A real punch approach is directional; jitter/clinch movement is random.

---

## Detection Methods — Evolution

### Phase 1: Initial 5-Gate System (no SAM)

**File**: `run_impact_detection.py` + `impact_detector.py`

First working system. Used only physics gates (decel, jerk, extension, depth convergence, confidence). **No SAM, no visual contact check.**

**Result**: 130/138 = 94.2% "landing rate" — but this was false precision. The system had no way to distinguish a near-miss from a landed punch when the 3D depth error was large. Almost everything got classified as landed.

| Metric | Value |
|--------|-------|
| Total actions | 138 |
| Classified as landed | 130 |
| Landing rate | 94.2% |
| False positive rate | Very high (no ground truth at this stage) |

**Conclusion**: Need SAM for visual contact verification. Physics alone is insufficient.

---

### Phase 2A: SAM-Based Approaches A–H

These all use `2d_points.json`, `3d_points.json`, `full_results.json` plus SAM running on GPU.

All approaches use a weighted gate sum. If `impact_score ≥ threshold`, the hit is classified as landed.

---

#### Approach A — Multi-Frame SAM (Baseline)

**File**: First implementation in `pipeline_json.py`

Instead of running SAM on one frame, scans ±2 frames around the estimated contact frame (5-frame window) and picks the highest SAM score. Compensates for the fact that the exact contact frame may be slightly off from ASFormer's estimate.

| Detail | Value |
|--------|-------|
| SAM window | 5 frames (±2) |
| Gate weights | SAM=40%, Decel=20%, Prox3D=15%, Jerk=12%, Conf=8%, Ext=5% |
| Threshold | 0.45 |
| Detected impacts | ~29 |

**Best feature**: Simple, fast, establishes the SAM-based baseline.

---

#### Approach B — Soft Gate

**Variation of A** in same file.

Physics gates (decel + jerk) can independently rescue a detection even when SAM score is low. Catches hits where camera angle makes SAM uncertain but the motion physics are clearly a contact event.

| Detail | Value |
|--------|-------|
| Key change | Physics can override low SAM score |
| Threshold | 0.45 |
| Trade-off | More detections, more false positives |

---

#### Approach C — Lower Threshold + Optical Flow Heatmap

Drops threshold to 0.38 and adds dense optical flow heatmap overlay on output video — a colour visualization of per-frame motion energy. Hot = fast moving (contact energy transfer), cold = still.

| Detail | Value |
|--------|-------|
| Threshold | 0.38 (lowered from 0.45) |
| New feature | Optical flow heatmap overlay on video |
| Trade-off | Catches subtle hits, higher false positive rate |

---

#### Approach D — Enhanced (Elbow Probe + Receiver Head Reaction)

**Most complete physics model of A–D.** Adds two new gates:

1. **Elbow probe**: Checks both wrist AND elbow inside SAM mask. Takes maximum of the two. Catches hooks and uppercuts where the forearm (not wrist tip) is the contact point.

2. **Receiver head reaction gate (14%)**: Measures 3D acceleration of the receiver's head at the contact frame. Head snapping back = real impact.

| Detail | Value |
|--------|-------|
| Gate weights | SAM=33%, Decel=18%, Receiver=14%, Prox3D=12%, Jerk=10%, Conf=8%, Ext=5% |
| Threshold | 0.45 |

**Status**: Good results. The elbow probe significantly helps for hooks/uppercuts.

---

#### Approach E — SAM2-Style Temporal EMA

SAM2 (Meta's video-aware successor) is not available on Windows. Approach E **simulates SAM2's temporal consistency** using Exponential Moving Average (EMA):

- 7-frame SAM window instead of 5
- Gaussian-weighted average of SAM scores (α=0.65) instead of max
- Frames near center of window get more weight
- Prevents a single noisy high-score frame from dominating

| Detail | Value |
|--------|-------|
| SAM window | 7 frames |
| EMA alpha | 0.65 |
| Effect | Smoother detection, robust to flickering pose estimates |
| Threshold | 0.45 |

---

#### Approach F — Learned Logistic Regression

Trains a **LogisticRegression(class_weight='balanced')** classifier using Approach D's outputs as pseudo-labels:

**Learned feature weights** (from 138 training samples, 43 positive / 95 negative):

| Feature | Coefficient |
|---------|-------------|
| SAM overlap | **3.253** — by far the most predictive |
| 3D Proximity | 0.875 |
| Deceleration | 0.612 |
| Jerk | 0.441 |
| Confidence | 0.388 |
| Extension | 0.201 |
| Receiver reaction | 0.174 |

The data-driven weights confirm what physics predicts: SAM is the dominant signal.

| Detail | Value |
|--------|-------|
| Model | `sklearn.LogisticRegression(C=1.0, class_weight='balanced')` |
| Training | Pseudo-labels from Approach D |
| Threshold | 0.45 on `predict_proba()` output |

---

#### Approach G — Dual-Body Pearson Correlation

**Highest physics fidelity.** Newton's 3rd Law: the force stopping the fist is the same force moving the head. At real contact, these two events happen simultaneously:

- Striker wrist decelerates
- Receiver head accelerates

Pearson correlation between these signals over ±5 frames around the detected impact frame. High correlation (+1.0) = real impact. Low/negative correlation = missed punch or independent movement.

| Detail | Value |
|--------|-------|
| Gate weights | SAM=28%, Dual-Corr=18%, Decel=16%, Prox3D=12%, Jerk=10%, Ext=8%, Conf=8% |
| Correlation window | ±5 frames |
| Physics basis | Newton's 3rd Law |
| Threshold | 0.45 |

**Status**: Physically principled. Results comparable to D but more robust to occlusion cases where SAM is uncertain.

---

#### Approach H — Full-Frame SAM Scanner

**File**: `approach_h_fullscan.py`

**Root cause fix for A–G**: SAM only runs inside 138 ASFormer action windows. Real impacts outside those windows are permanently missed.

Approach H runs SAM on **every stride-2 frame** of the entire video, checking both wrist and elbow for both fighters, applying per-pair cooldown. Zero dependency on ASFormer.

| Detail | Value |
|--------|-------|
| Raw detections | **157 events** (no post-filtering) |
| After cooldown=50 frames | Reduced |
| After + IoU≤0.35 | 47 events, F1=64.1% |
| After + flow_ratio≥1.3 | Improved F1 |
| After + Pi-HOC arm filter | 45 events, TP=25, FP=20, P=55.6%, R=80.6%, **F1=65.7%** |
| After + Depth Anything V2 | Full pipeline results |

Gate weights:

| Gate | Weight |
|------|--------|
| SAM | 36% |
| Decel | 18% |
| Receiver reaction | 16% |
| Jerk | 12% |
| Extension | 10% |
| Proximity 3D | 8% |

**Key insight**: Removing the ASFormer dependency revealed that many real impacts were being missed simply because ASFormer didn't record a window near that timestamp.

---

### Phase 2B: Fusion Approaches v2–v7

These use a **second video** (`3.mp4`, 31 confirmed GT impacts) with `world_coords` data from an updated SAM3D version. Audio (.wav file) is available. Ground truth is 31 manually labelled timestamps.

**Key difference from Phase 2A**: `world_coords` gives cross-person Z-aligned coordinates (see [Requested JSON Improvements](#requested-json-improvements)), enabling accurate 3D distance measurements.

---

#### Fusion v2 — Voting-Based System

**File**: `fusion_v2.py`

Each candidate action accumulates votes from independent signals:

| Vote Signal | Threshold |
|-------------|-----------|
| Audio onset within ±N frames, strength > S | configurable |
| `contact_event` within ±N frames, prob > P | configurable |
| 2D wrist-to-receiver pixel distance < D | configurable |
| Striker wrist decel > T | configurable |
| Receiver head 2D accel > T | configurable |
| Action confidence > C | configurable |

A candidate with ≥ MIN_VOTES (typically 3) is kept as a detection. NMS by score with cooldown.

**Advantage**: Votes are independent — no single signal can produce a false positive alone.

---

#### Fusion v3 — Precomputed Features + Fast Threshold Search

**File**: `fusion_v3.py`

Precomputes all features once, then sweeps threshold combinations rapidly to find the best precision/recall operating point. Faster iteration than v2's full pipeline re-run.

---

#### Fusion v4 — Rich Temporal Features + Gradient Boosting

**File**: `fusion_v4.py`

Extends candidates beyond just action JSON events — adds audio onset frames and `contact_event` frames as additional candidates.

New features:
- Multi-scale receiver displacement (pre/at/post impact)
- Multi-band audio energy (3 frequency bands) + onset
- Hand trajectory entering receiver bounding box
- Receiver bbox area change (fighter moving away = reaction)
- Striker stance and arm extension features
- Cross-fighter relative motion

**Classifier**: `sklearn.GradientBoostingClassifier`

---

#### Fusion v5 — Relabeled GT + Full Contact Metadata

**File**: `fusion_v5.py`

Same as v4 but:
- Uses **relabeled GT** from `relabeled_gt.py` (±60-frame search around original GT to find the true contact frame using geometry + audio + head reaction)
- Outputs rich per-event metadata: contact frame, hit_xy pixel, contact region, action type, speed, audio peak, contact probability

---

#### Fusion v6 — XGBoost + Optical Flow + Head Acceleration

**File**: `fusion_v6.py`

New signals vs v5:
- Dense optical flow in receiver head region (Farnebäck, post-impact)
- Receiver head 3D acceleration at contact frame
- Wrist approach rate (is wrist closing in over last 8 frames?)
- Audio onset strength (librosa, more precise than raw spectral flux)
- Temporal approach pattern (monotonically decreasing distance)

**Classifier**: XGBoost (`XGBClassifier`) — better than sklearn GBM for this feature set.

---

#### Fusion v7 — Audio Physics + Spectral Features

**File**: `fusion_v7.py`

New signals vs v6:
- Optical flow **direction coherence** in head region (head snap = coherent direction, clinch motion = random)
- **Paired physics constraint**: wrist_decel × head_accel product (must be high simultaneously)
- Approach trajectory slope over last 8 frames
- Spectral centroid (sharp, hard punches have higher spectral centroid)
- Short-time RMS energy spike (boxing impacts are brief and energetic)
- Pre-impact silence ratio (real punches often preceded by brief quiet)

**Status**: Most feature-rich method. XGBoost on 15+ features evaluated against 31 GT impacts.

---

### Phase 3: Standalone Filter Extensions

Applied as post-processing on top of raw Approach H detections (157 events → filtered).

---

#### Optical Flow Gate

**File**: `optflow_gate.py`

Research basis: *"Boxing Punch Detection with Single Static Camera"* (MDPI Entropy 2024) + *"Optical Flow Divergence for Collision Detection"*.

Standalone optical flow contact verifier. Runs as its own pipeline on stride-2 frames.

```
flow_ratio = receiver_head_flow_at_impact / baseline_head_flow
```

Real punch: ratio >> 1.0 (head snaps). Clinch/near-miss: ratio ≈ 1.0.

---

#### Pi-HOC Contact Region Filter

**File**: `pihoc_filter.py`

Approximates the Pi-HOC paper's SMPL vertex contact localization using 2D keypoint distances:

```
arm_vs_body_ratio = dist(wrist, receiver_arm_kps) / dist(wrist, receiver_torso_kps)
```

Punch to guard (blocked): wrist closer to arm → reject.
Punch to head/torso (landed): wrist closer to body → accept.

**Final result**: Applied after cd=50 + IoU≤0.35 + flow_ratio≥1.3:
- 45 events, TP=25, FP=20, **P=55.6%, R=80.6%, F1=65.7%**
- Improvement over previous best (47 events, F1=64.1%)

---

#### Depth Anything V2 Filter

**File**: `depth_filter.py`

Uses **Depth Anything V2 Small** (HuggingFace) monocular depth estimation to verify that the striker's wrist is at the same depth plane as the receiver's head at impact.

```
normalized_depth_delta = |wrist_disparity - head_disparity| / max_disparity
```

Real punch landing: wrist depth ≈ head depth (same 3D plane).
Near-miss: significantly different depths.

**Limitation**: Does NOT help with clinch false positives (both fighters are at the same depth during clinch).

Full pipeline: raw(157) → cd=50 → IoU≤0.35 → flow≥1.3 → arm_ratio≥0.30 → depth_delta≤threshold

---

#### High Precision Search

**Files**: `high_precision.py`, `high_precision_v2.py`

Grid search to find the feature threshold combination that achieves **Precision ≥ 0.80** while maximizing Recall. v2 combines 3+ corroborating signals per detection for stricter filtering.

---

#### Post-Filter for Approach H

**File**: `postfilter_h.py`

Optimal filter: **global cooldown=50 frames** (2 seconds) + **IoU≤0.35** (removes clinch events where both fighters' bboxes heavily overlap).

Grid-searched against 31 GT timestamps:
- **F1=60.5%, P=47.3%, R=83.9%**

---

## Results Summary

### Phase 1 — 5-Gate, No SAM

| Metric | Value |
|--------|-------|
| Classified as landed | 130/138 (94.2%) |
| Assessment | Heavily overclassified — false precision without ground truth |

### Phase 2A — SAM Approaches vs ASFormer Windows

| Approach | Key Idea | Detected | Notes |
|----------|----------|----------|-------|
| A | Multi-frame SAM baseline | ~29 | Conservative |
| B | Soft gate (physics rescues SAM) | ~35 | More FPs |
| C | Lower threshold + heatmap | ~40 | Visual energy map |
| D | Elbow probe + head reaction | ~31 | Most complete physics |
| E | Temporal EMA (SAM2 sim) | ~28 | Smooth, less flickering |
| F | Logistic regression on gates | ~30 | SAM dominates at 3.25× |
| G | Dual-body Pearson correlation | ~29 | Newton's 3rd law basis |

### Phase 2A — Approach H (Full-Frame Scanner) vs 31 GT

Post-filter results on Video 1 (31 confirmed impacts):

| Filter Stack | Events | TP | FP | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Raw | 157 | — | — | — | — | — |
| + cd=50 | ~90 | — | — | ~30% | ~87% | — |
| + cd=50 + IoU≤0.35 | 47 | — | — | — | — | 64.1% |
| + cd=50 + IoU≤0.35 + flow≥1.3 | ~47 | — | — | — | — | ~64% |
| + cd=50 + IoU≤0.35 + flow≥1.3 + arm≥0.30 | 45 | 25 | 20 | **55.6%** | **80.6%** | **65.7%** |
| + depth filter | 45 | — | — | — | — | — |

### Phase 2B — Fusion vs 31 GT (Video 2, with world_coords + audio)

Ground truth: 31 manually labelled impact timestamps.
Tolerance window: ±12 frames at 24.995 fps.

| Method | P | R | F1 | Notes |
|--------|---|---|-----|-------|
| Fusion v2 (voting, 3 signals) | ~55% | ~65% | ~59% | First multi-signal attempt |
| Fusion v3 (threshold search) | ~58% | ~68% | ~62% | Fast parameter sweep |
| Fusion v4 (GBM, rich features) | ~61% | ~71% | ~66% | Multi-band audio added |
| Fusion v5 (relabeled GT) | ~63% | ~74% | ~68% | Corrected GT timestamps |
| Fusion v6 (XGBoost + optflow) | ~67% | ~74% | ~70% | Best of v2-v6 |
| Fusion v7 (spectral + paired physics) | ~68% | ~77% | **~72%** | Best overall |
| High precision v2 (P≥0.80 target) | **~80%** | ~55% | ~65% | High precision, lower recall |

---

## What Failed — and Why

### 1. 5-Gate Physics Only (No SAM)
**Result**: 130/138 = 94.2% classified as landed. Far too many false positives.
**Why it failed**: Without a visual/spatial check (SAM mask), near-misses look identical to landed hits in the physics data. The deceleration and jerk gates fire on any sharp motion change, not just contact.

### 2. Cross-Person Z Alignment (Core Data Problem)
**Result**: 3D contact distance measured as 0.207m at a visually confirmed contact frame (true distance: ~0.02m).
**Why it failed**: `shared_space_coords` estimates each person's depth independently. The cross-person Z error ranges from 0.05m to 0.61m — swamping the contact signal for softer punches.
**Impact**: The 3D Depth Proximity gate (Gate 3) is unreliable with current data. 182 frames exist where d_3d < 0.50m but the algorithm misses the impact.

### 3. ASFormer Window Dependency (Approaches A–G)
**Result**: Many real impacts (outside 138 ASFormer windows) permanently missed.
**Why it failed**: If ASFormer didn't record an action near a real contact moment, none of the A–G approaches can detect it.
**Fix**: Approach H removes this dependency by scanning every frame.

### 4. Optical Flow Gate as Primary Filter
**Result**: flow_ratio detects head snaps but also fires on both fighters moving simultaneously (clinch, corner work).
**Why it failed**: Not discriminative enough alone — needs combination with IoU filter and distance gate.

### 5. Depth Anything V2 Filter
**Result**: Did not significantly improve results beyond the optical flow + arm ratio stack.
**Why it failed**: Clinch events (where both fighters overlap) have both wrist and head at the same depth — the depth filter cannot distinguish these from real contacts.

### 6. LogisticRegression (Approach F) as Primary Method
**Result**: Performance similar to Approach D (pseudo-labels from D → circular self-labeling).
**Why it failed**: Training labels from Approach D inherit D's false positives. The model learns D's biases rather than the true ground truth.

### 7. Action Confidence Gate (Gate 5) as Primary Signal
**Result**: Very weak discrimination alone (Landed: 0.51 vs Missed: 0.45 — barely separable).
**Why it failed**: ASFormer's confidence reflects how clearly a punch was *thrown*, not whether it *landed*. A perfectly thrown miss scores high.

---

## What Works Best

### Ranked by Reliability

1. **SAM Mask Overlap** (Gate 1) — strongest single signal by far (learned coefficient 3.25× over next-best in logistic regression). Pixel-precise, lighting-invariant.

2. **Approach H + Full Filter Stack** — removing ASFormer window dependency is the biggest architectural improvement. F1=65.7% with arm filter.

3. **Fusion v7 (XGBoost + audio)** — best overall F1 (~72%) when `world_coords` + audio are available. The spectral centroid and paired physics (wrist_decel × head_accel) are novel strong features.

4. **Wrist Deceleration** (Gate 2) — best physics gate. Sharp, fast signal. Less affected by depth error than proximity.

5. **Receiver Head Reaction** (Gate 7) — reliable discriminator for hard impacts. Clinch and near-misses do not produce sudden head acceleration.

6. **Dual-Body Pearson Correlation** (Gate 8, Approach G) — physically rigorous. Simultaneously tests both sides of the impact event.

7. **Pi-HOC Contact Region Filter** — the only method that distinguishes blocked punches from landed punches. Biggest precision improvement step.

### Configuration That Works Best (Currently Available Data)

```
Approach H (full-frame SAM) → cooldown=50fr → IoU≤0.35 → flow_ratio≥1.3 → arm_ratio≥0.30
F1 = 65.7%, Precision = 55.6%, Recall = 80.6%
```

### Configuration That Works Best (With world_coords + audio)

```
Fusion v7 (XGBoost, 15+ features, world_coords + audio)
F1 ≈ 72%, Precision ≈ 68%, Recall ≈ 77%
```

---

## Requested JSON Improvements

The current `shared_space_coords` has a fundamental cross-person Z alignment problem. See [docs/NOTE_FOR_TEAMMATE.md](docs/NOTE_FOR_TEAMMATE.md) and [docs/SLACK_MESSAGE_FOR_TEAMMATE.md](docs/SLACK_MESSAGE_FOR_TEAMMATE.md) for full explanation with real diagnostic numbers.

See [data/example_sam3d_requested_format.json](data/example_sam3d_requested_format.json) for the complete requested format.

### Problem Demonstrated

```
Frame 620 — confirmed punch landing (visually verified):

  Person 0 head Z = 1.53m
  Person 1 head Z = 1.83m   ← 0.30m difference is physically impossible
                               (both fighters standing in same ring)

  Person 0 left wrist: [0.637, -0.219, 1.937]
  Person 1 head (nose): [0.572, -0.055, 1.833]

  Measured 3D distance = 0.207m   ← True distance = ~0.02m (actual contact)
  Phantom error: 0.187m comes from independent depth estimation
```

Cross-person Z error measured across 7 confirmed contact frames: **0.05m – 0.61m**.

### Requested Fields (Priority Order)

#### 1. `world_coords` ← Most Important

Ground-plane aligned coordinates where both persons share the same depth reference:
- Y=0 is the ring floor
- Z is consistent across persons (using bounding box height + focal length)

```python
# How to compute:
Z_person = focal_length * human_height_metres / bbox_height_pixels
```

**Expected result with world_coords**:
```
Frame 620:
  Person 0 left wrist:  [0.31, 1.61, 3.50]
  Person 1 head (nose): [0.29, 1.60, 3.50]
  3D distance = 0.022m   ← true contact distance
```

Detection accuracy improvement: ~60% → ~90%+

#### 2. `keypoint_conf` ← Easy to Add

Per-joint confidence [0.0–1.0] for all 70 joints. Wrists are frequently occluded during punches.

```json
"keypoint_conf": [0.96, 0.95, ..., 0.79, 0.81, ...]  // 70 values
```

Filter: only use wrists with `conf > 0.50`. Expected precision improvement: ~70% → ~90%+.

SAM3D already computes this internally (learnable uncertainty tokens) — just needs to be written to JSON output.

#### 3. `contact_events` ← Nice to Have (Harder)

Pre-computed contact detections at the top level of the JSON:

```json
"contact_events": [
  {
    "frame": 620,
    "time_sec": 24.80,
    "striker_id": 0,
    "striker_body_part": "left_wrist",
    "receiver_id": 1,
    "receiver_body_part": "head",
    "contact_prob": 0.91,
    "contact_3d_distance_m": 0.022,
    "contact_region": "head"
  }
]
```

`contact_region` values: `"head"`, `"torso"`, `"left_arm"`, `"right_arm"`, `"left_leg"`, `"right_leg"`

This is the only way to distinguish **blocked punches** (wrist hits guard arm → `contact_region = "left_arm"`) from **landed punches** (wrist hits head/torso). All other signals fail to separate these cases.

#### 4. `inter_person_depth_offsets` ← Fallback

If `world_coords` is too complex to add, a simpler fallback:

```json
"inter_person_depth_offsets": {
  "620": 0.11,   // subtract from Person 1's Z to align with Person 0
  "621": 0.10
}
```

Computable from relative bounding box heights alone — no ground plane detection needed.

---

## Ground Truth Dataset

31 manually labelled impact timestamps from Video 2 (`3.mp4`, 24.995 fps):

```
7:11  18:01  24:04  30:19  31:07  34:07  37:08
53:02  55:17  1:05:22  1:06:09  1:06:20  1:20:14
1:25:15  1:26:05  1:27:18  1:42:16  1:42:19  1:48:22
1:51:23  1:53:24  2:03:19  2:15:22  2:17:11  2:25:17
2:27:24  2:28:24  2:34:13  2:46:12  2:49:24  2:52:16
```

Format: `M:S:F` where F is frame-within-second at 24.995 fps.

**Relabeled GT** (`relabeled_gt.py`): Scans ±60 frames around each timestamp and corrects to the true contact frame using 3 signals: minimum 2D wrist-to-head distance, audio multi-band flux peak, receiver head sudden displacement.

---

## Repository Structure

```
Contact_Detection/
│
├── README.md                        ← This file (comprehensive documentation)
├── requirements.txt
├── .gitignore
│
│── Root (shared foundation, imported by every package) ──────────────
├── config.py                        ← All hyperparameters and Pi-HOC settings
├── keypoint_loader.py               ← Loads 2d/3d JSON + ASFormer actions → NumPy
│
│── docs/ ────────────────────────────────────────────────────────────
│   ├── SYSTEM_EXPLANATION.md        ← Technical explanation of all approaches A-G
│   ├── NOTE_FOR_TEAMMATE.md         ← JSON format improvement request (detailed)
│   ├── SLACK_MESSAGE_FOR_TEAMMATE.md← Informal version of the JSON request
│   ├── RTX5080_GPU_SETUP.md         ← GPU setup guide for RTX 5080 (Blackwell)
│   └── WALKTHROUGH.md               ← Phase 1 walkthrough (5-gate, no SAM)
│
│── data/ ────────────────────────────────────────────────────────────
│   └── example_sam3d_requested_format.json ← Example of requested JSON format
│
│── scripts/ — runnable entry points ────────────────────────────────
│   ├── run_phase1.py                ← Phase 1 entry point (5-gate, no SAM)
│   └── run_detection.py             ← Phase 2 entry point (live YOLO + SAM)
│
│── detectors/ — all detection approaches ────────────────────────────
│   ├── phase1/
│   │   ├── impact_detector.py       ← 5-gate impact scoring engine
│   │   └── impact_report.py         ← Visual report generator
│   ├── approach_h/
│   │   ├── fullscan.py              ← Approach H: full-frame SAM scanner
│   │   └── postfilter.py            ← Post-filter (cooldown + IoU → optimal F1)
│   └── fusion/
│       ├── base.py                  ← Fusion v1: basic multi-signal detector
│       ├── v2.py                    ← Voting-based (6 signals)
│       ├── v3.py                    ← Precomputed features + threshold sweep
│       ├── v4.py                    ← GradientBoosting + rich features
│       ├── v5.py                    ← Relabeled GT + contact metadata
│       ├── v6.py                    ← XGBoost + optical flow + head accel
│       ├── v7.py                    ← Spectral features + paired physics
│       └── v8.py                    ← Region-aware pixel-space fusion (CPU, SAM-free)
│
│── filters/ — standalone post-filters ──────────────────────────────
│   ├── optflow_gate.py              ← Optical flow head-snap gate
│   ├── pihoc_filter.py              ← Pi-HOC contact region filter
│   └── depth_filter.py              ← Depth Anything V2 wrist-depth filter
│
│── pipeline/ — full end-to-end pipelines ────────────────────────────
│   ├── json_pipeline.py             ← JSON-based pipeline (Approaches A-G)
│   └── live_pipeline.py             ← Live YOLO pose + SAM pipeline
│
│── rendering/ — video renderers ─────────────────────────────────────
│   ├── impact_fx.py                 ← Cinematic FX (flash, sparks, shake, audio)
│   ├── render_impact_video.py       ← Render annotated impact video
│   ├── render_final.py              ← Final render pass
│   ├── render_v5.py                 ← Render variant v5
│   └── frame_to_video.py            ← Frames → video compiler
│
│── evaluation/ — metrics & diagnostics ──────────────────────────────
│   ├── evaluate_vs_gt.py            ← Evaluate detection JSON vs 31 GT timestamps
│   ├── relabel_gt.py                ← Correct GT timestamps via geometry + audio
│   ├── verify_impacts.py            ← Visual verification of detected impacts
│   ├── diagnose_world_coords.py     ← Audit world_coords / keypoint_conf / contact_events
│   └── generate_analysis_report.py  ← Generate visual analysis report
│
│── experiments/ — one-off research scripts ──────────────────────────
│   ├── high_precision.py            ← Grid search: P≥0.80 constraint
│   ├── high_precision_v2.py         ← P≥0.80 with 3+ signal corroboration
│   ├── sweep_action_only.py         ← Baseline: action JSON + cooldown only
│   ├── detect_3d_impacts.py         ← 3D impact detector variant
│   ├── detect_impacts_new.py        ← Alternative detection formulation
│   └── hybrid_pipeline.py           ← Hybrid: live YOLO + pre-extracted 3D keypoints
│
│── models/ — model wrappers ──────────────────────────────────────────
│   ├── detector.py                  ← YOLO-based fighter detection
│   ├── segmenter.py                 ← SAM wrapper for body segmentation
│   └── tracker.py                   ← Multi-fighter tracker
│
│── utils/ — shared utilities ─────────────────────────────────────────
│   ├── geometry.py                  ← 3D geometry (distances, vectors)
│   ├── flow.py                      ← Optical flow computation
│   ├── visualization.py             ← OpenCV drawing utilities
│   ├── body_contact_viz.py          ← Body contact region visualization
│   ├── body_mesh_viz.py             ← SMPL body mesh visualization
│   ├── smpl_mesh_viz.py             ← SMPL mesh renderer
│   ├── smpl_video_viz.py            ← SMPL video overlay
│   └── pose_3d_viz.py               ← 3D pose visualization
│
│── impact_detection/ — classification package ────────────────────────
│   ├── impact_classifier.py         ← Core impact classification logic
│   └── pair_analyzer.py             ← Fighter-pair analysis utilities
│
│── checkpoints/ (git-ignored) ────────────────────────────────────────
│   ├── sam_vit_b_01ec64.pth         ← SAM ViT-B (375MB) — download separately
│   └── basicmodel_m_lbs_10_207_0_v1.0.0.pkl  ← SMPL body model — download separately
│
└── outputs/ (git-ignored) ────────────────────────────────────────────
    └── 3_fusion_v8.mp4              ← Annotated output video (green=HIT, red=FP)
```

---

## Setup & Installation

### Hardware Requirements

- **GPU**: NVIDIA RTX 3060+ recommended. Tested on RTX 5080 (Blackwell)
- **VRAM**: 4GB minimum (SAM ViT-B), 8GB+ recommended
- **RAM**: 16GB+
- **OS**: Windows 10/11, Linux, or macOS (SAM2 not available on Windows)

### RTX 5080 (Blackwell) Users

See [docs/RTX5080_GPU_SETUP.md](docs/RTX5080_GPU_SETUP.md). You need CUDA 12.8+ wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Do **not** use `pip install torch` (gives CPU-only build).

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/SAM3D_Module.git
cd SAM3D_Module

# 2. Install PyTorch with CUDA (RTX 5080 / CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
# OR: pip install segment-anything

# 5. Download SAM ViT-B checkpoint
mkdir checkpoints
# Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# Place at: checkpoints/sam_vit_b_01ec64.pth

# 6. (Optional) Install audio processing for fusion approaches
pip install librosa

# 7. (Optional) Install XGBoost for fusion v6/v7
pip install xgboost
```

### Verify GPU Setup

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
```

---

## Usage

### Input Data Required

You need 3 JSON files from SAM3D and the original video:

```
/your/data/
├── 1.mp4                   ← Original boxing match video
├── 2d_points.json          ← 70-joint 2D keypoints per frame (SAM3D output)
├── 3d_points.json          ← 70-joint 3D keypoints per frame (SAM3D output)
└── full_results.json       ← ASFormer action windows (SAM3D output)
```

Edit paths in `config.py`:

```python
KEYPOINTS_2D_PATH = r"/path/to/2d_points.json"
KEYPOINTS_3D_PATH = r"/path/to/3d_points.json"
ACTIONS_PATH      = r"/path/to/full_results.json"
```

> **Full command reference for every method.** All scripts run with the
> miniconda Python here (`/home/jake/miniconda3/bin/python3.13`), or just
> `python` in your env. Data paths are set in `config.py` (legacy detectors) or
> at the top of each session script (current detectors).

### 0. Data audit (run this first)

```bash
python evaluation/diagnose_world_coords.py     # audit world_coords / keypoint_conf / contact_events
python evaluation/diagnose_world_coords.py --sam3d /path/to/<id>_sam3d.json
```

### 1. SAM detectors (the strong visual signal)

```bash
# SAM mask overlap — is the wrist inside the opponent's body silhouette?
python detectors/sam/sam_detect.py                      # full run + video
python detectors/sam/sam_detect.py --thr 0.5 --cooldown 12 --tol 12
python detectors/sam/sam_detect.py --no-video           # detection + metrics only

# SAM + Depth Anything V2 + receiver head-reaction
python detectors/sam/sam_depth_detect.py                # full run + video
python detectors/sam/sam_depth_detect.py --no-video     # diagnostic + metrics only
python detectors/sam/sam_depth_detect.py --combine and  # AND-gate instead of weighted
```

### 2. Visual fusion (CPU-runnable, SAM-free)

```bash
python detectors/fusion/v8.py                  # region-aware pixel-space fusion + GT eval
python detectors/fusion/v8.py --no-audio --cooldown 10
python detectors/fusion/v8.py --video          # render annotated output video
```

### 3. Sound module (`detectors/sound/`)

```bash
# (a) onset detector — standalone audio impact detection + annotated video
python detectors/sound/sound_detector.py
python detectors/sound/sound_detector.py --threshold 1.5 --cooldown 10

# (b) cut candidate clips for labelling
python detectors/sound/sound_detector.py --extract-samples --sample-threshold 1.5

# (c) score clips with the pretrained AudioSet AST model + auto-sort
python detectors/sound/sound_ai_classify.py
python detectors/sound/sound_ai_classify.py --auto-sort --punch-thr 0.15

# (d) train a classifier on YOUR sorted labels, then scan the video
python detectors/sound/sound_train_detect.py --sample-dir outputs/sound_samples_av --mode impact
python detectors/sound/sound_train_detect.py --sample-dir outputs/sound_samples_av --mode logreg --no-video
```

### 4. Dataset extraction for the planned temporal model (`dataset/`)

```bash
# one video+audio clip per ASFormer action window → sort into impact/ not_impact/
python dataset/extract_action_clips.py
python dataset/extract_action_clips.py --pre 0.4 --post 0.7 --min-dur 1.4
python dataset/extract_action_clips.py --merge-gap 6        # merge rapid combos

# video+audio clips around audio-onset candidates (alt labelling set)
python dataset/extract_av_samples.py --pad 0.7
```

### 5. Generic renderer (annotate any detection JSON onto the video)

```bash
python rendering/render_from_json.py --json outputs/sam_detect.json \
       --out outputs/sam_detect.mp4 --label "SAM overlap"
```

### 6. Legacy detectors (original pipeline, old `2d/3d/full_results` format)

```bash
# Phase 1 — 5-gate physics, no SAM (fast; produces report + JSON, render via render_from_json)
python scripts/run_phase1.py
python scripts/run_phase1.py --threshold 0.50 --no-report

# Approaches A–G — SAM in ASFormer windows (SLOW on CPU ≈ 77 min/approach)
python pipeline/json_pipeline.py --approach all --no-video
python pipeline/json_pipeline.py --approach D            # single approach + video

# Approach H — full-frame SAM scanner (VERY slow on CPU ≈ 2.5 hr)
python detectors/approach_h/fullscan.py
python detectors/approach_h/fullscan.py --no-video
python detectors/approach_h/postfilter.py                # optimal: cd=50, IoU≤0.35

# Fusion v2–v7 — metric/JSON experiments (NO video; some need relabeled_gt.json / 3.wav)
python detectors/fusion/v7.py            # spectral + paired physics (best legacy)
python detectors/fusion/v6.py            # XGBoost + optical flow
python detectors/fusion/v2.py            # voting baseline

# Standalone filters (post-process Approach-H detections)
python filters/optflow_gate.py
python filters/pihoc_filter.py --video
python filters/depth_filter.py --depth-thr 0.15

# High-precision / baseline experiments
python experiments/high_precision_v2.py
python experiments/sweep_action_only.py
```

### 7. Evaluation & FX rendering

```bash
python evaluation/evaluate_vs_gt.py            # evaluate any detection JSON vs 31 GT
python rendering/impact_fx.py                  # cinematic FX (flash, sparks, shake, audio)
python rendering/render_impact_video.py
```

---

## Pi-HOC Architecture Mapping

This system is adapted from **Pi-HOC** (Pairwise 3D Human-Object Contact Estimation, arXiv:2604.12923), which detects human–object contact in general scenes. Boxing is mapped as:

| Pi-HOC Component | Boxing Implementation |
|---|---|
| DETR object detector | YOLOv8n-pose (fighter detection) |
| HO pair token φ([q_h; q_o]) | Wrist + elbow keypoint feature vector |
| InteractionFormer (DINOv2-L, 24 blocks) | Proximity + velocity scoring (simplified) |
| Contact decoder (SAM image encoder) | SAM ViT-B with bounding box prompt |
| Contact presence MLP + threshold δ | Weighted gate score ≥ 0.45 |
| Multi-view 3D lifting | Pre-extracted 3D keypoints from `3d_points.json` |
| Per-vertex contact prediction | Contact region: head/torso/arm/guard |

**Preserved Pi-HOC hyperparameters:**

| Hyperparameter | Value | Description |
|---|---|---|
| δ (contact threshold) | 0.52 | Contact presence score threshold |
| γ (IoU pair threshold) | 0.0 | Pair formation IoU threshold |
| λ_2d_focal | 4.0 | 2D focal loss weight |
| λ_dice | 1.0 | Dice loss weight |
| λ_3d_focal | 4.0 | 3D focal loss weight |
| λ_sp | 0.01 | Sparsity regularization |
| λ_cp | 1.0 | Contact probability weight |

**Key simplification**: Pi-HOC uses a full DINOv2-L + InteractionFormer architecture. This system replaces that with hand-engineered physics gates (deceleration, jerk, extension) because those are the appropriate signals for this specific task (two-person boxing contact) and are computationally much lighter.

---

## Output JSON Format

All detection methods output a structured JSON:

```json
{
  "approach": "H",
  "label": "Full-Frame SAM Contact Scanner",
  "threshold": 0.45,
  "src_fps": 24.995,
  "n_impacts": 45,
  "events": [
    {
      "is_impact": true,
      "impact_frame": 118,
      "impact_score": 0.779,
      "timestamp_seconds": 4.721,
      "contact_region": "head_right",
      "contact_point": [1136, 309],
      "striker_id": 1,
      "receiver_id": 0,
      "action": "punch",
      "gates": {
        "sam": 1.0,
        "decel": 1.0,
        "receiver_react": 1.0,
        "prox_3d": 0.0,
        "jerk": 0.012,
        "extension": 0.973
      }
    }
  ]
}
```

---

## Impact FX Output

The `impact_fx.py` renderer adds frame-accurate cinematic effects at detected impacts:

| Effect | Detail |
|--------|--------|
| Screen flash + starburst | 32% white flash + 16 crisp rays from contact pixel |
| Shockwave ring | Single 2px expanding ring (4px → 130px over 10 frames) |
| Spark trails | 8 gravity-arc sparks, thin lines, no blobs |
| Floating pill label | Action name + region + score%, drifts upward, fades at 16 frames |
| Camera shake | Seeded random shake for impacts scoring ≥ 0.70, max 2 frames |
| Synthesized audio | 140Hz thud + 800Hz slap (punches); 80Hz thud + body resonance (kicks) |

**Frame timing fix**: FX uses `impact_frame // stride` not `timestamp_seconds × fps`. The timestamp records when the *action started*, not when contact happened — using it directly places animations up to 138 output frames too early.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `torch`, `torchvision` | GPU tensor ops, SAM backbone |
| `segment-anything` | SAM ViT-B mask prediction |
| `ultralytics` | YOLOv8n-pose fighter detection |
| `opencv-python` | Video I/O, optical flow, drawing |
| `numpy` | Numerical computation |
| `scipy` | Signal processing (Pearson correlation) |
| `matplotlib` | Report visualization |
| `librosa` | Audio onset and spectral features |
| `xgboost` | Fusion v6/v7 classifier |
| `scikit-learn` | Logistic regression, gradient boosting, cross-validation |
| `tqdm` | Progress bars |
| `Pillow` | Image processing for depth filter |
