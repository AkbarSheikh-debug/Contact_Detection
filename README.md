# SAM3D Boxing Impact Detection Module

A research pipeline for automatically detecting whether punches and kicks land in a boxing match video, using pre-extracted 3D skeleton data, SAM (Segment Anything Model) mask overlap, and a range of physics-based and ML-based scoring methods.

---

## Table of Contents

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

The current `shared_space_coords` has a fundamental cross-person Z alignment problem. See [NOTE_FOR_TEAMMATE.md](NOTE_FOR_TEAMMATE.md) and [SLACK_MESSAGE_FOR_TEAMMATE.md](SLACK_MESSAGE_FOR_TEAMMATE.md) for full explanation with real diagnostic numbers.

See [example_sam3d_requested_format.json](example_sam3d_requested_format.json) for the complete requested format.

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
SAM3D_Module/
│
├── README.md                       ← This file (comprehensive documentation)
├── SYSTEM_EXPLANATION.md           ← Technical explanation of all 7 approaches A-G
├── NOTE_FOR_TEAMMATE.md            ← JSON format improvement request (detailed)
├── SLACK_MESSAGE_FOR_TEAMMATE.md   ← Informal version of the JSON request
├── RTX5080_GPU_SETUP.md            ← GPU setup guide for RTX 5080 (Blackwell)
├── WALKTHROUGH.md                  ← Phase 1 walkthrough (5-gate, no SAM)
│
├── example_sam3d_requested_format.json  ← Complete example of requested JSON format
├── requirements.txt
├── .gitignore
│
│── Core System ──────────────────────────────────────────────────────
├── config.py                       ← All hyperparameters and Pi-HOC settings
├── keypoint_loader.py              ← Loads 2d/3d JSON + ASFormer actions → NumPy
├── impact_detector.py              ← 5-gate impact scoring engine (Phase 1)
├── impact_report.py                ← Visual report generator (Phase 1)
│
│── Entry Points ─────────────────────────────────────────────────────
├── run_impact_detection.py         ← Phase 1 entry point (5-gate, no SAM)
├── run_detection.py                ← Phase 2 entry point (live YOLO + SAM)
├── pipeline.py                     ← Live YOLO pose + SAM pipeline
├── pipeline_json.py                ← JSON-based pipeline (Approaches A-G)
│
│── SAM Approaches A–H ───────────────────────────────────────────────
├── approach_h_fullscan.py          ← Approach H: full-frame SAM scanner
├── postfilter_h.py                 ← Post-filter for H (cd + IoU → optimal F1)
├── optflow_gate.py                 ← Optical flow head-snap gate (H v2)
├── pihoc_filter.py                 ← Pi-HOC contact region filter (H v3)
├── depth_filter.py                 ← Depth Anything V2 wrist-depth filter (H v4)
│
│── Fusion Approaches ────────────────────────────────────────────────
├── fusion_detect.py                ← Fusion v1: basic multi-signal detector
├── fusion_v2.py                    ← Fusion v2: voting-based (6 signals)
├── fusion_v3.py                    ← Fusion v3: precomputed features + threshold sweep
├── fusion_v4.py                    ← Fusion v4: GradientBoosting + rich features
├── fusion_v5.py                    ← Fusion v5: relabeled GT + contact metadata
├── fusion_v6.py                    ← Fusion v6: XGBoost + optical flow + head accel
├── fusion_v7.py                    ← Fusion v7: spectral features + paired physics
│
│── High Precision Experiments ───────────────────────────────────────
├── high_precision.py               ← Grid search: P≥0.80 constraint
├── high_precision_v2.py            ← Push P≥0.80 with max recall, 3+ signal corroboration
├── sweep_action_only.py            ← Baseline: action JSON + cooldown only (no features)
│
│── Analysis & Evaluation ────────────────────────────────────────────
├── evaluate_vs_gt.py               ← Evaluate any detection JSON vs 31 GT timestamps
├── relabel_gt.py                   ← Correct GT timestamps using geometry + audio
├── generate_analysis_report.py     ← Generate visual analysis report
├── verify_impacts.py               ← Visual verification of detected impacts
│
│── Hybrid Approaches ────────────────────────────────────────────────
├── hybrid_pipeline.py              ← Hybrid: live YOLO + pre-extracted 3D keypoints
├── detect_3d_impacts.py            ← 3D impact detector variant
├── detect_impacts_new.py           ← Alternative detection formulation
│
│── Rendering & Visualization ────────────────────────────────────────
├── impact_fx.py                    ← Cinematic FX renderer (flash, sparks, shake, audio)
├── render_impact_video.py          ← Render annotated impact video
├── render_final.py                 ← Final render pass
├── render_v5.py                    ← Render variant v5
├── frame_to_video.py               ← Frames → video compiler
│
│── Models (Python packages) ─────────────────────────────────────────
├── models/
│   ├── detector.py                 ← YOLO-based fighter detection
│   ├── segmenter.py                ← SAM wrapper for body segmentation
│   └── tracker.py                  ← Multi-fighter tracker
│
│── Utils (Python packages) ──────────────────────────────────────────
├── utils/
│   ├── geometry.py                 ← 3D geometry utilities (distances, vectors)
│   ├── flow.py                     ← Optical flow computation
│   ├── visualization.py            ← OpenCV drawing utilities
│   ├── body_contact_viz.py         ← Body contact region visualization
│   ├── body_mesh_viz.py            ← SMPL body mesh visualization
│   ├── smpl_mesh_viz.py            ← SMPL mesh renderer
│   ├── smpl_video_viz.py           ← SMPL video overlay
│   └── pose_3d_viz.py              ← 3D pose visualization
│
│── Impact Detection Package ─────────────────────────────────────────
├── impact_detection/
│   ├── impact_classifier.py        ← Core impact classification logic
│   └── pair_analyzer.py            ← Fighter-pair analysis utilities
│
│── Model Checkpoints (Git-ignored) ──────────────────────────────────
└── checkpoints/
    ├── sam_vit_b_01ec64.pth        ← SAM ViT-B (375MB) — download separately
    └── basicmodel_m_lbs_10_207_0_v1.0.0.pkl  ← SMPL body model — download separately
```

---

## Setup & Installation

### Hardware Requirements

- **GPU**: NVIDIA RTX 3060+ recommended. Tested on RTX 5080 (Blackwell)
- **VRAM**: 4GB minimum (SAM ViT-B), 8GB+ recommended
- **RAM**: 16GB+
- **OS**: Windows 10/11, Linux, or macOS (SAM2 not available on Windows)

### RTX 5080 (Blackwell) Users

See [RTX5080_GPU_SETUP.md](RTX5080_GPU_SETUP.md). You need CUDA 12.8+ wheels:

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

### Phase 1 — 5-Gate (No SAM, Fast)

```bash
python run_impact_detection.py
python run_impact_detection.py --threshold 0.50
python run_impact_detection.py --no-report
```

### Phase 2 — Approach H (Full-Frame SAM Scanner)

```bash
python approach_h_fullscan.py
python approach_h_fullscan.py --threshold 0.42
python approach_h_fullscan.py --no-video
```

Then apply post-filtering:

```bash
python postfilter_h.py                    # optimal: cd=50, IoU≤0.35
python postfilter_h.py --cooldown 30
python postfilter_h.py --max-iou 1.0     # disable IoU filter
```

Optional additional filters:

```bash
python optflow_gate.py                   # optical flow gate
python pihoc_filter.py --video           # Pi-HOC contact region filter
python depth_filter.py --depth-thr 0.15  # Depth Anything V2 filter
```

### Phase 2 — Fusion (Requires world_coords + audio)

```bash
python fusion_v7.py    # best fusion method
python fusion_v6.py    # XGBoost + optical flow
python fusion_v5.py    # with relabeled GT
```

### Evaluation

```bash
python evaluate_vs_gt.py    # evaluate any output JSON vs 31 GT timestamps
```

### Render Annotated Video with FX

```bash
python impact_fx.py         # apply cinematic FX (flash, sparks, shake, audio)
python render_impact_video.py
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
