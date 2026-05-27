# Boxing Impact Detection System — Full Explanation

## What This Repository Does

This is a **boxing punch/kick impact detection system**. It takes a video of a boxing match and automatically detects every moment a punch or kick actually makes contact — not just when one was thrown — and produces annotated output videos with visual overlays, impact scores, and synthesized sound effects. It uses 7 different detection methods (Approaches A–G) so you can compare which method is most accurate.

---

## The Full Data Pipeline (What Happens Before Impact Detection)

The system does **not** run pose estimation live from scratch. All the heavy pre-processing was done by a separate external system called **SAM3D** that pre-analyzed the video and saved results to 3 JSON files.

### Input JSON Files

#### `2d_points.json`
Contains the 2D pixel position of 70 body joints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) for every frame of the video, for both fighters. Coordinates are in original 1920×1080 resolution. This gives you an exact pixel location like "fighter 0's right wrist is at pixel (843, 412) in frame 344."

#### `3d_points.json`
The same 70 joints but in 3D world-space coordinates (in metres). These come from a stereo reconstruction or depth-lifting model. Gives you real-world positions like "fighter 0's right wrist is 0.45 metres from fighter 1's head." There are two coordinate systems stored:
- `shared_space_coords` — absolute world coordinates
- `normalized_coords` — body-centred, so camera position does not affect the values

#### `full_results.json`
Contains 138 action recognition windows produced by **ASFormer** (an AI action segmentation model). Each window says: "Fighter 0 threw a cross between frames 340–358, confidence 72%, estimated speed 7.2 km/h, estimated power 148 watts, targeting Head." These are the *candidates* — the system knows a punch was thrown, but not whether it landed.

---

## The Core Problem This System Solves

ASFormer can tell you a punch was thrown. It cannot tell you if it actually connected. A fighter can throw a jab that misses completely and ASFormer still records it. **The impact detection system's job is to score every one of those 138 action windows and decide: did this punch/kick actually land?**

---

## The 7 Detection Gates (Shared Across All Approaches)

Each approach runs a set of scoring "gates." Each gate produces a score from 0.0 to 1.0. The gates are weighted and summed into a final `impact_score`. If `impact_score ≥ 0.45`, the hit is classified as a landed impact.

### Gate 1 — SAM Mask Overlap (weight: 40%)

This is the Pi-HOC-inspired gate and the heart of the system. **SAM (Segment Anything Model, Meta AI, ViT-B ~375MB)** runs on GPU. For each action window, the system:

1. Takes the receiver's bounding box from `2d_points.json`
2. Feeds it as a prompt to SAM, which segments a pixel-precise body mask of the opponent (like a perfect silhouette)
3. Takes the striker's wrist pixel position from `2d_points.json`
4. Checks: does the wrist pixel fall inside that mask?

If the wrist is **inside** the mask → score = 1.0 (contact). If it is outside, the score falls off with Euclidean distance from the mask edge. A wrist 60px away from the mask scores 0.0. This is the most geometrically accurate gate because SAM's mask is pixel-precise regardless of clothing or lighting.

### Gate 2 — Wrist Deceleration (weight: 20%)

A fist decelerates sharply the moment it hits something. The system computes the 3D speed of the striker's wrist across the action window using `3d_points.json`, then looks for a velocity drop. It finds the frame where velocity goes from high to low most abruptly. A sharp deceleration → high score. This runs on GPU as a CUDA tensor operation.

### Gate 3 — 3D Depth Proximity (weight: 15%)

Even if SAM gives a low score (e.g., the wrist is just outside the mask edge), if the 3D world-space distance between the wrist and the opponent's body centroid is < 0.60 metres, this gate gives a partial score. It is a backup gate that handles cases where the camera angle makes the 2D projection misleading but the 3D data shows the fighter was actually close.

### Gate 4 — 3D Jerk (weight: 12%)

Jerk is the rate of change of acceleration (third derivative of position). At the moment of contact, there is a sudden force spike — the wrist goes from free-swinging to impacting a body, which creates an extremely sharp acceleration change. The system computes this from `3d_points.json` on GPU and looks for the peak jerk value within the window.

### Gate 5 — Action Confidence (weight: 8%)

Uses ASFormer's own confidence score combined with the estimated speed and power. A confident, fast, powerful action is more likely to land with meaningful force. Formula:

```
min(1.0, confidence × (1.0 + speed_kmh/50) × (1.0 + power_watts/500))
```

### Gate 6 — Arm Extension (weight: 5%)

Measures how extended the arm is at the moment of contact. Calculated as `wrist-to-shoulder distance / full arm length`. If the arm is bent (< 0.68 extension ratio), it is probably a guard or retraction — not a landed punch. A fully extended arm confirms this is an outgoing strike making contact.

---

## The 7 Approaches (A–G)

All approaches use the same 6 gates above but with different configurations, extra gates, or different scoring strategies.

---

### Approach A — Multi-Frame SAM

**The baseline.** Instead of running SAM on just one frame, it scans ±2 frames around the estimated best contact frame (a 5-frame window) and picks the frame with the **highest SAM score**. This compensates for the fact that the "best" contact moment might be a frame earlier or later than what ASFormer detected.

| Detail | Value |
|---|---|
| SAM window | 5 frames (±2) |
| Gate weights | SAM=40%, Decel=20%, Prox3D=15%, Jerk=12%, Conf=8%, Ext=5% |
| Threshold | 0.45 |
| Detected impacts | ~29 |

---

### Approach B — Soft Gate

Same as A but with **relaxed scoring logic**. In the standard approach, if SAM scores below a minimum, physics gates are not enough to push the total over threshold. In Approach B, the physics gates (deceleration + jerk) can independently rescue a detection even if the SAM score is low. This catches hits where the camera angle made SAM uncertain but the motion physics are clearly a contact event.

| Detail | Value |
|---|---|
| Key change | Physics gates can override a low SAM score |
| Gate weights | Same as A |
| Threshold | 0.45 |
| Trade-off | More detections, some extra false positives |

---

### Approach C — Lower Threshold + Heatmap

Drops the impact score threshold from 0.45 to 0.38 so more borderline events are classified as impacts. Also adds a **dense optical-flow heatmap overlay** on the output video — a colour visualization of where the most motion is happening in each frame (hot = fast moving, cold = still). This is useful for visually seeing where energy is being transferred across the body even on subtle hits.

| Detail | Value |
|---|---|
| Key change | Lower threshold + optical flow heatmap visualization |
| Gate weights | Same as A |
| Threshold | 0.38 (lowered from 0.45) |
| Trade-off | Catches subtle impacts, higher false positive rate |

---

### Approach D — Enhanced (Elbow Probe + Receiver Reaction)

Adds two completely new gates on top of A's 6:

**Elbow probe:** In A/B/C, only the wrist is probed into the SAM mask. In D, the **elbow position is also probed**. If either wrist or elbow falls inside the opponent's mask, the SAM score is taken as the maximum of both. This catches hooks and uppercuts where the point of contact might be the forearm/elbow, not the wrist tip.

**Receiver reaction gate (weight: 14%):** At the moment of a real impact, the opponent's head snaps backward. The system measures the 3D acceleration of the receiver's head centroid (average of nose + eyes + ears keypoints) at the detected contact frame. A large head acceleration = real hit. This is a physics ground-truth check from the *receiving* body.

| Detail | Value |
|---|---|
| New gate 1 | Elbow probe into SAM mask (max of wrist vs elbow) |
| New gate 2 | Receiver head acceleration at impact frame |
| Gate weights | SAM=33%, Decel=18%, Receiver=14%, Prox3D=12%, Jerk=10%, Conf=8%, Ext=5% |
| Threshold | 0.45 |
| Notes | Most physically complete of Approaches A–D |

---

### Approach E — SAM2-Style Temporal EMA

SAM2 (Meta's video-aware successor to SAM) is not available on Windows, so this approach **simulates SAM2's temporal consistency** using Exponential Moving Average (EMA) smoothing. Instead of picking the single best frame in the SAM window, it computes a Gaussian-weighted average of SAM scores across a 7-frame window (alpha=0.65). Frames near the center of the window get more weight, frames at the edges get less.

This prevents a single noisy high-score frame from dominating, giving a temporally smoother detection that is more robust to flickering pose estimates.

| Detail | Value |
|---|---|
| SAM window | 7 frames |
| EMA alpha | 0.65 |
| Key change | Gaussian-weighted average of SAM scores instead of max |
| Gate weights | Same as A |
| Threshold | 0.45 |

---

### Approach F — Learned Logistic Regression Classifier

This approach trains a **machine learning classifier** on top of the other approaches. It uses Approach D's detection results as pseudo-labels (impacts D detected = positive examples, non-impacts = negatives). Then it fits a `LogisticRegression(class_weight='balanced')` model using the 7 gate scores as features:

```
features = [sam, decel, jerk, ext, prox_3d, conf, receiver_reaction]
```

**Learned feature weights from training on this video:**

| Feature | Coefficient |
|---|---|
| SAM overlap | **3.253** (by far the most predictive) |
| 3D Proximity | 0.875 |
| Deceleration | 0.612 |
| Jerk | 0.441 |
| Confidence | 0.388 |
| Extension | 0.201 |
| Receiver reaction | 0.174 |

The classifier then scores all 138 events using `predict_proba()`, giving a calibrated probability instead of a hand-tuned weighted sum. Training set: 138 samples, 43 positive, 95 negative.

| Detail | Value |
|---|---|
| Model | sklearn LogisticRegression(C=1.0, class_weight='balanced') |
| Training labels | Pseudo-labels from Approach D |
| Features | 7 gate sub-scores |
| Threshold | 0.45 on predict_proba output |

---

### Approach G — Dual-Body Correlation

This is the most physics-principled approach. At the moment of real contact, there is a causal physical relationship: **the striker's wrist decelerates exactly when and because the receiver's head accelerates**. These two events are linked by Newton's third law — the force stopping the fist is the same force moving the head.

The system computes a **Pearson correlation** between:
- Striker's wrist deceleration signal over ±5 frames around impact
- Receiver's head acceleration signal over the same window

A high positive correlation (close to +1.0) means decel and accel happened together → real impact. A low or negative correlation means the two events are independent → missed punch. The correlation is mapped from [-1, 1] to [0, 1] and used as a new gate.

| Detail | Value |
|---|---|
| New gate | Pearson correlation: striker wrist decel ↔ receiver head accel |
| Correlation window | ±5 frames around detected impact frame |
| Gate weights | SAM=28%, Dual-Corr=18%, Decel=16%, Prox3D=12%, Jerk=10%, Ext=8%, Conf=8% |
| Threshold | 0.45 |
| Physics basis | Newton's third law — action and reaction are simultaneous |

---

## The Output Videos

For each approach, the pipeline produces two outputs:

### Annotated Video (`1_X_approach_real.mp4`)
The original video with overlays:
- Fighter skeletons drawn from 70-joint keypoints
- Wrist position trails
- SAM body masks (Approach C: optical flow heatmap instead)
- Contact point marker at detected hit location
- HUD panel: impact score, gate breakdown bar chart, event log with timestamps

### FX Video (`X_approach_fx.mp4`)
The annotated video post-processed by the Impact FX system, which adds frame-accurate cinematic effects at every detected impact:

| Effect | Description |
|---|---|
| Screen flash + starburst | 32% white flash on frame 0, 16 crisp rays radiating from contact pixel |
| Shockwave ring | Single 2px ring expanding from 4px to 130px over 10 frames |
| Spark trails | 8 gravity-arc sparks radiating outward, no tip blobs, thin lines |
| Floating pill label | Action name + body region + score %, drifts upward, fades after 16 frames |
| Camera shake | Seeded random shake for impacts scoring ≥ 0.70, max 2 frames duration |
| Synthesized audio | 140Hz thud + 800Hz slap transient for punches; 80Hz thud + body resonance for kicks; stereo with 1ms decorrelation delay; volume scaled by impact score |

**Frame timing fix:** The FX system maps impact frames using `impact_frame // stride` (where stride=2) rather than `timestamp_seconds × output_fps`. This is critical because `timestamp_seconds` in the JSON records when the action *started*, not when contact happened — using it directly would place animations up to 138 output frames (11 seconds) too early.

---

## Approach Comparison Summary

| Approach | Key Idea | Gates | Threshold | Notes |
|---|---|---|---|---|
| A | Multi-frame SAM baseline | 6 | 0.45 | Best balance of precision and recall |
| B | Soft gate — physics rescues SAM failures | 6 | 0.45 | More detections, some false positives |
| C | Lower threshold + optical flow heatmap | 6 | 0.38 | Visual energy map, catches subtle hits |
| D | Elbow probe + receiver head reaction | 7 | 0.45 | Most complete physics model of A–D |
| E | Temporal EMA smoothing (SAM2 simulation) | 6+EMA | 0.45 | Robust to flickering pose estimates |
| F | Learned logistic regression on gate scores | ML | 0.45 | Data-driven weights, SAM dominates at 3.25x |
| G | Dual-body Pearson correlation | 7 | 0.45 | Newton's 3rd law — highest physics fidelity |

---

## Architecture Reference (Pi-HOC → Boxing Mapping)

This system is adapted from **Pi-HOC (Pairwise 3D Human-Object Contact Estimation)**, originally designed to detect human–object contact. The boxing adaptation maps the concepts as follows:

| Pi-HOC Component | Boxing Equivalent |
|---|---|
| DETR object detector | YOLOv8n-pose (fighter detection) |
| HO pair token φ([q_h; q_o]) | Wrist + elbow keypoint feature vector |
| InteractionFormer (DINOv2-L, 24 blocks) | Proximity + velocity scoring (simplified) |
| Contact decoder (SAM image encoder) | SAM ViT-B with bounding box prompt |
| Contact presence MLP with threshold δ | Weighted gate score ≥ 0.45 |
| Multi-view 3D lifting | Pre-extracted 3D keypoints from `3d_points.json` |

**Preserved Pi-HOC hyperparameters:** δ=0.52, γ=0.0 (IoU pair formation threshold), λ_2d_focal=4.0, λ_dice=1.0, λ_3d_focal=4.0, λ_sp=0.01, λ_cp=1.0.
