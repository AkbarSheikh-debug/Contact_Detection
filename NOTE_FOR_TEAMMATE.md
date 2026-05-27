# SAM3D Output Format Request
### From: [Your Name]  
### To: SAM3D Developer  
### Re: Changes needed to the SAM3D JSON output for reliable boxing impact detection

---

## What I Need (Short Version)

Please add **3 new fields** to the SAM3D output JSON:

1. `world_coords` — ground-plane aligned 3D coordinates, same reference frame for both persons
2. `keypoint_conf` — confidence score [0.0–1.0] per joint
3. `contact_events` — pre-computed list of contact detections per frame (optional but very helpful)

The example file `example_sam3d_requested_format.json` (attached) shows exactly where these fields go and what values they should contain.

---

## Why the Current Output Is Causing Problems

### The Core Problem: `shared_space_coords` are NOT actually shared

Right now, the JSON has `shared_space_coords` for each person. I assumed these were in a single shared 3D world coordinate frame — meaning if Person 0 is at depth Z=3.5m and Person 1 is also at Z=3.5m, they are at the same depth from the camera.

**But this is not what is happening.**

I ran a diagnostic on the output for a confirmed contact frame (frame 620, where I can visually see a punch landing). Here is what the current JSON gives:

```
Frame 620  — Person 0 LEFT WRIST hits Person 1 HEAD (visually confirmed contact)
  Person 0 head Z = 1.53m
  Person 1 head Z = 1.83m     <-- these should be the same! Both fighters are standing
                                   in the same ring at roughly the same distance from camera.
  Z difference = 0.30m
```

A 0.30m depth gap between the two fighters' heads is impossible in real life — they are standing next to each other. This means each person's depth (Z) is estimated independently from the other person, using only that person's bounding box and body proportions. There is no cross-person Z alignment.

**The impact on my detection algorithm:**

When I compute the 3D distance between Person 0's wrist and Person 1's head at a contact frame:

```
Wrist position (Person 0, kp9):  [0.637, -0.219, 1.937]
Head position  (Person 1, kp0):  [0.572, -0.055, 1.833]

3D distance = sqrt((0.065)^2 + (0.164)^2 + (0.104)^2) = 0.207m
```

The real physical distance should be ~0.02m (2cm — actual contact). The extra 0.10m in Z comes purely from the independent depth estimation error. This inflates all my distance measurements.

**Across 7 confirmed contact frames, I measured:**
- Cross-person head Z difference: ranges from **0.05m to 0.61m** (should be ~0m)
- This adds between 0.05m and 0.61m of phantom distance to every measurement
- The algorithm still works for the clearest impacts (d_3d peaks at 0.06–0.14m even with the error)
- But it **misses impacts where the Z noise is large** — the distance appears 0.40–0.60m even at true contact

---

## What Each New Field Should Be

### 1. `world_coords`  ← Most Important

**What it is:** The same 3D keypoint coordinates as `shared_space_coords`, BUT normalized to a shared ground-plane coordinate system where:

- **Y = 0** is the ring floor (ground plane, estimated from where the fighters' feet contact the floor)
- **Y increases upward** (head is at Y ≈ 1.75m for a standing boxer)
- **Z is depth from camera**, but both persons have their Z estimated relative to the SAME reference — so if Person 0 head is at Z = 3.5m and Person 1 head is at Z = 3.5m, they are truly at the same depth

**How to compute it:**
1. Detect the ground plane (Y = 0) for each person using their foot keypoints (kp15, kp16 for ankles)
2. Use the hip keypoints (kp11, kp12) as the body height reference to establish the metric scale
3. Align both persons to the same Y=0 ground plane
4. Use the relative bounding box sizes and the focal length to compute a consistent Z ratio between persons:
   ```
   If Person 0 bbox height = h0 pixels and Person 1 bbox height = h1 pixels,
   and focal_length = fl, and a human is approximately H=1.75m tall:
   
   Z_person0 = fl * H / h0
   Z_person1 = fl * H / h1
   
   These give consistent metric depths for both persons in the same world frame.
   ```

**Why it matters:** With `world_coords`, the 3D distance between Person 0's wrist and Person 1's head will be a true metric distance. At contact it should be ≈ 0.02–0.05m. Without contact it will be > 0.50m. This is the clearest possible signal for impact detection.

**Example of what I expect:**
```
Frame 620 with world_coords:
  Person 0 left wrist (kp9):  [0.31, 1.61, 3.50]
  Person 1 head (kp0):        [0.29, 1.60, 3.50]
  3D distance = 0.022m   <-- TRUE contact distance, no depth ambiguity
```

---

### 2. `keypoint_conf`

**What it is:** An array of 70 float values (one per joint), representing the model's confidence that the 3D position of that joint is accurate. Range: 0.0 (no confidence, joint is occluded or badly estimated) to 1.0 (high confidence).

**Format:**
```json
"keypoint_conf": [0.96, 0.95, 0.94, 0.88, 0.86, 0.92, 0.91, 0.85, 0.83, 0.79, 0.81, ...]
```
(70 values, one per joint in the same order as `shared_space_coords` and `world_coords`)

**Why it matters:**
- Wrist keypoints (kp9, kp10) are frequently occluded during punches — the striker's wrist goes behind the receiver's body
- When a wrist is occluded, the model still outputs a coordinate, but it may be far from the true position
- By filtering to only use wrists with `keypoint_conf[wrist_kp] > 0.50`, I avoid false positives from phantom wrist positions
- This will likely improve precision from ~70% to ~90%+

---

### 3. `contact_events` (Top-Level)  ← Optional but Very Helpful

**What it is:** A pre-computed list of contact detections at the top level of the JSON (not inside each person's entry), listing all frames where a limb from one person contacts the body of the other person.

**Format:**
```json
"contact_events": [
  {
    "frame": 620,
    "time_sec": 24.80,
    "striker_id": 0,
    "receiver_id": 1,
    "striker_keypoint": 9,
    "striker_body_part": "left_wrist",
    "receiver_body_part": "head",
    "contact_prob": 0.91,
    "contact_3d_distance_m": 0.022,
    "contact_region": "head"
  }
]
```

**`contact_region` values:** `"head"`, `"torso"`, `"left_arm"`, `"right_arm"`, `"left_leg"`, `"right_leg"`

**Why it matters:**
This distinguishes a punch that **lands** (contact region = `"head"` or `"torso"`) from a punch that gets **blocked** (contact region = `"left_arm"` or `"right_arm"` — wrist hits receiver's guard). Blocked punches currently appear identical to landed punches in the coordinate data, causing false positives. The Pi-HOC paper (arXiv:2604.12923) proposes exactly this approach using SMPL mesh contact regions.

**If implementing the full contact model is too complex**, even just adding the `contact_3d_distance_m` (computed from `world_coords`, not `shared_space_coords`) and the `striker_body_part` / `receiver_body_part` labels would be enough.

---

### 4. `inter_person_depth_offsets` (Fallback if `world_coords` is hard)

If adding `world_coords` is too complex for now, this is a simpler fallback:

```json
"inter_person_depth_offsets": {
  "620": 0.11,
  "621": 0.10
}
```

This is a per-frame single value: how much to subtract from Person 1's Z coordinates to align them with Person 0's Z scale. It can be computed simply from the relative bounding box heights (no ground plane detection needed).

---

## Summary — Priority Order

| Priority | Field | Effort | Impact |
|----------|-------|--------|--------|
| **#1 Must have** | `world_coords` | Medium | Fixes the fundamental cross-person Z problem. Detection accuracy goes from ~60% to ~90%+ |
| **#2 Must have** | `keypoint_conf` | Low — model already computes this internally | Removes detections based on occluded/noisy wrists. Improves precision significantly |
| **#3 Nice to have** | `contact_events` | High — needs contact detection head | Near-perfect detection. Eliminates blocked-punch false positives |
| **#4 Fallback** | `inter_person_depth_offsets` | Low | Partial fix for Z problem if `world_coords` is too complex |

---

## Reference

The example JSON file `example_sam3d_requested_format.json` (in the same folder as this note) shows exactly:
- Where each new field appears in the JSON structure
- What realistic values look like for a contact frame (frame 620) and a non-contact frame (frame 480)
- How `world_coords` values differ from `shared_space_coords` values at the same frame
- How the `contact_events` section is structured

**Current JSON structure (per person entry):**
```
track_id, frame, bbox, shared_space_coords (70x3), normalized_coords,
focal_normalized_coords, normalization_root, normalization_scale,
pred_cam_t, focal_length, frame_dims
```

**Requested JSON structure (additions in brackets):**
```
track_id, frame, bbox, shared_space_coords (70x3), normalized_coords,
focal_normalized_coords, normalization_root, normalization_scale,
pred_cam_t, focal_length, frame_dims,
[world_coords (70x3)],
[keypoint_conf (70 floats)],
[world_coords_reliable (bool)]
+ top-level: [contact_events], [inter_person_depth_offsets]
```

---


