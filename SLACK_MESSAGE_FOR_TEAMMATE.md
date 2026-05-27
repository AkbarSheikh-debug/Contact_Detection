# Slack Message — Copy and send this to your teammate

---

Hey! I need to talk to you about the SAM3D JSON output format. I've been working on the boxing impact detection pipeline and I hit a fundamental problem with the current `shared_space_coords` data. I want to explain exactly what the issue is with real numbers, show you what I need, and ask if any of it is possible from SAM3D.

---

**🔴 The Problem — shared_space_coords are not actually "shared"**

Right now the JSON gives me `shared_space_coords` for each person separately. I assumed both persons were in the same 3D coordinate system — meaning if Person 0 is at depth Z = 3.5m and Person 1 is also at Z = 3.5m, they are truly at the same depth from the camera.

But when I tested this on a confirmed contact frame (frame 620 — I can visually see the punch landing in the video), here is what the JSON gives:

```
Frame 620 — confirmed punch landing:

  Person 0  head Z  =  1.53m
  Person 1  head Z  =  1.83m
  
  Difference = 0.30m
```

These two fighters are standing right next to each other in the same boxing ring. Their heads should be at nearly the same Z depth. A 0.30m difference is impossible physically — it is coming from the fact that SAM3D estimates each person's depth independently from their own bounding box crop, with no cross-person alignment.

When I compute the 3D distance between Person 0's wrist and Person 1's head at that contact frame to check if impact happened:

```
Person 0 left wrist:  [0.637,  -0.219,  1.937]
Person 1 head (nose): [0.572,  -0.055,  1.833]

3D distance = 0.207m
```

The real physical distance should be around 0.02m (actual contact). The extra ~0.10m in Z is pure noise from independent depth estimation. This inflates all my distance measurements.

I measured this across all 7 confirmed contact frames and the cross-person Z difference ranged from **0.05m to 0.61m**. That is the error I am fighting against in every single detection.

---

**📊 Why it still partially works — and why it still misses impacts**

Even with this error, the detection works for clear/hard punches because the 3D distance at contact drops to 0.06–0.14m (still much smaller than the 0.5–2.0m background distance when fighters are apart). But for softer or faster combinations, the Z noise of 0.30–0.61m completely swamps the contact signal and the impact is missed.

I ran a full scan and found **182 frames** where d_3d < 0.50m — many of them are real impacts I can see visually but the algorithm misses because the depth error makes the distance appear too large.

---

**✅ What I need — 3 new fields in the JSON**

I have prepared a full example JSON file that shows exactly what I am asking for, with real values for a contact frame and a non-contact frame:
📎 `example_sam3d_requested_format.json`

Here is a summary of the 3 new fields:

---

**Field 1 — `world_coords`  (most important)**

Same 70-joint 3D coordinates as `shared_space_coords`, but normalized so BOTH persons share the same depth reference frame. The key requirement: if both fighters are standing in the same ring at approximately the same distance from camera, their head Z values should be within 0.05m of each other, not 0.30–0.61m apart.

The way to compute this: detect the ground plane for each person using their foot keypoints (ankles kp15 and kp16), set Y=0 at the ring floor, and use the relative bounding box sizes + focal length to compute consistent metric depths for both persons:

```
Z_person = focal_length × human_height_metres / bbox_height_pixels
```

This removes the independent depth estimation error between persons. With this field, the contact distance at frame 620 would be ~0.022m (true contact) instead of 0.207m (noise-inflated).

```json
"[NEW] world_coords": [
  [0.42, 1.71, 3.50],   ← nose,  Y=1.71m above ground, Z=3.50m from camera
  [0.44, 1.64, 3.50],   ← left eye
  ...  (70 joints total, same order as shared_space_coords)
]
```

The critical difference from `shared_space_coords`: both Person 0 and Person 1 will have Z ≈ 3.50m at this frame because they are at the same depth. Not 1.53m vs 1.83m.

---

**Field 2 — `keypoint_conf`  (easy to add)**

A confidence score between 0.0 and 1.0 for each of the 70 joints. I need this because during a punch, the striker's wrist often goes partially behind the receiver's body — SAM3D still outputs a coordinate for that wrist, but it is an unreliable guess. When I use a low-confidence wrist to measure contact distance, I get false detections.

```json
"[NEW] keypoint_conf": [
  0.96,  ← nose (usually visible, high conf)
  0.95,  ← left eye
  0.94,  ← right eye
  0.88,  ← left ear
  0.86,  ← right ear
  0.92,  ← left shoulder
  0.91,  ← right shoulder
  0.85,  ← left elbow
  0.83,  ← right elbow
  0.79,  ← left wrist   ← lower confidence during punch (partially occluded)
  0.81,  ← right wrist
  ...    (70 values total, one per joint)
]
```

I would then only use wrist keypoints with `keypoint_conf[wrist] > 0.50` for distance calculations. This should significantly reduce false positives.

I looked into SAM3D and it seems the model already computes per-joint uncertainty internally during inference (learnable uncertainty tokens) — it just does not write this to the JSON output. If that is correct, this should be a small change to expose it.

---

**Field 3 — `contact_events`  (nice to have, harder)**

A pre-computed list at the top level of the JSON that directly tells me which frames have contact, which person is the striker, and which body part was hit. This would give me near-perfect detection without any manual threshold tuning.

```json
"[NEW] contact_events": [
  {
    "frame": 620,
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

The `contact_region` field is especially valuable — it would tell me if the punch landed on the head/torso (real scored impact) vs landed on the arm/guard (blocked punch, not a scored impact). Right now I cannot distinguish these two cases from coordinates alone.

I know this one is significantly harder — it probably needs a separate contact detection head. So I understand if this is not possible short-term. Fields 1 and 2 are the priority.

---

**Quick reference — what I am asking for**

| Field | Where it goes | Priority | Why I need it |
|---|---|---|---|
| `world_coords` | Inside each person's entry, same format as `shared_space_coords` | 🔴 High | Fixes cross-person Z inconsistency — the core problem |
| `keypoint_conf` | Inside each person's entry, 70 float values | 🟡 Medium | Filters unreliable wrist positions, reduces false positives |
| `contact_events` | New top-level section | 🟢 Low / nice-to-have | Direct contact labels, eliminates manual threshold tuning |

---

**My question for you:**

After looking at the example JSON (`example_sam3d_requested_format.json`) and this explanation, can you tell me:

1. Is `world_coords` (ground-plane aligned shared coordinates) something that SAM3D can output, or would it need a separate step like WHAM to be added on top of SAM3D?
2. Is `keypoint_conf` already computed inside SAM3D and just needs to be exposed in the JSON output?
3. Is `contact_events` feasible or is it too large a change for now?

Even just `keypoint_conf` alone would help a lot since it seems like it might already be there internally. And if `world_coords` requires a pipeline change like WHAM integration, I want to understand the effort involved so I can plan around it.

Thanks a lot — let me know what is and is not possible and I will adjust the detection pipeline accordingly.

---
📎 Files attached: `example_sam3d_requested_format.json`, `NOTE_FOR_TEAMMATE.md` (detailed technical reference)
