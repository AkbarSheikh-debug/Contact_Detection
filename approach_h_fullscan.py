#!/usr/bin/env python3
"""
Approach H  —  Full-Frame SAM Contact Scanner
===============================================
Root cause of low precision in A-G: SAM (the strongest gate, weight 33-40%)
only runs inside 138 ASFormer action windows.  Many real impacts fall OUTSIDE
those windows → missed.  Many false windows score >threshold even without real
contact → false positives.

Fix: run SAM on EVERY stride-2 source frame, check BOTH wrist+elbow directions,
apply per-pair cooldown.  No dependency on ASFormer action windows.

Output:
    outputs/fullscan/1_H_fullscan_real.mp4
    outputs/fullscan/results_H_fullscan.json

Usage:
    python approach_h_fullscan.py
    python approach_h_fullscan.py --threshold 0.42
    python approach_h_fullscan.py --no-video
"""

import os, sys, json, argparse, time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    KEYPOINTS_2D_PATH, KEYPOINTS_3D_PATH,
    SAM_CHECKPOINT, SAM_MODEL_TYPE,
    OUTPUT_DIR, PROCESS_EVERY_N_FRAMES,
    ACTION_ARM_CHAIN,
    KP70_LEFT_WRIST, KP70_RIGHT_WRIST,
    KP70_LEFT_ELBOW, KP70_RIGHT_ELBOW,
    IMPACT_COOLDOWN_FRAMES, GLOBAL_COOLDOWN_FRAMES,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_VIDEO = (
    r"C:\Users\XRIG\Downloads\for_impact_detection_experiment_2 (1)"
    r"\for_impact_detection_experiment_2\1.mp4"
)
OUTPUT_SUBDIR = os.path.join(OUTPUT_DIR, "fullscan")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps"  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

# ── Gate weights (approach D formula) ─────────────────────────────────────────
W_SAM      = 0.36   # SAM mask overlap — most reliable
W_DECEL    = 0.18   # wrist deceleration
W_RECV     = 0.16   # receiver head reaction (new discriminator vs clinch)
W_PROX_3D  = 0.12   # 3D depth backup
W_JERK     = 0.10   # 3D jerk
W_EXT      = 0.08   # arm extension

IMPACT_THRESHOLD = 0.45
STRIDE           = PROCESS_EVERY_N_FRAMES   # 2
PROX_2D_MAX      = 60.0   # px: SAM falloff distance
PROX_3D_MAX      = 0.60   # metres in shared_space_coords

HEAD_KPS  = [0, 1, 2, 3, 4]
TORSO_KPS = [5, 6, 11, 12]
BODY_KPS  = HEAD_KPS + TORSO_KPS

# Both wrist-elbow pairs
ARM_PAIRS = [
    (KP70_LEFT_WRIST,  KP70_LEFT_ELBOW),
    (KP70_RIGHT_WRIST, KP70_RIGHT_ELBOW),
]

SKELETON_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
COL_F0    = (0, 255, 180)
COL_F1    = (255, 140, 0)
COL_WRIST = (0, 180, 255)
COL_IMP   = (0, 50, 255)
COL_HUD   = (15, 15, 20)
FLASH_FR  = 15


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Frame2D:
    frame:  int
    bbox:   np.ndarray   # (4,)
    joints: np.ndarray   # (70, 2) original resolution

@dataclass
class Frame3D:
    frame:     int
    shared:    np.ndarray   # (70, 3) world-space
    normalized: np.ndarray  # (70, 3) body-centred

@dataclass
class ImpactH:
    impact_frame:    int
    impact_score:    float
    striker_id:      int
    receiver_id:     int
    wrist_idx:       int
    contact_region:  str
    contact_point:   list
    timestamp_sec:   float
    gate_sam:        float = 0.0
    gate_decel:      float = 0.0
    gate_recv:       float = 0.0
    gate_prox3d:     float = 0.0
    gate_jerk:       float = 0.0
    gate_ext:        float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_2d(path: str) -> dict[int, dict[int, Frame2D]]:
    print(f"[Load] 2D: {path}")
    with open(path) as f: raw = json.load(f)
    persons: dict[int, dict[int, Frame2D]] = {}
    for pid_str, entries in raw.items():
        pid = int(pid_str); persons[pid] = {}
        for e in entries:
            fn   = e["frame"]
            dims = e.get("frame_dims", {})
            sx   = dims.get("original_width",  1920) / dims.get("resized_width",  640)
            sy   = dims.get("original_height", 1080) / dims.get("resized_height", 360)
            j    = np.array(e["joints_2d"], dtype=np.float64)
            j[:, 0] *= sx;  j[:, 1] *= sy
            bbox = np.array(e["bbox"], dtype=np.float64)
            persons[pid][fn] = Frame2D(frame=fn, bbox=bbox, joints=j)
    for pid, fd in persons.items():
        if fd: print(f"  Person {pid}: {len(fd)} 2D frames")
    return persons

def load_3d(path: str) -> dict[int, dict[int, Frame3D]]:
    print(f"[Load] 3D: {path}")
    with open(path) as f: raw = json.load(f)
    persons: dict[int, dict[int, Frame3D]] = {}
    for pid_str, entries in raw.items():
        pid = int(pid_str); persons[pid] = {}
        for e in entries:
            fn = e["frame"]
            persons[pid][fn] = Frame3D(
                frame      = fn,
                shared     = np.array(e["shared_space_coords"], dtype=np.float64),
                normalized = np.array(e["normalized_coords"],   dtype=np.float64),
            )
    for pid, fd in persons.items():
        if fd: print(f"  Person {pid}: {len(fd)} 3D frames")
    return persons


# ─────────────────────────────────────────────────────────────────────────────
# SAM Gate (reused from pipeline_json.py pattern)
# ─────────────────────────────────────────────────────────────────────────────

class SamGate:
    def __init__(self, checkpoint, model_type, device):
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(torch.device(device))
        self.predictor = SamPredictor(sam)
        self.device    = device
        print(f"[SAM] {model_type} loaded on {device}.")

    def score(self, frame_bgr, rec_bbox, wrist_xy, rec_joints, elbow_xy=None):
        if frame_bgr is None: return 0.0, "torso", []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        x1,y1,x2,y2 = [float(v) for v in rec_bbox]
        box   = np.array([[x1,y1,x2,y2]], dtype=np.float32)
        masks,_,_ = self.predictor.predict(box=box, multimask_output=False)
        mask  = masks[0]
        H, W  = mask.shape

        def _probe(xy):
            px = int(np.clip(xy[0], 0, W-1))
            py = int(np.clip(xy[1], 0, H-1))
            if mask[py, px]: return 1.0
            inv = (~mask).astype(np.uint8)
            dt  = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
            return max(0.0, 1.0 - float(dt[py, px]) / PROX_2D_MAX)

        sc_w = _probe(wrist_xy)
        best_sc, best_pt = sc_w, wrist_xy.astype(int).tolist()
        if elbow_xy is not None and not np.allclose(elbow_xy, 0):
            sc_e = _probe(elbow_xy)
            if sc_e > sc_w: best_sc, best_pt = sc_e, elbow_xy.astype(int).tolist()

        region = self._region(wrist_xy, rec_joints)
        return best_sc, region, best_pt

    @staticmethod
    def _region(wrist_xy, rec_joints):
        def valid(k): return k < len(rec_joints) and not np.allclose(rec_joints[k], 0)
        wx, wy = float(wrist_xy[0]), float(wrist_xy[1])
        shoulders = [rec_joints[k] for k in [5,6] if valid(k)]
        hips      = [rec_joints[k] for k in [11,12] if valid(k)]
        center_x  = float(np.mean([s[0] for s in shoulders]+[h[0] for h in hips])) \
                    if (shoulders or hips) else None
        side = "left" if (center_x and wx < center_x) else "right"
        head_d = [float(np.linalg.norm(wrist_xy - rec_joints[k]))
                  for k in [0,1,2,3,4] if valid(k)]
        if head_d and min(head_d) < 100: return f"head_{side}"
        sh_y = float(np.mean([s[1] for s in shoulders])) if shoulders else None
        hi_y = float(np.mean([h[1] for h in hips]))      if hips      else None
        if sh_y and hi_y:
            mid = (sh_y + hi_y) / 2
            return (f"upper_torso_{side}" if wy < mid else f"lower_torso_{side}")
        return "torso"


# ─────────────────────────────────────────────────────────────────────────────
# Physics gates (GPU tensors)
# ─────────────────────────────────────────────────────────────────────────────

def gate_decel(f3d_pid, wrist_idx, ws, we):
    frange = sorted(f for f in f3d_pid if ws <= f <= we)
    if len(frange) < 3: return 0.0
    pos = torch.tensor(
        np.array([f3d_pid[f].normalized[wrist_idx] for f in frange]),
        dtype=torch.float32, device=DEVICE,
    )
    dt  = torch.tensor(np.diff(frange).astype(np.float32), device=DEVICE).clamp(min=1.0)
    vel = torch.norm(torch.diff(pos, dim=0) / dt.unsqueeze(1), dim=1)
    if len(vel) < 2: return 0.0
    pk  = float(vel.max())
    if pk < 1e-6: return 0.0
    acc = torch.diff(vel) / dt[1:len(torch.diff(vel))+1].clamp(min=1.0)
    max_dec = float(torch.clamp(-acc.min(), min=0.0))
    return min(1.0, (max_dec / pk) / 0.6)

def gate_jerk(f3d_pid, wrist_idx, ws, we):
    frange = sorted(f for f in f3d_pid if ws <= f <= we)
    if len(frange) < 4: return 0.0
    pos = torch.tensor(
        np.array([f3d_pid[f].normalized[wrist_idx] for f in frange]),
        dtype=torch.float32, device=DEVICE,
    )
    dt   = torch.tensor(np.diff(frange).astype(np.float32), device=DEVICE).clamp(min=1.0)
    vel  = torch.norm(torch.diff(pos, dim=0) / dt.unsqueeze(1), dim=1)
    if len(vel) < 2: return 0.0
    acc  = torch.diff(vel) / dt[1:len(torch.diff(vel))+1].clamp(min=1.0)
    if len(acc) < 2: return 0.0
    jerk = torch.diff(acc).abs()
    return min(1.0, float(jerk.max()) / 1.5)

def gate_extension(f2d_pid, wrist_idx, elbow_idx, shoulder_idx, fn):
    fd = f2d_pid.get(fn)
    if fd is None: return 0.0
    j = fd.joints
    if any(i >= len(j) for i in [wrist_idx, elbow_idx, shoulder_idx]): return 0.0
    w, e, s = j[wrist_idx], j[elbow_idx], j[shoulder_idx]
    if any(np.allclose(x, 0) for x in [w, e, s]): return 0.0
    dws = np.linalg.norm(w - s)
    dwe = np.linalg.norm(w - e)
    des = np.linalg.norm(e - s)
    tot = dwe + des
    if tot < 1e-6: return 0.0
    ratio = dws / tot
    return min(1.0, max(0.0, (ratio - 0.50) / (0.90 - 0.50)))

def gate_prox_3d(f3d_str, f3d_rec, fn):
    fs = f3d_str.get(fn); fr = f3d_rec.get(fn)
    if fs is None or fr is None: return 0.0
    wrist_pts = [fs.shared[i] for i in [KP70_LEFT_WRIST, KP70_RIGHT_WRIST]
                 if not np.allclose(fs.shared[i], 0)]
    body_pts  = [fr.shared[k] for k in BODY_KPS
                 if k < len(fr.shared) and not np.allclose(fr.shared[k], 0)]
    if not wrist_pts or not body_pts: return 0.0
    min_d = min(float(np.linalg.norm(wp - bp)) for wp in wrist_pts for bp in body_pts)
    return max(0.0, 1.0 - min_d / PROX_3D_MAX)

def gate_receiver_reaction(f3d_rec, fn, half=5):
    """Receiver head centroid acceleration at impact frame."""
    frames_3d = f3d_rec
    def _hc(f):
        fd = frames_3d.get(f)
        if fd is None: return None
        pts = [fd.normalized[k] for k in HEAD_KPS
               if k < len(fd.normalized) and not np.allclose(fd.normalized[k], 0)]
        return np.mean(pts, axis=0) if pts else None
    hc_prev = _hc(fn - half); hc_now = _hc(fn); hc_next = _hc(fn + half)
    if hc_prev is None or hc_now is None or hc_next is None: return 0.0
    accel = float(np.linalg.norm(hc_next - 2*hc_now + hc_prev))
    return min(1.0, accel / 0.06)


# ─────────────────────────────────────────────────────────────────────────────
# Main scanner
# ─────────────────────────────────────────────────────────────────────────────

def scan_all_frames(
    video_path:  str,
    f2d:         dict[int, dict[int, Frame2D]],
    f3d:         dict[int, dict[int, Frame3D]],
    sam_gate:    Optional[SamGate],
    threshold:   float,
    src_fps:     float,
    window_pad:  int = 10,
) -> list[ImpactH]:
    """
    Iterate every stride-2 source frame.  For each (striker, receiver, wrist)
    combination, run SAM + physics gates, fire if score >= threshold.
    Per-pair cooldown applied separately for each striker->receiver direction.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Cooldown state per direction
    last_det: dict[tuple[int,int], int] = {(0,1): -9999, (1,0): -9999}
    global_last = -9999

    results: list[ImpactH] = []

    with tqdm(total=total//STRIDE, unit="fr", desc="Approach H full-frame SAM scan") as pbar:
        for frame_idx in range(total):
            ret, frame_bgr = cap.read()
            if not ret: break
            if frame_idx % STRIDE != 0: continue   # only stride-2 frames

            pbar.update(1)

            for striker_id in [0, 1]:
                receiver_id = 1 - striker_id
                pair_key = (striker_id, receiver_id)

                # Per-pair cooldown
                if frame_idx - last_det[pair_key] < IMPACT_COOLDOWN_FRAMES: continue
                # Global cooldown (allow rapid exchanges)
                if frame_idx - global_last < GLOBAL_COOLDOWN_FRAMES // 2: continue

                fd_s = f2d.get(striker_id,  {}).get(frame_idx)
                fd_r = f2d.get(receiver_id, {}).get(frame_idx)
                if fd_s is None or fd_r is None: continue

                # Try both wrists; keep the higher-scoring one
                best_score  = 0.0
                best_region = "torso"
                best_pt     = []
                best_widx   = KP70_RIGHT_WRIST

                for wrist_idx, elbow_idx in ARM_PAIRS:
                    wrist_xy = fd_s.joints[wrist_idx]
                    elbow_xy = fd_s.joints[elbow_idx]
                    if np.allclose(wrist_xy, 0): continue

                    # ── Gate 1: SAM mask ──────────────────────────────────
                    if sam_gate is not None:
                        sc_sam, region, cp = sam_gate.score(
                            frame_bgr, fd_r.bbox, wrist_xy, fd_r.joints,
                            elbow_xy=elbow_xy,
                        )
                    else:
                        # Keypoint fallback
                        head_d = [float(np.linalg.norm(wrist_xy - fd_r.joints[k]))
                                  for k in BODY_KPS if not np.allclose(fd_r.joints[k], 0)]
                        min_d = min(head_d) if head_d else 9999.0
                        sc_sam = max(0.0, 1.0 - min_d / PROX_2D_MAX)
                        region = "torso"; cp = wrist_xy.astype(int).tolist()

                    # ── Gate 2: Wrist deceleration ────────────────────────
                    ws = frame_idx - window_pad; we = frame_idx + window_pad
                    sc_dec = gate_decel(f3d.get(striker_id, {}), wrist_idx, ws, we)

                    # ── Gate 3: Receiver reaction ─────────────────────────
                    sc_rec = gate_receiver_reaction(f3d.get(receiver_id, {}), frame_idx)

                    # ── Gate 4: 3D proximity ──────────────────────────────
                    sc_p3d = gate_prox_3d(f3d.get(striker_id, {}), f3d.get(receiver_id, {}), frame_idx)

                    # ── Gate 5: 3D jerk ───────────────────────────────────
                    sc_jrk = gate_jerk(f3d.get(striker_id, {}), wrist_idx, ws, we)

                    # ── Gate 6: Arm extension ─────────────────────────────
                    shoulder_idx = ACTION_ARM_CHAIN.get(wrist_idx, (elbow_idx, 5))[1]
                    sc_ext = gate_extension(f2d.get(striker_id, {}), wrist_idx, elbow_idx, shoulder_idx, frame_idx)

                    score = (W_SAM * sc_sam + W_DECEL * sc_dec + W_RECV * sc_rec +
                             W_PROX_3D * sc_p3d + W_JERK * sc_jrk + W_EXT * sc_ext)

                    if score > best_score:
                        best_score  = score
                        best_region = region
                        best_pt     = cp
                        best_widx   = wrist_idx
                        best_gates  = (sc_sam, sc_dec, sc_rec, sc_p3d, sc_jrk, sc_ext)

                if best_score >= threshold:
                    results.append(ImpactH(
                        impact_frame   = frame_idx,
                        impact_score   = best_score,
                        striker_id     = striker_id,
                        receiver_id    = receiver_id,
                        wrist_idx      = best_widx,
                        contact_region = best_region,
                        contact_point  = best_pt,
                        timestamp_sec  = frame_idx / src_fps,
                        gate_sam       = best_gates[0],
                        gate_decel     = best_gates[1],
                        gate_recv      = best_gates[2],
                        gate_prox3d    = best_gates[3],
                        gate_jerk      = best_gates[4],
                        gate_ext       = best_gates[5],
                    ))
                    last_det[pair_key] = frame_idx
                    global_last = frame_idx

    cap.release()
    print(f"\n  Scan complete: {len(results)} impacts detected.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# JSON output
# ─────────────────────────────────────────────────────────────────────────────

def save_json(results: list[ImpactH], path: str, threshold: float, src_fps: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    events = []
    for r in results:
        events.append({
            "is_impact":          True,
            "impact_frame":       r.impact_frame,
            "impact_score":       round(r.impact_score, 4),
            "timestamp_seconds":  round(r.timestamp_sec, 3),
            "contact_region":     r.contact_region,
            "contact_point":      r.contact_point,
            "striker_id":         r.striker_id,
            "receiver_id":        r.receiver_id,
            "action":             "punch",
            "gates": {
                "sam":              round(r.gate_sam,    3),
                "decel":            round(r.gate_decel,  3),
                "receiver_react":   round(r.gate_recv,   3),
                "prox_3d":          round(r.gate_prox3d, 3),
                "jerk":             round(r.gate_jerk,   3),
                "extension":        round(r.gate_ext,    3),
            },
        })
    out = {
        "approach": "H",
        "label":    "Full-Frame SAM Contact Scanner",
        "threshold": threshold,
        "src_fps":   src_fps,
        "n_impacts": len(results),
        "events":    events,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  JSON saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Video renderer  (simplified HUD, same visual language as pipeline_json.py)
# ─────────────────────────────────────────────────────────────────────────────

def render_video(
    video_path: str,
    f2d: dict[int, dict[int, Frame2D]],
    results: list[ImpactH],
    output_path: str,
    src_fps: float,
):
    impact_map: dict[int, ImpactH] = {r.impact_frame: r for r in results}

    cap = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = src_fps / STRIDE

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))

    flash_q   = deque()
    event_log = deque(maxlen=6)
    n_impacts = 0
    COLORS    = {0: COL_F0, 1: COL_F1}

    with tqdm(total=total//STRIDE, unit="fr", desc="  Rendering") as pbar:
        for fi in range(total):
            ret, frame = cap.read()
            if not ret: break
            if fi % STRIDE != 0: continue
            pbar.update(1)
            canvas = frame.copy()

            # Skeletons
            for pid, col in [(0, COL_F0), (1, COL_F1)]:
                fd = f2d.get(pid, {}).get(fi)
                if fd is None: continue
                j = fd.joints
                for a, b in SKELETON_PAIRS:
                    if a<len(j) and b<len(j) and not np.allclose(j[a],0) and not np.allclose(j[b],0):
                        cv2.line(canvas, tuple(j[a].astype(int)), tuple(j[b].astype(int)), col, 1, cv2.LINE_AA)
                for wi in [KP70_LEFT_WRIST, KP70_RIGHT_WRIST]:
                    if wi<len(j) and not np.allclose(j[wi],0):
                        cv2.circle(canvas, tuple(j[wi].astype(int)), 5, COL_WRIST, -1, cv2.LINE_AA)

            # Flash on impact
            if fi in impact_map:
                r = impact_map[fi]
                flash_q.clear(); flash_q.append(fi)
                n_impacts += 1
                event_log.appendleft(
                    f"F{r.striker_id}->F{r.receiver_id} {r.contact_region} {r.impact_score:.2f}"
                )
                cp = r.contact_point
                if len(cp) >= 2:
                    cv2.circle(canvas, (int(cp[0]), int(cp[1])), 18, COL_IMP, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (int(cp[0]), int(cp[1])),  8, (0,255,255), -1, cv2.LINE_AA)
                    cv2.putText(canvas, f"{r.impact_score:.2f}", (int(cp[0])+12, int(cp[1])-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            if flash_q and fi < flash_q[-1] + FLASH_FR:
                alpha = max(0.0, 1.0 - (fi - flash_q[-1]) / FLASH_FR)
                red = np.zeros_like(canvas); red[:, :] = (0, 0, 200)
                cv2.addWeighted(red, alpha * 0.35, canvas, 1.0, 0, canvas)

            # HUD
            hud_h = 36 + 22 * len(event_log)
            cv2.rectangle(canvas, (0, 0), (420, hud_h), COL_HUD, -1)
            cv2.putText(canvas, f"Approach H  |  Impacts: {n_impacts}  |  Frame: {fi}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,215,255), 1, cv2.LINE_AA)
            for k, ev_txt in enumerate(event_log):
                cv2.putText(canvas, ev_txt, (12, 42 + k*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200,200,200), 1, cv2.LINE_AA)

            writer.write(canvas)

    cap.release(); writer.release()
    print(f"  Video saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth evaluator (for self-testing)
# ─────────────────────────────────────────────────────────────────────────────

GT_TIMESTAMPS = [
    "7:11","18:01","24:04","30:19","31:07","34:07","37:08",
    "53:02","55:17","1:05:22","1:06:09","1:06:20","1:20:14",
    "1:25:15","1:26:05","1:27:18","1:42:16","1:42:19","1:48:22",
    "1:51:23","1:53:24","2:03:19","2:15:22","2:17:11","2:25:17",
    "2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16",
]

def _parse_ts(ts, fps):
    p = ts.split(":")
    if len(p) == 2: return int(p[0]) + int(p[1])/fps
    return int(p[0])*60 + int(p[1]) + int(p[2])/fps

def evaluate_gt(results: list[ImpactH], src_fps: float, tol_frames: int = 30):
    gt_frames = [int(_parse_ts(ts, src_fps) * src_fps) for ts in GT_TIMESTAMPS]
    det_frames = [r.impact_frame for r in results]

    matched_gt:  set[int] = set()
    matched_det: set[int] = set()
    for di, df in enumerate(det_frames):
        for gi, gf in enumerate(gt_frames):
            if gi in matched_gt: continue
            if abs(df - gf) <= tol_frames:
                matched_gt.add(gi); matched_det.add(di); break

    tp = len(matched_gt)
    fp = len(det_frames) - len(matched_det)
    fn = len(gt_frames)  - tp
    p  = tp/(tp+fp) if (tp+fp) else 0.0
    r  = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0

    print(f"\n{'='*60}")
    print(f"  Ground-truth evaluation  (tol ±{tol_frames} frames = ±{tol_frames/src_fps:.1f}s)")
    print(f"  GT total : {len(gt_frames)}")
    print(f"  Detected : {len(det_frames)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision={p:.1%}  Recall={r:.1%}  F1={f1:.1%}")
    print(f"\n  Matched GT timestamps:")
    for gi in sorted(matched_gt):
        gf = gt_frames[gi]
        best_df = min((df for di,df in enumerate(det_frames) if di in matched_det
                       if abs(df-gf)<=tol_frames), default=gf, key=lambda x: abs(x-gf))
        ts_s = best_df / src_fps
        print(f"    {gi+1:2d}. {GT_TIMESTAMPS[gi]:8s} GT:{gf:5d}  det:{best_df:5d}  score={results[next(di for di,df in enumerate(det_frames) if df==best_df)].impact_score:.3f}")
    print(f"\n  Missed GT timestamps:")
    for gi in range(len(gt_frames)):
        if gi not in matched_gt:
            print(f"    {gi+1:2d}. {GT_TIMESTAMPS[gi]:8s} frame={gt_frames[gi]}")
    print(f"{'='*60}\n")
    return tp, fp, fn, p, r, f1


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",     default=DEFAULT_VIDEO)
    ap.add_argument("--threshold", type=float, default=IMPACT_THRESHOLD)
    ap.add_argument("--no-video",  action="store_true")
    ap.add_argument("--no-sam",    action="store_true", help="Use keypoint fallback (fast, less accurate)")
    ap.add_argument("--eval-gt",   action="store_true", default=True, help="Evaluate against GT timestamps")
    args = ap.parse_args()

    t0 = time.time()
    print()
    print("=" * 68)
    print("  Approach H  —  Full-Frame SAM Contact Scanner")
    print("=" * 68)

    # ── Detect source FPS ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"  Source video : {args.video}")
    print(f"  Source FPS   : {src_fps:.3f}")
    print(f"  Threshold    : {args.threshold}")
    print(f"  Stride       : {STRIDE}")
    print(f"  Device       : {DEVICE}")

    # ── Load keypoints ────────────────────────────────────────────────────
    f2d = load_2d(KEYPOINTS_2D_PATH)
    f3d = load_3d(KEYPOINTS_3D_PATH)

    # ── Load SAM ──────────────────────────────────────────────────────────
    sam_gate = None
    if not args.no_sam:
        try:
            sam_gate = SamGate(SAM_CHECKPOINT, SAM_MODEL_TYPE, DEVICE)
        except Exception as e:
            print(f"  [WARN] SAM unavailable ({e}), using keypoint fallback.")

    # ── Scan ──────────────────────────────────────────────────────────────
    results = scan_all_frames(
        args.video, f2d, f3d, sam_gate, args.threshold, src_fps
    )

    # ── Evaluate against GT ───────────────────────────────────────────────
    if args.eval_gt:
        evaluate_gt(results, src_fps)

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_SUBDIR, "results_H_fullscan.json")
    save_json(results, json_path, args.threshold, src_fps)

    # ── Render video ──────────────────────────────────────────────────────
    if not args.no_video:
        vid_path = os.path.join(OUTPUT_SUBDIR, "1_H_fullscan_real.mp4")
        render_video(args.video, f2d, results, vid_path, src_fps)

    elapsed = time.time() - t0
    print(f"\n  Approach H complete in {elapsed:.1f}s")
    print(f"  Results  : {json_path}")
    if not args.no_video:
        print(f"  Video    : {os.path.join(OUTPUT_SUBDIR, '1_H_fullscan_real.mp4')}")
    print()


if __name__ == "__main__":
    main()
