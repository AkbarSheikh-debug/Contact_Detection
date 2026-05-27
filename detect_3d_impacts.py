#!/usr/bin/env python3
"""
2D Kinematics Impact Detection Pipeline  --  New Video (3.mp4)
===============================================================
Single-camera solution. Replaces unreliable monocular 3D depth gates with
purely 2D pixel-space kinematics derived from the projected keypoints.

Gates:
  Gate 1  -- Wrist-in-Bbox : projected wrist inside receiver bbox (2D)
  Gate 1c -- Head Separation: two persons' heads >= HEAD_SEP_PX apart (tracker swap filter)
  Gate 2  -- Wrist Velocity : striker wrist moved >= VEL_PX_MIN px/frame over VEL_WINDOW frames
  Gate 3  -- Direction Align: wrist velocity vector aligned with wrist->receiver-head (cosine >= DIR_ALIGN_MIN)
  Gate 4  -- Global Cooldown: suppress duplicates within COOLDOWN frames

Output:
  new_outputs/results_2d_vel_<tag>.json   -- detected impacts
  new_outputs/4_impacts_2d_vel_<tag>.mp4  -- annotated video

Usage:
    python detect_3d_impacts.py
    python detect_3d_impacts.py --vel-px-min 50 --dir-align-min 0.3 --cooldown 25
    python detect_3d_impacts.py --sweep --no-video
"""

import os, json, argparse
from collections import deque
import cv2
import numpy as np
from tqdm import tqdm

os.environ["PYTHONIOENCODING"] = "utf-8"

# ── Paths ─────────────────────────────────────────────────────────────────────
VIDEO_IN   = r"C:\Users\XRIG\Downloads\2ndfight_3rdRound\3.mp4"
JSON_3D    = r"C:\Users\XRIG\Downloads\2ndfight_3rdRound\1876d63a-6cc9-4ebd-9507-c73f7729d1ee_sam3d.json"
OUTPUT_DIR = r"C:\Users\XRIG\Desktop\Impact_Detection\SAM3D_Module\new_outputs"

# ── Default gate parameters ────────────────────────────────────────────────────
BBOX_MARGIN   = 0.05   # Gate 1:  fractional bbox expansion per side
HEAD_SEP_PX   = 80     # Gate 1c: min pixel separation between the two persons' heads
VEL_PX_MIN    = 40.0   # Gate 2:  min wrist pixel speed (px/frame averaged over VEL_WINDOW)
DIR_ALIGN_MIN = 0.20   # Gate 3:  min cosine between velocity and wrist->receiver-head
VEL_WINDOW    = 4      # frames for velocity / direction measurement
COOLDOWN      = 20     # Gate 4:  frames between events (~0.8 s at 25 fps)

# ── Keypoint indices (COCO-17 subset of 70-joint skeleton) ────────────────────
HEAD_KPS     = [0, 1, 2, 3, 4]
WRIST_KPS    = [9, 10]
ELBOW_KPS    = [7, 8]
SHOULDER_KPS = [5, 6]

# ── Colours ───────────────────────────────────────────────────────────────────
COL_P0    = (0, 255, 180)
COL_P1    = (255, 140, 0)
COL_WRIST = (0, 180, 255)
COL_IMP   = (0, 50, 255)
COL_HUD   = (15, 15, 20)
FLASH_FR  = 20

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# Camera intrinsics from JSON focal_length field
FOCAL_LENGTH = 983.272705078125


def project_kp(X: float, Y: float, Z: float, fl: float, cx: float, cy: float):
    """Standard perspective: x=fl*X/Z+cx, y=-fl*Y/Z+cy (Y flipped camera→image)."""
    if Z < 0.05:
        return None
    return (int(fl * X / Z + cx), int(-fl * Y / Z + cy))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_3d(path: str) -> dict[int, dict[int, np.ndarray]]:
    """Returns {person_id: {frame: (70,3) array of shared_space_coords}}"""
    with open(path) as f:
        raw = json.load(f)
    persons = {}
    for pid_str, entries in raw.items():
        pid = int(pid_str)
        persons[pid] = {}
        for e in entries:
            persons[pid][e["frame"]] = np.array(e["shared_space_coords"], dtype=np.float32)
    return persons


def load_bboxes(path: str) -> dict[int, dict[int, list]]:
    """Returns {person_id: {frame: [x1,y1,x2,y2]}} in original image coords."""
    with open(path) as f:
        raw = json.load(f)
    persons = {}
    for pid_str, entries in raw.items():
        pid = int(pid_str)
        persons[pid] = {}
        for e in entries:
            fn   = e["frame"]
            dims = e.get("frame_dims", {})
            ow   = dims.get("original_width",  1920)
            oh   = dims.get("original_height", 1080)
            rw   = dims.get("resized_width",   1920)
            rh   = dims.get("resized_height",  1080)
            b    = e["bbox"]
            persons[pid][fn] = [b[0]*ow/rw, b[1]*oh/rh, b[2]*ow/rw, b[3]*oh/rh]
    return persons


# ─────────────────────────────────────────────────────────────────────────────
# Gate helpers
# ─────────────────────────────────────────────────────────────────────────────

def head_sep_px(j0: np.ndarray, j1: np.ndarray) -> float:
    """2D pixel distance between the two persons' average head positions."""
    def head_2d(j):
        pts = [project_kp(*j[k], FOCAL_LENGTH, 960.0, 540.0) for k in HEAD_KPS]
        pts = [p for p in pts if p is not None]
        if not pts:
            return None
        return np.mean(pts, axis=0)
    c0 = head_2d(j0)
    c1 = head_2d(j1)
    if c0 is None or c1 is None:
        return 9999.0
    return float(np.linalg.norm(np.array(c0) - np.array(c1)))


def wrist_in_bbox_2d(
    js: np.ndarray,
    recv_bb: list,
    margin: float,
) -> tuple[bool, int, float]:
    """
    Returns (passes, best_wrist_kp, norm_dist).
    norm_dist: normalised distance from wrist projection to the expanded bbox edge.
    <= 0 means inside the expanded box.
    """
    bw   = recv_bb[2] - recv_bb[0]
    bh   = recv_bb[3] - recv_bb[1]
    diag = (bw**2 + bh**2)**0.5 + 1e-6
    ex1  = recv_bb[0] - margin * bw
    ey1  = recv_bb[1] - margin * bh
    ex2  = recv_bb[2] + margin * bw
    ey2  = recv_bb[3] + margin * bh
    best_dist = 9999.0
    best_wk   = WRIST_KPS[0]
    for wk in WRIST_KPS:
        p = project_kp(*js[wk], FOCAL_LENGTH, 960.0, 540.0)
        if p is None:
            continue
        dx = max(ex1 - p[0], 0.0, p[0] - ex2)
        dy = max(ey1 - p[1], 0.0, p[1] - ey2)
        nd = (dx**2 + dy**2)**0.5 / diag
        if nd < best_dist:
            best_dist = nd
            best_wk   = wk
    return best_dist <= 0.0, best_wk, best_dist


def wrist_velocity_2d(
    f3d_striker: dict[int, np.ndarray],
    fn: int,
    wk: int,
    vel_window: int,
) -> tuple[float, np.ndarray]:
    """
    Returns (speed_px_per_frame, unit_velocity_vector_2d) for a specific wrist KP.
    Measures 2D pixel displacement from fn-vel_window to fn.
    """
    js_now  = f3d_striker.get(fn)
    js_prev = f3d_striker.get(fn - vel_window)
    if js_now is None or js_prev is None:
        return 0.0, np.zeros(2)
    p_now  = project_kp(*js_now[wk],  FOCAL_LENGTH, 960.0, 540.0)
    p_prev = project_kp(*js_prev[wk], FOCAL_LENGTH, 960.0, 540.0)
    if p_now is None or p_prev is None:
        return 0.0, np.zeros(2)
    vec  = np.array(p_now, dtype=float) - np.array(p_prev, dtype=float)
    spd  = float(np.linalg.norm(vec)) / vel_window
    unit = vec / (np.linalg.norm(vec) + 1e-6)
    return spd, unit


def direction_alignment(
    unit_vel: np.ndarray,
    js: np.ndarray,
    jr: np.ndarray,
    wk: int,
) -> float:
    """
    Cosine similarity between the wrist velocity vector and the
    wrist→receiver-head vector. >0 = punch moving toward receiver.
    """
    p_wrist = project_kp(*js[wk], FOCAL_LENGTH, 960.0, 540.0)
    if p_wrist is None:
        return 0.0
    head_pts = [project_kp(*jr[k], FOCAL_LENGTH, 960.0, 540.0) for k in HEAD_KPS]
    head_pts = [p for p in head_pts if p is not None]
    if not head_pts:
        return 0.0
    head_center = np.mean(head_pts, axis=0)
    to_head = head_center - np.array(p_wrist, dtype=float)
    norm = np.linalg.norm(to_head)
    if norm < 1e-6:
        return 0.0
    return float(np.dot(unit_vel, to_head / norm))


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────

def detect_impacts(
    f3d: dict[int, dict[int, np.ndarray]],
    bboxes: dict[int, dict[int, list]],
    total_frames: int,
    src_fps: float,
    bbox_margin: float,
    head_sep_min: int,
    vel_px_min: float,
    dir_align_min: float,
    vel_window: int,
    cooldown: int,
) -> list[dict]:
    last_event = -9999
    events = []

    for fn in tqdm(range(total_frames), desc="  Scanning frames", unit="fr"):
        j0 = f3d[0].get(fn)
        j1 = f3d[1].get(fn)
        if j0 is None or j1 is None:
            continue
        if fn - last_event < cooldown:
            continue
        b0 = bboxes[0].get(fn)
        b1 = bboxes[1].get(fn)
        if b0 is None or b1 is None:
            continue

        # Gate 1c: heads must be separated enough to be two distinct people
        if head_sep_px(j0, j1) < head_sep_min:
            continue

        best_ev = None
        for sid, rid, js, jr, recv_bb in [
            (0, 1, j0, j1, b1),
            (1, 0, j1, j0, b0),
        ]:
            # Gate 1: projected wrist inside (margin-expanded) receiver bbox
            passes, wk, nd = wrist_in_bbox_2d(js, recv_bb, bbox_margin)
            if not passes:
                continue

            # Gate 2: wrist must be moving fast enough to be a punch
            spd_px, unit_vel = wrist_velocity_2d(f3d[sid], fn, wk, vel_window)
            if spd_px < vel_px_min:
                continue

            # Gate 3: velocity must be directed toward receiver's head
            da = direction_alignment(unit_vel, js, jr, wk)
            if da < dir_align_min:
                continue

            t  = fn / src_fps
            ev = {
                "impact_frame":          fn,
                "time_sec":              round(t, 3),
                "time_str":              f"{int(t//60)}:{t%60:05.2f}",
                "striker_id":            sid,
                "receiver_id":           rid,
                "wrist_kp":              wk,
                "bbox_norm_dist":        round(float(nd), 4),
                "wrist_vel_px_per_frame": round(float(spd_px), 2),
                "direction_alignment":   round(float(da), 4),
                "wrist_3d":              [round(float(v), 4) for v in js[wk].tolist()],
            }

            # Prefer the candidate with wrist deepest inside bbox
            if best_ev is None or nd < best_ev["bbox_norm_dist"]:
                best_ev = ev

        if best_ev is not None:
            events.append(best_ev)
            last_event = fn

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Video rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_video(
    events: list[dict],
    f3d: dict[int, dict[int, np.ndarray]],
    video_path: str,
    out_path: str,
    src_fps: float,
):
    impact_map = {e["impact_frame"]: e for e in events}
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))

    flash_q   = deque()
    event_log = deque(maxlen=5)
    n_impacts = 0

    with open(JSON_3D) as f:
        raw_json = json.load(f)
    bbox_map: dict[int, dict[int, list]] = {}
    for pid_str, entries in raw_json.items():
        pid = int(pid_str)
        bbox_map[pid] = {}
        for e in entries:
            dims = e.get("frame_dims", {})
            sx = dims.get("original_width", W) / dims.get("resized_width", W)
            sy = dims.get("original_height", H) / dims.get("resized_height", H)
            b  = e["bbox"]
            bbox_map[pid][e["frame"]] = [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]

    cx_img, cy_img = W / 2.0, H / 2.0

    with tqdm(total=total, unit="fr", desc="  Rendering") as pbar:
        for fi in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)
            canvas = frame.copy()

            # Bounding boxes
            for pid, col in [(0, COL_P0), (1, COL_P1)]:
                bb = bbox_map[pid].get(fi)
                if bb:
                    x1, y1, x2, y2 = [int(v) for v in bb]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
                    cv2.putText(canvas, f"P{pid}", (x1+4, y1+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

            # 3D skeletons projected to 2D
            for pid, col in [(0, COL_P0), (1, COL_P1)]:
                j3 = f3d[pid].get(fi)
                if j3 is None:
                    continue
                pts = {}
                for k in range(min(17, len(j3))):
                    p = project_kp(*j3[k], FOCAL_LENGTH, cx_img, cy_img)
                    if p and 0 <= p[0] < W and 0 <= p[1] < H:
                        pts[k] = p
                for a, b in SKELETON:
                    if a in pts and b in pts:
                        cv2.line(canvas, pts[a], pts[b], col, 2, cv2.LINE_AA)
                for k, p in pts.items():
                    r = 5 if k in HEAD_KPS else 4
                    cv2.circle(canvas, p, r, col, -1, cv2.LINE_AA)
                for wk in WRIST_KPS:
                    if wk in pts:
                        cv2.circle(canvas, pts[wk], 8, COL_WRIST, 2, cv2.LINE_AA)

            # Impact flash & annotation
            if fi in impact_map:
                e = impact_map[fi]
                flash_q.clear()
                flash_q.append(fi)
                n_impacts += 1
                sid = e["striker_id"]
                event_log.appendleft(
                    f"P{sid}->P{e['receiver_id']}  "
                    f"vel={e['wrist_vel_px_per_frame']:.0f}px/fr  "
                    f"align={e['direction_alignment']:.2f}  "
                    f"t={e['time_str']}"
                )

                rid = e["receiver_id"]
                rbb = bbox_map[rid].get(fi)
                if rbb:
                    rx = int((rbb[0] + rbb[2]) / 2)
                    ry = int(rbb[1] + (rbb[3] - rbb[1]) * 0.15)
                    cv2.circle(canvas, (rx, ry), 30, COL_IMP, 3, cv2.LINE_AA)
                    cv2.circle(canvas, (rx, ry), 10, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(canvas,
                                f"IMPACT  {e['wrist_vel_px_per_frame']:.0f}px/fr",
                                (rx - 60, ry - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            if flash_q and fi < flash_q[-1] + FLASH_FR:
                alpha = max(0.0, 1.0 - (fi - flash_q[-1]) / FLASH_FR)
                red = np.zeros_like(canvas)
                red[:, :] = (0, 0, 200)
                cv2.addWeighted(red, alpha * 0.40, canvas, 1.0, 0, canvas)

            # HUD
            hud_h = 36 + 22 * len(event_log)
            cv2.rectangle(canvas, (0, 0), (640, hud_h), COL_HUD, -1)
            cv2.putText(canvas,
                        f"2D Kinematics Impact | Impacts: {n_impacts} | Frame: {fi}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1, cv2.LINE_AA)
            for k, ev_txt in enumerate(event_log):
                cv2.putText(canvas, ev_txt, (12, 42 + k * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

            writer.write(canvas)

    cap.release()
    writer.release()
    print(f"  Video saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sweep(
    f3d: dict[int, dict[int, np.ndarray]],
    bboxes: dict[int, dict[int, list]],
    total_frames: int,
    src_fps: float,
):
    print("\n  === Threshold Sweep (2D velocity + direction gates) ===")
    print(f"  {'vel_min':>8}  {'dir_min':>7}  {'cd':>4}  {'N':>5}  Times (first 8)")

    for vel_min in [20.0, 30.0, 40.0, 50.0, 60.0]:
        for dir_min in [0.0, 0.10, 0.20, 0.30]:
            for cd in [15, 20, 25]:
                evs = detect_impacts(
                    f3d, bboxes, total_frames, src_fps,
                    BBOX_MARGIN, HEAD_SEP_PX, vel_min, dir_min, VEL_WINDOW, cd,
                )
                times = "  ".join(e["time_str"] for e in evs[:8])
                print(f"  vel>{vel_min:.0f}  dir>{dir_min:.2f}  cd={cd:2d}  {len(evs):5d}  {times}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox-margin",    type=float, default=BBOX_MARGIN,
                    help="Fractional bbox expansion for Gate 1 (default %(default)s)")
    ap.add_argument("--head-sep-px",    type=int,   default=HEAD_SEP_PX,
                    help="Min head separation in pixels Gate 1c (default %(default)s)")
    ap.add_argument("--vel-px-min",     type=float, default=VEL_PX_MIN,
                    help="Min wrist velocity px/frame Gate 2 (default %(default)s)")
    ap.add_argument("--dir-align-min",  type=float, default=DIR_ALIGN_MIN,
                    help="Min cosine alignment wrist-vel->recv-head Gate 3 (default %(default)s)")
    ap.add_argument("--vel-window",     type=int,   default=VEL_WINDOW,
                    help="Frames for velocity measurement (default %(default)s)")
    ap.add_argument("--cooldown",       type=int,   default=COOLDOWN,
                    help="Min frames between events (default %(default)s)")
    ap.add_argument("--sweep",          action="store_true",
                    help="Run threshold sweep before detection")
    ap.add_argument("--no-video",       action="store_true",
                    help="Skip video rendering")
    args = ap.parse_args()

    print()
    print("=" * 70)
    print("  2D Kinematics Impact Detection  --  New Video (3.mp4)")
    print("=" * 70)
    print(f"  Gate 1:  projected wrist inside receiver bbox  (margin={args.bbox_margin:.2f})")
    print(f"  Gate 1c: head separation >= {args.head_sep_px} px")
    print(f"  Gate 2:  wrist 2D velocity >= {args.vel_px_min} px/frame  (window={args.vel_window}fr)")
    print(f"  Gate 3:  direction alignment >= {args.dir_align_min:.2f}  (cosine wrist-vel -> recv-head)")
    print(f"  Gate 4:  cooldown = {args.cooldown} frames")
    print()

    print("  Loading 3D keypoints & bboxes ...")
    f3d    = load_3d(JSON_3D)
    bboxes = load_bboxes(JSON_3D)
    print(f"  Persons: {list(f3d.keys())}  |  Frames per person: {len(f3d[0])}")

    cap      = cv2.VideoCapture(VIDEO_IN)
    src_fps  = cap.get(cv2.CAP_PROP_FPS)
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"  Video: {total_fr} frames @ {src_fps:.3f} fps  ({W}x{H})")
    print(f"  Duration: {total_fr/src_fps:.1f}s\n")

    if args.sweep:
        threshold_sweep(f3d, bboxes, total_fr, src_fps)
        print()

    print(f"  Running detection ({total_fr} frames) ...")
    events = detect_impacts(
        f3d, bboxes, total_fr, src_fps,
        args.bbox_margin, args.head_sep_px,
        args.vel_px_min, args.dir_align_min, args.vel_window,
        args.cooldown,
    )

    print()
    print(f"  === Results: {len(events)} impacts detected ===")
    for i, e in enumerate(events, 1):
        print(f"  [{i:2d}] frame={e['impact_frame']:5d}  t={e['time_str']}  "
              f"P{e['striker_id']}->P{e['receiver_id']}  "
              f"vel={e['wrist_vel_px_per_frame']:.1f}px/fr  "
              f"align={e['direction_alignment']:.3f}  "
              f"bbox_nd={e['bbox_norm_dist']:.3f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = (f"bm{int(args.bbox_margin*100)}"
           f"_vel{int(args.vel_px_min)}"
           f"_da{int(args.dir_align_min*100)}"
           f"_cd{args.cooldown}")
    json_out = os.path.join(OUTPUT_DIR, f"results_2d_vel_{tag}.json")
    with open(json_out, "w") as f:
        json.dump({
            "video":        VIDEO_IN,
            "json_3d":      JSON_3D,
            "src_fps":      src_fps,
            "total_frames": total_fr,
            "n_impacts":    len(events),
            "gates": {
                "bbox_margin":         args.bbox_margin,
                "head_sep_px":         args.head_sep_px,
                "vel_px_min":          args.vel_px_min,
                "dir_align_min":       args.dir_align_min,
                "vel_window_frames":   args.vel_window,
                "cooldown_frames":     args.cooldown,
            },
            "events": events,
        }, f, indent=2)
    print(f"\n  JSON saved: {json_out}")

    if not args.no_video:
        vid_out = os.path.join(OUTPUT_DIR, f"4_impacts_2d_vel_{tag}.mp4")
        print(f"\n  Rendering annotated video ...")
        render_video(events, f3d, VIDEO_IN, vid_out, src_fps)

    print(f"\n  Output: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
