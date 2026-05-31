#!/usr/bin/env python3
"""
Verification Script — 3D Impact Detection
==========================================
Extracts frames at each detected impact, overlays proper 2D projections
of 3D keypoints (using focal_length from JSON, NOT the buggy bbox-scale hack),
and saves screenshots to new_outputs/verification_frames/.

Also computes: are the fighters actually visually close at the impact frame?
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import os, json
import cv2
import numpy as np

VIDEO_IN   = r"C:\Users\XRIG\Downloads\2ndfight_3rdRound\3.mp4"
JSON_3D    = r"C:\Users\XRIG\Downloads\2ndfight_3rdRound\1876d63a-6cc9-4ebd-9507-c73f7729d1ee_sam3d.json"
RESULTS    = r"C:\Users\XRIG\Desktop\Impact_Detection\SAM3D_Module\new_outputs\results_3d_d30_asp5_cd50.json"
OUT_DIR    = r"C:\Users\XRIG\Desktop\Impact_Detection\SAM3D_Module\new_outputs\verification_frames"

os.makedirs(OUT_DIR, exist_ok=True)

HEAD_KPS     = [0, 1, 2, 3, 4]
WRIST_KPS    = [9, 10]
SHOULDER_KPS = [5, 6]
ELBOW_KPS    = [7, 8]
HIP_KPS      = [11, 12]
KNEE_KPS     = [13, 14]
ANKLE_KPS    = [15, 16]

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),          # face
    (5,6),                             # shoulders
    (5,7),(7,9),                       # left arm
    (6,8),(8,10),                      # right arm
    (5,11),(6,12),(11,12),             # torso
    (11,13),(13,15),                   # left leg
    (12,14),(14,16),                   # right leg
]

COL_P0 = (0, 220, 120)      # green
COL_P1 = (30, 130, 255)     # orange-blue
COL_WRIST_STRIKER = (0, 0, 255)     # red
COL_HEAD_RECEIVER = (255, 255, 0)   # yellow
FONT   = cv2.FONT_HERSHEY_SIMPLEX


def project_3d(X, Y, Z, fl, cx, cy):
    """Standard perspective projection: x=fl*X/Z+cx, y=-fl*Y/Z+cy (Y flipped)."""
    if Z < 0.01:
        return None
    px = int(fl * X / Z + cx)
    py = int(-fl * Y / Z + cy)
    return (px, py)


def draw_skeleton(canvas, coords, fl, cx, cy, col, striker_wrists=None, receiver_heads=None, W=1920, H=1080):
    """Draw projected 3D skeleton. Highlights strike wrist and target head."""
    pts = {}
    for i in range(min(17, len(coords))):   # COCO-17 first 17 joints
        p = project_3d(*coords[i], fl, cx, cy)
        if p and 0 <= p[0] < W and 0 <= p[1] < H:
            pts[i] = p

    # Draw bones
    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(canvas, pts[a], pts[b], col, 2, cv2.LINE_AA)

    # Draw joints
    for i, p in pts.items():
        r = 5 if i in HEAD_KPS else 4
        cv2.circle(canvas, p, r, col, -1, cv2.LINE_AA)

    # Highlight striker wrists
    if striker_wrists:
        for wk in striker_wrists:
            if wk in pts:
                cv2.circle(canvas, pts[wk], 10, COL_WRIST_STRIKER, 3, cv2.LINE_AA)
                cv2.circle(canvas, pts[wk], 5, (255, 255, 255), -1, cv2.LINE_AA)

    # Highlight receiver head
    if receiver_heads:
        for hk in receiver_heads:
            if hk in pts:
                cv2.circle(canvas, pts[hk], 10, COL_HEAD_RECEIVER, 3, cv2.LINE_AA)

    return pts


def bbox_iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def main():
    print("Loading JSON data ...")
    with open(JSON_3D) as f:
        raw = json.load(f)
    with open(RESULTS) as f:
        results = json.load(f)

    # Build lookup: pid -> frame -> entry
    entries = {}
    for pid_str, elist in raw.items():
        pid = int(pid_str)
        entries[pid] = {}
        for e in elist:
            entries[pid][e["frame"]] = e

    events = results["events"]
    fl = 983.272705078125
    cx, cy = 960.0, 540.0

    cap = cv2.VideoCapture(VIDEO_IN)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total} frames @ {fps:.2f} fps  ({W}x{H})")
    print(f"Output dir: {OUT_DIR}")
    print()

    # Collect all frame numbers to extract: impact frame ± 3
    WINDOW = 3
    frame_targets = {}   # frame_num -> list of event indices that reference it
    for idx, ev in enumerate(events):
        fi = ev["impact_frame"]
        for delta in range(-WINDOW, WINDOW + 1):
            fn = fi + delta
            if 0 <= fn < total:
                if fn not in frame_targets:
                    frame_targets[fn] = []
                frame_targets[fn].append((idx, delta))

    sorted_targets = sorted(frame_targets.keys())
    print(f"Extracting {len(sorted_targets)} frames (7 events × {2*WINDOW+1} frames each) ...")

    # Seek-read each target frame
    for fn in sorted_targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue

        canvas = frame.copy()

        # Get 3D data at this frame
        e0 = entries[0].get(fn)
        e1 = entries[1].get(fn)

        # Draw bboxes
        for pid, entry, col in [(0, e0, COL_P0), (1, e1, COL_P1)]:
            if entry is None:
                continue
            dims = entry.get("frame_dims", {})
            sx = dims.get("original_width", W) / dims.get("resized_width", W)
            sy = dims.get("original_height", H) / dims.get("resized_height", H)
            b = entry["bbox"]
            bb = [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]
            x1,y1,x2,y2 = [int(v) for v in bb]
            cv2.rectangle(canvas, (x1,y1), (x2,y2), col, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"P{pid}", (x1+4, y1+20), FONT, 0.65, col, 2, cv2.LINE_AA)

        # Draw skeletons using proper camera projection
        for pid, entry, col in [(0, e0, COL_P0), (1, e1, COL_P1)]:
            if entry is None:
                continue
            coords = np.array(entry["shared_space_coords"])
            draw_skeleton(canvas, coords, fl, cx, cy, col, W=W, H=H)

        # Highlight impact-specific keypoints for events near this frame
        for ev_idx, delta in frame_targets[fn]:
            if delta == 0:  # only at exact impact frame
                ev = events[ev_idx]
                sid = ev["striker_id"]
                rid = ev["receiver_id"]
                s_entry = entries[sid].get(fn)
                r_entry = entries[rid].get(fn)

                if s_entry and r_entry:
                    s_coords = np.array(s_entry["shared_space_coords"])
                    r_coords = np.array(r_entry["shared_space_coords"])

                    # Striker wrist (red dot)
                    wk = ev["wrist_kp"]
                    wp = project_3d(*s_coords[wk], fl, cx, cy)
                    if wp:
                        cv2.circle(canvas, wp, 14, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.circle(canvas, wp, 7, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(canvas, f"WRIST P{sid}", (wp[0]+12, wp[1]-5), FONT, 0.55, (0,0,255), 2, cv2.LINE_AA)

                    # Receiver head kps (yellow rings)
                    for hk in HEAD_KPS:
                        hp = project_3d(*r_coords[hk], fl, cx, cy)
                        if hp:
                            cv2.circle(canvas, hp, 12, (0, 255, 255), 2, cv2.LINE_AA)

                    # Distance line between striker wrist and receiver nose
                    rp = project_3d(*r_coords[0], fl, cx, cy)
                    if wp and rp:
                        cv2.line(canvas, wp, rp, (255, 0, 255), 2, cv2.LINE_AA)

                    # Bbox overlap check
                    def get_bb(entry_):
                        dims = entry_.get("frame_dims", {})
                        sx = dims.get("original_width", W) / dims.get("resized_width", W)
                        sy = dims.get("original_height", H) / dims.get("resized_height", H)
                        b = entry_["bbox"]
                        return [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]

                    bb_s = get_bb(s_entry)
                    bb_r = get_bb(r_entry)
                    iou = bbox_iou(bb_s, bb_r)

                    # Check if striker wrist is inside receiver bbox
                    wrist_in_receiver = False
                    if wp:
                        wrist_in_receiver = (bb_r[0] <= wp[0] <= bb_r[2] and
                                              bb_r[1] <= wp[1] <= bb_r[3])

                    # Impact info overlay
                    info = [
                        f"IMPACT #{ev_idx+1}  P{sid}->P{rid}  delta={delta}",
                        f"d_3d={ev['d_3d_metres']:.3f}m  spd={ev['approach_speed_m_per_frame']:.3f}m/fr",
                        f"bbox_IoU={iou:.3f}  wrist_in_recv_bbox={wrist_in_receiver}",
                        f"t={ev['time_str']}  frame={fn}",
                    ]
                    y0 = 45
                    cv2.rectangle(canvas, (0, 0), (600, y0 + len(info)*28), (10, 10, 20), -1)
                    cv2.putText(canvas, "IMPACT FRAME", (10, 30), FONT, 0.75, (0, 215, 255), 2, cv2.LINE_AA)
                    for k, txt in enumerate(info):
                        col_t = (0, 255, 100) if k == 0 else (200, 200, 200)
                        cv2.putText(canvas, txt, (10, y0 + k*28 + 22), FONT, 0.50, col_t, 1, cv2.LINE_AA)

                    print(f"  Impact #{ev_idx+1} | frame {fn} | P{sid}->P{rid} | "
                          f"d={ev['d_3d_metres']:.3f}m | IoU={iou:.3f} | wrist_in_recv={wrist_in_receiver}")

        # Frame label for non-impact context frames
        any_impact = any(d == 0 for _, d in frame_targets[fn])
        if not any_impact:
            ev_idx, delta = frame_targets[fn][0]
            ev = events[ev_idx]
            cv2.rectangle(canvas, (0, 0), (380, 40), (10, 10, 20), -1)
            cv2.putText(canvas, f"Event #{ev_idx+1} | frame {fn} (delta={delta:+d}) | t={ev['time_str']}",
                        (8, 26), FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Save
        ev_idx_main = frame_targets[fn][0][0]
        delta_main  = frame_targets[fn][0][1]
        fname = f"ev{ev_idx_main+1:02d}_fr{fn:05d}_d{delta_main:+d}.jpg"
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    cap.release()

    # ── Coordinate system sanity check ─────────────────────────────────────────
    print()
    print("=== Coordinate System Sanity Check ===")
    for idx, ev in enumerate(events):
        fn   = ev["impact_frame"]
        sid  = ev["striker_id"]
        rid  = ev["receiver_id"]
        wk   = ev["wrist_kp"]
        e_s  = entries[sid].get(fn)
        e_r  = entries[rid].get(fn)
        if e_s is None or e_r is None:
            print(f"  Impact #{idx+1}: missing data at frame {fn}")
            continue

        s_coords = np.array(e_s["shared_space_coords"])
        r_coords = np.array(e_r["shared_space_coords"])

        # Project striker wrist
        wp   = project_3d(*s_coords[wk], fl, cx, cy)
        # Project receiver head kps
        h_projs = [project_3d(*r_coords[hk], fl, cx, cy) for hk in HEAD_KPS]
        h_projs = [p for p in h_projs if p is not None]

        # Bbox check
        dims_r = e_r.get("frame_dims", {})
        sr_x = dims_r.get("original_width", W) / dims_r.get("resized_width", W)
        sr_y = dims_r.get("original_height", H) / dims_r.get("resized_height", H)
        br = e_r["bbox"]
        bb_r = [br[0]*sr_x, br[1]*sr_y, br[2]*sr_x, br[3]*sr_y]

        wrist_in_recv = (wp and bb_r[0] <= wp[0] <= bb_r[2] and bb_r[1] <= wp[1] <= bb_r[3])

        # How many receiver head kps are inside receiver bbox?
        dims_s = e_s.get("frame_dims", {})
        ss_x = dims_s.get("original_width", W) / dims_s.get("resized_width", W)
        ss_y = dims_s.get("original_height", H) / dims_s.get("resized_height", H)
        bs = e_s["bbox"]
        bb_s = [bs[0]*ss_x, bs[1]*ss_y, bs[2]*ss_x, bs[3]*ss_y]
        iou = bbox_iou(bb_s, bb_r)

        # Z values
        wZ = s_coords[wk][2]
        rZ = np.mean([r_coords[k][2] for k in HEAD_KPS if not np.allclose(r_coords[k], 0)])

        print(f"  Impact #{idx+1} | frame {fn:5d} | P{sid}->P{rid} | "
              f"d={ev['d_3d_metres']:.3f}m | wrist_Z={wZ:.2f}m | recv_head_Z={rZ:.2f}m | "
              f"wrist_proj={wp} | wrist_in_recv={wrist_in_recv} | bbox_IoU={iou:.3f}")

    print()
    print(f"Saved {len(sorted_targets)} verification frames to:")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    main()
