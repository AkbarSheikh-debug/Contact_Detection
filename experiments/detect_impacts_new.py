"""
Boxing Impact Detection — uses BOTH JSON files to full potential:
  - contact_events  -> landed / blocked impacts (red flash + banner)
  - world_coords    -> per-person 3D motion
  - keypoint_conf   -> skeleton drawn with confidence shading
  - actions JSON    -> punch type, speed, power, force; live sidebar timeline
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import cv2
import numpy as np
import os
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
FOLDER       = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
VIDEO_IN     = os.path.join(FOLDER, "3.mp4")
SAM3D_JSON   = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")
ACTION_JSON  = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")
VIDEO_OUT    = os.path.join(FOLDER, "3_impacts_detected.mp4")

# ── Detection thresholds ───────────────────────────────────────────────────────
LANDED_REGIONS   = {"head", "torso"}
BLOCKED_REGIONS  = {"left_arm", "right_arm"}
MIN_PROB_LANDED  = 0.30
MIN_PROB_BLOCKED = 0.60
MAX_DIST_M       = 0.15
COOLDOWN_FRAMES  = 15
ACTION_MATCH_WIN = 15

# ── Visual settings ────────────────────────────────────────────────────────────
FLASH_FRAMES     = 25         # screen-wide flash duration
BANNER_FRAMES    = 35         # how long the "IMPACT!" banner stays
PERSISTENT_LAST  = True       # always show last impact info

# Colours (BGR)
COL_P0       = (60,  255, 80)      # bright green
COL_P1       = (60,  160, 255)     # bright orange
COL_LANDED   = (0,   0,   255)     # vivid red
COL_BLOCKED  = (0,   220, 255)     # yellow
COL_TEXT_BG  = (15,  15,  15)
COL_WHITE    = (255, 255, 255)
COL_PANEL    = (25,  25,  30)

# COCO skeleton (first 17 of MHR keypoints follow COCO body convention)
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),               # head
    (5,6),(5,7),(7,9),(6,8),(8,10),        # arms
    (5,11),(6,12),(11,12),                 # torso
    (11,13),(13,15),(12,14),(14,16),       # legs
]
KP_NAMES = ["nose","l_eye","r_eye","l_ear","r_ear",
            "l_sh","r_sh","l_elb","r_elb","l_wr","r_wr",
            "l_hip","r_hip","l_knee","r_knee","l_ank","r_ank"]


# ── Load JSON data ─────────────────────────────────────────────────────────────
print("Loading JSON files...")
with open(SAM3D_JSON) as f:
    sam3d = json.load(f)
with open(ACTION_JSON) as f:
    act_data = json.load(f)

contact_events = sam3d.get("contact_events", [])
actions        = act_data.get("actions", [])

# Build frame -> {track_id -> entry}
frame_persons = {}
for tid_key in ("0", "1"):
    if tid_key not in sam3d:
        continue
    for entry in sam3d[tid_key]:
        fn  = entry["frame"]
        tid = int(entry["track_id"])
        frame_persons.setdefault(fn, {})[tid] = entry

print(f"  Frames with persons: {len(frame_persons)}")
print(f"  Contact events: {len(contact_events)}")
print(f"  Actions: {len(actions)}")


# ── Project 3D keypoints to 2D pixel coords using focal_length ────────────────
def project_kp(entry, kp_idx, W, H):
    """Project a 3D keypoint (world_coords) to pixel coordinates."""
    wc = entry.get("world_coords")
    if wc is None or kp_idx >= len(wc):
        return None
    X, Y, Z = wc[kp_idx]
    fl = entry.get("focal_length", 1500.0)
    if Z <= 0.1:
        return None
    cx, cy = W / 2.0, H / 2.0
    u = fl * X / Z + cx
    v = fl * Y / Z + cy
    if not (0 <= u < W and 0 <= v < H):
        return None
    return (int(u), int(v))


def project_normalized(entry, kp_idx, W, H):
    """Fallback: project normalized_coords using pred_cam_t."""
    nc = entry.get("normalized_coords")
    if nc is None or kp_idx >= len(nc):
        return None
    pred_cam = entry.get("pred_cam_t", [0, 0, 5])
    fl = entry.get("focal_length", 1500.0)
    x, y, z = nc[kp_idx]
    X = x + pred_cam[0]
    Y = y + pred_cam[1]
    Z = z + pred_cam[2]
    if Z <= 0.1:
        return None
    cx, cy = W / 2.0, H / 2.0
    u = fl * X / Z + cx
    v = fl * Y / Z + cy
    if not (0 <= u < W and 0 <= v < H):
        return None
    return (int(u), int(v))


# ── Filter & deduplicate contact events ───────────────────────────────────────
def filter_events(events):
    landed, blocked = [], []
    last_landed  = {}
    last_blocked = {}

    for e in sorted(events, key=lambda x: x["frame"]):
        fn     = e["frame"]
        region = e["contact_region"]
        prob   = e["contact_prob"]
        dist   = e["contact_3d_distance_m"]
        key    = (e["striker_id"], e["receiver_id"])

        if region in LANDED_REGIONS and prob >= MIN_PROB_LANDED and dist <= MAX_DIST_M:
            if fn - last_landed.get(key, -9999) >= COOLDOWN_FRAMES:
                landed.append(e)
                last_landed[key] = fn
        elif region in BLOCKED_REGIONS and prob >= MIN_PROB_BLOCKED and dist <= MAX_DIST_M:
            if fn - last_blocked.get(key, -9999) >= COOLDOWN_FRAMES:
                blocked.append(e)
                last_blocked[key] = fn

    return landed, blocked


landed_events, blocked_events = filter_events(contact_events)
print(f"\nDetected {len(landed_events)} LANDED impacts, {len(blocked_events)} BLOCKED punches\n")


# ── Match contact events with action JSON ─────────────────────────────────────
def match_action(contact_frame, striker_id):
    fighter_key = f"fighter_{striker_id}"
    best, best_d = None, ACTION_MATCH_WIN + 1
    for a in actions:
        if a["fighter_type"] != fighter_key:
            continue
        d = abs(a["frame"] - contact_frame)
        if d < best_d:
            best_d, best = d, a
    return best


# ── Per-fighter cumulative stats ──────────────────────────────────────────────
stats = {0: {"landed": 0, "blocked": 0, "total_pwr": 0.0, "max_spd": 0.0},
         1: {"landed": 0, "blocked": 0, "total_pwr": 0.0, "max_spd": 0.0}}


# ── Build event lookups by frame ──────────────────────────────────────────────
# active_events: frame -> [(event, is_landed, action, age)]
active_events = defaultdict(list)
for e in landed_events:
    a = match_action(e["frame"], e["striker_id"])
    for df in range(BANNER_FRAMES):
        active_events[e["frame"] + df].append((e, True, a))
for e in blocked_events:
    a = match_action(e["frame"], e["striker_id"])
    for df in range(BANNER_FRAMES):
        active_events[e["frame"] + df].append((e, False, a))

# action timeline: frame -> [(action, age)]  active for action_match_win frames
active_actions = defaultdict(list)
for a in actions:
    for fn in range(a["window_start"], a["window_end"] + 1):
        active_actions[fn].append(a)


# ── Pre-print summary ─────────────────────────────────────────────────────────
print("=" * 80)
print("LANDED IMPACTS:")
print("=" * 80)
for e in landed_events:
    a = match_action(e["frame"], e["striker_id"])
    spd = f"{a['speed_estimation']['estimated_speed_kmh']:5.1f} km/h" if a else "    ?    "
    pwr = f"{a['power_estimation']['estimated_power_watts']:7.0f}W"    if a else "    ?  "
    act = a["action"].replace("_", " ")                                if a else "punch"
    print(f"  F{e['frame']:4d}  t={e['time_sec']:6.2f}s  P{e['striker_id']}->P{e['receiver_id']} "
          f"{e['striker_body_part']:11s}->{e['contact_region']:6s}  "
          f"prob={e['contact_prob']:.2f}  d={e['contact_3d_distance_m']:.3f}m  "
          f"{act:14s}  {spd}  {pwr}")

print()
print("=" * 80)
print("BLOCKED PUNCHES:")
print("=" * 80)
for e in blocked_events:
    a = match_action(e["frame"], e["striker_id"])
    print(f"  F{e['frame']:4d}  t={e['time_sec']:6.2f}s  P{e['striker_id']}->P{e['receiver_id']} blocked  "
          f"prob={e['contact_prob']:.2f}  d={e['contact_3d_distance_m']:.3f}m")


# ── Drawing helpers ───────────────────────────────────────────────────────────
def put_text_bg(img, text, xy, scale=0.6, thickness=1, fg=COL_WHITE, bg=COL_TEXT_BG, pad=4):
    x, y = xy
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


def draw_bbox(img, bbox, color, label):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    # corner accents
    L = 18
    for (cx, cy) in [(x1,y1), (x2,y1), (x1,y2), (x2,y2)]:
        dx = L if cx == x1 else -L
        dy = L if cy == y1 else -L
        cv2.line(img, (cx, cy), (cx+dx, cy), color, 5)
        cv2.line(img, (cx, cy), (cx, cy+dy), color, 5)
    put_text_bg(img, label, (x1, y1 - 8), scale=0.75, thickness=2, fg=color, bg=(0,0,0))


def draw_skeleton(img, entry, color, W, H):
    """Draw 17-keypoint skeleton using world_coords projected to 2D."""
    pts = []
    confs = entry.get("keypoint_conf", [1.0] * 70)
    for k in range(17):
        p = project_kp(entry, k, W, H)
        if p is None:
            p = project_normalized(entry, k, W, H)
        pts.append(p)

    # bones
    for (a, b) in SKELETON:
        if pts[a] and pts[b]:
            c = min(confs[a] if a < len(confs) else 1.0,
                    confs[b] if b < len(confs) else 1.0)
            thickness = 2 if c > 0.5 else 1
            cv2.line(img, pts[a], pts[b], color, thickness, cv2.LINE_AA)
    # joints
    for k, p in enumerate(pts):
        if p:
            c = confs[k] if k < len(confs) else 1.0
            r = 4 if c > 0.7 else 3
            cv2.circle(img, p, r, color, -1)
            cv2.circle(img, p, r + 1, (0, 0, 0), 1)


def draw_screen_flash(img, color, alpha):
    overlay = img.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_screen_border(img, color, thickness):
    H, W = img.shape[:2]
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), color, thickness)


def draw_impact_banner(img, text, color, age, max_age):
    """Big top-of-screen IMPACT banner."""
    H, W = img.shape[:2]
    # ease-out alpha
    alpha = max(0.3, 1.0 - age / max_age)
    bh = 90
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, bh), color, -1)
    cv2.addWeighted(overlay, alpha * 0.75, img, 1 - alpha * 0.75, 0, img)
    cv2.rectangle(img, (0, bh), (W, bh + 3), color, -1)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)
    x = (W - tw) // 2
    y = bh // 2 + th // 2 - 4
    # shadow
    cv2.putText(img, text, (x + 3, y + 3), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),         cv2.FONT_HERSHEY_DUPLEX, 1.8, COL_WHITE, 3, cv2.LINE_AA)


def draw_receiver_target(img, bbox, color, age, max_age):
    """Draw expanding crosshair on the receiver during impact."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r_base = min(x2 - x1, y2 - y1) // 3
    r = int(r_base + age * 4)
    cv2.circle(img, (cx, cy), r, color, 3)
    cv2.circle(img, (cx, cy), r // 2, color, 2)
    cv2.line(img, (cx - r - 8, cy), (cx + r + 8, cy), color, 2)
    cv2.line(img, (cx, cy - r - 8), (cx, cy + r + 8), color, 2)


def draw_event_panel(img, event, action, is_landed, x, y, w=380):
    """Detailed event info panel."""
    color = COL_LANDED if is_landed else COL_BLOCKED
    label = "LANDED" if is_landed else "BLOCKED"
    lines = []
    lines.append((f"{label}: {event['contact_region'].replace('_',' ').upper()}", color, 0.75, 2))
    a = action
    if a:
        lines.append((f"Action: {a['action'].replace('_',' ').upper()}", COL_WHITE, 0.62, 1))
        lines.append((f"Speed:  {a['speed_estimation']['estimated_speed_kmh']:5.1f} km/h", (200,255,200), 0.58, 1))
        lines.append((f"Power:  {a['power_estimation']['estimated_power_watts']:7.0f} W", (200,200,255), 0.58, 1))
        lines.append((f"Force:  {a['force_estimation']['estimated_force_newtons']:6.0f} N", (255,220,200), 0.58, 1))
    lines.append((f"Striker: P{event['striker_id']}  ({event['striker_body_part']})", COL_WHITE, 0.55, 1))
    lines.append((f"Contact prob: {event['contact_prob']:.2f}", COL_WHITE, 0.55, 1))
    lines.append((f"3D distance:  {event['contact_3d_distance_m']:.3f} m", COL_WHITE, 0.55, 1))
    lines.append((f"Frame: {event['frame']}   t={event['time_sec']:.2f}s", (180,180,180), 0.50, 1))

    line_h = 26
    panel_h = len(lines) * line_h + 18
    panel = img[y:y+panel_h, x:x+w]
    if panel.size == 0:
        return
    overlay = np.full(panel.shape, 20, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.78, panel, 0.22, 0, panel)
    cv2.rectangle(img, (x, y), (x + w - 1, y + panel_h - 1), color, 2)
    cv2.rectangle(img, (x, y), (x + 6, y + panel_h - 1), color, -1)

    cy = y + 24
    for text, col, sc, th in lines:
        cv2.putText(img, text, (x + 16, cy), cv2.FONT_HERSHEY_SIMPLEX, sc, (0,0,0), th + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x + 16, cy), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
        cy += line_h


def draw_live_action(img, fn, x, y, w=300):
    """Show currently active action predictions from action JSON."""
    acts = active_actions.get(fn, [])
    if not acts:
        return
    h = 24 + len(acts) * 22 + 6
    panel = img[y:y+h, x:x+w]
    if panel.size == 0:
        return
    overlay = np.full(panel.shape, 30, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.7, panel, 0.3, 0, panel)
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (120, 120, 200), 1)
    cv2.putText(img, "LIVE ACTIONS", (x + 8, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1, cv2.LINE_AA)
    cy = y + 38
    for a in acts[:5]:
        col = COL_P0 if a["fighter_type"] == "fighter_0" else COL_P1
        txt = f"P{a['fighter_type'][-1]}: {a['action'].replace('_',' ')}  ({a['confidence']:.2f})"
        cv2.putText(img, txt, (x + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
        cy += 22


def draw_score_panel(img, x, y):
    """Cumulative fighter stats panel."""
    w, h = 320, 110
    panel = img[y:y+h, x:x+w]
    if panel.size == 0:
        return
    overlay = np.full(panel.shape, 25, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.8, panel, 0.2, 0, panel)
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (180, 180, 180), 1)
    cv2.putText(img, "FIGHTER STATS", (x + 8, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1, cv2.LINE_AA)

    for i, tid in enumerate([0, 1]):
        col = COL_P0 if tid == 0 else COL_P1
        s = stats[tid]
        cy = y + 50 + i * 28
        txt = f"P{tid}:  landed {s['landed']:2d}   blocked {s['blocked']:2d}   max {s['max_spd']:5.1f} km/h"
        cv2.putText(img, txt, (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def draw_timeline(img, total, current, landed, blocked):
    H, W = img.shape[:2]
    bar_y = H - 30
    x1, x2 = 30, W - 30
    bw = x2 - x1
    cv2.rectangle(img, (x1, bar_y - 6), (x2, bar_y + 6), (45, 45, 45), -1)
    cv2.rectangle(img, (x1, bar_y - 6), (x2, bar_y + 6), (90, 90, 90), 1)
    # markers
    for e in blocked:
        mx = x1 + int(e["frame"] / max(total - 1, 1) * bw)
        cv2.rectangle(img, (mx - 2, bar_y - 8), (mx + 2, bar_y + 8), COL_BLOCKED, -1)
    for e in landed:
        mx = x1 + int(e["frame"] / max(total - 1, 1) * bw)
        cv2.rectangle(img, (mx - 3, bar_y - 10), (mx + 3, bar_y + 10), COL_LANDED, -1)
    # playhead
    px = x1 + int(current / max(total - 1, 1) * bw)
    cv2.rectangle(img, (px - 2, bar_y - 14), (px + 2, bar_y + 14), COL_WHITE, -1)
    put_text_bg(img, "TIMELINE  (red=landed, yellow=blocked)",
                (x1, bar_y - 22), scale=0.45, fg=(200, 200, 200))


# ── Open video ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_IN)
fps   = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\nVideo: {W}x{H}  fps={fps:.2f}  frames={total}")
print(f"Rendering -> {VIDEO_OUT}\n")

# Try H.264 first, fall back to mp4v
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (W, H))
if not out.isOpened():
    print("  avc1 codec unavailable, falling back to mp4v")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (W, H))

# Track last impact for persistent panel
last_impact = None    # (event, is_landed, action, frame_seen)

# Pre-mark landed/blocked frames to update stats once
counted = set()

fn = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    persons = frame_persons.get(fn, {})

    # ── Person bboxes + skeleton ─────────────────────────────────────────────
    for tid, entry in persons.items():
        color = COL_P0 if tid == 0 else COL_P1
        draw_bbox(frame, entry["bbox"], color, f"FIGHTER {tid}")
        draw_skeleton(frame, entry, color, W, H)

    # ── Active impact events ─────────────────────────────────────────────────
    events_here = active_events.get(fn, [])
    fresh = [(e, l, a) for (e, l, a) in events_here if e["frame"] == fn]

    # Update stats once per event
    for (e, is_landed, a) in fresh:
        ek = (e["frame"], e["striker_id"], e["receiver_id"], e["contact_region"])
        if ek not in counted:
            counted.add(ek)
            sid = e["striker_id"]
            if is_landed:
                stats[sid]["landed"] += 1
            else:
                stats[sid]["blocked"] += 1
            if a:
                stats[sid]["total_pwr"] += a["power_estimation"]["estimated_power_watts"]
                stats[sid]["max_spd"] = max(stats[sid]["max_spd"],
                                            a["speed_estimation"]["estimated_speed_kmh"])
            last_impact = (e, is_landed, a, fn)

    # Process each active event for visual effects
    for (e, is_landed, a) in events_here:
        age = fn - e["frame"]
        color = COL_LANDED if is_landed else COL_BLOCKED
        recv_entry = persons.get(e["receiver_id"])
        recv_bbox  = recv_entry["bbox"] if recv_entry else None

        # screen flash (only landed, only at start)
        if is_landed and age < FLASH_FRAMES:
            alpha = 0.35 * (1.0 - age / FLASH_FRAMES) ** 2
            draw_screen_flash(frame, color, alpha)

        # crosshair on receiver
        if recv_bbox is not None and age < 18:
            draw_receiver_target(frame, recv_bbox, color, age, 18)

    # Big banner: prefer landed over blocked
    banner_event = None
    for (e, is_landed, a) in events_here:
        age = fn - e["frame"]
        if age < BANNER_FRAMES:
            if banner_event is None or (is_landed and not banner_event[1]):
                banner_event = (e, is_landed, a, age)
    if banner_event:
        e, is_landed, a, age = banner_event
        color = COL_LANDED if is_landed else COL_BLOCKED
        label = "IMPACT LANDED" if is_landed else "PUNCH BLOCKED"
        if a:
            label += f"  -  {a['action'].replace('_',' ').upper()}"
        draw_impact_banner(frame, label, color, age, BANNER_FRAMES)
        # screen border pulse
        if age < 12:
            draw_screen_border(frame, color, 10 - age // 2)

    # ── Persistent last-impact panel ─────────────────────────────────────────
    if PERSISTENT_LAST and last_impact:
        e, is_landed, a, _ = last_impact
        draw_event_panel(frame, e, a, is_landed, x=20, y=110)

    # ── Live action sidebar (top right) ──────────────────────────────────────
    draw_live_action(frame, fn, x=W - 320, y=110)

    # ── Cumulative stats (right side, below live actions) ────────────────────
    draw_score_panel(frame, x=W - 340, y=H - 180)

    # ── Frame counter top right ──────────────────────────────────────────────
    t_str = f"{int(fn/fps//60):02d}:{(fn/fps)%60:05.2f}"
    put_text_bg(frame, f"Frame {fn}/{total}   t={t_str}",
                (W - 340, 30), scale=0.65, thickness=2)

    # ── Timeline bar ─────────────────────────────────────────────────────────
    draw_timeline(frame, total, fn, landed_events, blocked_events)

    out.write(frame)
    fn += 1
    if fn % 500 == 0:
        print(f"  frame {fn}/{total}")

cap.release()
out.release()
print(f"\nDONE. Output: {VIDEO_OUT}")
print(f"  Landed:  {len(landed_events)}    Blocked: {len(blocked_events)}")
print(f"  P0: landed={stats[0]['landed']}  blocked={stats[0]['blocked']}  max_spd={stats[0]['max_spd']:.1f} km/h")
print(f"  P1: landed={stats[1]['landed']}  blocked={stats[1]['blocked']}  max_spd={stats[1]['max_spd']:.1f} km/h")
