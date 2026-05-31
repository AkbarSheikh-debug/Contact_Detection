"""
Render v5 - visually correct impact zones and timing.

Key improvements over render_final.py:
  - Crosshair/ring drawn at hit_xy (2D projected wrist->receiver keypoint)
    instead of arbitrary bbox center or head estimate
  - Visual effects centered on contact_frame (true min-distance frame)
    instead of the detection frame
  - Expanding ring + radial spikes at hit point for dramatic effect
  - Panel shows contact_region (head/torso/arm) and correct speed/audio values
  - Screen flash correctly peaks at actual contact moment
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, os, cv2, numpy as np, sys

FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
VIDEO_IN   = os.path.join(FOLDER, "3.mp4")
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")

DET_JSON  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FOLDER, "3_fusion_v5.json")
VIDEO_OUT = os.path.join(FOLDER, os.path.basename(DET_JSON).replace(".json","") + ".mp4")

GT_TS_ORIG = ["7:11","18:01","24:04","30:19","31:07","34:07","37:08","53:02","55:17",
              "1:05:22","1:06:09","1:06:20","1:20:14","1:25:15","1:26:05","1:27:18",
              "1:42:16","1:42:19","1:48:22","1:51:23","1:53:24","2:03:19","2:15:22",
              "2:17:11","2:25:17","2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16"]
FPS = 24.995
RELABEL_JSON_R = os.path.join(FOLDER, "relabeled_gt.json")

def to_f(ts):
    p = ts.split(":")
    s = int(p[0])+int(p[1])/FPS if len(p)==2 else int(p[0])*60+int(p[1])+int(p[2])/FPS
    return int(round(s*FPS))

# Load relabeled GT for evaluation (same as used in training)
if os.path.exists(RELABEL_JSON_R):
    with open(RELABEL_JSON_R) as _f:
        _rl = json.load(_f)
    GT_TS     = [r["suggested_ts"]    for r in _rl]
    GT_FRAMES = [r["suggested_frame"] for r in _rl]
else:
    GT_TS     = GT_TS_ORIG
    GT_FRAMES = [to_f(t) for t in GT_TS_ORIG]

GT_ORIG_FRAMES = [to_f(t) for t in GT_TS_ORIG]  # for timeline purple markers

with open(DET_JSON) as f: det_data = json.load(f)
detections = det_data["detections"]

with open(SAM3D_JSON) as f: sam3d = json.load(f)
fp = {}
for tid in ("0","1"):
    if tid in sam3d:
        for e in sam3d[tid]:
            fp.setdefault(e["frame"], {})[int(e["track_id"])] = e

def proj_kp(e, k):
    """Map normalized_coords keypoint k to image (u,v) using bbox as linear reference."""
    nc = e.get("normalized_coords")
    if nc is None or k >= len(nc): return None
    nc_arr = _np.array(nc)
    x1,y1,x2,y2 = e["bbox"]
    Y_min,Y_max = nc_arr[:,1].min(), nc_arr[:,1].max()
    X_min,X_max = nc_arr[:,0].min(), nc_arr[:,0].max()
    if Y_max<=Y_min or X_max<=X_min: return None
    u = x1 + (nc_arr[k,0]-X_min)/(X_max-X_min)*(x2-x1)
    v = y1 + (nc_arr[k,1]-Y_min)/(Y_max-Y_min)*(y2-y1)
    return (int(u), int(v))


# ── Tag detections TP/FP ──────────────────────────────────────────────────
def tag_dets(dets, gts, tol=30):
    cands = []
    for di, d in enumerate(dets):
        for gi, g in enumerate(gts):
            if abs(d["frame"]-g) <= tol:
                cands.append((abs(d["frame"]-g), di, gi))
    cands.sort()
    md, mg = set(), set(); pair = {}
    for dist, di, gi in cands:
        if di in md or gi in mg: continue
        md.add(di); mg.add(gi); pair[di] = gi
    tags = []
    for di, d in enumerate(dets):
        tags.append(("TP", GT_TS[pair[di]]) if di in pair else ("FP", None))
    missed = [GT_TS[gi] for gi in range(len(gts)) if gi not in mg]
    return tags, missed

tags, missed = tag_dets(detections, GT_FRAMES)
n_tp = sum(1 for t,_ in tags if t=="TP")
n_fp = len(detections) - n_tp
n_fn = len(missed)
P  = n_tp/(n_tp+n_fp) if (n_tp+n_fp) else 0
R  = n_tp/(n_tp+n_fn) if (n_tp+n_fn) else 0
F1 = 2*P*R/(P+R) if (P+R) else 0
print(f"Detections: {len(detections)}  TP={n_tp} FP={n_fp} FN={n_fn}  P={P:.2f} R={R:.2f} F1={F1:.2f}")


# ── Event timing: anchor visual effects on contact_frame ──────────────────
# EVENT_LEAD: start effects this many frames before contact_frame
# EVENT_LINGER: continue this many frames after contact_frame
EVENT_LEAD   = 1
EVENT_LINGER = 28

events_by_frame = {}
for di, d in enumerate(detections):
    tag, gt_label = tags[di]
    cf = d.get("contact_frame", d["frame"])   # true contact frame
    for df in range(-EVENT_LEAD, EVENT_LINGER):
        events_by_frame.setdefault(cf+df, []).append((d, df, tag, gt_label))

gt_by_frame = {}
for gi, gf in enumerate(GT_FRAMES):
    for df in range(EVENT_LINGER):
        gt_by_frame.setdefault(gf+df, []).append((gi, df, GT_TS[gi]))


# ── Video I/O ─────────────────────────────────────────────────────────────
cap   = cv2.VideoCapture(VIDEO_IN)
fps   = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out   = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (W, H))
if not out.isOpened():
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out    = cv2.VideoWriter(VIDEO_OUT.replace(".mp4",".avi"), fourcc, fps, (W, H))


# ── Color palette ─────────────────────────────────────────────────────────
COL_P0 = (60,  255, 80)    # green: fighter 0 bbox
COL_P1 = (60,  160, 255)   # orange: fighter 1 bbox
COL_TP = (40,  220, 40)    # green flash: true positive
COL_FP = (30,  80,  255)   # red flash: false positive
COL_GT = (180, 60,  220)   # purple: GT marker in timeline


# ── Drawing helpers ───────────────────────────────────────────────────────
def put_bg(img, txt, xy, scale=0.65, fg=(255,255,255), bg=(15,15,15), th=1):
    x, y = xy
    (tw, h), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    cv2.rectangle(img, (x-4, y-h-5), (x+tw+4, y+bl+3), bg, -1)
    cv2.putText(img, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, th, cv2.LINE_AA)

def draw_bbox(img, bbox, color, label):
    x1,y1,x2,y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    put_bg(img, label, (x1, y1-6), scale=0.6, fg=color, th=2)

def draw_screen_flash(img, color, a):
    o = img.copy(); o[:] = color
    cv2.addWeighted(o, a, img, 1-a, 0, img)

def draw_banner(img, txt, color, age, max_age):
    bh = 100
    a = max(0.3, 1.0 - age/max_age)
    o = img.copy()
    cv2.rectangle(o, (0,0), (W,bh), color, -1)
    cv2.addWeighted(o, a*0.75, img, 1-a*0.75, 0, img)
    cv2.rectangle(img, (0,bh), (W,bh+4), color, -1)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)
    x = (W-tw)//2; y = bh//2 + th//2 - 4
    cv2.putText(img, txt, (x+3,y+3), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0,0,0), 5, cv2.LINE_AA)
    cv2.putText(img, txt, (x,y),     cv2.FONT_HERSHEY_DUPLEX, 1.6, (255,255,255), 3, cv2.LINE_AA)

def draw_event_panel(img, d, tag, gt_label, x, y):
    color = COL_TP if tag == "TP" else COL_FP
    region = d.get("contact_region","?")
    lines = [
        (f"{tag}  {d['action'].upper().replace('_',' ')}", color, 0.70, 2),
        (f"Fighter: P{d['sid']}", (255,255,255), 0.55, 1),
        (f"Score: {d.get('score',0):.2f}", (255,255,255), 0.55, 1),
        (f"Speed: {d.get('speed_kmh',0):.1f} km/h", (200,255,200), 0.55, 1),
        (f"Audio: {d.get('audio',0):.3f}", (200,200,255), 0.50, 1),
        (f"Contact: {d.get('ce',0):.3f}", (200,200,255), 0.50, 1),
        (f"Region: {region}", (255,220,160), 0.50, 1),
        (f"Wrist-px: {d.get('dist_px',-1):.0f}", (200,200,255), 0.50, 1),
        (f"Det frame: {d['frame']}", (160,160,160), 0.48, 1),
        (f"Hit frame: {d.get('contact_frame', d['frame'])}", (180,220,180), 0.48, 1),
    ]
    if gt_label:
        lines.append((f"matches GT: {gt_label}", (180,255,180), 0.50, 1))
    w = 340; lh = 24; h = len(lines)*lh + 14
    panel = img[y:y+h, x:x+w]
    if panel.size == 0: return
    o = np.full(panel.shape, 18, dtype=np.uint8)
    cv2.addWeighted(o, 0.8, panel, 0.2, 0, panel)
    cv2.rectangle(img, (x,y), (x+w-1, y+h-1), color, 2)
    cv2.rectangle(img, (x,y), (x+6, y+h-1), color, -1)
    cy = y + 22
    for t, c, s, th in lines:
        cv2.putText(img, t, (x+14, cy), cv2.FONT_HERSHEY_SIMPLEX, s, (0,0,0), th+2, cv2.LINE_AA)
        cv2.putText(img, t, (x+14, cy), cv2.FONT_HERSHEY_SIMPLEX, s, c, th, cv2.LINE_AA)
        cy += lh

def draw_impact_at_point(img, hit_xy, color, age, max_age=EVENT_LEAD+EVENT_LINGER):
    """Draw expanding impact ring + crosshair at the exact hit point."""
    if hit_xy is None: return
    cx, cy = int(hit_xy[0]), int(hit_xy[1])

    # Fade alpha: peaks at age=0 (contact_frame), fades out
    t = max(0, age) / max_age          # 0 -> 1 over lifetime
    alpha = max(0.0, 1.0 - t*1.2)

    # Expanding outer ring
    base_r = 18
    ring_r = base_r + int(age * 3.5) if age >= 0 else base_r
    ring_r = max(base_r, min(ring_r, 120))
    thickness = max(1, int(4 * alpha))
    if thickness > 0 and ring_r > 0:
        cv2.circle(img, (cx,cy), ring_r, color, thickness, cv2.LINE_AA)

    # Inner solid circle that shrinks (most prominent right at impact)
    inner_r = max(0, base_r - int(max(0,age)*2))
    if inner_r > 0:
        overlay = img.copy()
        cv2.circle(overlay, (cx,cy), inner_r, color, -1)
        cv2.addWeighted(overlay, alpha*0.55, img, 1-alpha*0.55, 0, img)

    # Crosshair lines
    arm = ring_r + 15
    lw  = max(1, int(3 * alpha))
    cv2.line(img, (cx-arm, cy), (cx-ring_r-2, cy), color, lw, cv2.LINE_AA)
    cv2.line(img, (cx+ring_r+2, cy), (cx+arm, cy), color, lw, cv2.LINE_AA)
    cv2.line(img, (cx, cy-arm), (cx, cy-ring_r-2), color, lw, cv2.LINE_AA)
    cv2.line(img, (cx, cy+ring_r+2), (cx, cy+arm), color, lw, cv2.LINE_AA)

    # Radial spikes (only at impact moment ±3 frames)
    if -3 <= age <= 5:
        n_spikes = 8
        spike_len = int(60 * alpha)
        for i in range(n_spikes):
            angle = i * (2*np.pi/n_spikes)
            x0 = cx + int(ring_r * np.cos(angle))
            y0 = cy + int(ring_r * np.sin(angle))
            x1 = cx + int((ring_r+spike_len) * np.cos(angle))
            y1 = cy + int((ring_r+spike_len) * np.sin(angle))
            cv2.line(img, (x0,y0), (x1,y1), color, max(1,lw-1), cv2.LINE_AA)

    # Label at hit point (only early frames)
    if age <= 8:
        region = ""   # caller can pass if needed
        put_bg(img, "IMPACT", (cx+ring_r+5, cy+6), scale=0.55, fg=color, th=2)

def draw_timeline(img, fn):
    bar_y = H-30; x1, x2 = 30, W-30; bw = x2-x1
    cv2.rectangle(img, (x1, bar_y-6), (x2, bar_y+6), (50,50,50), -1)
    # Original GT markers (purple, small ticks)
    for gf in GT_ORIG_FRAMES:
        mx = x1 + int(gf/(total-1)*bw)
        cv2.rectangle(img, (mx-1, bar_y-8), (mx+1, bar_y+8), COL_GT, -1)
    # Relabeled GT markers (cyan, slightly larger)
    for gf in GT_FRAMES:
        mx = x1 + int(gf/(total-1)*bw)
        cv2.rectangle(img, (mx-2, bar_y-12), (mx+2, bar_y+12), (200,220,60), -1)
    # Detection markers
    for di, d in enumerate(detections):
        cf = d.get("contact_frame", d["frame"])
        mx = x1 + int(cf/(total-1)*bw)
        col = COL_TP if tags[di][0]=="TP" else COL_FP
        cv2.rectangle(img, (mx-3, bar_y-10), (mx+3, bar_y+10), col, -1)
    px = x1 + int(fn/(total-1)*bw)
    cv2.rectangle(img, (px-2, bar_y-16), (px+2, bar_y+16), (255,255,255), -1)
    put_bg(img, "purple=origGT  yellow=relabledGT  green=TP  red=FP",
           (x1, bar_y-22), scale=0.42, fg=(200,200,200))


# ── Main render loop ───────────────────────────────────────────────────────
print(f"Rendering {total} frames -> {VIDEO_OUT}")
fn = 0
last_event = None

while True:
    ret, frame = cap.read()
    if not ret: break

    # Draw person bboxes
    persons = fp.get(fn, {})
    for tid, e in persons.items():
        color = COL_P0 if tid == 0 else COL_P1
        draw_bbox(frame, e["bbox"], color, f"P{tid}")

    # GT marker near current frame (show both original and relabeled labels)
    for gi, gf in enumerate(GT_FRAMES):
        if abs(fn - gf) < 8:
            orig_label = GT_TS_ORIG[gi] if gi < len(GT_TS_ORIG) else "?"
            put_bg(frame, f"GT: {orig_label} -> {GT_TS[gi]}", (W-380, 70), scale=0.55, fg=(200,220,60), th=2)

    events = events_by_frame.get(fn, [])
    if events:
        d, age, tag, gt_label = events[0]
        color = COL_TP if tag == "TP" else COL_FP
        hit_xy = d.get("hit_xy")

        # Screen flash: peak at age=0 (contact_frame), fade quickly
        if -2 <= age <= 12:
            d_from_peak = abs(age)
            alpha = max(0.0, 0.38 * (1.0 - d_from_peak / 10.0))
            draw_screen_flash(frame, color, alpha)

        # Banner: visible from lead through post
        if -EVENT_LEAD <= age < 20:
            label = f"{'IMPACT' if tag=='TP' else 'DETECT'}  -  {d['action'].upper().replace('_',' ')}"
            ba = max(0, age + EVENT_LEAD)
            draw_banner(frame, label, color, ba, EVENT_LEAD + 20)

        # Impact ring at actual hit point
        if -EVENT_LEAD <= age < EVENT_LINGER:
            draw_impact_at_point(frame, hit_xy, color, age)

        last_event = (d, tag, gt_label)

    if last_event:
        draw_event_panel(frame, *last_event, x=20, y=110)

    # Frame counter + metrics
    t_str = f"{int(fn/fps//60):02d}:{(fn/fps)%60:05.2f}"
    put_bg(frame, f"Frame {fn}/{total}  t={t_str}", (W-380, 30), scale=0.65, th=2)
    put_bg(frame, f"P={P:.2f} R={R:.2f} F1={F1:.2f}  TP={n_tp} FP={n_fp} FN={n_fn}",
           (W-580, H-60), scale=0.6, fg=(200,255,200))

    draw_timeline(frame, fn)
    out.write(frame)
    fn += 1
    if fn % 500 == 0: print(f"  frame {fn}/{total}")

cap.release(); out.release()
print(f"\nDONE -> {VIDEO_OUT}")
print(f"  P={P:.2f}  R={R:.2f}  F1={F1:.2f}  TP={n_tp}  FP={n_fp}  FN={n_fn}")
if missed:
    print(f"  Missed GTs: {', '.join(missed)}")
