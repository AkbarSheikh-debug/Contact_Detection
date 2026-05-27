"""
Render impact-detection video using a selected detection JSON.
Highlights each detection with banner + receiver crosshair + persistent panel.
"""
import json, os, cv2, numpy as np

FOLDER     = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
VIDEO_IN   = os.path.join(FOLDER, "3.mp4")
SAM3D_JSON = os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json")

import sys
DET_JSON = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FOLDER, "3_fusion_v3.json")
VIDEO_OUT = os.path.join(FOLDER, os.path.basename(DET_JSON).replace(".json","") + ".mp4")

GT_TS = ["7:11","18:01","24:04","30:19","31:07","34:07","37:08","53:02","55:17",
         "1:05:22","1:06:09","1:06:20","1:20:14","1:25:15","1:26:05","1:27:18",
         "1:42:16","1:42:19","1:48:22","1:51:23","1:53:24","2:03:19","2:15:22",
         "2:17:11","2:25:17","2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16"]
FPS = 24.995
def to_f(ts):
    p=ts.split(":")
    s=int(p[0])+int(p[1])/FPS if len(p)==2 else int(p[0])*60+int(p[1])+int(p[2])/FPS
    return int(round(s*FPS))
GT_FRAMES = [to_f(t) for t in GT_TS]

with open(DET_JSON) as f:
    det_data = json.load(f)
detections = det_data["detections"]
metrics = det_data.get("metrics", {})

with open(SAM3D_JSON) as f:
    sam3d = json.load(f)
fp = {}
for tid in ("0","1"):
    if tid in sam3d:
        for e in sam3d[tid]:
            fp.setdefault(e["frame"], {})[int(e["track_id"])] = e

# Mark each detection as TP/FP vs GT
def tag_dets(dets, gts, tol=30):
    cands = []
    for di,d in enumerate(dets):
        for gi,g in enumerate(gts):
            if abs(d["frame"]-g) <= tol:
                cands.append((abs(d["frame"]-g), di, gi))
    cands.sort()
    md, mg = set(), set()
    pair = {}
    for d,di,gi in cands:
        if di in md or gi in mg: continue
        md.add(di); mg.add(gi); pair[di] = gi
    tags = []
    for di,d in enumerate(dets):
        if di in pair:
            tags.append(("TP", GT_TS[pair[di]]))
        else:
            tags.append(("FP", None))
    missed = [GT_TS[gi] for gi in range(len(gts)) if gi not in mg]
    return tags, missed

tags, missed = tag_dets(detections, GT_FRAMES)
n_tp = sum(1 for t,_ in tags if t == "TP")
n_fp = len(detections) - n_tp
n_fn = len(missed)
P = n_tp/(n_tp+n_fp) if (n_tp+n_fp) else 0
R = n_tp/(n_tp+n_fn) if (n_tp+n_fn) else 0
F1 = 2*P*R/(P+R) if (P+R) else 0
print(f"Detections: {len(detections)}  TP={n_tp} FP={n_fp} FN={n_fn}  P={P:.2f} R={R:.2f} F1={F1:.2f}")

# Per-frame event index — start the flash 8 frames BEFORE the detection frame
# (the detection frame from action/audio/contact lands ~5–10 frames after the actual punch)
EVENT_LEAD   = 8     # start visual effects this many frames BEFORE detection
EVENT_LINGER = 30    # continue this many frames AFTER detection
events_by_frame = {}
for di, d in enumerate(detections):
    tag, gt_label = tags[di]
    for df in range(-EVENT_LEAD, EVENT_LINGER):
        # age is measured from the detection frame; negative = before
        events_by_frame.setdefault(d["frame"]+df, []).append((d, df, tag, gt_label))

# GT markers for visual ref
gt_by_frame = {}
for gi, gf in enumerate(GT_FRAMES):
    for df in range(EVENT_LINGER):
        gt_by_frame.setdefault(gf+df, []).append((gi, df, GT_TS[gi]))

cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (W, H))

COL_P0 = (60, 255, 80)
COL_P1 = (60, 160, 255)
COL_TP = (40, 220, 40)   # green
COL_FP = (40, 100, 255)  # red-orange
COL_GT = (180, 60, 220)  # purple for GT marker

last_event = None

def put_bg(img, txt, xy, scale=0.65, fg=(255,255,255), bg=(15,15,15), th=1):
    x,y = xy
    (tw,h),bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    cv2.rectangle(img, (x-4, y-h-5), (x+tw+4, y+bl+3), bg, -1)
    cv2.putText(img, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, th, cv2.LINE_AA)

def draw_bbox(img, bbox, color, label):
    x1,y1,x2,y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    put_bg(img, label, (x1, y1-6), scale=0.6, fg=color, th=2)

def draw_target(img, bbox, color, age, max_age=18):
    x1,y1,x2,y2 = [int(v) for v in bbox]
    cx, cy = (x1+x2)//2, (y1+y2)//2
    r = min(x2-x1, y2-y1)//3 + age*4
    cv2.circle(img, (cx,cy), r, color, 3)
    cv2.circle(img, (cx,cy), r//2, color, 2)

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
    cv2.putText(img, txt, (x+3, y+3), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0,0,0), 5, cv2.LINE_AA)
    cv2.putText(img, txt, (x,y),     cv2.FONT_HERSHEY_DUPLEX, 1.6, (255,255,255), 3, cv2.LINE_AA)

def draw_event_panel(img, d, tag, gt_label, x, y):
    color = COL_TP if tag == "TP" else COL_FP
    lines = [
        (f"{tag}  {d['action'].upper().replace('_',' ')}", color, 0.7, 2),
        (f"Fighter: P{d['sid']}", (255,255,255), 0.55, 1),
        (f"Score: {d.get('score', 0):.2f}", (255,255,255), 0.55, 1),
        (f"Speed: {d.get('speed_kmh', 0):.1f} km/h", (200,255,200), 0.55, 1),
        (f"Audio: {d.get('audio', 0):.2f}", (200,200,255), 0.50, 1),
        (f"Contact: {d.get('ce', 0):.2f}", (200,200,255), 0.50, 1),
        (f"Wrist-px: {d.get('dist_px', -1):.0f}", (200,200,255), 0.50, 1),
        (f"Frame: {d['frame']}", (160,160,160), 0.48, 1),
    ]
    if gt_label:
        lines.append((f"matches GT: {gt_label}", (180,255,180), 0.50, 1))
    w = 320; lh = 24; h = len(lines)*lh + 14
    panel = img[y:y+h, x:x+w]
    if panel.size == 0: return
    o = np.full(panel.shape, 18, dtype=np.uint8)
    cv2.addWeighted(o, 0.8, panel, 0.2, 0, panel)
    cv2.rectangle(img, (x,y), (x+w-1, y+h-1), color, 2)
    cv2.rectangle(img, (x,y), (x+6, y+h-1), color, -1)
    cy = y + 22
    for t,c,s,th in lines:
        cv2.putText(img, t, (x+14, cy), cv2.FONT_HERSHEY_SIMPLEX, s, (0,0,0), th+2, cv2.LINE_AA)
        cv2.putText(img, t, (x+14, cy), cv2.FONT_HERSHEY_SIMPLEX, s, c, th, cv2.LINE_AA)
        cy += lh

def draw_timeline(img, fn):
    bar_y = H - 30; x1, x2 = 30, W-30; bw = x2-x1
    cv2.rectangle(img, (x1, bar_y-6), (x2, bar_y+6), (50,50,50), -1)
    for gf in GT_FRAMES:
        mx = x1 + int(gf/(total-1)*bw)
        cv2.rectangle(img, (mx-2, bar_y-12), (mx+2, bar_y+12), COL_GT, -1)
    for di, d in enumerate(detections):
        mx = x1 + int(d["frame"]/(total-1)*bw)
        col = COL_TP if tags[di][0]=="TP" else COL_FP
        cv2.rectangle(img, (mx-3, bar_y-10), (mx+3, bar_y+10), col, -1)
    px = x1 + int(fn/(total-1)*bw)
    cv2.rectangle(img, (px-2, bar_y-16), (px+2, bar_y+16), (255,255,255), -1)
    put_bg(img, "purple=GT  green=TP  red=FP", (x1, bar_y-22), scale=0.42, fg=(200,200,200))

print(f"Rendering {total} frames -> {VIDEO_OUT}")
fn = 0
while True:
    ret, frame = cap.read()
    if not ret: break

    persons = fp.get(fn, {})
    for tid, e in persons.items():
        color = COL_P0 if tid == 0 else COL_P1
        draw_bbox(frame, e["bbox"], color, f"P{tid}")

    # Show GT marker (purple) for any GT within +/-15fr of current frame
    for gi, gf in enumerate(GT_FRAMES):
        if abs(fn - gf) < 8:
            put_bg(frame, f"GT: {GT_TS[gi]}", (W-260, 70), scale=0.6, fg=COL_GT, th=2)

    events = events_by_frame.get(fn, [])
    if events:
        d, age, tag, gt_label = events[0]   # age: -EVENT_LEAD .. EVENT_LINGER-1
        color = COL_TP if tag == "TP" else COL_FP
        sid = d["sid"]; rid = 1-sid
        recv = persons.get(rid)

        # Visual time relative to actual impact moment (5 frames before detection)
        # vt = 0 at expected impact, increases through the event
        vt = age + 5    # 0 .. EVENT_LEAD+EVENT_LINGER+4

        # Crosshair from start of lead through 20 frames after impact
        if recv is not None and vt >= 0 and vt < 25:
            draw_target(frame, recv["bbox"], color, vt)

        # Screen flash: peak at impact moment, fade out over 14 frames
        if -3 <= age <= 14:
            # alpha is highest near age=-3..2 (the actual punch moment)
            d_from_peak = abs(age + 1)   # peak at age=-1
            alpha = max(0.0, 0.35 * (1.0 - d_from_peak / 12))
            draw_screen_flash(frame, color, alpha)

        # Banner: visible the whole time the event is active
        if -EVENT_LEAD <= age < 22:
            label = f"{'IMPACT' if tag=='TP' else 'DETECT'}  -  {d['action'].upper().replace('_',' ')}"
            ba = max(0, age + EVENT_LEAD)
            draw_banner(frame, label, color, ba, EVENT_LEAD + 22)

        last_event = (d, tag, gt_label)

    if last_event:
        draw_event_panel(frame, *last_event, x=20, y=110)

    # frame counter
    t_str = f"{int(fn/fps//60):02d}:{(fn/fps)%60:05.2f}"
    put_bg(frame, f"Frame {fn}/{total}  t={t_str}", (W-360, 30), scale=0.65, th=2)
    put_bg(frame, f"P={P:.2f} R={R:.2f} F1={F1:.2f}  TP={n_tp} FP={n_fp} FN={n_fn}",
            (W-560, H-60), scale=0.6, fg=(200,255,200))

    draw_timeline(frame, fn)

    out.write(frame)
    fn += 1
    if fn % 500 == 0: print(f"  frame {fn}/{total}")

cap.release(); out.release()
print(f"\nDONE -> {VIDEO_OUT}")
print(f"  P={P:.2f}  R={R:.2f}  F1={F1:.2f}  TP={n_tp}  FP={n_fp}  FN={n_fn}")
if missed:
    print(f"  Missed GTs: {', '.join(missed)}")
