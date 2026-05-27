"""Sweep: what F1 can we achieve using ONLY the action JSON + cooldown?"""
import json, os
from itertools import product

FOLDER = r"C:\Users\XRIG\Downloads\sam3d_with_world_coords"
with open(os.path.join(FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d.json")) as f:
    actions = json.load(f)["actions"]

FPS = 24.995
GT = [
    "7:11","18:01","24:04","30:19","31:07","34:07","37:08",
    "53:02","55:17","1:05:22","1:06:09","1:06:20","1:20:14",
    "1:25:15","1:26:05","1:27:18","1:42:16","1:42:19","1:48:22",
    "1:51:23","1:53:24","2:03:19","2:15:22","2:17:11","2:25:17",
    "2:27:24","2:28:24","2:34:13","2:46:12","2:49:24","2:52:16",
]
def to_f(ts):
    p = ts.split(":")
    s = int(p[0]) + int(p[1])/FPS if len(p)==2 else int(p[0])*60 + int(p[1]) + int(p[2])/FPS
    return int(round(s*FPS))
gt_frames = [to_f(t) for t in GT]

def evaluate(det, gt, tol=30):
    mg, md = set(), set()
    for di, df in enumerate(det):
        best, bd = None, tol+1
        for gi, gf in enumerate(gt):
            if gi in mg: continue
            d = abs(df - gf)
            if d < bd: bd, best = d, gi
        if best is not None: mg.add(best); md.add(di)
    tp = len(mg); fp = len(det)-len(md); fn = len(gt)-tp
    p = tp/(tp+fp) if (tp+fp) else 0
    r = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return tp, fp, fn, p, r, f1

best = None
for conf_min in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    for cd in [1, 4, 6, 8, 10, 12, 15, 20]:
        # take actions sorted by frame, apply confidence + cooldown
        kept = []
        last = -10**9
        for a in sorted(actions, key=lambda x: x["frame"]):
            if a["confidence"] < conf_min: continue
            if a["frame"] - last < cd: continue
            kept.append(a["frame"])
            last = a["frame"]
        tp, fp, fn, p, r, f1 = evaluate(kept, gt_frames)
        if best is None or f1 > best[5]:
            best = (conf_min, cd, len(kept), tp, fp, fn, f1, p, r)
        if f1 >= 0.6:
            print(f"conf>={conf_min:.1f}  cd={cd:2d}  n={len(kept):3d}  TP={tp:2d} FP={fp:3d} FN={fn:2d}  P={p:.2f} R={r:.2f} F1={f1:.3f}")

cm, cd, n, tp, fp, fn, f1, p, r = best
print(f"\nBest action-only:  conf>={cm}  cd={cd}  n={n}  TP={tp} FP={fp} FN={fn}  P={p:.2f} R={r:.2f} F1={f1:.3f}")

# Also: NMS by confidence (keep highest-confidence in each cooldown window)
print("\n--- NMS by confidence (keep highest score within window) ---")
for conf_min in [0.0, 0.2, 0.3, 0.4, 0.5]:
    for cd in [4, 6, 8, 10, 12, 15, 20]:
        cand = [(a["frame"], a["confidence"]) for a in actions if a["confidence"] >= conf_min]
        cand.sort(key=lambda x: -x[1])
        kept_frames = []
        for f, _ in cand:
            if all(abs(f - k) >= cd for k in kept_frames):
                kept_frames.append(f)
        kept_frames.sort()
        tp, fp, fn, p, r, f1 = evaluate(kept_frames, gt_frames)
        if f1 >= 0.6:
            print(f"conf>={conf_min:.1f}  cd={cd:2d}  n={len(kept_frames):3d}  TP={tp:2d} FP={fp:3d} FN={fn:2d}  P={p:.2f} R={r:.2f} F1={f1:.3f}")
