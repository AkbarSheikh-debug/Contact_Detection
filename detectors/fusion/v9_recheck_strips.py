#!/usr/bin/env python3
"""
v9_recheck_strips.py — finer-grained strips for UNCLEAR vision verdicts
=======================================================================
Second stage of the VLM verification loop: for candidates the first-pass
3-panel strip couldn't resolve, produce a 5-panel strip (-4,-2,0,+2,+4)
at higher resolution so glove-contact and head reaction are visible.

Usage:
    python v9_recheck_strips.py --folder "<fight dir>" --frames 422 760 1330
"""
import os, glob, argparse, json
import numpy as np
import cv2

OFFS = (-4, -2, 0, 2, 4)
PANEL_W = 430


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    args = ap.parse_args()

    vids = [v for v in glob.glob(os.path.join(args.folder, "*.mp4"))
            if "_visualized" not in os.path.basename(v) and "_impacts" not in os.path.basename(v)]
    sam3d_p = glob.glob(os.path.join(args.folder, "*_sam3d.json"))[0]
    s = json.load(open(sam3d_p))
    persons = {0: {e["frame"]: e for e in s.get("0", [])},
               1: {e["frame"]: e for e in s.get("1", [])}}

    out_dir = os.path.join(args.folder, "impacts_v9_recheck")
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "*.jpg")):
        os.remove(old)

    need = {f + o for f in args.frames for o in OFFS}
    cap = cv2.VideoCapture(vids[0])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Wf, Hf = int(cap.get(3)), int(cap.get(4))
    store = {}
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if fi in need:
            store[fi] = frame.copy()
        if len(store) == len(need):
            break
    cap.release()

    for f in args.frames:
        boxes = [persons.get(p, {}).get(f, {}).get("bbox") for p in (0, 1)]
        boxes = [b for b in boxes if b]
        if boxes:
            x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
            pw, ph = (x2 - x1) * 0.22, (y2 - y1) * 0.15
            x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            x2, y2 = min(Wf, int(x2 + pw)), min(Hf, int(y2 + ph))
        else:
            x1, y1, x2, y2 = 0, 0, Wf, Hf
        panels = []
        for o in OFFS:
            fr = store.get(f + o)
            fr = (np.zeros((y2 - y1, x2 - x1, 3), np.uint8) if fr is None
                  else fr[y1:y2, x1:x2].copy())
            ph_, pw_ = fr.shape[:2]
            fr = cv2.resize(fr, (PANEL_W, int(ph_ * PANEL_W / max(1, pw_))))
            lbl = "IMPACT" if o == 0 else f"t{o:+d}"
            cv2.rectangle(fr, (0, 0), (PANEL_W, 24), (15, 15, 20), -1)
            cv2.putText(fr, lbl, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255) if o else (0, 80, 255), 1, cv2.LINE_AA)
            panels.append(fr)
        h = min(p.shape[0] for p in panels)
        strip = np.hstack([p[:h] for p in panels])
        fn = os.path.join(out_dir, f"recheck_frame{f}.jpg")
        cv2.imwrite(fn, strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("saved", fn)


if __name__ == "__main__":
    main()
