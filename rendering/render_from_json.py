#!/usr/bin/env python3
"""
Generic detection-JSON → annotated video renderer
==================================================
Renders ANY detector's output JSON onto the source video so all methods can be
compared in the same visual language:
  - both fighters' skeletons (real joints_2d)
  - a marker + score label + flash at each detected impact frame
  - green = within ±tol of a user-verified landing, red = not (so quality is visible)
  - original audio muxed back in

Reads events with a "frame" (or "impact_frame") and optional "contact_point".

Usage:
    python render_from_json.py --json outputs/sam_detect.json --out outputs/sam_detect.mp4 --label "SAM overlap"
"""
import os
import re
import glob
import json
import argparse
import subprocess

import cv2
import numpy as np

DATA = r"/home/jake/Downloads/sam3d_with_world_coords"
KP2D = r"/home/jake/Downloads/for_impact_detection_experiment_2/2d_points.json"
VIDEO = os.path.join(DATA, "3.mp4")
VERIFIED_DIR = r"/home/jake/Desktop/HITAI/Contact_Detection/outputs/sound_samples_av/punch"
FPS = 24.995
WRIST_KPS = [9, 10]
SKEL = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
COL = {0: (0, 255, 180), 1: (255, 140, 0)}


def load_2d():
    raw = json.load(open(KP2D))
    J = {}
    for pid_s, entries in raw.items():
        pid = int(pid_s); J[pid] = {}
        for e in entries:
            d = e.get("frame_dims", {})
            sx = d.get("original_width", 1920) / d.get("resized_width", 640)
            sy = d.get("original_height", 1080) / d.get("resized_height", 360)
            j = np.asarray(e["joints_2d"], float); j[:, 0] *= sx; j[:, 1] *= sy
            J[pid][e["frame"]] = j
    return J


def verified_frames():
    fs = []
    for f in glob.glob(os.path.join(VERIFIED_DIR, "*.mp4")):
        m = re.search(r"_f(\d+)\.", os.path.basename(f))
        if m:
            fs.append(int(m.group(1)))
    return sorted(fs)


def event_frame(e):
    return e.get("impact_frame", e.get("frame"))


def event_score(e):
    for k in ("score", "impact_score", "sam", "punch_prob", "onset_score"):
        if k in e:
            return float(e[k])
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="detector")
    ap.add_argument("--tol", type=int, default=15)
    args = ap.parse_args()

    data = json.load(open(args.json))
    events = data.get("events", [])
    imp = {}
    for e in events:
        if e.get("is_impact") is False:      # respect detectors that emit all candidates
            continue
        f = event_frame(e)
        if f is not None:
            imp[int(f)] = e
    VF = verified_frames()
    J = load_2d()
    print(f"[render] {args.label}: {len(imp)} detections, {len(VF)} verified landings")

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = args.out.replace(".mp4", "_noaudio.mp4")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    flash = None; n_tp = 0; n_fp = 0
    print(f"[render] {total} frames → {args.out}")
    for fi in range(total):
        ok, fr = cap.read()
        if not ok:
            break
        for pid in (0, 1):
            j = J.get(pid, {}).get(fi)
            if j is None:
                continue
            for a, b in SKEL:
                if not np.allclose(j[a], 0) and not np.allclose(j[b], 0):
                    cv2.line(fr, tuple(j[a].astype(int)), tuple(j[b].astype(int)),
                             COL[pid], 2, cv2.LINE_AA)
            for wk in WRIST_KPS:
                if not np.allclose(j[wk], 0):
                    cv2.circle(fr, tuple(j[wk].astype(int)), 6, (0, 180, 255), -1)

        if fi in imp:
            e = imp[fi]
            is_tp = any(abs(fi - v) <= args.tol for v in VF)
            flash = (fi, is_tp)
            n_tp += is_tp; n_fp += (not is_tp)
            cp = e.get("contact_point")
            col = (0, 230, 0) if is_tp else (0, 50, 255)
            sc = event_score(e)
            if cp and len(cp) >= 2:
                cv2.circle(fr, (int(cp[0]), int(cp[1])), 24, col, 3, cv2.LINE_AA)
                cv2.putText(fr, f"{'HIT' if is_tp else 'FP'} {sc:.2f}",
                            (int(cp[0])+14, int(cp[1])-14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(fr, f"{'HIT' if is_tp else 'FP'} {sc:.2f}", (W//2-60, H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)

        if flash and fi < flash[0] + 12:
            al = max(0.0, 1.0 - (fi - flash[0]) / 12) * 0.30
            t = np.zeros_like(fr); t[:, :] = (0, 200, 0) if flash[1] else (0, 0, 200)
            cv2.addWeighted(t, al, fr, 1.0, 0, fr)

        cv2.rectangle(fr, (0, 0), (560, 30), (15, 15, 20), -1)
        cv2.putText(fr, f"{args.label}  det:{len(imp)}  HIT(green):{n_tp} FP(red):{n_fp}  f:{fi}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
        wr.write(fr)
    cap.release(); wr.release()

    try:
        subprocess.run(["ffmpeg", "-y", "-i", tmp, "-i", VIDEO, "-map", "0:v",
                        "-map", "1:a?", "-c:v", "copy", "-shortest", args.out,
                        "-loglevel", "error"], check=True)
        os.remove(tmp)
        print(f"[render] saved {args.out}")
    except Exception as e:
        print(f"[render] mux failed ({e}); silent at {tmp}")


if __name__ == "__main__":
    main()
