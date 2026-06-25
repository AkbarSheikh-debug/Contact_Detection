#!/usr/bin/env python3
"""
Render tools/predict_keypoint_model.py's predictions.json onto its source
video: both fighters' bounding boxes, a marker + label at each candidate
action's contact frame.

Color-coded by the CALIBRATED 3-tier label (green=impact, yellow=borderline,
red=not_impact) computed from this video's own percentile ranking, not just
the raw 0.5-probability threshold -- the model's absolute threshold was
calibrated on the training video's distance scale and can run uniformly low
on a new, out-of-distribution video (see conversation), so the on-screen
label also shows the raw probability, percentile rank, and raw
wrist-to-torso distance / reach ratio so you can judge borderline cases
yourself instead of trusting one fixed cutoff.

Usage:
  python rendering/render_keypoint_predictions.py --predictions <predictions.json> --video <video.mp4> --out <out.mp4>
"""
import os
import sys
import json
import argparse
import subprocess

import cv2
import numpy as np

FFMPEG = r"C:\Users\XRIG\Downloads\ffmpeg_extracted\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))


def load_sam3d_bboxes(sam3d_path):
    d = json.load(open(sam3d_path))
    return {tid: {e["frame"]: e["bbox"] for e in d[tid]} for tid in ("0", "1")}


TIER_COLOR = {
    "impact": (0, 215, 0),        # green (BGR)
    "borderline": (0, 200, 255),  # yellow
    "not_impact": (0, 0, 230),    # red
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--marker-frames", type=int, default=12,
                     help="how many frames each marker stays visible")
    ap.add_argument("--freeze", type=float, default=0.0,
                     help="seconds to hold + green-tint the frame at each IMPACT-tier "
                          "event's contact frame (0 = disabled). Same technique as "
                          "detectors/fusion/v9.py's render_video: solid green "
                          "(0,180,0 BGR) blended at alpha 0.35 via cv2.addWeighted, "
                          "frame repeated freeze*fps times in the writer.")
    ap.add_argument("--model-label", default="ensemble",
                     help="label shown in the top-left counter overlay (e.g. 'TCN', 'GRU')")
    args = ap.parse_args()

    data = json.load(open(args.predictions))
    preds = data["predictions"]
    bboxes = load_sam3d_bboxes(data["sam3d_path"])
    n_imp = sum(1 for p in preds if p.get("calibrated_label") == "impact")
    n_bord = sum(1 for p in preds if p.get("calibrated_label") == "borderline")
    n_miss = len(preds) - n_imp - n_bord
    print(f"Rendering {len(preds)} predictions onto {args.video}  "
          f"(impact={n_imp} borderline={n_bord} not_impact={n_miss})")

    # index predictions by their window_end (closest to contact) for display timing
    by_frame = {}
    for p in preds:
        contact_frame = p["window_end"]
        by_frame.setdefault(contact_frame, []).append(p)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out.replace(".mp4", "_noaudio.mp4")
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    freeze_n = int(round(args.freeze * fps))
    if freeze_n > 0:
        print(f"Freeze-on-impact enabled: holding {args.freeze}s ({freeze_n} frames) "
              f"per IMPACT-tier event")

    active_markers = []  # list of (expire_frame, pred_dict)
    n_shown = 0
    n_frozen = 0

    for fi in range(total):
        ok, fr = cap.read()
        if not ok:
            break

        for tid, col in (("0", (255, 140, 0)), ("1", (0, 140, 255))):
            b = bboxes.get(tid, {}).get(fi)
            if b:
                x1, y1, x2, y2 = [int(v) for v in b]
                cv2.rectangle(fr, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

        if fi in by_frame:
            for p in by_frame[fi]:
                active_markers.append((fi + args.marker_frames, p))
                n_shown += 1

        active_markers = [(exp, p) for exp, p in active_markers if exp > fi]
        for row, (_, p) in enumerate(active_markers):
            prob = p["impact_probability"]
            tier = p.get("calibrated_label", p["predicted_label"])
            col = TIER_COLOR.get(tier, (200, 200, 200))
            pctile = p.get("video_percentile")
            min_dist = p.get("min_wrist_to_torso_dist")
            min_reach = p.get("min_reach_ratio")
            label = (f"{p['fighter_type']} {p['action']}  p={prob:.2f}  pct={pctile:.0f}  "
                     f"dist={min_dist:.2f}  reach={min_reach:.2f}  {tier.upper()}")
            y = H - 14 - row * 26
            cv2.rectangle(fr, (0, y - 18), (min(W, 30 + len(label) * 10), y + 8), (15, 15, 20), -1)
            cv2.putText(fr, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        cv2.rectangle(fr, (0, 0), (560, 28), (15, 15, 20), -1)
        cv2.putText(fr, f"{args.model_label}  impact:{n_imp} border:{n_bord} miss:{n_miss}  "
                        f"shown:{n_shown}/{len(preds)}  f:{fi}",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
        wr.write(fr)

        # ── FREEZE: same technique as detectors/fusion/v9.py's render_video --
        # solid green tint blended at alpha 0.35, frame held freeze_n times --
        # for each IMPACT-tier event landing on this frame.
        if freeze_n > 0 and fi in by_frame:
            for p in by_frame[fi]:
                if p.get("calibrated_label", p["predicted_label"]) != "impact":
                    continue
                n_frozen += 1
                hold = fr.copy()
                tint = np.zeros_like(hold)
                tint[:, :] = (0, 180, 0)  # BGR green
                cv2.addWeighted(tint, 0.35, hold, 1.0, 0, hold)
                cv2.rectangle(hold, (0, 0), (820, 34), (15, 15, 20), -1)
                cv2.putText(hold, f"IMPACT #{n_frozen}  {p['fighter_type']} {p['action']}  "
                                  f"frame:{fi}  p={p['impact_probability']:.2f}  "
                                  f"(holding {args.freeze}s for review)",
                            (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                for _ in range(freeze_n):
                    wr.write(hold)

        if (fi + 1) % 500 == 0:
            print(f"  ...{fi + 1}/{total} frames")

    cap.release()
    wr.release()

    try:
        mux_args = [FFMPEG, "-y", "-i", tmp, "-i", args.video, "-map", "0:v",
                    "-map", "1:a?", "-c:v", "copy"]
        # -shortest would truncate the video back down to the original audio
        # length, undoing every freeze we just inserted -- only safe when
        # there's no freezing going on.
        if freeze_n == 0:
            mux_args.append("-shortest")
        mux_args += [args.out, "-loglevel", "error"]
        subprocess.run(mux_args, check=True)
        os.remove(tmp)
        print(f"Saved (with audio) -> {args.out}")
    except Exception as e:
        print(f"mux failed ({e}); silent video at {tmp}")


if __name__ == "__main__":
    main()
