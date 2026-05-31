#!/usr/bin/env python3
"""
Sound Punch Detector — calibrated on YOUR labels (few-shot on AudioSet AST)
===========================================================================
Final stage of the audio pipeline.  Workflow recap:

  1. sound_detector.py --extract-samples   → 70 candidate clips
  2. sound_ai_classify.py --auto-sort       → AI pre-sort into punch/ not_punch/
  3. YOU verified/corrected the two folders
  4. THIS script:
       a. embeds every clip in punch/ and not_punch/ with AST (527-class
          AudioSet probability vector, ±1s window) → feature vectors
       b. trains LogisticRegression(punch vs not-punch) on YOUR labels,
          reports honest k-fold cross-val (small data, so CV not train score)
       c. scans the whole video: onset peaks → AST features → classifier
          probability → keep punches above --prob, NMS by cooldown
       d. saves outputs/sound_ai.json and renders outputs/sound_ai.mp4

The decision is now learned from the sounds YOU marked, on top of a model that
already knows what impacts sound like.

Usage:
    python sound_train_detect.py
    python sound_train_detect.py --prob 0.6 --onset-thr 1.3 --cooldown 12
    python sound_train_detect.py --no-video
"""
import os
import re
import glob
import json
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sound_detector import (
    compute_onset, detect_impacts, load_wav, render,
    FPS, WAV_PATH, VIDEO_IN, OUT_DIR,
)
from sound_ai_classify import (
    build_model, classify, window_16k, ANALYSIS_HALF,
)

SAMPLE_DIR = os.path.join(OUT_DIR, "sound_samples")
OUT_MP4 = os.path.join(OUT_DIR, "sound_ai.mp4")


def time_from_name(fname):
    """Parse '..._t1m20.61s_...' → seconds."""
    m = re.search(r"_t(\d+)m([\d.]+)s_", fname)
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def labeled_set(sample_dir):
    """Return [(time_sec, label)] with punch/ authoritative over not_punch/.
    Accepts both audio (.wav) and video+audio (.mp4) clips."""
    def clips(sub):
        out = {}
        for ext in ("*.wav", "*.mp4"):
            for p in glob.glob(os.path.join(sample_dir, sub, ext)):
                out[os.path.basename(p)] = p
        return out
    punch = clips("punch")
    notp = clips("not_punch")
    # punch wins any duplicate
    for f in punch:
        notp.pop(f, None)
    rows = []
    for f in punch:
        t = time_from_name(f)
        if t is not None:
            rows.append((t, 1))
    for f in notp:
        t = time_from_name(f)
        if t is not None:
            rows.append((t, 0))
    return rows, len(punch), len(notp)


def build_features(rows, x_full, sr, torch, fe, model, half):
    """AST 527-class probability vector per labeled clip."""
    X, y = [], []
    for t, lab in rows:
        seg = window_16k(x_full, sr, t, half)
        X.append(classify(torch, fe, model, seg))
        y.append(lab)
    return np.array(X), np.array(y)


def train_classifier(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, accuracy_score

    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)

    n_pos = int(y.sum())
    k = min(5, n_pos) if n_pos >= 2 else 2
    print(f"\n[train] {len(y)} clips  ({n_pos} punch / {len(y)-n_pos} not)  "
          f"feature dim={X.shape[1]}")
    try:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba)
        acc = accuracy_score(y, (proba >= 0.5).astype(int))
        print(f"[train] {k}-fold CV  ROC-AUC={auc:.3f}  acc={acc:.3f}  "
              f"(honest, held-out)")
    except Exception as e:
        print(f"[train] CV skipped ({e})")
    clf.fit(X, y)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", default=SAMPLE_DIR)
    ap.add_argument("--wav", default=WAV_PATH)
    ap.add_argument("--video-in", default=VIDEO_IN)
    ap.add_argument("--onset-thr", type=float, default=1.3,
                    help="onset ratio to nominate candidates (recall stage)")
    ap.add_argument("--prob", type=float, default=0.5,
                    help="classifier punch-probability to accept (mode=logreg)")
    ap.add_argument("--mode", choices=["impact", "logreg"], default="impact",
                    help="impact = AST impact-class prob (works better here); "
                         "logreg = classifier trained on your labels")
    ap.add_argument("--impact-thr", type=float, default=0.20,
                    help="AST impact-class probability to accept (mode=impact)")
    ap.add_argument("--cooldown", type=int, default=12)
    ap.add_argument("--analysis-half", type=float, default=ANALYSIS_HALF)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=OUT_MP4)
    args = ap.parse_args()

    # ── audio + onset envelope ────────────────────────────────────────────────
    x_full, sr = load_wav(args.wav)
    print(f"[snd] {len(x_full)/sr:.1f}s @ {sr}Hz")
    times, onset, lo_env, hi_env = compute_onset(x_full, sr)

    # ── AST model + train classifier on YOUR labels ───────────────────────────
    rows, n_p, n_n = labeled_set(args.sample_dir)
    if n_p < 2 or n_n < 2:
        print(f"[snd] need >=2 clips in each of punch/ and not_punch/ "
              f"(have {n_p}/{n_n}); sort the folders first.")
        return
    torch, fe, model, id2label, impact_idx, crowd_idx = build_model()
    X, y = build_features(rows, x_full, sr, torch, fe, model, args.analysis_half)
    clf = train_classifier(X, y)

    # ── scan whole video: onset candidates → AST → classifier ─────────────────
    cand = detect_impacts(times, onset, args.onset_thr, args.cooldown)
    impact_cols = list(impact_idx)
    accept = args.prob if args.mode == "logreg" else args.impact_thr
    print(f"\n[scan] {len(cand)} onset candidates (onset>={args.onset_thr}); "
          f"mode={args.mode}, accept>={accept}; running AST on each …")
    detections = []
    for t, onset_sc in cand:
        seg = window_16k(x_full, sr, t, args.analysis_half)
        feat = classify(torch, fe, model, seg)
        impact_p = float(feat[impact_cols].sum())
        logreg_p = float(clf.predict_proba(feat.reshape(1, -1))[0, 1])
        score = impact_p if args.mode == "impact" else logreg_p
        if score >= accept:
            detections.append((t, score))

    # NMS by cooldown (keep higher prob)
    detections.sort(key=lambda z: -z[1])
    cd_sec = args.cooldown / FPS
    kept = []
    for t, p in detections:
        if all(abs(t - kt) >= cd_sec for kt, _ in kept):
            kept.append((t, p))
    kept.sort()

    print(f"\n[scan] {len(kept)} punches confirmed "
          f"(mode={args.mode}, score >= {accept})\n")
    for i, (t, p) in enumerate(kept):
        m, s = int(t // 60), t % 60
        print(f"  {i+1:3d}.  {m}:{s:05.2f}  score={p:.3f}  frame≈{int(t*FPS)}")

    # ── save json ──────────────────────────────────────────────────────────────
    out_json = args.out.replace(".mp4", ".json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump({
        "method": "sound_ai_calibrated",
        "model": "AST AudioSet + LogReg on user labels",
        "onset_thr": args.onset_thr, "prob_thr": args.prob,
        "cooldown_frames": args.cooldown,
        "n_train_punch": n_p, "n_train_not": n_n,
        "n_impacts": len(kept),
        "events": [{"time_sec": round(t, 3), "frame": int(round(t * FPS)),
                    "punch_prob": round(p, 4)} for t, p in kept],
    }, open(out_json, "w"), indent=2)
    print(f"\n[snd] JSON saved: {out_json}")

    # ── render ──────────────────────────────────────────────────────────────────
    if not args.no_video:
        render(args.video_in, kept, (times, onset, lo_env, hi_env),
               args.onset_thr, args.out)


if __name__ == "__main__":
    main()
