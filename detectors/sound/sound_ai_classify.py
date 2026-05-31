#!/usr/bin/env python3
"""
AI Punch-Sound Classifier  (pretrained AudioSet model)
======================================================
Uses the Audio Spectrogram Transformer (AST) fine-tuned on AudioSet
(MIT/ast-finetuned-audioset-10-10-0.4593, 527 sound classes) to score each
candidate clip by how much it sounds like a real impact — independent of
loudness.  AudioSet already separates impact sounds from crowd/speech, so this
rejects crowd roar and commentary that the raw loudness detector fired on.

"Punch-ness" = sum of probabilities over the impact-related AudioSet classes:
    Slap, smack | Whack, thwack | Thump, thud | Smash, crash | Knock | Crack | Bang
minus a penalty for crowd/speech classes:
    Speech | Cheering | Applause | Crowd | Hubbub

Two modes
---------
  (1) score-samples : run on outputs/sound_samples/*.wav, rank by punch-ness,
      and auto-suggest a sort into punch/ vs not_punch/  (you then verify).
  (2) (later) the same model embeds your verified clips to calibrate a final
      threshold — run after you confirm the auto-sort.

Usage:
    python sound_ai_classify.py                       # score the 70 sample clips
    python sound_ai_classify.py --auto-sort           # also copy into punch/ not_punch/
    python sound_ai_classify.py --punch-thr 0.10
"""
import os
import json
import glob
import shutil
import argparse
import wave
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SAMPLE_DIR = r"/home/jake/Desktop/HITAI/Contact_Detection/outputs/sound_samples"
WAV_PATH = r"/home/jake/Downloads/sam3d_with_world_coords/3.wav"
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR = 16000

# AST needs ~1s of context to recognise an impact pattern; 0.4s dilutes it.
ANALYSIS_HALF = 1.0   # seconds on each side of the onset (→ 2s analysis window)

# Impact / percussive AudioSet classes a punch landing can fall under.
# (leather-on-body ≈ slap/smack/whack/thwack/chop/slam/whip/thump)
IMPACT_KEYWORDS = ["slap", "smack", "whack", "thwack", "thump", "thud",
                   "smash", "chop", "slam", "whip", "bang", "knock",
                   "crack", "punch", "hammer", "breaking", "crushing"]
# Speech is omnipresent (constant commentary) → NOT used as a penalty.
# Crowd shown for information only.
CROWD_KEYWORDS = ["cheer", "applause", "crowd", "hubbub", "battle cry"]


def load_full_wav(path):
    from scipy.signal import resample_poly
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def window_16k(x, sr, t_sec, half=ANALYSIS_HALF):
    """Cut a ±half-second window centred at t_sec, resampled to 16 kHz."""
    from math import gcd
    from scipy.signal import resample_poly
    c = int(t_sec * sr)
    seg = x[max(0, c - int(half * sr)): c + int(half * sr)]
    if sr != AST_SR:
        g = gcd(AST_SR, sr)
        seg = resample_poly(seg, AST_SR // g, sr // g)
    return seg.astype(np.float32)


def build_model():
    import torch
    from transformers import ASTForAudioClassification, AutoFeatureExtractor
    print(f"[ai] loading {MODEL_ID} (CPU) …")
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()
    id2label = model.config.id2label
    impact_idx = {i: l for i, l in id2label.items()
                  if any(k in l.lower() for k in IMPACT_KEYWORDS)}
    crowd_idx = {i: l for i, l in id2label.items()
                 if any(k in l.lower() for k in CROWD_KEYWORDS)}
    return torch, fe, model, id2label, impact_idx, crowd_idx


def classify(torch, fe, model, x):
    """Return full 527-class probability vector for one window."""
    inp = fe(x, sampling_rate=AST_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inp).logits[0]
    return torch.softmax(logits, dim=-1).numpy()


def punch_score(probs, impact_idx, crowd_idx):
    impact = sum(float(probs[i]) for i in impact_idx)
    crowd = sum(float(probs[i]) for i in crowd_idx)
    # score by impact-class probability alone (commentary speech is omnipresent
    # and would wrongly suppress real punches, so it is NOT subtracted)
    return impact, crowd, impact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", default=SAMPLE_DIR)
    ap.add_argument("--wav", default=WAV_PATH)
    ap.add_argument("--punch-thr", type=float, default=0.15,
                    help="impact-class probability >= this -> suggested punch")
    ap.add_argument("--analysis-half", type=float, default=ANALYSIS_HALF,
                    help="seconds each side of onset for AST context window")
    ap.add_argument("--auto-sort", action="store_true",
                    help="copy clips into punch/ and not_punch/ by the AI score")
    args = ap.parse_args()

    idx_path = os.path.join(args.sample_dir, "index.json")
    if not os.path.exists(idx_path):
        print(f"[ai] {idx_path} missing; run "
              f"`python sound_detector.py --extract-samples` first.")
        return
    samples = json.load(open(idx_path))["samples"]

    x_full, sr = load_full_wav(args.wav)
    torch, fe, model, id2label, impact_idx, crowd_idx = build_model()
    print(f"[ai] impact classes ({len(impact_idx)}): "
          f"{list(impact_idx.values())}")
    print(f"[ai] analysis window: ±{args.analysis_half}s\n")

    rows = []
    for s in samples:
        seg = window_16k(x_full, sr, s["time_sec"], args.analysis_half)
        probs = classify(torch, fe, model, seg)
        impact, crowd, net = punch_score(probs, impact_idx, crowd_idx)
        top3 = sorted(range(len(probs)), key=lambda i: -probs[i])[:3]
        rows.append({
            "file": s["file"],
            "timestamp": s["timestamp"],
            "impact": round(impact, 4),
            "crowd": round(crowd, 4),
            "net": round(net, 4),
            "top3": [(id2label[i], round(float(probs[i]), 3)) for i in top3],
        })

    rows.sort(key=lambda r: -r["net"])

    print(f"{'rank':>4}  {'net':>6}  {'impact':>6}  {'crowd':>6}  "
          f"{'verdict':>10}  file   |  top AudioSet class")
    print("-" * 100)
    for k, r in enumerate(rows):
        verdict = "PUNCH" if r["net"] >= args.punch_thr else "reject"
        print(f"{k+1:>4}  {r['net']:>6.3f}  {r['impact']:>6.3f}  "
              f"{r['crowd']:>6.3f}  {verdict:>10}  {r['file']}  "
              f"|  {r['top3'][0][0]} ({r['top3'][0][1]})")

    n_punch = sum(1 for r in rows if r["net"] >= args.punch_thr)
    print(f"\n[ai] {n_punch}/{len(rows)} clips suggested as PUNCH "
          f"(net >= {args.punch_thr})")

    out = os.path.join(args.sample_dir, "ai_scores.json")
    json.dump({"model": MODEL_ID, "punch_thr": args.punch_thr,
               "analysis_half": args.analysis_half,
               "impact_classes": impact_idx, "rows": rows},
              open(out, "w"), indent=2)
    print(f"[ai] scores saved: {out}")

    if args.auto_sort:
        pd = os.path.join(args.sample_dir, "punch")
        nd = os.path.join(args.sample_dir, "not_punch")
        os.makedirs(pd, exist_ok=True)
        os.makedirs(nd, exist_ok=True)
        for r in rows:
            src = os.path.join(args.sample_dir, r["file"])
            dst = pd if r["net"] >= args.punch_thr else nd
            shutil.copy2(src, os.path.join(dst, r["file"]))
        print(f"[ai] auto-sorted copies into {pd}/ and {nd}/ — "
              f"please VERIFY before training the final detector.")


if __name__ == "__main__":
    main()
