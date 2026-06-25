#!/usr/bin/env python3
"""
Render the impact-anchored keypoint model's predictions onto the actual clip
videos from 1st_fight and stitch them into a single results video.

Each clip gets a colored border showing correctness:
  Green  = True Positive  (GT=impact,     pred=impact)
  Orange = False Negative (GT=impact,     pred=not_impact)
  Gray   = True Negative  (GT=not_impact, pred=not_impact)
  Red    = False Positive (GT=not_impact, pred=impact)

A magenta vertical line on the progress bar marks the anchor frame
(exact impact_frame if marked, else window_end proxy).

Clips scored using leave-one-round-out: each clip is scored by the checkpoint
that never saw its round during training, giving honest out-of-distribution
predictions.

Output: outputs/1st_fight_anchored_predictions.mp4

Usage:
  python tools/render_anchored_predictions.py
  python tools/render_anchored_predictions.py --out outputs/my_output.mp4
  python tools/render_anchored_predictions.py --threshold 0.4
"""
import os, sys, json, argparse
import cv2
import numpy as np
import torch

_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

from keypoint_model import build_model
from dataset import fights

# ── paths ─────────────────────────────────────────────────────────────────────
NPZ_PATH  = os.path.join(_REPO_ROOT, "outputs", "keypoint_dataset",
                          "1st_fight_anchored.npz")
CKPT_DIR  = os.path.join(_REPO_ROOT, "outputs", "keypoint_model")
OUT_DIR   = os.path.join(_REPO_ROOT, "outputs")
CLIPS_DIR = os.path.join(_REPO_ROOT, "dataset", "1st_fight", "clips")

# Checkpoint per round: trained WITHOUT that round (leave-one-round-out)
ROUND_TO_CKPT = {
    1: os.path.join(CKPT_DIR, "tcn_1st_fight_R1_best.pt"),
    2: os.path.join(CKPT_DIR, "tcn_1st_fight_R2_best.pt"),
    3: os.path.join(CKPT_DIR, "tcn_1st_fight_R3_best.pt"),
}

# ── display config ─────────────────────────────────────────────────────────────
W, H       = 640, 360
FPS        = 25.0
BORDER     = 6
BAR_H      = 18     # timeline bar height at bottom of frame
FONT       = cv2.FONT_HERSHEY_SIMPLEX

COL_TP  = (30,  200,  30)    # green
COL_FN  = (30,  140, 255)    # orange
COL_TN  = (160, 160, 160)    # gray
COL_FP  = (30,   30, 200)    # red
COL_ANC = (220,   0, 220)    # magenta — anchor frame marker
COL_GT  = {1: (30, 200, 30), 0: (30, 30, 200)}
GROUP_COL = {"TP": COL_TP, "FN": COL_FN, "TN": COL_TN, "FP": COL_FP}
GROUP_LABEL = {
    "TP": "TRUE POSITIVE   GT=IMPACT    pred=IMPACT",
    "FN": "FALSE NEGATIVE  GT=IMPACT    pred=NOT IMPACT",
    "TN": "TRUE NEGATIVE   GT=NOT IMPACT  pred=NOT IMPACT",
    "FP": "FALSE POSITIVE  GT=NOT IMPACT  pred=IMPACT",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def txt(img, text, x, y, scale=0.52, fg=(255,255,255), bg=(0,0,0), thickness=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness + 1)
    cv2.rectangle(img, (x-2, y-th-3), (x+tw+2, y+3), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, fg, thickness+1, cv2.LINE_AA)


def draw_prob_bar(img, prob, x, y, w=140, h=12):
    cv2.rectangle(img, (x, y), (x+w, y+h), (50,50,50), -1)
    filled = int(w * prob)
    col = COL_TP if prob >= 0.5 else COL_FP
    if filled > 0:
        cv2.rectangle(img, (x, y), (x+filled, y+h), col, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (150,150,150), 1)
    cv2.putText(img, f"{prob:.2f}", (x+w+4, y+h-1), FONT, 0.42, (220,220,220), 1, cv2.LINE_AA)


def draw_timeline_bar(img, n_frames, current_f, anchor_f, border_col):
    """Thin timeline bar at the very bottom: filled to current_f, anchor in magenta."""
    bar_y = H - BAR_H
    # background
    cv2.rectangle(img, (0, bar_y), (W, H), (30,30,30), -1)
    # filled portion
    filled_w = max(0, int((current_f / max(n_frames-1, 1)) * W))
    col = tuple(int(c * 0.7) for c in border_col)
    cv2.rectangle(img, (0, bar_y), (filled_w, H), col, -1)
    # anchor marker
    ax = int((anchor_f / max(n_frames-1, 1)) * W)
    cv2.rectangle(img, (ax-2, bar_y), (ax+2, H), COL_ANC, -1)
    # current position
    cx = max(0, min(W-1, filled_w))
    cv2.line(img, (cx, bar_y), (cx, H), (255,255,255), 1)


def title_card(text, colour, n_frames=int(FPS)):
    frames = []
    for _ in range(n_frames):
        img = np.full((H, W, 3), 20, dtype=np.uint8)
        cv2.rectangle(img, (0,0), (W-1, H-1), colour, 4)
        for i, line in enumerate(text.split("\n")):
            (tw, th), _ = cv2.getTextSize(line, FONT, 0.8, 2)
            x = max(8, (W - tw) // 2)
            y = H//2 - 20 + i*(th+14)
            cv2.putText(img, line, (x, y), FONT, 0.8, colour, 2, cv2.LINE_AA)
        frames.append(img)
    return frames


def render_clip(clip_path, meta, true_label, pred_prob, threshold):
    """Read clip MP4, overlay prediction info, return list of BGR frames."""
    cap       = cv2.VideoCapture(clip_path)
    n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pred_label = 1 if pred_prob >= threshold else 0
    if true_label == 1 and pred_label == 1:   group = "TP"
    elif true_label == 1 and pred_label == 0: group = "FN"
    elif true_label == 0 and pred_label == 0: group = "TN"
    else:                                      group = "FP"

    border_col = GROUP_COL[group]

    # Compute local anchor frame index (0-based position in the clip)
    window_start = meta["window_start"]
    clip_start   = max(0, window_start - 8)   # PAD_BEFORE=8 matches prepare_fight_dataset
    anchor_global = meta.get("impact_frame") or meta["window_end"]
    anchor_local  = max(0, min(n_frames-1, anchor_global - clip_start))
    has_exact_anchor = meta.get("impact_frame") is not None

    gt_text   = "GT: IMPACT" if true_label == 1 else "GT: NOT IMPACT"
    pred_text = f"PRED: {'IMPACT' if pred_label else 'NOT IMPACT'}"
    anchor_tag = "(exact)" if has_exact_anchor else "(proxy)"
    action_txt = f"{meta['action']}  R{meta['round']} F{meta['fighter_id']}"

    out_frames = []
    fi = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        bgr = cv2.resize(bgr, (W, H))

        is_anchor = (fi == anchor_local)

        # colored border
        thick = BORDER + 4 if is_anchor else BORDER
        col   = COL_ANC if is_anchor else border_col
        cv2.rectangle(bgr, (0,0), (W-1, H-1), col, thick)

        # top-left: GT label
        txt(bgr, gt_text, BORDER+4, BORDER+20,
            fg=COL_GT[true_label], scale=0.58)
        # top-right: prediction
        txt(bgr, pred_text, W - 210, BORDER+20,
            fg=(30,200,30) if pred_label==1 else (80,80,200), scale=0.58)

        # mid-left: probability bar
        draw_prob_bar(bgr, pred_prob, BORDER+4, BORDER+36)

        # bottom-left: action info
        txt(bgr, action_txt, BORDER+4, H - BAR_H - 22, scale=0.46, fg=(200,200,200))
        # anchor label
        anchor_label = f"anchor fr{anchor_global} {anchor_tag}"
        txt(bgr, anchor_label, BORDER+4, H - BAR_H - 6, scale=0.42,
            fg=COL_ANC if has_exact_anchor else (150,150,150))

        # timeline bar
        draw_timeline_bar(bgr, n_frames, fi, anchor_local, border_col)

        out_frames.append(bgr)
        fi += 1

    cap.release()
    return out_frames, group


# ── inference ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path, num_features):
    m = build_model("tcn", num_features)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.eval()
    return m


def predict(model, x_row, mask_row):
    x    = torch.from_numpy(x_row).float().unsqueeze(0)
    mask = torch.from_numpy(mask_row).float().unsqueeze(0)
    with torch.no_grad():
        logit = model(x, mask)
        return float(torch.sigmoid(logit).item())


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",   default=NPZ_PATH)
    ap.add_argument("--out",   default=os.path.join(OUT_DIR,
                                "1st_fight_anchored_predictions.mp4"))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    # load dataset
    d         = np.load(args.npz, allow_pickle=True)
    X         = d["X"]
    mask      = d["mask"]
    y         = d["y"]
    rounds    = d["round"]
    clip_names = d["clip_name"]
    num_features = X.shape[-1]

    # load manifest for per-clip metadata
    cfg_dict  = fights.get_fight("1st_fight")
    manifest  = json.load(open(cfg_dict["manifest_path"]))
    meta_by_clip = {c["clip"]: c for c in manifest["clips"]}

    # load one model per round
    models = {}
    for r, ckpt in ROUND_TO_CKPT.items():
        if not os.path.exists(ckpt):
            print(f"  [WARN] checkpoint missing: {ckpt}")
            continue
        models[r] = load_model(ckpt, num_features)
        print(f"  Loaded checkpoint for Round{r}: {os.path.basename(ckpt)}")

    # score every clip using its OOD checkpoint
    results = []
    n_skip  = 0
    for i in range(len(y)):
        clip_file = str(clip_names[i])
        clip_path = os.path.join(CLIPS_DIR, clip_file)
        r         = int(rounds[i])
        model     = models.get(r)

        if model is None:
            print(f"  [SKIP no ckpt] {clip_file}")
            n_skip += 1
            continue
        if not os.path.exists(clip_path):
            print(f"  [SKIP no mp4]  {clip_file}")
            n_skip += 1
            continue

        prob = predict(model, X[i], mask[i])
        results.append({
            "clip_file":  clip_file,
            "clip_path":  clip_path,
            "true_label": int(y[i]),
            "pred_prob":  prob,
            "meta":       meta_by_clip.get(clip_file, {}),
        })

    if n_skip:
        print(f"  {n_skip} clips skipped")

    # assign TP/FN/TN/FP
    for r in results:
        pl = 1 if r["pred_prob"] >= args.threshold else 0
        tl = r["true_label"]
        r["group"] = ("TP" if tl==1 and pl==1 else
                       "FN" if tl==1 and pl==0 else
                       "TN" if tl==0 and pl==0 else "FP")

    counts = {g: sum(1 for r in results if r["group"]==g)
              for g in ("TP","FN","TN","FP")}
    total  = len(results)
    n_impact_pred = sum(1 for r in results if r["pred_prob"] >= args.threshold)
    print(f"\n{total} clips scored   threshold={args.threshold}")
    print(f"  TP={counts['TP']}  FN={counts['FN']}  TN={counts['TN']}  FP={counts['FP']}")
    prec = counts['TP'] / max(counts['TP']+counts['FP'], 1)
    rec  = counts['TP'] / max(counts['TP']+counts['FN'], 1)
    f1   = 2*prec*rec / max(prec+rec, 1e-9)
    print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    print(f"  Predicted impact: {n_impact_pred}/{total}")

    # sort: TP (desc prob), FN (asc prob), TN (asc prob), FP (desc prob)
    order = {"TP": 0, "FN": 1, "TN": 2, "FP": 3}
    for r in results:
        p = r["pred_prob"]
        r["sort_key"] = (order[r["group"]],
                          -p if r["group"] in ("TP","FP") else p)
    results.sort(key=lambda r: r["sort_key"])

    # write video
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, FPS, (W, H))

    print(f"\nRendering {total} clips -> {args.out}")
    current_group = None
    n_rendered    = 0

    for r in results:
        # section title card on group change
        if r["group"] != current_group:
            current_group = r["group"]
            col   = GROUP_COL[current_group]
            label = GROUP_LABEL[current_group]
            n_in_group = counts[current_group]
            tc = title_card(f"{label}\n({n_in_group} clips)", col, n_frames=int(FPS*1.5))
            for f in tc:
                writer.write(f)

        frames, group = render_clip(
            r["clip_path"], r["meta"], r["true_label"], r["pred_prob"], args.threshold
        )
        for f in frames:
            writer.write(f)
        n_rendered += 1
        if n_rendered % 20 == 0:
            print(f"  {n_rendered}/{total} clips rendered...")

    writer.release()
    print(f"\nDone. {n_rendered} clips rendered.")
    print(f"Output -> {args.out}")
    print(f"  magenta bar = anchor frame (exact impact mark or window_end proxy)")
    print(f"  Green border = TP, Orange = FN, Gray = TN, Red = FP")


if __name__ == "__main__":
    main()
