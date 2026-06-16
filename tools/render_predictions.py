#!/usr/bin/env python3
"""
Render per-clip prediction results video.

Trains r3d_18 (±10, layer4 finetune, temporal-block 5-fold) so every clip
gets exactly one OOD prediction score.  Reads the original full-res video and
renders each 21-frame clip with overlaid:
  - Colored border  (green=correct, red=wrong)
  - GT label        (LANDED / MISS, top-left)
  - Predicted score (bar + number, bottom-right)
  - Body part       (bottom-left)
  - Frame number    (top-right)

Clips are grouped into TP / FN / TN / FP sections, each section sorted by
descending confidence so the "most-certain" examples come first.

Outputs
-------
  outputs/prediction_clips/tp_<frame>.mp4  etc.
  outputs/prediction_clips/results_reel.mp4

Usage
-----
  python tools/render_predictions.py \\
      --data   outputs/gt_dataset/fight1_gt.npz \\
      --gt     "/mnt/c/Users/XRIG/Desktop/anno_vids/(BLUE) Cameron O'Callaghan VS (RED) Liam McElhinney- round2_gt.json" \\
      --epochs 30
"""

import argparse, json, os, sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import torchvision

# ── constants ──────────────────────────────────────────────────────────────────
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 112
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3,1,1,1)
STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3,1,1,1)
T_HALF   = 10      # ±10 frames — best window from ablation
T_FRAMES = 16      # sample 16 from 21

DISP_W, DISP_H = 640, 360   # render resolution per frame
BORDER = 8                   # border width in pixels
FPS_OUT = 25.0

# group colours: TP=green, FN=orange, TN=blue, FP=red
GROUP_COLOUR = {
    "TP": (0,  200, 80),
    "FN": (20, 140, 255),
    "TN": (200, 200, 200),
    "FP": (0,  50,  220),
}
GROUP_LABEL  = {
    "TP": "TRUE POSITIVE   (LANDED, hit correct)",
    "FN": "FALSE NEGATIVE  (LANDED, predicted miss)",
    "TN": "TRUE NEGATIVE   (MISS,   predicted miss)",
    "FP": "FALSE POSITIVE  (MISS,   predicted hit)",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def win_to_wsl(path: str) -> str:
    """Convert Windows C:/... paths to /mnt/c/... when running under Linux."""
    if sys.platform.startswith("linux") and len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        rest  = path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return path


def temporal_folds(frames: np.ndarray, k=5):
    order  = np.argsort(frames)
    blocks = np.array_split(order, k)
    return [(np.concatenate([blocks[j] for j in range(k) if j != i]), blocks[i])
            for i in range(k)]


# ── dataset & model ────────────────────────────────────────────────────────────

class ClipDS(Dataset):
    def __init__(self, clips, labels, train):
        self.c, self.y, self.train = clips, labels, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        clip  = self.c[i]                        # (21, 128, 128, 3)
        jitter = clip.shape[0] - T_FRAMES
        t0 = np.random.randint(0, jitter + 1) if self.train else jitter // 2
        clip = clip[t0 : t0 + T_FRAMES]

        x = torch.from_numpy(clip).float() / 255.0   # (T, 128, 128, 3)
        x = x.permute(3, 0, 1, 2)                     # (3, T, 128, 128)

        if self.train:
            i0 = np.random.randint(0, 128 - SIZE + 1)
            j0 = np.random.randint(0, 128 - SIZE + 1)
            x  = x[:, :, i0:i0+SIZE, j0:j0+SIZE]
            if np.random.rand() < 0.5:
                x = torch.flip(x, [3])
            x = (x * np.random.uniform(0.8, 1.2)).clamp(0, 1)
        else:
            off = (128 - SIZE) // 2
            x   = x[:, :, off:off+SIZE, off:off+SIZE]

        return (x - MEAN) / STD, self.y[i]


def make_model():
    m = torchvision.models.video.r3d_18(weights="KINETICS400_V1")
    for p in m.parameters():
        p.requires_grad = False
    for p in m.layer4.parameters():
        p.requires_grad = True
    m.fc = nn.Linear(512, 1)
    return m.to(DEV)


def train_fold(clips, labels, tr_i, te_i, epochs):
    tr_ds = ClipDS(clips[tr_i], labels[tr_i], train=True)
    te_ds = ClipDS(clips[te_i], labels[te_i], train=False)
    model = make_model()
    opt   = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if n.startswith("fc")],     "lr": 1e-3},
        {"params": [p for n,p in model.named_parameters()
                    if p.requires_grad and not n.startswith("fc")], "lr": 1e-4},
    ], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr    = DataLoader(tr_ds, batch_size=8,  shuffle=True, num_workers=2)
    te    = DataLoader(te_ds, batch_size=16, num_workers=2)
    for _ in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEV), y.float().to(DEV)
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    ps = []
    with torch.no_grad():
        for x, _ in te:
            ps += torch.sigmoid(model(x.to(DEV)).squeeze(1)).cpu().tolist()
    return np.array(ps)


# ── rendering helpers ──────────────────────────────────────────────────────────

def add_border(frame_bgr: np.ndarray, colour_bgr, thickness=BORDER):
    f = frame_bgr.copy()
    h, w = f.shape[:2]
    cv2.rectangle(f, (0,0), (w-1, h-1), colour_bgr, thickness)
    return f


def put_text_bg(img, text, pos, font_scale=0.65, fg=(255,255,255), bg=(0,0,0)):
    """Draw text with a dark backing rect for legibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    tw, th = cv2.getTextSize(text, font, font_scale, 2)[0]
    x, y = pos
    cv2.rectangle(img, (x-2, y-th-4), (x+tw+2, y+4), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, fg, 2, cv2.LINE_AA)


def draw_score_bar(img, score, x, y, w=120, h=14):
    """Draw a horizontal [0→1] bar at (x,y)."""
    cv2.rectangle(img, (x,y), (x+w, y+h), (60,60,60), -1)
    fill = int(w * score)
    col  = (0,200,80) if score >= 0.5 else (0,100,200)
    cv2.rectangle(img, (x,y), (x+fill, y+h), col, -1)
    cv2.rectangle(img, (x,y), (x+w, y+h), (180,180,180), 1)


def render_clip_frames(
    video_path, centre_frame, body_part, gt_label, score, group, total_frames
):
    """
    Extract 21 frames from the original video centred on `centre_frame`,
    add overlays, and return a list of BGR numpy arrays resized to DISP_W×DISP_H.
    """
    cap   = cv2.VideoCapture(video_path)
    start = max(0, centre_frame - T_HALF)
    end   = min(total_frames - 1, centre_frame + T_HALF)
    # pad to always yield 21 frames
    frames_to_read = list(range(start, end + 1))
    while len(frames_to_read) < 21:
        frames_to_read.append(frames_to_read[-1])

    colour  = GROUP_COLOUR[group]
    gt_text = f"GT: {gt_label}"
    sc_text = f"score {score:.2f}"
    pred_text = "pred: HIT" if score >= 0.5 else "pred: MISS"
    gt_colour = (0, 220, 80) if gt_label == "LANDED" else (0, 80, 220)

    rendered = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_read[0])
    prev_pos = frames_to_read[0]

    for fno in frames_to_read:
        if fno != prev_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            bgr = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)
        prev_pos = fno + 1

        bgr = cv2.resize(bgr, (DISP_W, DISP_H))
        bgr = add_border(bgr, colour)

        # highlight the centre (contact) frame
        is_centre = (fno == centre_frame)
        if is_centre:
            cv2.rectangle(bgr, (BORDER, BORDER),
                          (DISP_W-BORDER-1, DISP_H-BORDER-1), (255,255,255), 1)

        # top-left: GT label
        put_text_bg(bgr, gt_text, (BORDER+4, BORDER+20), fg=gt_colour)
        # top-right: frame number
        fn_text = f"fr {fno}" + (" *" if is_centre else "")
        put_text_bg(bgr, fn_text, (DISP_W - 100, BORDER+20))
        # bottom-left: body part + pred
        put_text_bg(bgr, body_part, (BORDER+4, DISP_H - BORDER - 30))
        put_text_bg(bgr, pred_text, (BORDER+4, DISP_H - BORDER - 10))
        # bottom-right: score bar
        draw_score_bar(bgr, score, DISP_W - 136, DISP_H - BORDER - 28)
        put_text_bg(bgr, sc_text, (DISP_W - 136, DISP_H - BORDER - 10))

        rendered.append(bgr)

    cap.release()
    return rendered


def title_card(text, colour, n_frames=int(FPS_OUT)):
    """Return n_frames identical frames showing a group title."""
    img = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.rectangle(img, (0,0), (DISP_W-1, DISP_H-1), colour, 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for line_i, line in enumerate(text.split("\n")):
        tw, th = cv2.getTextSize(line, font, 0.9, 2)[0]
        x = (DISP_W - tw) // 2
        y = DISP_H//2 - 20 + line_i * (th + 12)
        cv2.putText(img, line, (x, y), font, 0.9, colour, 2, cv2.LINE_AA)
    return [img.copy() for _ in range(n_frames)]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   required=True,
                    help="Path to fight1_gt.npz")
    ap.add_argument("--gt",     required=True,
                    help="Path to the GT JSON, or @/path/to/file to read path from file")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out",    default="outputs/prediction_clips")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for sub in ("tp","fn","tn","fp"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    # ── load data ──────────────────────────────────────────────────────────────
    if args.gt.startswith("@"):
        with open(args.gt[1:]) as _f:
            args.gt = _f.read().strip()

    d      = np.load(args.data, allow_pickle=True)
    clips  = d["clips"]          # (93, 21, 128, 128, 3)
    labels = d["labels"]         # (93,) int
    frames = d["frames"]         # (93,) int  — annotated frame index
    body_parts = d["body_parts"] if "body_parts" in d else ["unknown"] * len(labels)

    with open(args.gt) as f:
        gt_data = json.load(f)

    video_path  = win_to_wsl(gt_data["video"])
    total_frames = gt_data.get("total_frames", 4425)
    # build frame → verdict map
    ann_map = {a["frame"]: a for a in gt_data["annotations"]}

    print(f"Loaded {len(labels)} clips  pos={labels.sum()}  neg={(labels==0).sum()}")
    print(f"Video : {video_path}")
    print(f"Device: {DEV}")
    print(f"Training {5}-fold CV (±10, layer4 finetune, {args.epochs} epochs)...")

    # ── train 5 folds, collect OOD predictions ────────────────────────────────
    folds  = temporal_folds(frames)
    scores = np.full(len(labels), np.nan)

    for fi, (tr_i, te_i) in enumerate(folds):
        print(f"  fold {fi+1}/5  train {len(tr_i)}  test {len(te_i)}", flush=True)
        ps = train_fold(clips, labels, tr_i, te_i, args.epochs)
        scores[te_i] = ps

    assert not np.any(np.isnan(scores)), "Some clips were never predicted"

    # metrics
    preds = (scores >= 0.5).astype(int)
    auc   = roc_auc_score(labels, scores)
    acc   = (preds == labels).mean()
    print(f"\nOverall  AUC={auc:.3f}  acc={acc:.3f}")

    # ── assign groups ──────────────────────────────────────────────────────────
    groups = []
    for i in range(len(labels)):
        gt    = int(labels[i])
        pred  = int(preds[i])
        if   gt == 1 and pred == 1: groups.append("TP")
        elif gt == 1 and pred == 0: groups.append("FN")
        elif gt == 0 and pred == 0: groups.append("TN")
        else:                        groups.append("FP")

    from collections import Counter
    cnt = Counter(groups)
    print(f"TP={cnt['TP']}  FN={cnt['FN']}  TN={cnt['TN']}  FP={cnt['FP']}")

    # ── render individual clips + build reel ──────────────────────────────────
    reel_writer = None
    reel_path   = os.path.join(args.out, "results_reel.mp4")

    group_order = ["TP", "FN", "TN", "FP"]
    all_clips_by_group = {g: [] for g in group_order}
    for i in range(len(labels)):
        all_clips_by_group[groups[i]].append(i)

    # within each group: TP/FP sort descending score, FN/TN sort ascending
    for g in group_order:
        idxs = all_clips_by_group[g]
        if g in ("TP", "FP"):
            idxs.sort(key=lambda i: -scores[i])
        else:
            idxs.sort(key=lambda i: scores[i])
        all_clips_by_group[g] = idxs

    # gather annotation info per clip
    ann_list = list(gt_data["annotations"])
    frame_to_ann = {a["frame"]: a for a in ann_list}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print("\nRendering clips...")
    for group in group_order:
        idxs = all_clips_by_group[group]
        colour = GROUP_COLOUR[group]

        # write title card to reel
        if reel_writer is None and idxs:
            reel_writer = cv2.VideoWriter(reel_path, fourcc, FPS_OUT, (DISP_W, DISP_H))

        if idxs:
            tc_frames = title_card(GROUP_LABEL[group] + f"\n({len(idxs)} clips)", colour)
            for f in tc_frames:
                reel_writer.write(f)

        for rank, i in enumerate(idxs):
            frame_idx = int(frames[i])
            label_str = "LANDED" if labels[i] == 1 else "MISS"
            bp        = str(body_parts[i])
            score_i   = float(scores[i])

            ann = frame_to_ann.get(frame_idx, {})
            bp  = ann.get("body_part", bp)

            print(f"  {group} rank{rank+1:02d}  fr={frame_idx:4d}  score={score_i:.2f}  GT={label_str}  {bp}",
                  flush=True)

            rendered = render_clip_frames(
                video_path, frame_idx, bp, label_str, score_i, group, total_frames
            )

            # individual MP4
            tag  = f"{group.lower()}_fr{frame_idx:04d}_s{int(score_i*100):03d}"
            path = os.path.join(args.out, group.lower(), f"{tag}.mp4")
            w    = cv2.VideoWriter(path, fourcc, FPS_OUT, (DISP_W, DISP_H))
            for fr in rendered:
                w.write(fr)
            w.release()

            # append to reel
            for fr in rendered:
                reel_writer.write(fr)

    if reel_writer:
        reel_writer.release()

    print(f"\nDone.")
    print(f"Individual clips → {args.out}/{{tp,fn,tn,fp}}/")
    print(f"Results reel     → {reel_path}")
    print(f"\nQuick stats:")
    print(f"  Precision = {cnt['TP']/(cnt['TP']+cnt['FP']+ 1e-9):.2f}")
    print(f"  Recall    = {cnt['TP']/(cnt['TP']+cnt['FN']+ 1e-9):.2f}")
    print(f"  AUC       = {auc:.3f}")


if __name__ == "__main__":
    main()
