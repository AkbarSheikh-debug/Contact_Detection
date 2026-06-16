#!/usr/bin/env python3
"""
Temporal window ablation: ±3, ±6, ±8, ±10 frame clips vs AUC.
Reuses the existing fight1_gt.npz (±10 clips, 21 frames).
Temporally center-subsets to each smaller window — no re-extraction needed.

T_FRAMES used per window:
  ±3  → T=6   (all frames minus 1 for minimal train jitter)
  ±6  → T=12  (all frames minus 1)
  ±8  → T=16  (all frames minus 1, matches r3d_18 native)
  ±10 → T=16  (random 16-frame crop from 21 — current baseline)

CV: honest temporal-block 5-fold (test on unseen part of round).
Model: r3d_18 Kinetics pretrained, layer4 + fc fine-tuned.

Usage:
  python tools/ablation_window.py --data outputs/gt_dataset/fight1_gt.npz
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import torchvision

DEV  = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 112
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3,1,1,1)
STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3,1,1,1)

# (T_HALF, T_FRAMES) pairs to test
CONFIGS = [
    (3,  6),   # ±3  frames → 7 total, sample 6
    (6,  12),  # ±6  frames → 13 total, sample 12
    (8,  16),  # ±8  frames → 17 total, sample 16
    (10, 16),  # ±10 frames → 21 total, sample 16  ← baseline
]

# Middle index of the full 21-frame clip (T_HALF_MAX=10 → centre=10)
T_HALF_MAX = 10
CENTRE     = T_HALF_MAX          # index of the annotated frame inside a 21-frame clip


def subset_clips(clips_21, t_half):
    """Centre-crop clips to 2*t_half+1 frames around the annotated frame."""
    lo = CENTRE - t_half
    hi = CENTRE + t_half + 1
    return clips_21[:, lo:hi]   # (N, 2*t_half+1, H, W, C)


class ClipDS(Dataset):
    def __init__(self, clips, labels, t_frames, train):
        self.c, self.y, self.T, self.train = clips, labels, t_frames, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        clip  = self.c[i]                        # (win, 128, 128, 3)
        total = clip.shape[0]
        jitter = total - self.T
        if self.train:
            t0 = np.random.randint(0, jitter + 1)
        else:
            t0 = jitter // 2
        clip = clip[t0 : t0 + self.T]

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


def run_fold(clips_sub, labels, tr_i, te_i, t_frames, epochs=30):
    tr_ds = ClipDS(clips_sub[tr_i], labels[tr_i], t_frames, train=True)
    te_ds = ClipDS(clips_sub[te_i], labels[te_i], t_frames, train=False)
    model = make_model()
    opt   = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if n.startswith("fc")],     "lr": 1e-3},
        {"params": [p for n,p in model.named_parameters() if p.requires_grad
                    and not n.startswith("fc")], "lr": 1e-4},
    ], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr    = DataLoader(tr_ds, batch_size=8, shuffle=True, num_workers=2)
    te    = DataLoader(te_ds, batch_size=16, num_workers=2)
    for _ in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEV), y.float().to(DEV)
            loss  = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in te:
            ps += torch.sigmoid(model(x.to(DEV)).squeeze(1)).cpu().tolist()
            ys += y.tolist()
    ps, ys = np.array(ps), np.array(ys)
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    acc = ((ps > 0.5) == ys).mean()
    return auc, acc


def temporal_folds(frames, k=5):
    order  = np.argsort(frames)
    blocks = np.array_split(order, k)
    return [(np.concatenate([blocks[j] for j in range(k) if j != i]), blocks[i])
            for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    d      = np.load(args.data, allow_pickle=True)
    clips21, labels, frames = d["clips"], d["labels"], d["frames"]
    print(f"Loaded {len(labels)} clips  pos {labels.sum()}  neg {(labels==0).sum()}  dev={DEV}")
    print(f"Running ablation: {[f'±{t}' for t,_ in CONFIGS]}\n")

    folds   = temporal_folds(frames)
    results = []

    for t_half, t_frames in CONFIGS:
        clips_sub = subset_clips(clips21, t_half)
        win       = 2 * t_half + 1
        print(f"Window ±{t_half:2d}  ({win:2d} frames total, sample {t_frames})", flush=True)
        aucs, accs = [], []
        for fi, (tr_i, te_i) in enumerate(folds):
            auc, acc = run_fold(clips_sub, labels, tr_i, te_i, t_frames, args.epochs)
            aucs.append(auc); accs.append(acc)
            print(f"  fold {fi+1}/5  AUC={auc:.3f}  acc={acc:.3f}", flush=True)
        mean_auc = np.nanmean(aucs)
        std_auc  = np.nanstd(aucs)
        print(f"  → mean AUC {mean_auc:.3f} ± {std_auc:.3f}\n", flush=True)
        results.append((t_half, t_frames, mean_auc, std_auc, aucs))

    # ── summary table ──────────────────────────────────────────────────────────
    print("=" * 62)
    print(f"{'Window':>8}  {'Frames':>6}  {'AUC mean':>10}  {'±std':>6}  folds")
    print("-" * 62)
    best_auc = max(r[2] for r in results)
    for t_half, t_frames, mean_auc, std_auc, aucs in results:
        tag = "  ← BEST" if abs(mean_auc - best_auc) < 1e-9 else ""
        fold_str = " ".join(f"{a:.2f}" for a in aucs)
        print(f"  ±{t_half:2d}     {2*t_half+1:3d}fr    {mean_auc:.3f}      {std_auc:.3f}   [{fold_str}]{tag}")
    print("=" * 62)


if __name__ == "__main__":
    main()
