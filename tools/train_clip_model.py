#!/usr/bin/env python3
"""
Temporal clip classifier: 21-frame clip (+/-10 around the annotated impact frame,
320px crop around the click, resized 128) -> LANDED vs MISS.

Model: torchvision r3d_18 pretrained on Kinetics-400.
  probe    : freeze all, train fc only
  finetune : freeze stem+layer1-3, train layer4 + fc (low LR)

CV: random stratified 5-fold AND temporal-block 5-fold.

Usage (inside WSL mamma env):
  python tools/train_clip_model.py --data outputs/gt_dataset/fight1_gt.npz
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torchvision

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
T_FRAMES = 16   # r3d_18 native clip length
SIZE = 112      # r3d_18 native spatial size


class ClipDS(Dataset):
    def __init__(self, clips, labels, train):
        self.c, self.y, self.train = clips, labels, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        clip = self.c[i]                      # (21,128,128,3) uint8
        T_total = clip.shape[0]
        if self.train:
            t0 = np.random.randint(0, T_total - T_FRAMES + 1)
        else:
            t0 = (T_total - T_FRAMES) // 2
        clip = clip[t0:t0 + T_FRAMES]

        x = torch.from_numpy(clip).float() / 255.0   # (16,128,128,3)
        x = x.permute(3, 0, 1, 2)                    # (3,16,128,128)

        if self.train:
            # random spatial crop 112 + hflip + brightness jitter
            i0 = np.random.randint(0, 128 - SIZE + 1)
            j0 = np.random.randint(0, 128 - SIZE + 1)
            x = x[:, :, i0:i0 + SIZE, j0:j0 + SIZE]
            if np.random.rand() < 0.5:
                x = torch.flip(x, [3])
            x = (x * np.random.uniform(0.8, 1.2)).clamp(0, 1)
        else:
            off = (128 - SIZE) // 2
            x = x[:, :, off:off + SIZE, off:off + SIZE]

        return (x - MEAN) / STD, self.y[i]


def make_model(mode):
    m = torchvision.models.video.r3d_18(weights="KINETICS400_V1")
    for p in m.parameters():
        p.requires_grad = False
    if mode == "finetune":
        for p in m.layer4.parameters():
            p.requires_grad = True
    m.fc = nn.Linear(512, 1)
    return m.to(DEV)


def run_fold(mode, tr_ds, te_ds, epochs):
    model = make_model(mode)
    head = [p for n, p in model.named_parameters() if n.startswith("fc")]
    rest = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("fc")]
    opt = torch.optim.AdamW(
        [{"params": head, "lr": 1e-3},
         {"params": rest, "lr": 1e-4}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr = DataLoader(tr_ds, batch_size=8, shuffle=True, num_workers=2)
    te = DataLoader(te_ds, batch_size=16, num_workers=2)
    for _ in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEV), y.float().to(DEV)
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in te:
            ps += torch.sigmoid(model(x.to(DEV)).squeeze(1)).cpu().tolist()
            ys += y.tolist()
    ps, ys = np.array(ps), np.array(ys)
    acc = ((ps > 0.5) == ys).mean()
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    return acc, auc


def folds_random(y, k=5, seed=0):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def folds_temporal(frames, k=5):
    order = np.argsort(frames)
    blocks = np.array_split(order, k)
    return [(np.concatenate([blocks[j] for j in range(k) if j != i]), blocks[i])
            for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    clips, labels, frames = d["clips"], d["labels"], d["frames"]
    print(f"{len(labels)} clips {clips.shape}  pos {labels.sum()}  "
          f"neg {(labels==0).sum()}  dev={DEV}")

    for mode in ("probe", "finetune"):
        for scheme, folds in (("random", folds_random(labels)),
                              ("temporal", folds_temporal(frames))):
            accs, aucs = [], []
            for tr_i, te_i in folds:
                tr_ds = ClipDS(clips[tr_i], labels[tr_i], train=True)
                te_ds = ClipDS(clips[te_i], labels[te_i], train=False)
                acc, auc = run_fold(mode, tr_ds, te_ds, args.epochs)
                accs.append(acc); aucs.append(auc)
            print(f"[{mode:8s}|{scheme:8s}] acc {np.mean(accs):.3f}±{np.std(accs):.3f}  "
                  f"AUC {np.nanmean(aucs):.3f}±{np.nanstd(aucs):.3f}  "
                  f"(folds: {['%.2f' % a for a in aucs]})", flush=True)


if __name__ == "__main__":
    main()
