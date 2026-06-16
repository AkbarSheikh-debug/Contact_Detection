#!/usr/bin/env python3
"""
Patch classifier: 128x128 crop around the annotated click point -> LANDED vs MISS.

Two models:
  scratch : small 4-block CNN trained from scratch
  resnet  : ImageNet resnet18, frozen backbone, linear head

Two CV schemes (both 5-fold):
  random   : stratified random folds (optimistic - nearby punches leak)
  temporal : contiguous time blocks (honest for a single fight)

Usage (inside WSL mamma env):
  python tools/train_patch_cnn.py --data outputs/gt_dataset/fight1_gt.npz
"""
import argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torchvision
from torchvision import transforms as T

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class PatchDS(Dataset):
    def __init__(self, patches, labels, train):
        self.p, self.y, self.train = patches, labels, train
        self.aug = T.Compose([
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            T.RandomRotation(10),
            T.Resize(128, antialias=True),
        ])
        self.eval_t = T.Compose([T.CenterCrop(224), T.Resize(128, antialias=True)])
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.p[i]).permute(2, 0, 1).float() / 255.0
        x = self.aug(x) if self.train else self.eval_t(x)
        return self.norm(x), self.y[i]


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [3, 32, 64, 128, 256]
        self.blocks = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1),
                          nn.BatchNorm2d(ch[i+1]), nn.ReLU(),
                          nn.MaxPool2d(2))
            for i in range(4)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.5), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.blocks(x)).squeeze(1)


def make_model(kind):
    if kind == "scratch":
        return SmallCNN().to(DEV)
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(512, 1)
    return m.to(DEV)


def run_fold(kind, tr_ds, te_ds, epochs=60):
    model = make_model(kind)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=2)
    te = DataLoader(te_ds, batch_size=32, num_workers=2)
    for _ in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEV), y.float().to(DEV)
            out = model(x).reshape(-1)
            loss = F.binary_cross_entropy_with_logits(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in te:
            ps += torch.sigmoid(model(x.to(DEV)).reshape(-1)).cpu().tolist()
            ys += y.tolist()
    ps, ys = np.array(ps), np.array(ys)
    acc = ((ps > 0.5) == ys).mean()
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    return acc, auc


def folds_random(y, k=5, seed=0):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def folds_temporal(frames, k=5):
    """Contiguous time blocks by annotation frame order."""
    order = np.argsort(frames)
    blocks = np.array_split(order, k)
    out = []
    for i in range(k):
        te = blocks[i]
        tr = np.concatenate([blocks[j] for j in range(k) if j != i])
        out.append((tr, te))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    patches, labels, frames = d["patches"], d["labels"], d["frames"]
    print(f"{len(labels)} samples  pos {labels.sum()}  neg {(labels==0).sum()}  dev={DEV}")

    for kind in ("scratch", "resnet"):
        for scheme, folds in (("random", folds_random(labels)),
                              ("temporal", folds_temporal(frames))):
            accs, aucs = [], []
            for tr_i, te_i in folds:
                tr_ds = PatchDS(patches[tr_i], labels[tr_i], train=True)
                te_ds = PatchDS(patches[te_i], labels[te_i], train=False)
                acc, auc = run_fold(kind, tr_ds, te_ds, args.epochs)
                accs.append(acc); aucs.append(auc)
            print(f"[{kind:7s}|{scheme:8s}] acc {np.mean(accs):.3f}±{np.std(accs):.3f}  "
                  f"AUC {np.nanmean(aucs):.3f}±{np.nanstd(aucs):.3f}  "
                  f"(folds: {['%.2f' % a for a in aucs]})", flush=True)


if __name__ == "__main__":
    main()
