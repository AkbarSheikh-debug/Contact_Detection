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
import os
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
    # num_workers=0: data is already fully in RAM (no disk I/O to parallelize),
    # and on Windows every DataLoader with workers>0 pays a fresh process-spawn
    # cost (re-imports torch/torchvision per worker) -- with 6 folds x 2
    # loaders that overhead dominated wall time and produced no output for
    # 30+ minutes. In-process loading is faster here.
    tr = DataLoader(tr_ds, batch_size=8, shuffle=True, num_workers=0)
    te = DataLoader(te_ds, batch_size=16, num_workers=0)
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEV), y.float().to(DEV)
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        print(f"    epoch {ep+1}/{epochs} done", flush=True)
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


def train_final(mode, clips, labels, epochs):
    """Train on the FULL dataset (all matches mixed together, no held-out
    fold) for a deployable checkpoint. This is NOT an accuracy measurement
    -- that's what the CV schemes above are for -- it's meant to be run
    after the CV has shown the approach is worth deploying."""
    ds = ClipDS(clips, labels, train=True)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    model = make_model(mode)
    head = [p for n, p in model.named_parameters() if n.startswith("fc")]
    rest = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("fc")]
    opt = torch.optim.AdamW(
        [{"params": head, "lr": 1e-3},
         {"params": rest, "lr": 1e-4}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for ep in range(epochs):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(DEV), y.float().to(DEV)
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        print(f"    epoch {ep+1}/{epochs}  train_loss={np.mean(losses):.4f}", flush=True)
    return model.state_dict()


def folds_random(y, k=5, seed=0):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def folds_temporal(frames, k=5):
    order = np.argsort(frames)
    blocks = np.array_split(order, k)
    return [(np.concatenate([blocks[j] for j in range(k) if j != i]), blocks[i])
            for i in range(k)]


def folds_match(groups):
    """Leave-one-match-out: each fold tests on one whole match, trains on
    the rest. The only honest split when clips come from multiple fights --
    random/temporal folds let clips from the same match leak across
    train/test (see MODULE_EXPLAINER.md's v6/v7 post-mortem)."""
    uniq = sorted(set(groups.tolist()))
    return [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in uniq], uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--schemes", nargs="+", default=["random", "temporal", "match"],
                     choices=["random", "temporal", "match", "none"],
                     help="pass 'none' to skip CV entirely (e.g. when only --train-final is wanted)")
    ap.add_argument("--train-final", action="store_true",
                     help="after CV (if any), also train on the FULL dataset (all matches "
                          "mixed, no held-out split) and save a deployable checkpoint")
    ap.add_argument("--final-mode", choices=["probe", "finetune"], default="finetune",
                     help="which backbone-freezing mode to use for the final checkpoint "
                          "(finetune scored higher in match-blocked CV: avg AUC 0.609 vs 0.564)")
    ap.add_argument("--out", default=None,
                     help="path for the final checkpoint (default: outputs/clip_model_FINAL_alldata.pt)")
    args = ap.parse_args()
    if "none" in args.schemes:
        args.schemes = []

    d = np.load(args.data, allow_pickle=True)
    clips, labels, frames = d["clips"], d["labels"], d["frames"]
    groups = d["groups"] if "groups" in d else None
    print(f"{len(labels)} clips {clips.shape}  pos {labels.sum()}  "
          f"neg {(labels==0).sum()}  dev={DEV}")

    schemes = []
    for name in args.schemes:
        if name == "random":
            schemes.append(("random", folds_random(labels)))
        elif name == "temporal":
            schemes.append(("temporal", folds_temporal(frames)))
        elif name == "match":
            if groups is None:
                print("[skip] 'match' scheme requested but .npz has no 'groups' array")
                continue
            folds, match_names = folds_match(groups)
            schemes.append(("match", folds))
            print(f"  match folds (leave-one-out): {match_names}")

    for mode in ("probe", "finetune"):
        for scheme, folds in schemes:
            accs, aucs = [], []
            for i, (tr_i, te_i) in enumerate(folds):
                tr_ds = ClipDS(clips[tr_i], labels[tr_i], train=True)
                te_ds = ClipDS(clips[te_i], labels[te_i], train=False)
                print(f"  [{mode}|{scheme}] fold {i+1}/{len(folds)} "
                      f"(train={len(tr_i)} test={len(te_i)})...", flush=True)
                acc, auc = run_fold(mode, tr_ds, te_ds, args.epochs)
                print(f"  [{mode}|{scheme}] fold {i+1}/{len(folds)} -> acc={acc:.3f} auc={auc:.3f}", flush=True)
                accs.append(acc); aucs.append(auc)
            print(f"[{mode:8s}|{scheme:8s}] acc {np.mean(accs):.3f}±{np.std(accs):.3f}  "
                  f"AUC {np.nanmean(aucs):.3f}±{np.nanstd(aucs):.3f}  "
                  f"(folds: {['%.2f' % a for a in aucs]})", flush=True)

    if args.train_final:
        out_path = args.out or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs", "clip_model_FINAL_alldata.pt")
        print(f"\n=== Final full-data training ({args.final_mode}, all {len(labels)} clips, "
              f"matches mixed, no held-out split) ===", flush=True)
        state = train_final(args.final_mode, clips, labels, args.epochs)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save(state, out_path)
        print(f"  saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
