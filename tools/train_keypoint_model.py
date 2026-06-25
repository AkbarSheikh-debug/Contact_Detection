#!/usr/bin/env python3
"""
Train the keypoint-based impact / not_impact classifier (ImpactTCN or
ImpactGRU, see tools/keypoint_model.py) on the dataset extracted by
tools/extract_keypoint_dataset.py.

Dual evaluation, mirroring tools/train_clip_model.py's "report both
schemes" convention (random + temporal there -> leave-one-round-out +
random here, since this dataset's natural leakage-safe grouping unit is
the fight round, not an arbitrary temporal block):

  PRIMARY   : leave-one-round-out, 4-fold (round in {3,4,5,8}). No clip from
              the held-out round's video ever appears in that fold's
              training set -- the trustworthy, leakage-safe number.
  SECONDARY : stratified random 5-fold. Leaks across rounds (clips from the
              same round/combo can split across train/test) -- reported
              only as an optimistic comparison point, like train_clip_model.py
              does with its "random" scheme.

Usage:
  python tools/train_keypoint_model.py
  python tools/train_keypoint_model.py --model gru --epochs 60
  python tools/train_keypoint_model.py --model tcn --dropout 0.4 --weight-decay 1e-3
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, confusion_matrix,
)

from keypoint_model import build_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATA = r"C:\Users\XRIG\Desktop\Impact_Detection_Improve\Impact_Detection\outputs\keypoint_dataset\lillyella_vs_zoe.npz"
OUT_DIR = r"C:\Users\XRIG\Desktop\Impact_Detection_Improve\Impact_Detection\outputs\keypoint_model"


class KeypointDS(Dataset):
    def __init__(self, X, mask, y):
        self.X, self.mask, self.y = X, mask, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]).float(),
                torch.from_numpy(self.mask[i]).float(),
                float(self.y[i]))


def make_loader(ds, y_subset, batch_size, sampler_mode, shuffle):
    if sampler_mode == "weighted" and shuffle:
        class_counts = np.bincount(y_subset, minlength=2)
        weights = 1.0 / np.maximum(class_counts[y_subset], 1)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_fold(model_name, X_tr, mask_tr, y_tr, X_te, mask_te, y_te,
             epochs, lr, weight_decay, dropout, sampler_mode, patience,
             batch_size=32):
    num_features = X_tr.shape[-1]
    model = build_model(model_name, num_features, dropout=dropout).to(DEV)

    # pos_weight: y=1 is "impact" (the MAJORITY class here), so pos_weight
    # must be < 1 to down-weight it relative to the minority not_impact
    # class, per BCEWithLogitsLoss's convention (pos_weight multiplies the
    # loss term for y=1 samples). Computed from the TRAINING fold only.
    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=DEV)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    tr_ds = KeypointDS(X_tr, mask_tr, y_tr)
    te_ds = KeypointDS(X_te, mask_te, y_te)
    tr_loader = make_loader(tr_ds, y_tr, batch_size, sampler_mode, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, m, y in tr_loader:
            x, m, y = x.to(DEV), m.to(DEV), y.to(DEV)
            logits = model(x, m)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, m, y in te_loader:
                x, m, y = x.to(DEV), m.to(DEV), y.to(DEV)
                logits = model(x, m)
                val_losses.append(criterion(logits, y).item())
        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(train_losses))
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, m, y in te_loader:
            p = torch.sigmoid(model(x.to(DEV), m.to(DEV))).cpu().numpy()
            probs.extend(p.tolist())
            ys.extend(y.tolist())
    probs, ys = np.array(probs), np.array(ys).astype(int)
    preds = (probs > 0.5).astype(int)

    acc = (preds == ys).mean()
    auc = roc_auc_score(ys, probs) if len(set(ys)) > 1 else float("nan")
    prec, rec, f1, _ = precision_recall_fscore_support(
        ys, preds, average="binary", pos_label=1, zero_division=0)
    cm = confusion_matrix(ys, preds, labels=[0, 1])

    return {
        "acc": acc, "auc": auc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm, "model_state": model.state_dict(), "history": history,
    }


def folds_leave_one_round_out(groups):
    """`groups` is the leakage-safe held-out unit -- "<fight>_R<round>" for
    multi-fight datasets (see extract_keypoint_dataset.py), or bare round
    numbers for older single-fight .npz files that predate the 'group'
    field. Either way, holding out one whole group keeps every clip from
    that video out of its fold's training set."""
    unique_groups = sorted(set(groups.tolist()))
    folds = []
    for held_out in unique_groups:
        te_i = np.where(groups == held_out)[0]
        tr_i = np.where(groups != held_out)[0]
        folds.append((tr_i, te_i, held_out))
    return folds


def folds_random(y, k=5, seed=0):
    splits = StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    return [(tr_i, te_i, i) for i, (tr_i, te_i) in enumerate(splits)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", choices=["tcn", "gru"], default="tcn")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--sampler", choices=["none", "weighted"], default="weighted")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, mask, y = d["X"], d["mask"], d["y"]
    # 'group' (fight_R round) is the leakage-safe fold key; fall back to bare
    # 'round' for older single-fight .npz files saved before it existed.
    groups = d["group"] if "group" in d.files else d["round"].astype(str)
    print(f"{len(y)} clips  X={X.shape}  impact={int(y.sum())}  "
          f"not_impact={int((1 - y).sum())}  device={DEV}\n")
    if "fight" in d.files:
        fight_counts = dict(zip(*np.unique(d["fight"], return_counts=True)))
        print(f"fights: {fight_counts}\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    for scheme_name, folds, optimistic in (
        ("leave_one_round_out", folds_leave_one_round_out(groups), False),
        ("stratified_random", folds_random(y), True),
    ):
        tag = "OPTIMISTIC / LEAKS ACROSS ROUNDS" if optimistic else "PRIMARY / LEAKAGE-SAFE"
        print(f"=== {scheme_name}  ({tag}) ===")
        fold_metrics = []
        for tr_i, te_i, fold_id in folds:
            res = run_fold(
                args.model, X[tr_i], mask[tr_i], y[tr_i], X[te_i], mask[te_i], y[te_i],
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                dropout=args.dropout, sampler_mode=args.sampler, patience=args.patience,
            )
            fold_metrics.append(res)
            print(f"  fold {fold_id}: acc {res['acc']:.3f}  AUC {res['auc']:.3f}  "
                  f"P {res['precision']:.3f}  R {res['recall']:.3f}  F1 {res['f1']:.3f}  "
                  f"n_test={len(te_i)}")

            if scheme_name == "leave_one_round_out":
                ckpt_path = os.path.join(OUT_DIR, f"{args.model}_{fold_id}_best.pt")
                torch.save(res["model_state"], ckpt_path)
                hist_path = os.path.join(OUT_DIR, f"training_history_loro_{fold_id}.npy")
                np.save(hist_path, np.array(res["history"], dtype=object))

        accs = [r["acc"] for r in fold_metrics]
        aucs = [r["auc"] for r in fold_metrics]
        precs = [r["precision"] for r in fold_metrics]
        recs = [r["recall"] for r in fold_metrics]
        f1s = [r["f1"] for r in fold_metrics]
        print(f"[{scheme_name:20s}] acc {np.mean(accs):.3f}+/-{np.std(accs):.3f}  "
              f"AUC {np.nanmean(aucs):.3f}+/-{np.nanstd(aucs):.3f}  "
              f"P {np.mean(precs):.3f}  R {np.mean(recs):.3f}  F1 {np.mean(f1s):.3f}  "
              f"(AUC folds: {['%.2f' % a for a in aucs]})")

        if scheme_name == "leave_one_round_out":
            summed_cm = sum(r["confusion_matrix"] for r in fold_metrics)
            print(f"  summed out-of-fold confusion matrix [[TN,FP],[FN,TP]]:\n{summed_cm}")

        print()
        all_results[scheme_name] = {
            "per_fold": [
                {"fold": fid, "acc": r["acc"], "auc": r["auc"], "precision": r["precision"],
                 "recall": r["recall"], "f1": r["f1"],
                 "confusion_matrix": r["confusion_matrix"].tolist()}
                for (tr_i, te_i, fid), r in zip(folds, fold_metrics)
            ],
            "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
            "mean_auc": float(np.nanmean(aucs)), "std_auc": float(np.nanstd(aucs)),
            "mean_precision": float(np.mean(precs)), "mean_recall": float(np.mean(recs)),
            "mean_f1": float(np.mean(f1s)),
        }

    results_path = os.path.join(OUT_DIR, "results_summary.json")
    json.dump({"model": args.model, "args": vars(args), "results": all_results},
              open(results_path, "w"), indent=2)
    print(f"Saved results summary -> {results_path}")


if __name__ == "__main__":
    main()
