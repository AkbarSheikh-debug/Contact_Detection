#!/usr/bin/env python3
"""
Train + compare 3 sequence-model architectures on the impact/not_impact
keypoint dataset (tools/extract_keypoint_dataset.py's combined.npz):

  tcn       ImpactTCN       existing baseline (tools/keypoint_model.py) --
                             pure dilated Conv1d, local receptive field only.
  asformer  ImpactASFormer  new (tools/svtas_models/asformer_binary.py) --
                             dilated conv + self-attention per block, global
                             receptive field. Inspired by D:\\SVTAS\\SVTAS's
                             ASFormer.
  brt       ImpactBRT       new (tools/svtas_models/brt_binary.py) --
                             block-local attention with a recurrent memory
                             state carried block-to-block. Inspired by
                             D:\\SVTAS\\SVTAS's BRT (Block Recurrent Transformer).

Same leave-one-round-out CV as train_keypoint_model.py (held-out unit =
"<fight>_R<round>", never split across train/test). Two independent factors
are ablated:
  - augmentation:  mirror / time-warp / scale-jitter / Gaussian-jitter,
                   applied fresh every epoch to the TRAINING fold only
                   (tools/svtas_models/augmentation.py).
  - class balance: WeightedRandomSampler so each epoch sees impact/not_impact
                   roughly 50/50, vs plain shuffling (the dataset is
                   imbalanced: ~70/30 impact/not_impact).

Usage:
  python tools/svtas_models/train_compare.py --data outputs/keypoint_dataset/combined.npz
  python tools/svtas_models/train_compare.py --epochs 50 --ablation-model tcn
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_factory import build_model, count_params, MODEL_NAMES  # noqa: E402
from augmentation import mirror, time_warp, scale_jitter, gaussian_jitter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + r"\..")
from train_keypoint_model import folds_leave_one_round_out  # noqa: E402


def folds_leave_one_match_out(fight):
    """Leave-one-MATCH-out (not round-out): holds out an entire fight, so no
    sibling round from the same match stays in training. The round-level
    grouping in folds_leave_one_round_out lets lillyella_vs_zoe's 4 rounds
    leak across each other's folds (same fighters/camera/ring in train and
    test), which inflates the AUC -- see MODULE_EXPLAINER.md's v6/v7
    leakage post-mortem for why that pattern is untrustworthy."""
    uniq = sorted(set(fight.tolist()))
    return [(np.where(fight != f)[0], np.where(fight == f)[0], f) for f in uniq]

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATA = r"C:\Users\XRIG\Desktop\Impact_Detection_Improve\Impact_Detection\outputs\keypoint_dataset\combined.npz"
OUT_DIR = r"C:\Users\XRIG\Desktop\Impact_Detection_Improve\Impact_Detection\outputs\svtas_models"


# ── Dataset with optional on-the-fly augmentation ───────────────────────────

class KeypointDS(Dataset):
    def __init__(self, X, mask, y, t_max, col_std, augment=False):
        self.X, self.mask, self.y = X, mask, y
        self.t_max, self.col_std, self.augment = t_max, col_std, augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        seq, m = self.X[i], self.mask[i]
        if self.augment:
            if np.random.random() < 0.5:
                seq, m = mirror(seq, m)
            if np.random.random() < 0.5:
                seq, m = time_warp(seq, m, self.t_max)
            if np.random.random() < 0.5:
                seq, m = scale_jitter(seq, m)
            if np.random.random() < 0.5:
                seq, m = gaussian_jitter(seq, m, self.col_std)
        return torch.from_numpy(seq).float(), torch.from_numpy(m).float(), float(self.y[i])


def make_loader(ds, y_subset, batch_size, balance, shuffle):
    if balance and shuffle:
        class_counts = np.bincount(y_subset, minlength=2)
        weights = 1.0 / np.maximum(class_counts[y_subset], 1)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── Single fold ──────────────────────────────────────────────────────────────

def run_fold(model_name, X_tr, mask_tr, y_tr, X_te, mask_te, y_te, t_max,
             augment, balance, epochs, lr, weight_decay, dropout, patience, batch_size=32):
    num_features = X_tr.shape[-1]
    model = build_model(model_name, num_features, dropout=dropout).to(DEV)

    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=DEV)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    col_std = X_tr.std(axis=(0, 1)) + 1e-6
    tr_ds = KeypointDS(X_tr, mask_tr, y_tr, t_max, col_std, augment=augment)
    te_ds = KeypointDS(X_te, mask_te, y_te, t_max, col_std, augment=False)
    tr_loader = make_loader(tr_ds, y_tr, batch_size, balance, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for x, m, y in tr_loader:
            x, m, y = x.to(DEV), m.to(DEV), y.to(DEV)
            loss = criterion(model(x, m), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, m, y in te_loader:
                x, m, y = x.to(DEV), m.to(DEV), y.to(DEV)
                val_losses.append(criterion(model(x, m), y).item())
        val_loss = float(np.mean(val_losses))
        sched.step(val_loss)

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
    prec, rec, f1, _ = precision_recall_fscore_support(ys, preds, average="binary", pos_label=1, zero_division=0)
    cm = confusion_matrix(ys, preds, labels=[0, 1])
    return {"acc": acc, "auc": auc, "precision": prec, "recall": rec, "f1": f1,
            "confusion_matrix": cm.tolist(), "n_params": count_params(model),
            "model_state": model.state_dict()}


def run_config(model_name, X, mask, y, groups, t_max, augment, balance,
                epochs, lr, weight_decay, dropout, patience, save_ckpt_dir=None,
                fold_fn=folds_leave_one_round_out):
    folds = fold_fn(groups)
    fold_metrics = []
    for tr_i, te_i, fold_id in folds:
        res = run_fold(model_name, X[tr_i], mask[tr_i], y[tr_i], X[te_i], mask[te_i], y[te_i],
                        t_max, augment, balance, epochs, lr, weight_decay, dropout, patience)
        res["fold"] = fold_id
        res["n_test"] = int(len(te_i))
        fold_metrics.append(res)
        print(f"    fold {fold_id}: acc {res['acc']:.3f}  AUC {res['auc']:.3f}  "
              f"P {res['precision']:.3f}  R {res['recall']:.3f}  F1 {res['f1']:.3f}")
        if save_ckpt_dir:
            os.makedirs(save_ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(save_ckpt_dir, f"{model_name}_{fold_id}_best.pt")
            torch.save(res["model_state"], ckpt_path)
        del res["model_state"]  # don't keep weights in the JSON-serialized summary

    accs = [r["acc"] for r in fold_metrics]
    aucs = [r["auc"] for r in fold_metrics]
    f1s = [r["f1"] for r in fold_metrics]
    summary = {
        "model": model_name, "augment": augment, "balance": balance,
        "n_params": fold_metrics[0]["n_params"],
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "mean_auc": float(np.nanmean(aucs)), "std_auc": float(np.nanstd(aucs)),
        "mean_precision": float(np.mean([r["precision"] for r in fold_metrics])),
        "mean_recall": float(np.mean([r["recall"] for r in fold_metrics])),
        "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
        "per_fold": fold_metrics,
    }
    print(f"  [{model_name:9s} aug={augment!s:5s} bal={balance!s:5s}] "
          f"acc {summary['mean_acc']:.3f}+/-{summary['std_acc']:.3f}  "
          f"AUC {summary['mean_auc']:.3f}+/-{summary['std_auc']:.3f}  "
          f"F1 {summary['mean_f1']:.3f}+/-{summary['std_f1']:.3f}  "
          f"params={summary['n_params']:,}")
    return summary


def train_final(model_name, X, mask, y, t_max, augment, balance,
                 epochs, lr, weight_decay, dropout, batch_size=32):
    """Train on the FULL dataset (no held-out fold) for a deployable
    checkpoint. No early stopping (there's no held-out val set to stop on
    by design) -- just runs the fixed epoch budget. This is meant to be run
    AFTER the CV above has told you the model is worth deploying; it is not
    itself an accuracy measurement."""
    num_features = X.shape[-1]
    model = build_model(model_name, num_features, dropout=dropout).to(DEV)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=DEV)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    col_std = X.std(axis=(0, 1)) + 1e-6
    ds = KeypointDS(X, mask, y, t_max, col_std, augment=augment)
    loader = make_loader(ds, y, batch_size, balance, shuffle=True)

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, m, yb in loader:
            x, m, yb = x.to(DEV), m.to(DEV), yb.to(DEV)
            loss = criterion(model(x, m), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch+1}/{epochs}  train_loss={np.mean(losses):.4f}", flush=True)

    return model.state_dict(), count_params(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--eval-scheme", choices=["round", "match"], default="round",
                     help="round: leave-one-round-out (lets sibling rounds of the same "
                          "match leak across folds). match: leave-one-MATCH-out (every "
                          "round of the held-out fight is excluded from training -- the "
                          "honest comparison point against tools/train_clip_model.py and "
                          "tools/train_xgboost_impact.py's match-blocked CV).")
    ap.add_argument("--skip-ablation", action="store_true",
                     help="only run the headline aug+balance-on 3-model comparison, skip the ablation grid")
    ap.add_argument("--ablation-model", default="tcn", choices=MODEL_NAMES,
                     help="which model gets the full 4-cell aug x balance ablation grid "
                          "(the other models only get the no-aug/no-balance vs full endpoints)")
    ap.add_argument("--train-final", action="store_true",
                     help="after CV, also train each model on the FULL combined dataset "
                          "(no held-out fold) and save a deployable checkpoint")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, mask, y = d["X"], d["mask"], d["y"]
    if args.eval_scheme == "match":
        groups = d["fight"]
        fold_fn = folds_leave_one_match_out
    else:
        groups = d["group"] if "group" in d.files else d["round"].astype(str)
        fold_fn = folds_leave_one_round_out
    t_max = X.shape[1]
    print(f"{len(y)} clips  X={X.shape}  impact={int(y.sum())}  not_impact={int((1 - y).sum())}  "
          f"eval_scheme={args.eval_scheme}  device={DEV}\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []

    print("=== Headline 3-model comparison (augmentation + class-balance ON) ===")
    for model_name in ("tcn", "asformer", "brt"):
        print(f"\n-- {model_name} --")
        summary = run_config(model_name, X, mask, y, groups, t_max,
                              augment=True, balance=True,
                              epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                              dropout=args.dropout, patience=args.patience, fold_fn=fold_fn)
        all_results.append(summary)

    if not args.skip_ablation:
        print(f"\n=== Ablation: augmentation / class-balance, full grid on '{args.ablation_model}', "
              f"endpoints on the other two ===")
        for model_name in ("tcn", "asformer", "brt"):
            configs = [(False, False), (True, True)]
            if model_name == args.ablation_model:
                configs = [(False, False), (True, False), (False, True), (True, True)]
            for augment, balance in configs:
                if augment and balance:
                    continue  # already ran as part of the headline comparison above
                print(f"\n-- {model_name}  aug={augment} bal={balance} --")
                summary = run_config(model_name, X, mask, y, groups, t_max,
                                      augment=augment, balance=balance,
                                      epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                      dropout=args.dropout, patience=args.patience, fold_fn=fold_fn)
                all_results.append(summary)

    if args.train_final:
        print("\n=== Final full-data training (no held-out fold, for deployment) ===")
        for model_name in ("tcn", "asformer", "brt"):
            print(f"\n-- {model_name} (final, all {len(y)} clips) --")
            state, n_params = train_final(model_name, X, mask, y, t_max,
                                           augment=True, balance=True,
                                           epochs=args.epochs, lr=args.lr,
                                           weight_decay=args.weight_decay, dropout=args.dropout)
            ckpt_path = os.path.join(OUT_DIR, f"{model_name}_FINAL_alldata.pt")
            torch.save(state, ckpt_path)
            print(f"  saved -> {ckpt_path}  ({n_params:,} params)")

    out_name = "comparison_results.json" if args.eval_scheme == "round" else f"comparison_results_{args.eval_scheme}.json"
    out_path = os.path.join(OUT_DIR, out_name)
    json.dump({"data": args.data, "args": vars(args), "results": all_results},
               open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
