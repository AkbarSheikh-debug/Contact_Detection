#!/usr/bin/env python3
"""
Combine v9's hand-tuned score with each trained model's score, on the same
fight folder, and check honestly whether the combination beats either alone.

Evaluation uses the SAME real ground truth established earlier in this
project: cameron_vs_liam Round2 clips manually labeled via
tools/annotate_clips.py, fuzzy-matched (<=8 frame center distance, same
fighter) to this folder's ASFormer action candidates -- not a held-out
match-blocked split (that's what the training scripts already report), but
real human-verified labels on never-trained-on candidates from this video.

Combination method: RANK-average (no fitted parameters, so no extra
overfitting risk on top of the n~65 sample). v9's scores live in a narrower
band (0-~0.75, since its hard pre-filters reject many candidates outright --
those get v9_score=0 here) than the trained models' (0-1), so a raw
score-average would just let v9 drag everything down; rank-averaging fixes
that by comparing relative ORDER within each model's own candidate pool
instead of raw magnitude.

Usage:
    python tools/ensemble_v9_with_models.py --folder "C:/Users/XRIG/Downloads/1st_Impact_detection_Fixed_05062026"
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import fights
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

MODELS = ["xgb", "tcn", "asformer", "brt", "r3d18"]


def load_candidates(folder, tag):
    matches = [f for f in os.listdir(folder) if f.endswith(f"_impacts_{tag}.json")]
    if not matches:
        return None
    d = json.load(open(os.path.join(folder, matches[0])))
    by_key = {}
    for e in d["all_scored_candidates"]:
        key = (e["striker_id"], e["window_start"], e["window_end"])
        by_key[key] = e["impact_score"]
    return by_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--fight", default="cameron_vs_liam")
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("--match-tol", type=int, default=8, help="max frame-center distance for GT fuzzy match")
    args = ap.parse_args()

    v9_scores = load_candidates(args.folder, "v9")
    model_scores = {m: load_candidates(args.folder, m) for m in MODELS}
    # canonical candidate identity list: union of all keys (xgb/tcn/etc cover all 104; v9 is a subset)
    all_keys = set()
    for d in [v9_scores] + list(model_scores.values()):
        if d:
            all_keys |= set(d.keys())
    all_keys = sorted(all_keys)
    print(f"canonical candidate pool: {len(all_keys)}  (v9 scored {len(v9_scores)}/{len(all_keys)} "
          f"-- the rest were hard-rejected by v9's precision filters -> treated as v9_score=0)")

    def center(ws, we):
        return (ws + we) / 2.0

    cfg = fights.get_fight(args.fight)
    manifest = json.load(open(cfg["manifest_path"]))
    labeled = [c for c in manifest["clips"] if c["round"] == args.round and c["label"] is not None]

    pairs = []
    for c in labeled:
        cc = center(c["window_start"], c["window_end"])
        best, bestd = None, 1e9
        for key in all_keys:
            sid, ws, we = key
            if sid != c["fighter_id"]:
                continue
            d = abs(center(ws, we) - cc)
            if d < bestd:
                bestd, best = d, key
        if best is not None and bestd <= args.match_tol:
            pairs.append((c, best))
    print(f"matched {len(pairs)} of {len(labeled)} labeled round{args.round} clips to candidates "
          f"(<= {args.match_tol} frame center distance)\n")

    y_true = np.array([1 if c["label"] == "impact" else 0 for c, key in pairs])

    def rank_normalize(score_dict, keys):
        vals = np.array([score_dict.get(k, 0.0) for k in keys])
        order = vals.argsort().argsort()  # rank 0..n-1
        return order / max(1, len(vals) - 1)

    v9_rank_all = rank_normalize(v9_scores or {}, all_keys)
    v9_rank_by_key = dict(zip(all_keys, v9_rank_all))

    def report(y, prob, label):
        pred = (prob >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
        try:
            auc = roc_auc_score(y, prob)
        except ValueError:
            auc = float("nan")
        acc = accuracy_score(y, pred)
        print(f"  {label:24s} acc={acc:.3f}  P={p:.3f}  R={r:.3f}  AUC={auc:.3f}")
        return auc

    v9_match_scores = np.array([v9_scores.get(key, 0.0) if v9_scores else 0.0 for c, key in pairs])
    v9_rank_match = np.array([v9_rank_by_key[key] for c, key in pairs])
    print("=== v9 alone (real score, on matched GT) ===")
    v9_auc = report(y_true, v9_match_scores, "v9")
    print()

    for m in MODELS:
        sd = model_scores[m]
        if sd is None:
            print(f"=== {m}: no json found, skipping ===\n")
            continue
        m_rank_all = rank_normalize(sd, all_keys)
        m_rank_by_key = dict(zip(all_keys, m_rank_all))
        m_match_scores = np.array([sd.get(key, 0.0) for c, key in pairs])
        m_rank_match = np.array([m_rank_by_key[key] for c, key in pairs])
        combined_rank = 0.5 * v9_rank_match + 0.5 * m_rank_match

        print(f"=== {m} ===")
        m_auc = report(y_true, m_match_scores, m)
        combo_auc = report(y_true, combined_rank, f"v9 + {m} (rank-avg)")
        best_individual = max(v9_auc, m_auc)
        verdict = "HELPS" if combo_auc > best_individual + 0.01 else (
            "NEUTRAL" if abs(combo_auc - best_individual) <= 0.01 else "HURTS")
        print(f"  -> combo vs best individual ({best_individual:.3f}): {verdict}\n")


if __name__ == "__main__":
    main()
