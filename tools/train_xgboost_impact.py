#!/usr/bin/env python3
"""
XGBoost impact classifier trained on manually-labeled clips from tools/annotate_clips.py.

This revisits the approach abandoned in detectors/fusion/v6.py / v7.py (see
MODULE_EXPLAINER.md): a classifier trained on ~31 GT labels from one match
overfit badly (good CV AUC, no better than chance held out). That data
problem is largely fixed now -- the annotation tool has produced labels
across THREE matches:
    lillyella_vs_zoe  496 clips (4 rounds)
    cameron_vs_liam   259 clips (3 rounds, but 2 rounds are missing one
                       fighter's SAM3D keypoints -> bbox-only -> dropped)
    jamie_vs_ryan     234 clips (2 rounds)

Features reuse the v9 (detectors/fusion/v9.py) geometry/physics signal
extractors directly -- same wrist-body gap, arm extension, deceleration,
head-reaction, approach-rate, guard-interception, bbox-IoU code v9 uses for
its hand-tuned weighted sum. No audio / contact_events features: this
dataset has no full_analysis.json (CLAUDE.md already found audio dead for
this task and contact_events unreliable), so those columns would be
constant/zero -- pure noise to a tree model.

Evaluation is reported TWO ways to make the leakage risk visible:
  - random 5-fold CV (stratified by label)       -- inflated, do not trust
  - leave-one-match-out CV (train on 2, test on 1) -- honest generalization

Usage:
    python tools/train_xgboost_impact.py
    python tools/train_xgboost_impact.py --save outputs/xgb_impact_v1.json
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectors.fusion import v9
from dataset import fights

FEATURE_NAMES = [
    "gap_sw", "s_gap", "region_w", "is_head", "is_torso",
    "arm_ext", "s_ext", "wrist_peak", "s_speed",
    "wrist_decel", "head_react", "approach_rate", "guard_open",
    "bbox_iou", "action_confidence", "window_len_frames",
]

FIGHTS_TO_USE = ["lillyella_vs_zoe", "cameron_vs_liam", "jamie_vs_ryan", "1st_fight", "2nd_fight"]


def load_round_data(out_base, round_id):
    sam3d_path = os.path.join(out_base, f"Round{round_id}", "sam3d.json")
    actions_path = os.path.join(out_base, f"Round{round_id}", "actions.json")
    s = json.load(open(sam3d_path))
    persons = {0: {e["frame"]: e for e in s.get("0", [])},
               1: {e["frame"]: e for e in s.get("1", [])}}
    has_kp = {0: bool(s.get("0")) and "world_coords" in s["0"][0],
              1: bool(s.get("1")) and "world_coords" in s["1"][0]}
    a = json.load(open(actions_path))
    conf_by_key = {}
    for act in a["actions"]:
        key = (act["fighter_id"], act["window_start"], act["window_end"])
        conf_by_key[key] = act.get("confidence", 1.0)
    return persons, has_kp, conf_by_key


def extract_features(clip, persons, conf_by_key, return_meta=False):
    sid = clip["fighter_id"]
    rid = 1 - sid
    sp, rp = persons.get(sid, {}), persons.get(rid, {})
    ws, we = clip["window_start"] - 3, clip["window_end"] + 5

    best_f, best_gap, best_reg = clip["window_start"], 1e9, "torso"
    for f in range(ws, we + 1):
        se = sp.get(f)
        re = rp.get(f)
        g, reg = v9.min_wrist_body_gap(v9.wc(se) if se else None,
                                        v9.wc(re) if re else None)
        if g is not None and g < best_gap:
            best_gap, best_f, best_reg = g, f, reg
    if best_gap >= 1e9:
        return None  # no usable keypoints in this window (missing SAM3D)

    s_gap = max(0.0, min(1.0, (1.3 - best_gap) / 1.0))
    region_w = v9.REGION_W.get(best_reg, 0.5)

    ext = max((v9.arm_extension(sp.get(f)) for f in range(best_f - 1, best_f + 2)
               if sp.get(f)), default=0.0)
    s_ext = max(0.0, min(1.0, (ext - 0.45) / 0.40))
    iou = v9.mean_iou(sp, rp, best_f)
    peak = v9.wrist_peak_speed(sp, best_f)
    s_speed = max(0.0, min(1.0, (peak - 0.40) / 0.60))
    decel = v9.wrist_decel(sp, best_f)
    react = v9.head_reaction(rp, best_f)
    approach = v9.approach_rate(sp, rp, best_f)
    guard = v9.guard_interception(sp, rp, best_f)

    conf = conf_by_key.get((sid, clip["window_start"], clip["window_end"]), 1.0)
    win_len = clip["window_end"] - clip["window_start"]

    feats = [
        best_gap, s_gap, region_w, float(best_reg == "head"), float(best_reg == "torso"),
        ext, s_ext, peak, s_speed,
        decel, react, approach, guard,
        iou, conf, win_len,
    ]
    if return_meta:
        return feats, best_f, best_reg
    return feats


def build_dataset():
    rows, labels, groups, meta = [], [], [], []
    skipped_no_kp = 0
    for fight_name in FIGHTS_TO_USE:
        cfg = fights.get_fight(fight_name)
        manifest = json.load(open(cfg["manifest_path"]))
        clips_by_round = defaultdict(list)
        for c in manifest["clips"]:
            if c["label"] is not None:
                clips_by_round[c["round"]].append(c)

        for round_id, clips in clips_by_round.items():
            persons, has_kp, conf_by_key = load_round_data(cfg["out_base"], round_id)
            if not (has_kp[0] and has_kp[1]):
                print(f"  [{fight_name} Round{round_id}] missing SAM3D keypoints for "
                      f"fighter {'0' if not has_kp[0] else '1'} -> skipping "
                      f"{len(clips)} clips")
                skipped_no_kp += len(clips)
                continue
            for c in clips:
                feats = extract_features(c, persons, conf_by_key)
                if feats is None:
                    skipped_no_kp += 1
                    continue
                rows.append(feats)
                labels.append(1 if c["label"] == "impact" else 0)
                groups.append(fight_name)
                meta.append({"fight": fight_name, "round": round_id, "clip": c["clip"]})

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    print(f"\nBuilt dataset: {len(y)} clips usable ({skipped_no_kp} skipped: "
          f"missing keypoints / bbox-only rounds)")
    for fname in FIGHTS_TO_USE:
        n = sum(1 for g in groups if g == fname)
        n_pos = sum(1 for g, lab in zip(groups, y) if g == fname and lab == 1)
        print(f"  {fname:18s} n={n:4d}  impact={n_pos:4d}  not_impact={n - n_pos:4d}")
    return X, y, np.array(groups), meta


def report(y_true, y_pred, y_prob, label):
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  roc_auc_score, accuracy_score)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    print(f"  {label:28s} n={len(y_true):4d}  acc={acc:.3f}  P={p:.3f}  "
          f"R={r:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    return dict(n=len(y_true), accuracy=acc, precision=p, recall=r, f1=f1, auc=auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str, default=None,
                     help="path to save the final model (trained on all data), e.g. outputs/xgb_impact.json")
    ap.add_argument("--thr", type=float, default=0.5, help="probability threshold for P/R/F1")
    args = ap.parse_args()

    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold

    X, y, groups, meta = build_dataset()
    if len(y) < 20:
        raise SystemExit("Not enough labeled clips with usable keypoints to train.")

    def make_model():
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1,
        )

    print("\n=== Random 5-fold CV (stratified by label, NOT by match) ===")
    print("    (inflated -- clips from the same match leak across train/test)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    rand_metrics = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        model = make_model()
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        pred = (prob >= args.thr).astype(int)
        rand_metrics.append(report(y[te], pred, prob, f"fold {fold}"))
    avg = {k: float(np.mean([m[k] for m in rand_metrics])) for k in rand_metrics[0] if k != "n"}
    print(f"  {'AVERAGE':28s} P={avg['precision']:.3f}  R={avg['recall']:.3f}  "
          f"F1={avg['f1']:.3f}  AUC={avg['auc']:.3f}")

    print("\n=== Leave-one-match-out CV (train on 2 matches, test on held-out match) ===")
    print("    (honest -- this is the number that matters)")
    lomo_metrics = {}
    for test_fight in FIGHTS_TO_USE:
        te = groups == test_fight
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        model = make_model()
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        pred = (prob >= args.thr).astype(int)
        lomo_metrics[test_fight] = report(y[te], pred, prob, f"test={test_fight}")
    if lomo_metrics:
        avg = {k: float(np.mean([m[k] for m in lomo_metrics.values()])) for k in
               next(iter(lomo_metrics.values())) if k != "n"}
        print(f"  {'AVERAGE':28s} P={avg['precision']:.3f}  R={avg['recall']:.3f}  "
              f"F1={avg['f1']:.3f}  AUC={avg['auc']:.3f}")

    print("\n=== Feature importance (model trained on ALL data) ===")
    final_model = make_model()
    final_model.fit(X, y)
    importances = final_model.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda t: -t[1]):
        print(f"  {name:20s} {imp:.4f}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        final_model.save_model(args.save)
        meta_path = os.path.splitext(args.save)[0] + "_meta.json"
        json.dump({
            "feature_names": FEATURE_NAMES,
            "fights_used": FIGHTS_TO_USE,
            "n_train": len(y),
            "random_cv": avg if False else {k: float(np.mean([m[k] for m in rand_metrics])) for k in rand_metrics[0] if k != "n"},
            "lomo_cv": {f: m for f, m in lomo_metrics.items()},
        }, open(meta_path, "w"), indent=2)
        print(f"\nSaved model -> {args.save}")
        print(f"Saved metrics/meta -> {meta_path}")


if __name__ == "__main__":
    main()
