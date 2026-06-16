#!/usr/bin/env python3
"""
Evaluate v9 detector (raw + VLM-verified) against human frame-level ground truth
from the annotation tool (<video>_gt.json).

Matching: a detection at frame d matches a GT LANDED at frame g if |d-g| <= tol.
Greedy 1-to-1 matching (each GT matches at most one detection and vice versa),
closest pairs first.

Usage:
  python tools/eval_v9_vs_gt.py --gt "<gt.json>" --folder "<fight folder>" [--tols 4 8 12]
"""
import os, json, glob, argparse


def load_gt(path):
    d = json.load(open(path, encoding="utf-8"))
    anns = d["annotations"]
    landed = sorted(a["frame"] for a in anns if a["verdict"] == "LANDED")
    missed = sorted(a["frame"] for a in anns if a["verdict"] == "MISS")
    other  = sorted(a["frame"] for a in anns if a["verdict"] not in ("LANDED", "MISS"))
    return landed, missed, other


def greedy_match(dets, gts, tol):
    """Return (matched_pairs, unmatched_dets, unmatched_gts). 1-to-1, closest first."""
    pairs = []
    for d in dets:
        for g in gts:
            if abs(d - g) <= tol:
                pairs.append((abs(d - g), d, g))
    pairs.sort()
    used_d, used_g, matched = set(), set(), []
    for dist, d, g in pairs:
        if d in used_d or g in used_g:
            continue
        used_d.add(d); used_g.add(g)
        matched.append((d, g))
    un_d = [d for d in dets if d not in used_d]
    un_g = [g for g in gts if g not in used_g]
    return matched, un_d, un_g


def prf(n_match, n_det, n_gt):
    p = n_match / n_det if n_det else 0.0
    r = n_match / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def eval_set(name, dets, gt_landed, gt_missed, tol):
    matched, fp_dets, fn_gts = greedy_match(dets, gt_landed, tol)
    p, r, f = prf(len(matched), len(dets), len(gt_landed))
    # of the false-positive detections, how many sit on an annotated MISS?
    m2, _, _ = greedy_match(fp_dets, gt_missed, tol)
    print(f"\n  {name}  (n_det={len(dets)}, tol=±{tol})")
    print(f"    precision {p:.3f}   recall {r:.3f}   F1 {f:.3f}")
    print(f"    TP {len(matched)}  FP {len(fp_dets)}  FN {len(fn_gts)}")
    print(f"    of the {len(fp_dets)} FPs, {len(m2)} sit on an annotated MISS "
          f"(detector fired on a thrown-but-missed punch)")
    if fn_gts:
        print(f"    missed GT frames: {fn_gts}")
    return dict(name=name, tol=tol, n_det=len(dets), precision=round(p, 4),
                recall=round(r, 4), f1=round(f, 4),
                tp=len(matched), fp=len(fp_dets), fn=len(fn_gts),
                fp_on_annotated_miss=len(m2),
                matched=[list(x) for x in matched], missed_gt=fn_gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--tols", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt_landed, gt_missed, gt_other = load_gt(args.gt)
    print(f"GT: {len(gt_landed)} LANDED, {len(gt_missed)} MISS, {len(gt_other)} other")

    raw_p = glob.glob(os.path.join(args.folder, "*_impacts_v9.json"))[0]
    raw = json.load(open(raw_p, encoding="utf-8"))
    dets_raw = sorted(i["impact_frame"] for i in raw["impacts"])
    cands = raw.get("all_scored_candidates", [])

    ver_p = glob.glob(os.path.join(args.folder, "*_impacts_v9_verified.json"))
    dets_ver = []
    if ver_p:
        ver = json.load(open(ver_p[0], encoding="utf-8"))
        dets_ver = sorted(i["impact_frame"] for i in ver["impacts"])

    # detections at the previously-tuned threshold 0.45
    dets_045 = sorted(i["impact_frame"] for i in raw["impacts"]
                      if i["impact_score"] >= 0.45)

    results = []
    for tol in args.tols:
        print(f"\n{'='*64}\nTOLERANCE ±{tol} frames")
        results.append(eval_set("v9 raw @ thr 0.30", dets_raw, gt_landed, gt_missed, tol))
        results.append(eval_set("v9 @ thr 0.45", dets_045, gt_landed, gt_missed, tol))
        if dets_ver:
            results.append(eval_set("v9 + VLM verified (LANDED only)",
                                    dets_ver, gt_landed, gt_missed, tol))

    # candidate-level recall ceiling: even the full 42-candidate scored list
    if cands:
        all_cand_frames = sorted(c["impact_frame"] for c in cands)
        print(f"\n{'='*64}\nRECALL CEILING (all {len(all_cand_frames)} scored candidates, "
              f"any threshold)")
        for tol in args.tols:
            m, _, fn = greedy_match(all_cand_frames, gt_landed, tol)
            print(f"  tol ±{tol}: best possible recall {len(m)}/{len(gt_landed)} "
                  f"= {len(m)/len(gt_landed):.3f}")

    if args.out:
        json.dump(dict(gt=args.gt, folder=args.folder, results=results),
                  open(args.out, "w"), indent=1)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
