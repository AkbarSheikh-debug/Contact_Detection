#!/usr/bin/env python3
"""
SAM3D Data Quality Diagnostic
=============================
Checks whether the *new* SAM3D export (the one that added `world_coords`,
`keypoint_conf`, `world_coords_reliable`, and top-level `contact_events`)
actually fixes the cross-person depth problem documented in NOTE_FOR_TEAMMATE.md.

Run this BEFORE trusting any 3D-distance gate.  It prints, with hard numbers:

  1. world_coords cross-person head-Z agreement   (should be ~0 m for two
     fighters standing in the same ring; large values == still broken)
  2. keypoint_conf variability                     (constant 1.0 == placeholder,
     carries no occlusion signal)
  3. contact_events probability / distance / region distribution
     (the one genuinely usable new field)

Usage:
    python diagnose_world_coords.py
    python diagnose_world_coords.py --sam3d /path/to/<id>_sam3d.json
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import os
import json
import argparse
from collections import Counter

import numpy as np

DEFAULT_FOLDER = r"/home/jake/Downloads/sam3d_with_world_coords"
DEFAULT_SAM3D = os.path.join(
    DEFAULT_FOLDER, "fd7a77fd-588f-43ff-925f-ff5a648a246d_sam3d.json"
)
HEAD_KP = 0  # nose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3d", default=DEFAULT_SAM3D)
    args = ap.parse_args()

    print(f"[diag] loading {args.sam3d} ...")
    with open(args.sam3d) as f:
        d = json.load(f)

    print(f"[diag] top-level keys: {list(d.keys())}")
    p0 = {e["frame"]: e for e in d.get("0", [])}
    p1 = {e["frame"]: e for e in d.get("1", [])}
    common = sorted(set(p0) & set(p1))
    print(f"[diag] persons 0/1 common frames: {len(common)} "
          f"({common[0]}-{common[-1]})\n")

    # ── 1. world_coords cross-person depth agreement ──────────────────────────
    has_world = "world_coords" in (p0[common[0]] if common else {})
    print("=" * 64)
    print(" 1. world_coords cross-person head-Z agreement (target ~0 m)")
    print("=" * 64)
    if has_world:
        diffs = np.array([
            abs(p0[f]["world_coords"][HEAD_KP][2] - p1[f]["world_coords"][HEAD_KP][2])
            for f in common
        ])
        rel = np.array([
            bool(p0[f].get("world_coords_reliable", False))
            and bool(p1[f].get("world_coords_reliable", False))
            for f in common
        ])
        print(f"  mean   = {diffs.mean():7.2f} m")
        print(f"  median = {np.median(diffs):7.2f} m")
        print(f"  p90    = {np.percentile(diffs, 90):7.2f} m")
        print(f"  max    = {diffs.max():7.2f} m")
        print(f"  frames with phantom depth > 1.0 m : {100*(diffs>1.0).mean():.1f}%")
        print(f"  frames flagged world_coords_reliable=True : {100*rel.mean():.1f}%")
        verdict = ("STILL BROKEN — do NOT use world_coords for cross-person distance"
                   if np.median(diffs) > 0.25 else "usable")
        print(f"  >> VERDICT: {verdict}")
    else:
        print("  world_coords not present in this export.")

    # ── 2. keypoint_conf variability ──────────────────────────────────────────
    print("\n" + "=" * 64)
    print(" 2. keypoint_conf variability (constant 1.0 == no real signal)")
    print("=" * 64)
    if common and "keypoint_conf" in p0[common[0]]:
        conf = np.array([p0[f]["keypoint_conf"] for f in common])
        print(f"  shape={conf.shape}  min={conf.min():.4f} "
              f"max={conf.max():.4f} mean={conf.mean():.4f}")
        print(f"  fraction of values < 0.99 : {100*(conf<0.99).mean():.2f}%")
        verdict = ("PLACEHOLDER — effectively constant, cannot gate occluded wrists"
                   if (conf < 0.99).mean() < 0.02 else "usable")
        print(f"  >> VERDICT: {verdict}")
    else:
        print("  keypoint_conf not present.")

    # ── 3. contact_events — the usable new field ──────────────────────────────
    print("\n" + "=" * 64)
    print(" 3. contact_events (region-aware contact prior)")
    print("=" * 64)
    ce = d.get("contact_events", [])
    if ce:
        probs = np.array([e["contact_prob"] for e in ce])
        dist = np.array([e["contact_3d_distance_m"] for e in ce])
        print(f"  n events       = {len(ce)}  (frames {min(e['frame'] for e in ce)}"
              f"-{max(e['frame'] for e in ce)})")
        print(f"  contact_prob   : min={probs.min():.3f} mean={probs.mean():.3f} "
              f"max={probs.max():.3f}  (#>0.5={int((probs>0.5).sum())}, "
              f"#>0.3={int((probs>0.3).sum())})")
        print(f"  distance_m     : min={dist.min():.3f} median={np.median(dist):.3f} "
              f"max={dist.max():.3f}")
        print(f"  regions        : {dict(Counter(e['contact_region'] for e in ce))}")
        print("  >> VERDICT: USABLE — real variance + region labels "
              "(arm=blocked vs head/torso=landed)")
    else:
        print("  contact_events not present.")

    print("\n[diag] done.\n")


if __name__ == "__main__":
    main()
