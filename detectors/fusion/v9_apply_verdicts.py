#!/usr/bin/env python3
"""
v9_apply_verdicts.py — merge VLM (Claude vision) verdicts into v9 detections
=============================================================================
The vision-verification loop:
  1. v9.py runs HIGH-RECALL (--thr 0.30 --shots) -> impacts_v9_strips/ has one
     before/impact/after strip per candidate.
  2. Claude reads every strip image and judges it:
       LANDED   - glove visibly connects, receiver head-snap / body fold
       BLOCKED  - punch thrown but caught on gloves/arms (guard)
       MISS     - punch thrown, no contact (slip / out of range / air)
       CLINCH   - fighters tangled, no strike
       UNCLEAR  - cannot tell from the strip
  3. Verdicts are written to verdicts.json in the fight folder (by this tool
     or by Claude directly), then this script merges them:
       <name>_impacts_v9_verified.json  - only LANDED (+ optionally BLOCKED)
     and prints precision stats of the raw detector vs the verified set.

Usage:
    python v9_apply_verdicts.py --folder "<fight dir>" \
        --verdicts verdicts.json [--keep-blocked]
verdicts.json format: {"<frame>": {"verdict": "LANDED", "note": "..."}, ...}
"""
import os, json, glob, argparse
from collections import Counter

VALID = {"LANDED", "BLOCKED", "MISS", "CLINCH", "UNCLEAR"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--verdicts", default="verdicts.json")
    ap.add_argument("--keep-blocked", action="store_true",
                    help="count BLOCKED as impact (punch reached the guard)")
    args = ap.parse_args()

    v9p = glob.glob(os.path.join(args.folder, "*_impacts_v9.json"))
    if not v9p:
        raise SystemExit("no *_impacts_v9.json in folder — run v9.py first")
    v9 = json.load(open(v9p[0]))

    vp = args.verdicts if os.path.isabs(args.verdicts) else \
        os.path.join(args.folder, args.verdicts)
    verdicts = {int(k): v for k, v in json.load(open(vp)).items()}

    keep_set = {"LANDED"} | ({"BLOCKED"} if args.keep_blocked else set())
    verified, dropped = [], []
    for imp in v9["impacts"]:
        f = imp["impact_frame"]
        v = verdicts.get(f, {"verdict": "UNCLEAR", "note": "no verdict"})
        if v["verdict"] not in VALID:
            raise SystemExit(f"bad verdict {v} for frame {f}")
        imp = dict(imp)
        imp["vlm_verdict"] = v["verdict"]
        imp["vlm_note"] = v.get("note", "")
        (verified if v["verdict"] in keep_set else dropped).append(imp)

    mix = Counter(verdicts[f]["verdict"] for f in verdicts)
    n_raw = len(v9["impacts"])
    out = {
        "approach": "fusion_v9 + VLM verification (Claude vision)",
        "label": "v9 high-recall candidates filtered by per-impact visual "
                 "verification of before/impact/after strips",
        "source": os.path.basename(v9p[0]),
        "fps": v9["fps"],
        "raw_detections": n_raw,
        "verdict_mix": dict(mix),
        "verified_impacts": len(verified),
        "detector_precision_vs_vlm": round(len(verified) / max(1, n_raw), 3),
        "impacts": verified,
        "rejected": dropped,
    }
    name = os.path.basename(v9p[0]).replace("_impacts_v9.json", "")
    outp = os.path.join(args.folder, f"{name}_impacts_v9_verified.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"verdict mix : {dict(mix)}")
    print(f"verified    : {len(verified)}/{n_raw} "
          f"(detector precision vs VLM = {out['detector_precision_vs_vlm']})")
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
