#!/usr/bin/env python3
"""Render the annotated video from a *_impacts_v9_verified.json (LANDED only).
Reuses v9's renderer; freeze frames show only vision-verified impacts.

Usage: python v9_render_verified.py --folder "<fight dir>" [--freeze 1.2]
"""
import os, sys, json, glob, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--freeze", type=float, default=1.2)
    args = ap.parse_args()

    vp = glob.glob(os.path.join(args.folder, "*_impacts_v9_verified.json"))
    if not vp:
        raise SystemExit("run v9_apply_verdicts.py first")
    ver = json.load(open(vp[0]))
    sam3d_p, _, vid = v9.find_files(args.folder)
    if not vid:
        raise SystemExit("no raw video in folder")
    s = json.load(open(sam3d_p))
    persons = {0: {e["frame"]: e for e in s.get("0", [])},
               1: {e["frame"]: e for e in s.get("1", [])}}
    kept = [dict(frame=i["impact_frame"], score=i["impact_score"],
                 contact_region=i["contact_region"],
                 striker_id=i["striker_id"], receiver_id=i["receiver_id"])
            for i in ver["impacts"]]
    print(f"rendering {len(kept)} VERIFIED impacts (of {ver['raw_detections']} raw)")
    out = v9.render_video(args.folder, vid, persons, kept, ver["fps"], args.freeze)
    if out:
        final = out.replace("_impacts_v9_frozen", "_impacts_VERIFIED")
        if os.path.exists(final):
            os.remove(final)
        os.replace(out, final)
        print(f"final -> {final}")


if __name__ == "__main__":
    main()
