#!/usr/bin/env python3
"""
Merge the per-fighter, per-part SAM3D keypoint exports (Label Studio export
folder, e.g. .../Gladius_Output_Label_Studio_with_SAM3D/Gladius_Output_Label_Studio/
<fight>/fighter{0,1}/<prefix>_fighter{0,1}_part{NN}_SAM3D.json) into the
merged-per-round sam3d.json format every other script in this repo expects
(dataset/<fight>/RoundN/sam3d.json: {"0": [...], "1": [...]}, frame-keyed,
GLOBAL frame numbers matching the round's full video).

Why this is needed: dataset/<fight>/RoundN/sam3d.json for 1st_fight/2nd_fight/
3rd_fight/4th_fight was built from raw CUTIE bbox tracking only (tracking_format
="raw_bbox" in dataset/fights.py) -- no skeleton keypoints, so XGBoost/TCN/
ASFormer/BRT (and r3d_18's contact-frame estimator) can't use any of it. This
Label Studio export folder has the real SAM3D keypoints, just chunked into
4 parts of 1500 frames each per round (verified: 3*1500 + 125 = 4625 exactly
matches Round1's total_frames for 1st_fight) instead of one merged file.

Frame mapping: part files use LOCAL frame numbers 1..1500 (1..N for the last,
shorter part). global_frame = (part_index - 1) * 1500 + (local_frame - 1),
0-indexed to match the existing bbox-only sam3d.json's frame numbering.

Usage:
    python tools/merge_sam3d_parts.py --fight 1st_fight
    python tools/merge_sam3d_parts.py --fight 2nd_fight --rounds 1 2 3
"""
import os
import sys
import re
import json
import glob
import argparse
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import fights

PART_SIZE = 1500
LABEL_STUDIO_ROOT = (
    r"C:\Users\XRIG\Downloads\Gladius_Output_Label_Studio_with_SAM3D"
    r"\Gladius_Output_Label_Studio"
)


def merge_fighter_parts(fight_dir, prefix, fighter_id):
    pattern = os.path.join(fight_dir, f"fighter{fighter_id}", f"{prefix}_fighter{fighter_id}_part*_SAM3D.json")
    part_files = sorted(glob.glob(pattern),
                         key=lambda p: int(re.search(r"_part(\d+)_SAM3D", p).group(1)))
    if not part_files:
        return None

    merged = {}
    for pf in part_files:
        part_idx = int(re.search(r"_part(\d+)_SAM3D", pf).group(1))
        d = json.load(open(pf))
        entries = d["keypoints_3d"].get(str(fighter_id), [])
        for e in entries:
            local_frame = e["frame"]
            global_frame = (part_idx - 1) * PART_SIZE + (local_frame - 1)
            e2 = dict(e)
            e2["frame"] = global_frame
            merged[global_frame] = e2
    return [merged[f] for f in sorted(merged)]


def merge_round(fight_name, fight_cfg, round_id):
    prefix = os.path.splitext(fight_cfg["video_filenames"][round_id])[0]
    fight_dir = os.path.join(LABEL_STUDIO_ROOT, fight_name)
    if not os.path.isdir(fight_dir):
        print(f"  [skip] no Label Studio folder for {fight_name}")
        return False

    f0 = merge_fighter_parts(fight_dir, prefix, 0)
    f1 = merge_fighter_parts(fight_dir, prefix, 1)
    if not f0 or not f1:
        print(f"  [skip] Round{round_id} ({prefix}_fighter*): no part files found")
        return False

    out_path = os.path.join(fight_cfg["out_base"], f"Round{round_id}", "sam3d.json")
    if not os.path.exists(os.path.dirname(out_path)):
        print(f"  [skip] Round{round_id}: {os.path.dirname(out_path)} doesn't exist "
              f"(run dataset/prepare_fight_dataset.py first)")
        return False

    if os.path.exists(out_path):
        bak_path = out_path + ".bboxonly.bak"
        if not os.path.exists(bak_path):
            shutil.copy(out_path, bak_path)
        old = json.load(open(out_path))
        contact_events = old.get("contact_events", [])
    else:
        contact_events = []

    merged = {"0": f0, "1": f1, "contact_events": contact_events}
    json.dump(merged, open(out_path, "w"))
    print(f"  [ok] Round{round_id}: fighter0={len(f0)} fighter1={len(f1)} frames -> {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fight", required=True, choices=fights.all_fight_names())
    ap.add_argument("--rounds", nargs="+", type=int, default=None,
                     help="default: all rounds registered for this fight in dataset/fights.py")
    args = ap.parse_args()

    cfg = fights.get_fight(args.fight)
    rounds = args.rounds or cfg["rounds"]
    print(f"=== {args.fight}: merging SAM3D parts for rounds {rounds} ===")
    ok = 0
    for r in rounds:
        if merge_round(args.fight, cfg, r):
            ok += 1
    print(f"\n{ok}/{len(rounds)} rounds merged successfully.")


if __name__ == "__main__":
    main()
