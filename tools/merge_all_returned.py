#!/usr/bin/env python3
"""
Batch-process the whole incoming/ inbox in one pass: every *.json dropped
under incoming/<fight>/ gets merged into that fight's live
dataset/<fight>/manifest.json (see incoming/README.md for how files land
there, and tools/manifest_merge_lib.py for the merge rules).

Multiple files for the same fight are applied IN FILENAME ORDER, each
diffed against the running state -- so if two teammates' files disagree
with each other (not just with the original live manifest), that shows up
as a conflict too, not just a silent "last one wins".

By default this is a DRY RUN across the whole inbox (report only, nothing
written, nothing moved). Pass --apply to actually merge: live manifests are
backed up (.bak) before any write, and any incoming file that merged with
ZERO conflicts gets moved into its fight's processed/ subfolder so it won't
be re-merged next time. Files that had conflicts are left in place in
incoming/<fight>/ so they keep showing up until you resolve them.

Usage:
  python tools/merge_all_returned.py
  python tools/merge_all_returned.py --apply
  python tools/merge_all_returned.py --apply --fight 3rd_fight   # just one fight's inbox
"""
import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + r"\..\dataset")
import fights  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifest_merge_lib import diff_manifest, print_report, update_manifest_stats  # noqa: E402

INCOMING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "incoming")


def process_fight_inbox(fight_name, apply_changes):
    cfg = fights.get_fight(fight_name)
    live_path = cfg["manifest_path"]
    inbox = os.path.join(INCOMING_DIR, fight_name)
    files = sorted(glob.glob(os.path.join(inbox, "*.json")))
    if not files:
        return None

    live = json.load(open(live_path))
    live_by_clip = {c["clip"]: c for c in live["clips"]}

    totals = {"new_labels": [], "agree": [], "conflicts": [], "frame_added": [],
              "frame_conflicts": [], "structure_warnings": []}
    clean_files = []

    for path in files:
        fname = os.path.basename(path)
        incoming = json.load(open(path))
        inc_by_clip = {c["clip"]: c for c in incoming["clips"]}
        result = diff_manifest(live_by_clip, inc_by_clip)  # mutates live_by_clip with non-conflicting adds
        print_report(f" [{fname}]", fight_name, result)

        for k in totals:
            totals[k].extend(result[k])
        if not result["conflicts"] and not result["frame_conflicts"]:
            clean_files.append(path)

    print(f"\n--- {fight_name} inbox summary: {len(files)} file(s), "
          f"{len(totals['new_labels'])} new labels, {len(totals['frame_added'])} new frames, "
          f"{len(totals['conflicts']) + len(totals['frame_conflicts'])} unresolved conflict(s) ---")

    if not apply_changes:
        return totals

    if totals["new_labels"] or totals["frame_added"]:
        backup_path = live_path + ".bak"
        shutil.copy2(live_path, backup_path)
        update_manifest_stats(live)
        json.dump(live, open(live_path, "w"), indent=2)
        print(f"  Updated -> {live_path}  (backup -> {backup_path})")

    processed_dir = os.path.join(inbox, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    for path in clean_files:
        shutil.move(path, os.path.join(processed_dir, os.path.basename(path)))
    if clean_files:
        print(f"  Moved {len(clean_files)} fully-clean file(s) -> {processed_dir}")
    left = len(files) - len(clean_files)
    if left:
        print(f"  Left {left} file(s) in {inbox} -- they had conflicts, resolve and re-run")

    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write merges + move clean files to processed/ (default: dry run)")
    ap.add_argument("--fight", default=None, choices=fights.all_fight_names(),
                    help="only process this fight's inbox (default: all fights)")
    args = ap.parse_args()

    fight_names = [args.fight] if args.fight else fights.all_fight_names()
    grand_total = {"new_labels": 0, "conflicts": 0, "frame_added": 0, "frame_conflicts": 0}
    any_inbox = False

    for fight_name in fight_names:
        totals = process_fight_inbox(fight_name, args.apply)
        if totals is None:
            continue
        any_inbox = True
        grand_total["new_labels"]     += len(totals["new_labels"])
        grand_total["conflicts"]      += len(totals["conflicts"])
        grand_total["frame_added"]    += len(totals["frame_added"])
        grand_total["frame_conflicts"] += len(totals["frame_conflicts"])

    if not any_inbox:
        print("Nothing in incoming/ to process -- drop returned manifest.json files under "
              "incoming/<fight>/ first (see incoming/README.md).")
        return

    print(f"\n{'='*60}")
    print(f"  GRAND TOTAL across all processed inboxes")
    print(f"  New labels       : {grand_total['new_labels']}")
    print(f"  New impact_frames: {grand_total['frame_added']}")
    print(f"  Unresolved conflicts: {grand_total['conflicts'] + grand_total['frame_conflicts']}")
    if not args.apply:
        print(f"\n  Dry run only -- re-run with --apply to actually merge.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
