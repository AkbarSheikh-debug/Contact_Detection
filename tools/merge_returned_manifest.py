#!/usr/bin/env python3
"""
Safely merge ONE teammate's returned dataset/<fight>/manifest.json (from a
copy of the distributed annotation tool) back into the live one.

For merging a whole inbox of returned files at once, see
tools/merge_all_returned.py instead -- this script handles a single file,
useful for a one-off or for testing a file before dropping it in incoming/.

Never blind-overwrites. Per clip, compares the incoming label/impact_frame
against the live one (see tools/manifest_merge_lib.py for the exact rules).

By default this is a DRY RUN -- it only prints the report. Pass --apply to
actually write the merged manifest (a .bak of the live one is made first).
Conflicts are never auto-resolved even with --apply.

Usage:
  python tools/merge_returned_manifest.py --fight 3rd_fight --incoming returned_manifest.json
  python tools/merge_returned_manifest.py --fight 3rd_fight --incoming returned_manifest.json --apply
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + r"\..\dataset")
import fights  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifest_merge_lib import diff_manifest, print_report, update_manifest_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fight", required=True, choices=fights.all_fight_names())
    ap.add_argument("--incoming", required=True, help="path to the returned manifest.json")
    ap.add_argument("--apply", action="store_true",
                    help="write the merge (default is dry-run, report only)")
    ap.add_argument("--reviewer", default=None,
                    help="name/initials to tag in the report, e.g. --reviewer Alex")
    args = ap.parse_args()

    cfg = fights.get_fight(args.fight)
    live_path = cfg["manifest_path"]

    live = json.load(open(live_path))
    incoming = json.load(open(args.incoming))

    live_by_clip = {c["clip"]: c for c in live["clips"]}
    inc_by_clip = {c["clip"]: c for c in incoming["clips"]}

    result = diff_manifest(live_by_clip, inc_by_clip)
    tag = f" [{args.reviewer}]" if args.reviewer else ""
    print_report(tag, args.fight, result)

    if not args.apply:
        print(f"\n  Dry run only -- nothing written. Re-run with --apply to merge "
              f"new labels + new impact_frame marks (conflicts are never auto-applied).")
        return

    if result["new_labels"] or result["frame_added"]:
        backup_path = live_path + ".bak"
        shutil.copy2(live_path, backup_path)
        update_manifest_stats(live)
        json.dump(live, open(live_path, "w"), indent=2)
        print(f"\n  Applied {len(result['new_labels'])} new labels + "
              f"{len(result['frame_added'])} new impact_frame marks.")
        print(f"  Backup of previous manifest -> {backup_path}")
        print(f"  Updated -> {live_path}")
    else:
        print(f"\n  Nothing new to apply (only conflicts/agreements/structure warnings).")

    n_conf = len(result["conflicts"]) + len(result["frame_conflicts"])
    if n_conf:
        print(f"\n  {n_conf} conflict(s) still need your manual review -- they were NOT applied. "
              f"Resolve by editing {live_path} directly, or by deciding which source wins and "
              f"re-running.")


if __name__ == "__main__":
    main()
