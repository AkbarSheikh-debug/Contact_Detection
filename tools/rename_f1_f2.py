"""
Renames F1_F2_Output files to include fight number + fighter name.

Mapping:
  fighter0  -> JP_OMeara
  fighter1  -> Jose_Manuel_Perez
  F1 fight  -> prefix "1"
  F2 fight  -> prefix "2"

Before: F1/fighter0/R1_fighter0_part01.mp4
After:  F1/JP_OMeara/1_R1_JP_OMeara_part01.mp4

Also updates Round JSON "file" fields and regenerates manifest CSV.
"""

import json
import csv
import os
from pathlib import Path

OUTPUT_DIR = Path(r"C:\Users\XRIG\Desktop\F1_F2_Output")

FIGHTER_NAMES = {
    "fighter0": "JP_OMeara",
    "fighter1": "Jose_Manuel_Perez",
}
FIGHT_NUMS = {
    "F1": "1",
    "F2": "2",
}


def new_video_name(fight_id: str, round_id: str, fighter_key: str, part: str) -> str:
    """1_R1_JP_OMeara_part01.mp4"""
    num = FIGHT_NUMS[fight_id]
    name = FIGHTER_NAMES[fighter_key]
    return f"{num}_{round_id}_{name}_{part}.mp4"


def rename_all():
    manifest_rows = []

    for fight_id in sorted(OUTPUT_DIR.iterdir()):
        if not fight_id.is_dir() or fight_id.name not in FIGHT_NUMS:
            continue

        for fighter_key, fighter_name in FIGHTER_NAMES.items():
            old_folder = fight_id / fighter_key
            if not old_folder.exists():
                print(f"[SKIP] {old_folder} not found")
                continue

            new_folder = fight_id / fighter_name
            print(f"\n{fight_id.name}/{fighter_key} -> {fight_id.name}/{fighter_name}")

            # Collect renames: old_path -> new_path
            renames = {}
            for f in old_folder.iterdir():
                if f.suffix == ".mp4":
                    # R1_fighter0_part01.mp4  ->  parse round + part
                    stem = f.stem  # e.g. R1_fighter0_part01
                    parts = stem.split("_")
                    # parts = ['R1', 'fighter0', 'part01']
                    round_id = parts[0]
                    part_id = parts[-1]
                    new_name = new_video_name(fight_id.name, round_id, fighter_key, part_id)
                    renames[f.name] = new_name
                elif f.suffix == ".json":
                    renames[f.name] = f.name  # keep R1.json etc unchanged

            # Rename the folder first (to new_folder)
            old_folder.rename(new_folder)
            print(f"  Renamed folder: {old_folder.name} -> {new_folder.name}")

            # Rename files inside new_folder
            for old_name, new_name in renames.items():
                src = new_folder / old_name
                dst = new_folder / new_name
                if src == dst:
                    continue
                if src.exists():
                    src.rename(dst)
                    print(f"  {old_name} -> {new_name}")

            # Update Round JSON files (their "file" fields point to old mp4 names)
            for json_file in sorted(new_folder.glob("*.json")):
                with open(json_file) as fh:
                    data = json.load(fh)

                changed = False
                if "parts" in data:
                    for p in data["parts"]:
                        old_file = p.get("file", "")
                        if old_file in renames:
                            p["file"] = renames[old_file]
                            changed = True
                if "fighter" in data:
                    old_fighter = data["fighter"]  # e.g. "fighter0"
                    if old_fighter in FIGHTER_NAMES:
                        data["fighter"] = FIGHTER_NAMES[old_fighter]
                        changed = True
                if "fighter_name" not in data:
                    data["fighter_name"] = FIGHTER_NAMES[fighter_key]
                    changed = True

                if changed:
                    with open(json_file, "w") as fh:
                        json.dump(data, fh, indent=2)

                # Collect manifest rows from parts
                if "parts" in data:
                    round_id = data.get("round", json_file.stem)
                    for p in data["parts"]:
                        manifest_rows.append({
                            "fight": fight_id.name,
                            "round": round_id,
                            "fighter": FIGHTER_NAMES[fighter_key],
                            "part": p["part"],
                            "path": str(new_folder / p["file"]),
                        })

    # Rewrite manifest CSV
    manifest_path = OUTPUT_DIR / "label_studio_manifest.csv"
    fieldnames = ["fight", "round", "fighter", "part", "path"]
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nUpdated manifest: {manifest_path} ({len(manifest_rows)} entries)")
    print("Done.")


if __name__ == "__main__":
    rename_all()
