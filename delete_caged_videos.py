"""
Delete all .mp4 videos whose filename contains 'caged' (case-insensitive)
from the chunked output directory.

Usage:
    python delete_caged_videos.py
    python delete_caged_videos.py --dry-run   # preview only, no deletion
"""

import os
import glob
import argparse

# ==============================
# CONFIG  — match frame_to_video.py
# ==============================
output_dir = r"C:\Users\XRIG\Desktop\New_Blender_Sprints\Synthetic_dataset\multi_test_2_mirror_video"

KEYWORD = "caged"   # case-insensitive match against filename


def find_caged_videos(base_dir: str, keyword: str) -> list[str]:
    """Return sorted list of .mp4 paths whose filename contains *keyword*."""
    matches = []
    pattern = os.path.join(base_dir, "**", "*.mp4")
    for path in glob.glob(pattern, recursive=True):
        if keyword.lower() in os.path.basename(path).lower():
            matches.append(path)
    return sorted(matches)


def main():
    parser = argparse.ArgumentParser(description="Delete videos containing a keyword in their filename.")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be deleted without deleting them.")
    parser.add_argument("--dir", default=output_dir, help="Root directory to search (overrides CONFIG).")
    parser.add_argument("--keyword", default=KEYWORD, help=f"Keyword to match (default: '{KEYWORD}').")
    args = parser.parse_args()

    search_dir = args.dir
    keyword = args.keyword

    print(f"Searching in : {search_dir}")
    print(f"Keyword      : '{keyword}'")
    print(f"Mode         : {'DRY RUN (no files deleted)' if args.dry_run else 'LIVE (files will be deleted)'}")
    print()

    if not os.path.isdir(search_dir):
        print(f"ERROR: Directory not found: {search_dir}")
        return

    videos = find_caged_videos(search_dir, keyword)

    if not videos:
        print("No matching videos found.")
        return

    total_bytes = sum(os.path.getsize(p) for p in videos)
    print(f"Found {len(videos)} video(s) totalling {total_bytes / (1024**3):.3f} GB:\n")

    for path in videos:
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"  {'[DRY RUN] would delete' if args.dry_run else 'Deleting'}: {path}  ({size_mb:.1f} MB)")
        if not args.dry_run:
            os.remove(path)

    if not args.dry_run:
        print(f"\nDeleted {len(videos)} file(s), freed {total_bytes / (1024**3):.3f} GB.")
    else:
        print(f"\nDry run complete — {len(videos)} file(s) would be deleted ({total_bytes / (1024**3):.3f} GB).")


if __name__ == "__main__":
    main()
