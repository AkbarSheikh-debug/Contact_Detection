"""
Convert all frame folders into videos, split into ~5GB chunks.

Each folder (e.g. .../angle_000/) containing frame_0001.jpg ... frame_XXXX.jpg
becomes one .mp4 video. Videos are distributed into chunk folders (chunk_01, chunk_02, ...)
so each chunk stays under 5GB.

Usage:
    python frames_to_videos.py

Install:
    pip install opencv-python numpy
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import os
import sys
import time
import glob
import re
import shutil

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: Install dependencies first:")
    print("  pip install opencv-python numpy")
    sys.exit(1)

# ==============================
# CONFIG
# ==============================
input_dir = r"C:\Users\XRIG\Desktop\New_Blender_Sprints\Synthetic_dataset\multi_test_2_mirror"
output_dir = r"C:\Users\XRIG\Desktop\New_Blender_Sprints\Synthetic_dataset\multi_test_2_mirror_video"

CHUNK_SIZE_GB = 5.0
CHUNK_SIZE_BYTES = int(CHUNK_SIZE_GB * 1024 * 1024 * 1024)

FPS = 30                    # video framerate
VIDEO_CODEC = 'mp4v'        # H.264 compatible codec
VIDEO_EXT = '.mp4'

NUM_WORKERS = 1             # sequential (opencv VideoWriter isn't thread-safe)

# ==============================
# FIND ALL FRAME FOLDERS
# ==============================
def find_frame_folders(base_dir):
    """
    Find all leaf folders that contain frame_XXXX.jpg files.
    Returns list of (folder_path, frame_files_sorted).
    """
    folders = []

    for root, dirs, files in os.walk(base_dir):
        # Find frame files in this folder
        frame_files = [f for f in files if re.match(r'frame_\d{4}\.jpg', f, re.IGNORECASE)]

        if not frame_files:
            continue

        # Sort by frame number
        frame_files.sort(key=lambda x: int(re.search(r'(\d{4})', x).group(1)))

        folders.append((root, frame_files))

    return folders


# ==============================
# CONVERT ONE FOLDER TO VIDEO
# ==============================
def folder_to_video(folder_path, frame_files, output_path):
    """
    Convert a folder of frame_XXXX.jpg files into a single .mp4 video.
    Returns the video file size in bytes.
    """
    if os.path.exists(output_path):
        return os.path.getsize(output_path)

    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(folder_path, frame_files[0]))
    if first_frame is None:
        print(f"    WARNING: Cannot read {frame_files[0]} in {folder_path}, skipping")
        return 0

    h, w = first_frame.shape[:2]

    # Create video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (w, h))

    if not writer.isOpened():
        print(f"    WARNING: Cannot create video writer for {output_path}")
        return 0

    # Write all frames
    for fname in frame_files:
        frame = cv2.imread(os.path.join(folder_path, fname))
        if frame is not None:
            writer.write(frame)

    writer.release()

    if os.path.exists(output_path):
        return os.path.getsize(output_path)
    return 0


# ==============================
# GENERATE VIDEO NAME FROM FOLDER PATH
# ==============================
def folder_to_video_name(folder_path, base_dir):
    """
    Convert folder path to a flat video filename.
    e.g. power_5500/CameraA/pos_0/h_0.5/color_0/dist_5/angle_000
      -> power_5500__CameraA__pos_0__h_0.5__color_0__dist_5__angle_000.mp4
    """
    rel = os.path.relpath(folder_path, base_dir)
    # Replace path separators with double underscore
    name = rel.replace(os.sep, "__").replace("/", "__")
    return name + VIDEO_EXT


# ==============================
# MAIN
# ==============================
def main():
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Chunk size: {CHUNK_SIZE_GB} GB")
    print()

    # Find all frame folders
    print("Scanning for frame folders...")
    frame_folders = find_frame_folders(input_dir)
    print(f"Found {len(frame_folders)} folders with frames")

    if not frame_folders:
        print("No frame folders found!")
        return

    # Count total frames
    total_frames = sum(len(files) for _, files in frame_folders)
    print(f"Total frames across all folders: {total_frames:,}")
    print()

    # Process folders into videos, distributing into chunks
    chunk_idx = 1
    chunk_size = 0
    videos_in_chunk = 0
    total_videos = 0
    total_bytes = 0
    skipped = 0

    t_start = time.time()

    chunk_dir = os.path.join(output_dir, f"chunk_{chunk_idx:02d}")
    os.makedirs(chunk_dir, exist_ok=True)

    for i, (folder_path, frame_files) in enumerate(frame_folders):

        video_name = folder_to_video_name(folder_path, input_dir)
        video_path = os.path.join(chunk_dir, video_name)

        # Check if video already exists in ANY chunk (resume support)
        already_exists = False
        for existing_chunk in glob.glob(os.path.join(output_dir, "chunk_*")):
            existing_path = os.path.join(existing_chunk, video_name)
            if os.path.exists(existing_path):
                already_exists = True
                skipped += 1
                break

        if already_exists:
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(frame_folders)}] Skipping (already exists)")
            continue

        # Convert frames to video
        video_size = folder_to_video(folder_path, frame_files, video_path)

        if video_size == 0:
            continue

        chunk_size += video_size
        videos_in_chunk += 1
        total_videos += 1
        total_bytes += video_size

        # Check if chunk is full -> start new chunk
        if chunk_size >= CHUNK_SIZE_BYTES:
            print(f"  ✅ chunk_{chunk_idx:02d}: {videos_in_chunk} videos, {chunk_size / (1024**3):.2f} GB")
            chunk_idx += 1
            chunk_dir = os.path.join(output_dir, f"chunk_{chunk_idx:02d}")
            os.makedirs(chunk_dir, exist_ok=True)
            chunk_size = 0
            videos_in_chunk = 0

        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = total_videos / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(frame_folders)}] {total_videos} videos created, "
                  f"{total_bytes / (1024**3):.2f} GB total, {rate:.1f} videos/sec")

    # Final chunk info
    if videos_in_chunk > 0:
        print(f"  ✅ chunk_{chunk_idx:02d}: {videos_in_chunk} videos, {chunk_size / (1024**3):.2f} GB")

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"  Total videos created: {total_videos}")
    print(f"  Total videos skipped: {skipped}")
    print(f"  Total size: {total_bytes / (1024**3):.2f} GB")
    print(f"  Chunks: {chunk_idx}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
