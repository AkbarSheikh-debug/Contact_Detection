"""
Re-encodes all mp4v videos in F1_F2_Output_SmoothTrack to H.264 (libx264)
so they play in Label Studio's browser-based player.

Encode flags: -c:v libx264 -preset fast -crf 23 -movflags +faststart -pix_fmt yuv420p
Each file is encoded to a temp file then replaces the original.
"""

import subprocess
import sys
from pathlib import Path

FFMPEG = Path(r"C:\Users\XRIG\Downloads\ffmpeg_extracted\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe")
INPUT_DIR = Path(r"C:\Users\XRIG\Desktop\F1_F2_Output_SmoothTrack")


def reencode(src: Path) -> bool:
    tmp = src.with_suffix(".h264_tmp.mp4")
    cmd = [
        str(FFMPEG), "-y",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode})")
        print(result.stderr.decode(errors="replace")[-500:])
        if tmp.exists():
            tmp.unlink()
        return False
    tmp.replace(src)
    return True


def main():
    if not FFMPEG.exists():
        print(f"ffmpeg not found: {FFMPEG}")
        sys.exit(1)

    videos = sorted(INPUT_DIR.rglob("*.mp4"))
    total = len(videos)
    print(f"Found {total} MP4 files to re-encode\n")

    ok = 0
    for i, v in enumerate(videos, 1):
        size_mb = v.stat().st_size / 1_048_576
        print(f"[{i:2d}/{total}] {v.relative_to(INPUT_DIR)}  ({size_mb:.1f} MB) ...", end=" ", flush=True)
        if reencode(v):
            new_mb = v.stat().st_size / 1_048_576
            print(f"OK  {size_mb:.1f} -> {new_mb:.1f} MB")
            ok += 1
        else:
            print("FAILED")

    print(f"\nDone: {ok}/{total} re-encoded to H.264.")


if __name__ == "__main__":
    main()
