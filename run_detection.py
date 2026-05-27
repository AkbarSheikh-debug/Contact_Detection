#!/usr/bin/env python3
"""
SAM3D Boxing Impact Detection — Entry Point
============================================
Usage:
    python run_detection.py --video path/to/video.mp4
    python run_detection.py --video path/to/video.mp4 --no-sam --max-frames 500
    python run_detection.py  (uses default test video)
"""
import argparse
import os
import sys

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import ImpactDetectionPipeline

DEFAULT_VIDEO = r"C:\Users\XRIG\Downloads\for_impact_detection_experiment_2\for_impact_detection_experiment_2\1.mp4"


def main():
    parser = argparse.ArgumentParser(
        description="SAM3D Boxing Impact Detection (Pi-HOC inspired)"
    )
    parser.add_argument(
        "--video", default=DEFAULT_VIDEO,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to output video (default: outputs/<name>_impact_detected.mp4)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit number of frames to process (default: all)",
    )
    parser.add_argument(
        "--no-sam", action="store_true",
        help="Skip SAM segmentation (faster, no checkpoint needed)",
    )
    parser.add_argument(
        "--download-sam", action="store_true",
        help="Download SAM ViT-B checkpoint before running",
    )
    args = parser.parse_args()

    # ── Optional SAM download ─────────────────────────────────────────────────
    if args.download_sam or (not args.no_sam and not _sam_checkpoint_exists()):
        _download_sam_checkpoint()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    use_sam = not args.no_sam
    pipeline = ImpactDetectionPipeline(use_sam=use_sam)
    summary = pipeline.run(
        video_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
    )

    print(f"\n✓ Done. Output: {summary['output_path']}")
    return 0


def _sam_checkpoint_exists() -> bool:
    from config import SAM_CHECKPOINT
    return os.path.exists(SAM_CHECKPOINT)


def _download_sam_checkpoint():
    from config import SAM_CHECKPOINT, SAM_CHECKPOINT_URL, CHECKPOINT_DIR
    import urllib.request

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(SAM_CHECKPOINT):
        print(f"[SAM] Checkpoint already exists: {SAM_CHECKPOINT}")
        return

    print(f"[SAM] Downloading SAM ViT-B checkpoint (~375 MB) …")
    print(f"      URL: {SAM_CHECKPOINT_URL}")
    print(f"      Destination: {SAM_CHECKPOINT}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 / total_size)
        bar = "#" * int(pct / 2) + "." * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:5.1f}%  ({downloaded // 1_000_000} MB / {total_size // 1_000_000} MB)",
              end="", flush=True)

    urllib.request.urlretrieve(SAM_CHECKPOINT_URL, SAM_CHECKPOINT, reporthook=_progress)
    print(f"\n[SAM] Download complete: {SAM_CHECKPOINT}")


if __name__ == "__main__":
    sys.exit(main())
