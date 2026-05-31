#!/usr/bin/env python3
import sys as _sys, os as _os
"""
SAM3D Impact Detection — Keypoint-Based Entry Point
=====================================================
Loads pre-extracted 2D/3D keypoints and ASFormer action results,
runs the 5-gate impact detection algorithm, and generates a full
analysis report.

Usage:
    python run_impact_detection.py
    python run_impact_detection.py --threshold 0.40
    python run_impact_detection.py --no-report
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict

# Ensure repo root is on path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

from keypoint_loader import KeypointLoader
from detectors.phase1.impact_detector import ImpactDetector, ImpactResult
from detectors.phase1.impact_report import generate_full_report
from config import (
    KEYPOINTS_2D_PATH, KEYPOINTS_3D_PATH, ACTIONS_PATH,
    IMPACT_SCORE_THRESHOLD, OUTPUT_DIR,
)


def main():
    parser = argparse.ArgumentParser(
        description="SAM3D Impact Detection from Pre-extracted Keypoints"
    )
    parser.add_argument(
        "--kp2d", default=KEYPOINTS_2D_PATH,
        help="Path to 2D keypoints JSON",
    )
    parser.add_argument(
        "--kp3d", default=KEYPOINTS_3D_PATH,
        help="Path to 3D keypoints JSON",
    )
    parser.add_argument(
        "--actions", default=ACTIONS_PATH,
        help="Path to ASFormer full_results.json",
    )
    parser.add_argument(
        "--threshold", type=float, default=IMPACT_SCORE_THRESHOLD,
        help=f"Impact score threshold (default: {IMPACT_SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Path for output JSON results (default: outputs/impact_results.json)",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip generating visual report",
    )
    args = parser.parse_args()

    print("")
    print("=" * 70)
    print("  SAM3D Impact Detection — Keypoint-Based Pipeline")
    print("=" * 70)

    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────
    loader = KeypointLoader()

    print("\n-- Loading Data " + "-" * 49)
    frames_2d = loader.load_2d(args.kp2d)
    frames_3d = loader.load_3d(args.kp3d)
    actions = loader.load_actions(args.actions)

    print(f"\n  2D frames available : {len(frames_2d)}")
    print(f"  3D frames available : {len(frames_3d)}")
    print(f"  Action events       : {len(actions)}")

    # ── 2. Run impact detection ──────────────────────────────────────────
    print(f"\n-- Impact Detection (threshold={args.threshold}) " + "-" * 20)
    detector = ImpactDetector(frames_2d, frames_3d, threshold=args.threshold)
    results = detector.analyze_all(actions)

    # ── 3. Print results ─────────────────────────────────────────────────
    summary = ImpactDetector.summary(results)
    _print_results_table(results)
    _print_summary(summary)

    # ── 4. Save JSON output ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = args.output_json or os.path.join(OUTPUT_DIR, "impact_results.json")
    _save_json(results, summary, json_path)

    # ── 5. Generate visual report ────────────────────────────────────────
    if not args.no_report:
        report_path = generate_full_report(results, summary)
        print(f"\n  Visual report : {report_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time    : {elapsed:.1f}s")
    print("=" * 70 + "\n")
    return 0


# ── Output helpers ───────────────────────────────────────────────────────────

def _print_results_table(results: list[ImpactResult]):
    """Print a compact console table of all results."""
    print(f"\n  {'#':>3}  {'Time':>6}  {'Type':<16}  {'Hand':<6}  "
          f"{'Score':>6}  {'Decel':>6}  {'Jerk':>6}  {'Ext':>6}  "
          f"{'Depth':>6}  {'Conf':>6}  {'Result':<10}")
    print("  " + "-" * 100)

    for i, r in enumerate(results, 1):
        m, s = divmod(int(r.timestamp_seconds), 60)
        tag = "* LANDED" if r.is_impact else "  missed"
        print(
            f"  {i:>3}  {m:02d}:{s:02d}  {r.action:<16}  {r.striking_hand:<6}  "
            f"{r.impact_score:>6.3f}  {r.decel_score:>6.3f}  {r.jerk_score:>6.3f}  "
            f"{r.extension_score:>6.3f}  {r.depth_score:>6.3f}  {r.confidence_score:>6.3f}  "
            f"{tag}"
        )


def _print_summary(summary: dict):
    """Print aggregate statistics."""
    print(f"\n{'-' * 70}")
    print(f"  SUMMARY")
    print(f"{'-' * 70}")
    print(f"  Total actions detected : {summary['total_actions']}")
    print(f"  Landed impacts         : {summary['total_landed']}")
    print(f"  Missed                 : {summary['total_missed']}")
    print(f"  Landing rate           : {summary['landing_rate']:.1%}")
    print(f"  Avg impact score       : {summary['avg_impact_score']:.3f}")
    print(f"  Avg peak velocity (3D) : {summary['avg_peak_velocity']:.4f}")
    print()
    print(f"  By punch type:")
    for ptype, stats in sorted(summary["by_type"].items()):
        rate = stats["landed"] / max(stats["total"], 1)
        print(f"    {ptype:<16}  {stats['landed']:>2} / {stats['total']:>2}  ({rate:.0%})")
    print(f"{'-' * 70}")


def _save_json(results: list[ImpactResult], summary: dict, path: str):
    """Save results and summary to JSON."""
    output = {
        "summary": summary,
        "events": [],
    }

    for r in results:
        event = {
            "action": r.action,
            "action_frame": r.action_frame,
            "window_start": r.window_start,
            "window_end": r.window_end,
            "timestamp_seconds": r.timestamp_seconds,
            "target": r.target,
            "striking_hand": r.striking_hand,
            "action_confidence": r.action_confidence,
            "speed_kmh": r.speed_kmh,
            "power_watts": r.power_watts,
            "is_impact": r.is_impact,
            "impact_score": r.impact_score,
            "impact_frame": r.impact_frame,
            "gate_scores": {
                "deceleration": r.decel_score,
                "jerk": r.jerk_score,
                "extension": r.extension_score,
                "depth_convergence": r.depth_score,
                "confidence": r.confidence_score,
            },
            "kinematics": {
                "peak_velocity_3d": r.peak_velocity_3d,
                "deceleration_magnitude": r.deceleration_magnitude,
                "arm_extension_at_impact": r.arm_extension_at_impact,
            },
        }
        output["events"].append(event)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results JSON  : {path}")


if __name__ == "__main__":
    sys.exit(main())
