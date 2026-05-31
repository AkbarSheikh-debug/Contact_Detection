"""
Pi-HOC + SAM3D style analysis report generator.

For every impact detected in the video this script produces:
  1. Pi-HOC Fig-1 style canonical body contact maps (per fighter, cumulative)
  2. SAM3D style 3D body pose at each impact frame (aggressor + receiver side-by-side)
  3. Impact timeline bar chart
  4. Diagnostic table: extension ratio + directed velocity per event

Outputs
-------
  outputs/analysis/
    contact_map_summary.jpg   — cumulative per-fighter body contact diagram
    impact_timeline.jpg       — timeline of all impacts
    3d_poses/                 — one 3D-pose image per impact event
    diagnostic_table.jpg      — per-event stats table
    full_report.jpg            — combined single-page report
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

from models.detector import PersonDetector
from models.tracker  import PersonTracker
from impact_detection.pair_analyzer      import PairInteractionAnalyzer
from impact_detection.impact_classifier  import ImpactClassifier, ImpactEvent
from utils.body_contact_viz import BodyContactDiagram, BW, BH
from utils.smpl_mesh_viz    import render_smpl_pair
from config import (
    YOLO_MODEL, PROCESS_EVERY_N_FRAMES,
    MIN_PERSON_AREA_RATIO, OUTPUT_DIR,
)

OUT_DIR   = os.path.join(OUTPUT_DIR, "analysis")
POSE3D_DIR = os.path.join(OUT_DIR, "3d_poses")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dirs():
    for d in [OUT_DIR, POSE3D_DIR]:
        os.makedirs(d, exist_ok=True)


def _filter_size(persons, frame_area):
    return [p for p in persons
            if ((p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1])/frame_area)
            >= MIN_PERSON_AREA_RATIO]


def _is_referee(person, frame):
    x1, y1, x2, y2 = person["bbox"]
    h = y2 - y1
    crop = frame[y1:min(int(y1+h*0.40), frame.shape[0]), x1:x2]
    return crop.size > 0 and float(crop.mean()) > 165


# ── Data collection pass ──────────────────────────────────────────────────────

def collect_data(video_path: str, max_frames: int | None = None):
    """Re-run pipeline (pose only, no SAM) collecting poses + impact events."""
    detector  = PersonDetector(YOLO_MODEL)
    tracker   = PersonTracker()
    analyzer  = PairInteractionAnalyzer()
    classifier = ImpactClassifier()

    cap        = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS)
    W          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_area = W * H
    if max_frames:
        total = min(total, max_frames)

    frame_idx = 0

    print(f"[Analysis] Collecting data from {total} frames …")
    while frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            frame_idx += 1
            continue

        persons = detector.detect(frame)
        persons = _filter_size(persons, frame_area)
        persons = tracker.update(persons)
        persons = sorted(
            persons,
            key=lambda p: (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]),
            reverse=True,
        )[:2]
        fighters = persons  # analysis pass: take 2 largest, no referee heuristic needed

        pairs    = analyzer.form_pairs(fighters)
        contacts = analyzer.analyze(fighters, pairs, frame_idx)
        classifier.process(contacts, frame_idx, fps)

        frame_idx += PROCESS_EVERY_N_FRAMES

    cap.release()
    return classifier.events, fps


# ── Contact body map summary ──────────────────────────────────────────────────

def _make_contact_map_summary(events: list[ImpactEvent]) -> np.ndarray:
    """
    Pi-HOC Figure-1 style: per-fighter cumulative contact diagram.
    Left panel = receiver hit map.  Right panel = aggressor strike map.
    """
    diag = BodyContactDiagram()

    # Collect hit/strike regions per fighter
    recv_hits:   dict[int, list[str]] = defaultdict(list)
    recv_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    strike_map:  dict[int, list[str]] = defaultdict(list)

    for ev in events:
        hit_r, strike_r = diag.regions_from_event(ev.contact_region, ev.striking_limb)
        recv_hits[ev.receiver_id].extend(hit_r)
        for r in hit_r:
            recv_counts[ev.receiver_id][r] += 1
        strike_map[ev.aggressor_id].extend(strike_r)

    fighters = sorted(set(list(recv_hits.keys()) + list(strike_map.keys())))

    if not fighters:
        blank = np.full((BH, BW, 3), 14, dtype=np.uint8)
        cv2.putText(blank, "No events", (5, BH//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140,140,140), 1)
        return blank

    columns = []
    gap_col = np.full((BH, 10, 3), 14, dtype=np.uint8)

    for fid in fighters[:4]:   # max 4 fighters shown
        recv_img   = diag.draw(
            hit_regions=list(set(recv_hits.get(fid, []))),
            hit_counts=recv_counts.get(fid, {}),
            label=f"ID{fid} received",
        )
        strike_img = diag.draw(
            strike_regions=list(set(strike_map.get(fid, []))),
            label=f"ID{fid} struck",
        )
        pair_block = np.hstack([recv_img, np.full((BH,4,3),14,np.uint8), strike_img])
        columns.append(pair_block)
        columns.append(gap_col)

    if columns:
        columns.pop()   # remove trailing gap
    return np.hstack(columns)


# ── 3D pose per-impact ────────────────────────────────────────────────────────

def _generate_3d_poses(events: list[ImpactEvent]):
    """Save one 3D side-by-side body image per impact event."""
    import shutil
    if os.path.isdir(POSE3D_DIR):
        shutil.rmtree(POSE3D_DIR)
    os.makedirs(POSE3D_DIR)

    print(f"[Analysis] Generating 3D pose images for {len(events)} events …")
    for i, ev in enumerate(events):
        hit_r, strike_r = BodyContactDiagram.regions_from_event(
            ev.contact_region, ev.striking_limb)
        img  = render_smpl_pair(
            contact_regions_a=strike_r,
            contact_regions_b=hit_r,
            label_a=f"ID{ev.aggressor_id} ({ev.striking_limb.replace('_',' ')})",
            label_b=f"ID{ev.receiver_id} ({ev.contact_region} hit)",
            width=340, height=420,
        )
        path = os.path.join(POSE3D_DIR, f"impact_{i+1:03d}_t{ev.time_sec:.1f}s.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"[Analysis] 3D pose images saved to {POSE3D_DIR}/")


# ── Impact timeline ───────────────────────────────────────────────────────────

def _make_timeline(events: list[ImpactEvent]) -> np.ndarray:
    """Returns a matplotlib-rendered timeline image as numpy array."""
    if not events:
        return np.zeros((200, 800, 3), dtype=np.uint8)

    times  = [ev.time_sec for ev in events]
    probs  = [ev.probability for ev in events]
    labels = [ev.contact_region for ev in events]

    colors = ["#00D7FF" if l == "head" else "#BE32FF" for l in labels]

    fig, ax = plt.subplots(figsize=(10, 2.5), facecolor="#111")
    ax.set_facecolor("#111")
    ax.scatter(times, probs, c=colors, s=60, zorder=5, edgecolors="none")
    ax.vlines(times, 0, probs, colors=colors, linewidth=1.2, alpha=0.6)
    ax.axhline(0.58, color="#888", linestyle="--", linewidth=0.8, label="threshold")

    ax.set_xlim(0, max(times) + 1)
    ax.set_ylim(0.50, 1.05)
    ax.set_xlabel("Time (s)", color="#aaa", fontsize=9)
    ax.set_ylabel("Contact prob", color="#aaa", fontsize=9)
    ax.set_title("Impact Event Timeline", color="white", fontsize=11, pad=6)
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    patches = [
        mpatches.Patch(color="#00D7FF", label="Head"),
        mpatches.Patch(color="#BE32FF", label="Torso"),
    ]
    ax.legend(handles=patches, facecolor="#222", edgecolor="#555",
              labelcolor="white", fontsize=8, loc="upper right")

    plt.tight_layout(pad=0.4)
    buf = __import__("io").BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img if img is not None else np.zeros((200, 800, 3), dtype=np.uint8)


# ── Diagnostic table ──────────────────────────────────────────────────────────

def _make_diagnostic_table(events: list[ImpactEvent]) -> np.ndarray:
    """Matplotlib table: per-event stats for false-positive diagnosis."""
    if not events:
        return np.zeros((100, 800, 3), dtype=np.uint8)

    rows = []
    for i, ev in enumerate(events[:30], 1):   # cap at 30 rows
        m, s = divmod(int(ev.time_sec), 60)
        rows.append([
            str(i),
            f"{m:02d}:{s:02d}",
            ev.striking_limb.replace("_", " ").title(),
            ev.contact_region.title(),
            f"{ev.probability:.0%}",
            f"{ev.velocity:.1f}",
            f"ID{ev.aggressor_id}→ID{ev.receiver_id}",
        ])

    cols = ["#", "Time", "Limb", "Region", "Prob", "Vel (px/fr)", "Pair"]
    nrows = len(rows)

    fig_h = max(2.5, 0.30 * nrows + 0.8)
    fig, ax = plt.subplots(figsize=(11, fig_h), facecolor="#111")
    ax.set_facecolor("#111")
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=cols,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)

    # Style header
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2a2a40")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#1a1a1a" if r % 2 == 0 else "#222230")
            cell.set_text_props(color="#cccccc")
        cell.set_edgecolor("#333")

    ax.set_title("Impact Event Diagnostics", color="white", fontsize=11, pad=8)
    plt.tight_layout(pad=0.3)
    buf = __import__("io").BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img if img is not None else np.zeros((100, 800, 3), dtype=np.uint8)


# ── Full report assembly ──────────────────────────────────────────────────────

def _assemble_report(contact_map, timeline, diag_table) -> np.ndarray:
    """Stack all panels into a single-page report."""
    W = 1200

    def _resize_w(img, w):
        h = int(img.shape[0] * w / img.shape[1])
        return cv2.resize(img, (w, h))

    def _pad_w(img, w):
        if img.shape[1] == w:
            return img
        canvas = np.full((img.shape[0], w, 3), 14, dtype=np.uint8)
        canvas[:, :img.shape[1]] = img[:, :w]
        return canvas

    # Title bar
    title_bar = np.full((50, W, 3), 20, dtype=np.uint8)
    cv2.putText(title_bar, "SAM3D Boxing Impact Detection - Analysis Report",
                (16, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 215, 255), 2)

    # Section: contact maps
    cm_h   = 30 + BH + 20
    cm_sec = np.full((cm_h, W, 3), 14, dtype=np.uint8)
    cv2.putText(cm_sec, "Pi-HOC Canonical Body Contact Maps (per fighter)",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cmap_rsz = _pad_w(contact_map, W - 24)
    cm_sec[30:30+min(BH, cmap_rsz.shape[0]), 12:12+min(W-24, cmap_rsz.shape[1])] = \
        cmap_rsz[:min(BH, cmap_rsz.shape[0]), :min(W-24, cmap_rsz.shape[1])]

    timeline_rsz   = _resize_w(timeline,   W)
    diag_table_rsz = _resize_w(diag_table, W)

    sep = np.full((6, W, 3), 40, dtype=np.uint8)

    report = np.vstack([
        title_bar, sep,
        cm_sec, sep,
        timeline_rsz, sep,
        diag_table_rsz,
    ])
    return report


# ── Entry point ───────────────────────────────────────────────────────────────

def run_analysis(video_path: str, max_frames: int | None = None):
    _ensure_dirs()

    print("\n" + "="*60)
    print("  Pi-HOC / SAM3D Impact Analysis Report")
    print("="*60)

    events, fps = collect_data(video_path, max_frames)

    print(f"\n  Events collected : {len(events)}")
    head  = sum(1 for e in events if e.contact_region == "head")
    torso = sum(1 for e in events if e.contact_region == "torso")
    print(f"  Head impacts     : {head}")
    print(f"  Torso impacts    : {torso}")

    # 1. Contact body maps
    contact_map = _make_contact_map_summary(events)
    cv2.imwrite(os.path.join(OUT_DIR, "contact_map_summary.jpg"), contact_map,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: contact_map_summary.jpg")

    # 2. 3D poses per impact
    _generate_3d_poses(events)

    # 3. Timeline
    timeline = _make_timeline(events)
    cv2.imwrite(os.path.join(OUT_DIR, "impact_timeline.jpg"), timeline,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: impact_timeline.jpg")

    # 4. Diagnostic table
    diag_table = _make_diagnostic_table(events)
    cv2.imwrite(os.path.join(OUT_DIR, "diagnostic_table.jpg"), diag_table,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: diagnostic_table.jpg")

    # 5. Full report
    report = _assemble_report(contact_map, timeline, diag_table)
    report_path = os.path.join(OUT_DIR, "full_report.jpg")
    cv2.imwrite(report_path, report, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: full_report.jpg")
    print(f"\n  All outputs in: {OUT_DIR}")
    print("="*60 + "\n")
    return events


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Pi-HOC / SAM3D analysis report generator")
    ap.add_argument("video",        help="Path to input video")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Process only first N frames (default: all)")
    args = ap.parse_args()

    run_analysis(args.video, args.max_frames)
