"""
Impact Detection Visual Report Generator
==========================================
Generates matplotlib-based analysis charts:
  1. Impact timeline — all actions color-coded by landed/missed
  2. Velocity profiles — per-impact wrist velocity curves
  3. Summary statistics panel
  4. Full combined report image
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

import os
import io
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from detectors.phase1.impact_detector import ImpactResult
from config import OUTPUT_DIR


REPORT_DIR = os.path.join(OUTPUT_DIR, "impact_analysis")


def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)


def _fig_to_array(fig, dpi=130) -> np.ndarray:
    """Render matplotlib figure to numpy BGR array (for cv2)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    import cv2
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img if img is not None else np.zeros((100, 800, 3), dtype=np.uint8)


# ── 1. Impact Timeline ──────────────────────────────────────────────────────

def make_timeline(results: list[ImpactResult]) -> np.ndarray:
    """
    Timeline chart: x = time, y = impact score.
    Green circles = landed, red X = missed.
    """
    if not results:
        return np.zeros((250, 1000, 3), dtype=np.uint8)

    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    for r in results:
        color = "#00e676" if r.is_impact else "#ff1744"
        marker = "o" if r.is_impact else "x"
        size = 80 if r.is_impact else 50
        alpha = 0.9 if r.is_impact else 0.5
        ax.scatter(r.timestamp_seconds, r.impact_score, c=color,
                   marker=marker, s=size, zorder=5, alpha=alpha,
                   edgecolors="white" if r.is_impact else "none",
                   linewidths=0.5)

    # Threshold line
    from config import IMPACT_SCORE_THRESHOLD
    ax.axhline(IMPACT_SCORE_THRESHOLD, color="#ffd740", linestyle="--",
               linewidth=1, alpha=0.7, label=f"threshold ({IMPACT_SCORE_THRESHOLD})")

    max_t = max(r.timestamp_seconds for r in results)
    ax.set_xlim(-1, max_t + 2)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Time (s)", color="#b0bec5", fontsize=10)
    ax.set_ylabel("Impact Score", color="#b0bec5", fontsize=10)
    ax.set_title("Impact Detection Timeline", color="white",
                 fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="#78909c", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#37474f")

    patches = [
        mpatches.Patch(color="#00e676", label=f"Landed ({sum(1 for r in results if r.is_impact)})"),
        mpatches.Patch(color="#ff1744", label=f"Missed ({sum(1 for r in results if not r.is_impact)})"),
    ]
    ax.legend(handles=patches, facecolor="#1a1a2e", edgecolor="#37474f",
              labelcolor="white", fontsize=9, loc="upper right")

    ax.grid(True, alpha=0.15, color="#37474f")
    plt.tight_layout(pad=0.5)
    return _fig_to_array(fig)


# ── 2. Gate Breakdown Bar Chart ──────────────────────────────────────────────

def make_gate_breakdown(results: list[ImpactResult]) -> np.ndarray:
    """
    Horizontal grouped bar chart showing average gate scores
    for landed vs missed punches.
    """
    landed = [r for r in results if r.is_impact]
    missed = [r for r in results if not r.is_impact]

    if not landed and not missed:
        return np.zeros((250, 800, 3), dtype=np.uint8)

    gates = ["Deceleration", "3D Jerk", "Extension", "Depth Conv.", "Confidence"]

    def avg_gates(group):
        if not group:
            return [0] * 5
        return [
            float(np.mean([r.decel_score for r in group])),
            float(np.mean([r.jerk_score for r in group])),
            float(np.mean([r.extension_score for r in group])),
            float(np.mean([r.depth_score for r in group])),
            float(np.mean([r.confidence_score for r in group])),
        ]

    landed_avgs = avg_gates(landed)
    missed_avgs = avg_gates(missed)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    y = np.arange(len(gates))
    h = 0.35

    bars1 = ax.barh(y - h/2, landed_avgs, h, label="Landed", color="#00e676", alpha=0.85)
    bars2 = ax.barh(y + h/2, missed_avgs, h, label="Missed", color="#ff1744", alpha=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels(gates, color="#b0bec5", fontsize=10)
    ax.set_xlabel("Average Gate Score", color="#b0bec5", fontsize=10)
    ax.set_title("Gate Score Breakdown: Landed vs Missed",
                 color="white", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0, 1.1)
    ax.tick_params(colors="#78909c", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#37474f")

    ax.legend(facecolor="#1a1a2e", edgecolor="#37474f",
              labelcolor="white", fontsize=9)
    ax.grid(True, axis="x", alpha=0.15, color="#37474f")
    plt.tight_layout(pad=0.5)
    return _fig_to_array(fig)


# ── 3. Punch Type Distribution ──────────────────────────────────────────────

def make_type_distribution(results: list[ImpactResult]) -> np.ndarray:
    """Stacked bar chart: punch types with landed/missed counts."""
    if not results:
        return np.zeros((250, 800, 3), dtype=np.uint8)

    types = sorted(set(r.action for r in results))
    landed_counts = []
    missed_counts = []
    for t in types:
        group = [r for r in results if r.action == t]
        landed_counts.append(sum(1 for r in group if r.is_impact))
        missed_counts.append(sum(1 for r in group if not r.is_impact))

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    x = np.arange(len(types))
    w = 0.5

    ax.bar(x, landed_counts, w, label="Landed", color="#00e676", alpha=0.85)
    ax.bar(x, missed_counts, w, bottom=landed_counts, label="Missed",
           color="#ff1744", alpha=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ").title() for t in types],
                       color="#b0bec5", fontsize=10, rotation=15)
    ax.set_ylabel("Count", color="#b0bec5", fontsize=10)
    ax.set_title("Punch Type Distribution",
                 color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="#78909c", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#37474f")

    ax.legend(facecolor="#1a1a2e", edgecolor="#37474f",
              labelcolor="white", fontsize=9)
    ax.grid(True, axis="y", alpha=0.15, color="#37474f")
    plt.tight_layout(pad=0.5)
    return _fig_to_array(fig)


# ── 4. Summary Statistics Panel ──────────────────────────────────────────────

def make_summary_panel(results: list[ImpactResult], summary: dict) -> np.ndarray:
    """Key metrics as a styled stats panel."""
    fig, ax = plt.subplots(figsize=(12, 2.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    total = summary["total_actions"]
    landed = summary["total_landed"]
    missed = summary["total_missed"]
    rate = summary["landing_rate"]
    avg_score = summary["avg_impact_score"]

    landed_results = [r for r in results if r.is_impact]
    avg_power = float(np.mean([r.power_watts for r in landed_results])) if landed_results else 0
    avg_speed = float(np.mean([r.speed_kmh for r in landed_results])) if landed_results else 0

    metrics = [
        ("Total\nPunches", str(total), "#64b5f6"),
        ("Landed", str(landed), "#00e676"),
        ("Missed", str(missed), "#ff1744"),
        ("Landing\nRate", f"{rate:.0%}", "#ffd740"),
        ("Avg Impact\nScore", f"{avg_score:.2f}", "#ce93d8"),
        ("Avg Power\n(Landed)", f"{avg_power:.0f}W", "#ffab40"),
        ("Avg Speed\n(Landed)", f"{avg_speed:.1f} km/h", "#80deea"),
    ]

    n = len(metrics)
    for i, (label, value, color) in enumerate(metrics):
        cx = (i + 0.5) / n
        ax.text(cx, 0.72, value, transform=ax.transAxes,
                ha="center", va="center", fontsize=22, fontweight="bold",
                color=color)
        ax.text(cx, 0.22, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#78909c")

    ax.set_title("Impact Detection Summary", color="white",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout(pad=0.5)
    return _fig_to_array(fig)


# ── 5. Detailed Event Table ──────────────────────────────────────────────────

def make_event_table(results: list[ImpactResult]) -> np.ndarray:
    """Matplotlib table with per-event details."""
    if not results:
        return np.zeros((100, 800, 3), dtype=np.uint8)

    rows = []
    for i, r in enumerate(results[:40], 1):
        m, s = divmod(int(r.timestamp_seconds), 60)
        rows.append([
            str(i),
            f"{m:02d}:{s:02d}",
            r.action.replace("_", " ").title(),
            r.striking_hand.title(),
            r.target,
            "LANDED" if r.is_impact else "Missed",
            f"{r.impact_score:.2f}",
            f"{r.decel_score:.2f}",
            f"{r.jerk_score:.2f}",
            f"{r.speed_kmh:.1f}",
            f"{r.power_watts:.0f}",
        ])

    cols = ["#", "Time", "Type", "Hand", "Target", "Result",
            "Score", "Decel", "Jerk", "Speed", "Power"]
    nrows = len(rows)

    fig_h = max(3.0, 0.28 * nrows + 1.0)
    fig, ax = plt.subplots(figsize=(14, fig_h), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=cols,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.3)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a237e")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            is_landed = rows[r - 1][5] == "LANDED"
            bg = "#1b5e20" if is_landed else "#1a1a2e"
            cell.set_facecolor(bg if r % 2 == 0 else
                               ("#2e7d32" if is_landed else "#16213e"))
            cell.set_text_props(color="#e8f5e9" if is_landed else "#b0bec5")
        cell.set_edgecolor("#37474f")

    ax.set_title("Detailed Event Log", color="white",
                 fontsize=13, fontweight="bold", pad=8)
    plt.tight_layout(pad=0.3)
    return _fig_to_array(fig)


# ── Full Report Assembly ─────────────────────────────────────────────────────

def generate_full_report(
    results: list[ImpactResult],
    summary: dict,
    save: bool = True,
) -> str:
    """
    Generate and save all report components + a combined full report.
    Returns path to the full report image.
    """
    import cv2

    ensure_dirs()
    print("\n[Report] Generating impact analysis report …")

    # Generate each panel
    summary_panel = make_summary_panel(results, summary)
    timeline = make_timeline(results)
    gate_chart = make_gate_breakdown(results)
    type_chart = make_type_distribution(results)
    event_table = make_event_table(results)

    if save:
        cv2.imwrite(os.path.join(REPORT_DIR, "summary_panel.png"), summary_panel,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(os.path.join(REPORT_DIR, "timeline.png"), timeline,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(os.path.join(REPORT_DIR, "gate_breakdown.png"), gate_chart,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(os.path.join(REPORT_DIR, "type_distribution.png"), type_chart,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(os.path.join(REPORT_DIR, "event_table.png"), event_table,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])

    # Assemble full report
    W = 1600

    def resize_w(img, w):
        h = int(img.shape[0] * w / max(img.shape[1], 1))
        return cv2.resize(img, (w, max(h, 1)))

    panels = [summary_panel, timeline, gate_chart, type_chart, event_table]
    resized = [resize_w(p, W) for p in panels]

    # Separators
    sep = np.full((4, W, 3), 30, dtype=np.uint8)

    # Title bar
    title = np.full((60, W, 3), 13, dtype=np.uint8)
    cv2.putText(title, "SAM3D Impact Detection Analysis Report",
                (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 230, 255), 2)

    parts = [title, sep]
    for p in resized:
        parts.extend([p, sep])

    report = np.vstack(parts)

    report_path = os.path.join(REPORT_DIR, "full_report.png")
    cv2.imwrite(report_path, report, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"[Report] Full report saved: {report_path}")
    print(f"[Report] All panels saved to: {REPORT_DIR}")

    return report_path
