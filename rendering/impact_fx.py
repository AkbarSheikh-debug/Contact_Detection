#!/usr/bin/env python3
"""
Impact FX System  —  cinematic post-processor
==============================================
Reads an existing pipeline video + results JSON and overlays
cinematic animations + synthesised sound at every detected impact.

Usage
-----
    python impact_fx.py --approach D        # single approach
    python impact_fx.py --approach all      # all 7 approaches
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import os, sys, json, argparse, tempfile, time
from dataclasses import dataclass
import cv2
import numpy as np
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

# All approaches share the same source video (used to compute stride)
SOURCE_VIDEO = r"C:\Users\XRIG\Downloads\for_impact_detection_experiment_2 (1)\for_impact_detection_experiment_2\1.mp4"

APPROACH_MAP = {
    "A": dict(video="outputs/1_A_multiframe_real.mp4",            json="outputs/results_A_multiframe.json",           label="A  Multi-frame SAM",               tag="A_multiframe"),
    "B": dict(video="outputs/1_B_softgate_real.mp4",              json="outputs/results_B_softgate.json",             label="B  Soft Gate",                     tag="B_softgate"),
    "C": dict(video="outputs/1_C_heatmap_real.mp4",               json="outputs/results_C_heatmap.json",              label="C  Lower Threshold + Heatmap",     tag="C_heatmap"),
    "D": dict(video="outputs/enhanced/1_D_enhanced_real.mp4",     json="outputs/enhanced/results_D_enhanced.json",    label="D  Enhanced (elbow + receiver)",   tag="D_enhanced"),
    "E": dict(video="outputs/sam2style/1_E_sam2style_real.mp4",   json="outputs/sam2style/results_E_sam2style.json",  label="E  SAM2-Style Temporal EMA",       tag="E_sam2style"),
    "F": dict(video="outputs/learned/1_F_learned_real.mp4",       json="outputs/learned/results_F_learned.json",      label="F  Learned LogReg",                tag="F_learned"),
    "G": dict(video="outputs/dual_corr/1_G_dual_corr_real.mp4",   json="outputs/dual_corr/results_G_dual_corr.json",  label="G  Dual-Body Correlation",         tag="G_dual_corr"),
}
FX_OUTPUT_DIR = os.path.join(_HERE, "FX_Outputs")

SAMPLE_RATE = 44100
KICK_ACTIONS = {"roundhouse_kick_right", "roundhouse_kick_left", "side_kick", "front_kick"}

# Per-zone accent colors (BGR) — used only for subtle accents, not filled blobs
ZONE_ACCENT = {
    "head_left":          (80,  80,  255),   # red
    "head_right":         (80,  80,  255),
    "upper_torso_left":   (60, 160, 255),    # orange
    "upper_torso_right":  (60, 160, 255),
    "lower_torso_left":   (40, 210, 255),    # amber
    "lower_torso_right":  (40, 210, 255),
    "head":               (80,  80,  255),
    "torso":              (60, 160, 255),
}
_DEFAULT_ACCENT = (60, 160, 255)

# Animation durations (frames at ~12.5 fps)
_FLASH_DUR     = 3    # screen flash + starburst
_RING_DUR      = 10   # expanding shockwave ring
_SPARK_DUR     = 8    # flying sparks
_TEXT_START    = 2    # text appears after flash settles
_TEXT_DUR      = 16   # text linger
_SHAKE_DUR     = 2    # camera shake
ANIM_WINDOW    = max(_RING_DUR, _SPARK_DUR, _TEXT_START + _TEXT_DUR, _SHAKE_DUR) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ImpactEvent:
    frame:     int      # source video frame (NOT used for output timing)
    timestamp: float    # seconds — ground truth for output frame & audio onset
    cx:        int
    cy:        int
    score:     float
    action:    str
    region:    str
    is_kick:   bool = False

    def __post_init__(self):
        self.is_kick = self.action in KICK_ACTIONS

    @property
    def accent(self):
        return ZONE_ACCENT.get(self.region, _DEFAULT_ACCENT)


def load_impacts(json_path: str) -> list[ImpactEvent]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for e in data.get("events", []):
        if not e.get("is_impact"):
            continue
        cp = e.get("contact_point", [])
        if len(cp) < 2:
            continue
        out.append(ImpactEvent(
            frame     = int(e["impact_frame"]),
            timestamp = float(e.get("timestamp_seconds", 0.0)),
            cx        = int(cp[0]),
            cy        = int(cp[1]),
            score     = float(e.get("impact_score", 0.5)),
            action    = e.get("action", "punch"),
            region    = e.get("contact_region", "torso"),
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Audio engine
# ─────────────────────────────────────────────────────────────────────────────
class AudioEngine:
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr

    def _punch(self, score: float) -> np.ndarray:
        dur = 0.20
        t   = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
        thud  = np.sin(2 * np.pi * 140 * t) * np.exp(-t * 40)
        crack = np.random.default_rng(42).standard_normal(len(t)) * np.exp(-t * 70)
        slap  = np.sin(2 * np.pi * 800 * t) * np.exp(-t * 130)
        w = 0.55 * thud + 0.28 * crack + 0.17 * slap
        return self._scale(w, score)

    def _kick(self, score: float) -> np.ndarray:
        dur = 0.26
        t   = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
        thud  = np.sin(2 * np.pi * 80 * t) * np.exp(-t * 22)
        body  = np.sin(2 * np.pi * 210 * t) * np.exp(-t * 50)
        noise = np.random.default_rng(7).standard_normal(len(t)) * np.exp(-t * 30)
        w = 0.52 * thud + 0.26 * body + 0.22 * noise
        return self._scale(w, score)

    @staticmethod
    def _scale(w, score):
        pk = np.abs(w).max()
        if pk > 1e-8:
            w = w / pk
        vol = 0.50 + score * 0.50
        return (w * vol).astype(np.float64)

    def build_track(self, events, total_frames, fps, stride: int = 1) -> np.ndarray:
        n_samp = int(total_frames / fps * self.sr) + self.sr
        mono   = np.zeros(n_samp, dtype=np.float64)
        for ev in events:
            # Use source impact_frame // stride to get output frame, then convert to seconds
            out_frame = ev.frame // stride
            onset = int(out_frame / fps * self.sr)
            clip  = self._kick(ev.score) if ev.is_kick else self._punch(ev.score)
            end   = min(onset + len(clip), n_samp)
            mono[onset:end] += clip[:end - onset]
        mono = np.tanh(mono * 0.85)
        delay = int(0.001 * self.sr)
        stereo = np.stack([mono, np.roll(mono, delay)], axis=1).astype(np.float32)
        return stereo


# ─────────────────────────────────────────────────────────────────────────────
# Cinematic FX Renderer
# ─────────────────────────────────────────────────────────────────────────────
class FXRenderer:
    """
    Each impact plays a sequenced animation:

      age 0      : global screen flash (very brief) + starburst rays at contact
      age 0-2    : starburst fades
      age 0-9    : thin shockwave ring expands (crisp, 2 px)
      age 0-7    : sparks fly outward (thin lines, no blobs)
      age 2-17   : clean pill text floats upward
      age 0-1    : camera shake (high-score only)
    """

    def render(
        self,
        canvas: np.ndarray,
        events_ages: list[tuple[ImpactEvent, int]],
    ) -> np.ndarray:
        """Apply all active animations to canvas (in-place)."""

        # Screen shake — apply first, before overlays, so text doesn't shake
        for ev, age in events_ages:
            if age < _SHAKE_DUR and ev.score >= 0.70:
                canvas = _shake(canvas, age, ev.score)
                break

        for ev, age in events_ages:
            # Layer order: ring (background) → sparks → starburst → text (foreground)
            if age < _RING_DUR:
                _ring(canvas, ev.cx, ev.cy, age, ev.score)
            if age < _SPARK_DUR:
                _sparks(canvas, ev.cx, ev.cy, age, ev.frame, ev.score, ev.accent)
            if age < _FLASH_DUR:
                _starburst(canvas, ev.cx, ev.cy, age, ev.score, ev.accent)
            text_age = age - _TEXT_START
            if 0 <= text_age < _TEXT_DUR:
                _text_pill(canvas, ev.cx, ev.cy, text_age, ev.score,
                           ev.action, ev.region, ev.accent)

        return canvas


# ── Individual effect functions ───────────────────────────────────────────────

def _shake(canvas: np.ndarray, age: int, score: float) -> np.ndarray:
    strength = max(1, int(5 * score * (1.0 - age / _SHAKE_DUR)))
    rng = np.random.default_rng(age + 100)
    dx  = int(rng.integers(-strength, strength + 1))
    dy  = int(rng.integers(-strength, strength + 1))
    M   = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def _starburst(
    canvas: np.ndarray,
    cx: int, cy: int,
    age: int,
    score: float,
    accent: tuple,
):
    """
    Frame 0: bright global screen flash + 16-ray starburst at contact.
    Frame 1: starburst only, half alpha.
    Frame 2: ghost starburst.
    """
    h, w = canvas.shape[:2]
    t = age / _FLASH_DUR  # 0→1

    # ── Global screen flash (age 0 only, very brief) ──
    if age == 0:
        flash_alpha = 0.32
        white = np.full_like(canvas, 255)
        cv2.addWeighted(white, flash_alpha, canvas, 1 - flash_alpha, 0, canvas)

    # ── Starburst rays ────────────────────────────────
    ray_alpha  = max(0.0, 1.0 - t ** 0.6)
    if ray_alpha < 0.03:
        return

    max_r = int(45 + 55 * score)
    n_all = 16
    overlay = canvas.copy()

    for i in range(n_all):
        angle    = i * (2 * np.pi / n_all)
        is_long  = (i % 2 == 0)
        ray_len  = max_r if is_long else max_r * 0.55
        x1 = int(cx + ray_len * np.cos(angle))
        y1 = int(cy + ray_len * np.sin(angle))
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        color  = (255, 255, 255) if is_long else accent
        thick  = 2 if is_long else 1
        cv2.line(overlay, (cx, cy), (x1, y1), color, thick, cv2.LINE_AA)

    # Bright core circle (small, crisp)
    core_r = max(3, int(7 + 8 * score))
    cv2.circle(overlay, (cx, cy), core_r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), core_r + 3, (200, 230, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, ray_alpha * 0.92, canvas, 1.0, 0, canvas)


def _ring(
    canvas: np.ndarray,
    cx: int, cy: int,
    age: int,
    score: float,
):
    """
    Single thin shockwave ring expanding outward.
    Starts small, grows to ~120 px, fades as it expands.
    """
    t        = age / _RING_DUR                    # 0→1
    r        = int(4 + (100 + 30 * score) * t)   # 4 → ~130 px
    alpha    = max(0.0, (1.0 - t) ** 1.4)
    if alpha < 0.03:
        return

    overlay = canvas.copy()
    # Primary ring — thin, white
    cv2.circle(overlay, (cx, cy), r, (255, 255, 255), 2, cv2.LINE_AA)
    # Secondary ring — slightly smaller, very faint blue-white
    if r > 14:
        cv2.circle(overlay, (cx, cy), r - 8, (180, 210, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha * 0.80, canvas, 1 - alpha * 0.80, 0, canvas)


def _sparks(
    canvas: np.ndarray,
    cx: int, cy: int,
    age: int,
    seed: int,
    score: float,
    accent: tuple,
):
    """
    8 spark trails radiating outward. Each is a thin line (no tip blobs).
    Colors: white + pale yellow + accent. Gravity causes a slight downward arc.
    """
    N    = 8
    rng  = np.random.default_rng(seed)
    angs = rng.uniform(0, 2 * np.pi, N)
    spds = rng.uniform(10, 22, N) * (0.65 + score * 0.55)
    t    = age / _SPARK_DUR
    alpha = max(0.0, 1.0 - t ** 0.9)
    if alpha < 0.03:
        return

    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    grav = 3.0 * (age ** 1.3)   # downward pull, pixels

    colors = [(255, 255, 255), (120, 230, 255), accent]   # white, pale yellow, accent

    for i in range(N):
        a, v = angs[i], spds[i]
        # Tail: position at age-1; Head: position at age
        t0 = max(0, age - 1)
        x0 = int(np.clip(cx + v * t0 * np.cos(a), 0, w - 1))
        y0 = int(np.clip(cy + v * t0 * np.sin(a) + 3.0 * (t0 ** 1.3), 0, h - 1))
        x1 = int(np.clip(cx + v * age * np.cos(a), 0, w - 1))
        y1 = int(np.clip(cy + v * age * np.sin(a) + grav,             0, h - 1))

        col   = colors[i % len(colors)]
        thick = 2 if i % 4 == 0 else 1
        cv2.line(overlay, (x0, y0), (x1, y1), col, thick, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha * 0.88, canvas, 1 - alpha * 0.88, 0, canvas)


def _text_pill(
    canvas: np.ndarray,
    cx: int, cy: int,
    text_age: int,
    score: float,
    action: str,
    region: str,
    accent: tuple,
):
    """
    Clean pill-shaped label floating upward from the contact point.
    Layout:  ┌─────────────────────────────┐
             │  CROSS   HEAD LEFT   74%    │
             └─────────────────────────────┘
    Appears at text_age=0, drifts up, fades after text_age=10.
    """
    alpha = float(np.clip(
        np.interp(text_age, [0, 2, _TEXT_DUR - 4, _TEXT_DUR], [0.0, 1.0, 1.0, 0.0]),
        0.0, 1.0,
    ))
    if alpha < 0.05:
        return

    rise  = int(text_age * 2.8)   # px upward drift
    h, w  = canvas.shape[:2]

    # Build label strings
    act_str = action.replace("_", " ").upper()
    reg_str = region.replace("_", " ").title()
    pct_str = f"{score:.0%}"

    font   = cv2.FONT_HERSHEY_DUPLEX
    f_act  = 0.52
    f_reg  = 0.42
    f_pct  = 0.72
    thick  = 1

    # Measure each segment
    (tw_act, th_act), _ = cv2.getTextSize(act_str, font, f_act, thick)
    (tw_reg, th_reg), _ = cv2.getTextSize(reg_str, font, f_reg, thick)
    (tw_pct, th_pct), _ = cv2.getTextSize(pct_str, font, f_pct, 2)

    pad_x, pad_y = 10, 6
    sep = 8
    pill_w = pad_x + tw_act + sep + tw_reg + sep + tw_pct + pad_x
    pill_h = max(th_act, th_reg, th_pct) + pad_y * 2

    # Position: centred on cx, above cy with drift
    px = int(np.clip(cx - pill_w // 2, 4, w - pill_w - 4))
    py = int(np.clip(cy - 65 - rise,   4, h - pill_h - 4))

    # ── Draw pill background ──────────────────────────────────────────────
    overlay = canvas.copy()
    # Dark semi-transparent pill
    pill_bg = (20, 20, 20)
    cv2.rectangle(overlay, (px, py), (px + pill_w, py + pill_h),
                  pill_bg, -1, cv2.LINE_AA)
    # Accent-colored left edge bar (3 px wide)
    cv2.rectangle(overlay, (px, py), (px + 3, py + pill_h),
                  accent, -1, cv2.LINE_AA)
    # Thin white border
    cv2.rectangle(overlay, (px, py), (px + pill_w, py + pill_h),
                  (180, 180, 180), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha * 0.82, canvas, 1 - alpha * 0.82, 0, canvas)

    # ── Draw text on top of blended pill ─────────────────────────────────
    text_y = py + pad_y + max(th_act, th_pct) - 2
    cursor = px + pad_x + 3  # offset for accent bar

    # Action name — white
    cv2.putText(canvas, act_str, (cursor, text_y),
                font, f_act, (255, 255, 255), thick, cv2.LINE_AA)
    cursor += tw_act + sep

    # Region — light accent
    light = tuple(min(255, int(c * 1.4 + 80)) for c in accent)
    cv2.putText(canvas, reg_str, (cursor, text_y),
                font, f_reg, light, thick, cv2.LINE_AA)
    cursor += tw_reg + sep

    # Score percent — bright white, bold
    cv2.putText(canvas, pct_str, (cursor, text_y),
                font, f_pct, (255, 255, 255), 2, cv2.LINE_AA)

    # Apply text alpha by blending just the text region (avoids double-blend)
    # Text is already drawn on canvas; if we need fade, blend with pre-text copy
    # (handled above via overlay alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_approach(label: str, video_path: str, json_path: str, out_path: str,
                     source_video: str = SOURCE_VIDEO):
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"  video   : {video_path}")
    print(f"  results : {json_path}")
    print(f"  output  : {out_path}")
    print(f"{'=' * 72}")

    events = load_impacts(json_path)
    print(f"  {len(events)} impact events  |  window={ANIM_WINDOW} frames each")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}");  return

    fps      = cap.get(cv2.CAP_PROP_FPS)
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {W}x{H}  {fps:.2f} fps  {n_frames} frames")

    # Compute stride from source video fps so we can map source frame -> output frame.
    # run_video writes every stride-th source frame, so: out_frame = impact_frame // stride
    # timestamp_seconds in JSON is the ACTION onset time (not the contact frame), so we
    # must use impact_frame // stride — NOT int(timestamp * output_fps).
    stride = 1
    src_cap = cv2.VideoCapture(source_video)
    if src_cap.isOpened():
        src_fps = src_cap.get(cv2.CAP_PROP_FPS)
        src_cap.release()
        stride = max(1, round(src_fps / fps))
        print(f"  Source fps={src_fps:.3f}  stride={stride}  (out_frame = impact_frame // {stride})")
    else:
        print(f"  [WARN] Cannot open source video for stride; using timestamp*fps fallback")

    # Build per-frame look-up: impact_frame // stride gives the exact output frame
    frame_events: dict[int, list[tuple[ImpactEvent, int]]] = {}
    for ev in events:
        out_frame = ev.frame // stride
        for age in range(ANIM_WINDOW):
            fi = out_frame + age
            if 0 <= fi < n_frames:
                frame_events.setdefault(fi, []).append((ev, age))

    renderer  = FXRenderer()
    fd, tmp   = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for fi in tqdm(range(n_frames), desc=f"  Rendering {label}", unit="fr", ncols=78):
        ret, frame = cap.read()
        if not ret:
            break
        ea = frame_events.get(fi)
        if ea:
            frame = renderer.render(frame, ea)
        writer.write(frame)

    cap.release()
    writer.release()

    # ── Audio ─────────────────────────────────────────────────────────────
    engine = AudioEngine()
    audio  = engine.build_track(events, n_frames, fps, stride=stride)
    print(f"  Audio: {audio.shape[0] / SAMPLE_RATE:.1f}s  stereo")

    # ── Mux with moviepy ──────────────────────────────────────────────────
    try:
        from moviepy import VideoFileClip, AudioArrayClip
        clip  = VideoFileClip(tmp)
        aclip = AudioArrayClip(audio, fps=SAMPLE_RATE).subclipped(0, clip.duration)
        final = clip.with_audio(aclip)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        final.write_videofile(out_path, codec="libx264", audio_codec="aac",
                              logger=None, ffmpeg_params=["-crf", "20"])
        clip.close(); aclip.close()
        print(f"  Saved: {out_path}")
    except Exception as e:
        import shutil
        print(f"  [WARN] audio mux failed ({e}) — saving silent video.")
        shutil.copy(tmp, out_path)
    finally:
        try: os.remove(tmp)
        except OSError: pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Impact FX System — cinematic animations + sound on detected impacts"
    )
    parser.add_argument("--approach", default="all",
                        choices=["all", "A", "B", "C", "D", "E", "F", "G"])
    parser.add_argument("--video",   default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--out",     default=None)
    parser.add_argument("--out-dir", default=FX_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    print()
    print("=" * 72)
    print("  SAM3D  |  Impact FX  (cinematic edition)")
    print("=" * 72)

    if args.video and args.results:
        out = args.out or os.path.join(
            args.out_dir,
            os.path.splitext(os.path.basename(args.video))[0] + "_fx.mp4",
        )
        vp = args.video if os.path.isabs(args.video) else os.path.join(_HERE, args.video)
        jp = args.results if os.path.isabs(args.results) else os.path.join(_HERE, args.results)
        process_approach("Custom", vp, jp, out)
        print(f"\n  Done in {time.time()-t0:.1f}s\n");  return 0

    keys = list(APPROACH_MAP) if args.approach == "all" else [args.approach]
    print(f"  Approaches : {keys}\n")

    table = []
    for key in keys:
        cfg  = APPROACH_MAP[key]
        vp   = os.path.join(_HERE, cfg["video"])
        jp   = os.path.join(_HERE, cfg["json"])
        out  = args.out or os.path.join(args.out_dir, f"{cfg['tag']}_fx.mp4")
        if not os.path.isfile(vp):
            print(f"  [SKIP] video not found: {vp}"); continue
        if not os.path.isfile(jp):
            print(f"  [SKIP] json  not found: {jp}"); continue
        t1 = time.time()
        process_approach(cfg["label"], vp, jp, out)
        table.append((cfg["label"], out, time.time() - t1))

    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  Impact FX complete in {elapsed:.1f}s")
    print(f"  {'Approach':<45}  {'Time':>5}  File")
    print(f"  {'-' * 70}")
    for lbl, path, dt in table:
        print(f"  {lbl:<45}  {dt:>4.0f}s  {os.path.basename(path)}")
    print(f"{'=' * 72}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
