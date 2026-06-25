#!/usr/bin/env python3
"""
Video clip annotation tool for impact detection labeling.
Uses tkinter + PIL — works on Windows without a GUI OpenCV build.

Controls (keyboard shortcuts active when window is focused):
    I / 1       -> IMPACT           (punch landed)
    N / 0       -> NOT IMPACT       (missed / blocked)
    M           -> MARK IMPACT FRAME at the currently displayed frame
                   (pauses first if playing, so you mark the exact frame
                   you're looking at, not whatever plays next)
    Space       -> PAUSE / RESUME
    Left arrow  -> step back 1 frame (while paused)
    Right arrow -> step forward 1 frame (while paused)
    , (comma)   -> step back 5 frames
    . (period)  -> step forward 5 frames
    Click/drag on the progress bar -> scrub directly to that frame (pauses)
    S           -> SKIP             (leave unlabeled, come back later)
    B           -> BACK             (step to the previous clip, even if already labeled)
    R           -> REPLAY           current clip from the start
    Q / Esc     -> QUIT             (saves all progress)

Labels (and marked impact frames) are written to manifest.json after every
decision. You can quit and resume at any time -- the tool auto-starts at the
first unlabeled clip, but [B] Back always works to step into clips you
already labeled, so you can review or correct any of them.

Marking the exact contact frame is independent of labeling -- you can mark
before or after pressing I/N, and re-marking just overwrites the previous
frame. Use --needs-impact-frame to build a review queue of every clip
that's already labeled "impact" but doesn't have an exact frame marked yet,
so you can go back through past annotation sessions and fill these in
without re-finding each one by hand.

Usage:
    python tools/annotate_clips.py --fight lillyella_vs_zoe
    python tools/annotate_clips.py --fight cameron_vs_liam --round 3
    python tools/annotate_clips.py --fight jamie_vs_ryan --fighter 0
    python tools/annotate_clips.py --fight 1st_fight --speed 0.5      # slow motion
    python tools/annotate_clips.py --fight 1st_fight --show-labeled   # start from clip #1
    python tools/annotate_clips.py --fight 1st_fight --jump-to 448    # start at a specific clip
    python tools/annotate_clips.py --fight lillyella_vs_zoe --needs-impact-frame
                                                       # review pass: only already-labeled
                                                       # "impact" clips missing a marked frame
"""

import os
import sys
import json
import argparse
import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

# os.path.join (not string-concatenated literal backslashes) so this
# resolves on Linux/macOS too, not just Windows -- tools/ and dataset/ are
# sibling folders under the repo root either way.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "dataset"))
import fights

# Windows has Consolas; it doesn't exist on Linux/macOS (tkinter would
# silently substitute *some* font, but picking a real monospace font per
# platform looks right instead of leaving it to chance).
if sys.platform == "win32":
    MONO_FONT = "Consolas"
elif sys.platform == "darwin":
    MONO_FONT = "Menlo"
else:
    MONO_FONT = "DejaVu Sans Mono"

# Set by main() from fights.get_fight(args.fight) before any of the helpers
# below are called -- left as None here so importing this module doesn't
# silently default to one fight's data.
MANIFEST_PATH = None
CLIPS_DIR     = None
FIGHTER_NAMES = None
FIGHTER_SHORT_NAMES = None
ROUNDS_DIR    = None
VIDEO_FOLDER  = None
IDENTITY_MARKER = None

LABEL_IMPACT     = "impact"
LABEL_NOT_IMPACT = "not_impact"

_sam3d_bbox_cache = {}


def load_sam3d_bboxes_for_round(round_id):
    """Lazy-load + cache per-fighter bbox-by-frame from each round's merged
    sam3d.json (Gladius's own per-fighter CUTIE run)."""
    if round_id in _sam3d_bbox_cache:
        return _sam3d_bbox_cache[round_id]
    path = os.path.join(ROUNDS_DIR, f"Round{round_id}", "sam3d.json")
    by_fighter = None
    if os.path.exists(path):
        d = json.load(open(path))
        by_fighter = {}
        for fid_str in ("0", "1"):
            entries = d.get(fid_str, [])
            by_fighter[int(fid_str)] = {e["frame"]: e["bbox"] for e in entries}
    _sam3d_bbox_cache[round_id] = by_fighter
    return by_fighter


def lookup_bbox_with_fallback(frame_dict, frame):
    """Exact-frame bbox lookup ONLY -- no carried-forward fallback. A
    forward-filled stale position was tried here and reverted: during a real
    CUTIE tracking gap the fighter has usually moved by the time tracking
    picks back up, so a several-frames-old box renders in a location that no
    longer corresponds to anyone on screen (confirmed concretely: frame 4424
    fighter1 was at [806,216,890,304], but by frame 4430+ the real fighter
    had moved well past that, leaving the carried-forward tag floating over
    empty ring/turnbuckle). Showing no tag for the untracked fighter -- while
    the other fighter's tag, which has no gap, keeps showing correctly -- is
    more honest than a confidently-wrong stale guess."""
    return frame_dict.get(frame)


def green_score(frame_rgb, bbox):
    """Fraction of green-dominant pixels in the lower (shorts) region of bbox.
    Zoe's shorts have a bright green dragon trim; Lillyella's are plain black.
    This appearance check is independent of tracker identity/continuity, so it
    can't accumulate drift the way frame-to-frame tracking correction can --
    it directly answers 'does this region look like Zoe's shorts' each time.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    ly1 = y1 + int(h * 0.55)
    region = frame_rgb[max(0, ly1):y2, max(0, x1):x2].astype("float32")
    if region.size == 0:
        return 0.0
    r, g, b = region[..., 0], region[..., 1], region[..., 2]
    green_mask = (g > r + 15) & (g > b + 15) & (g > 60)
    return float(green_mask.mean())


_swap_timeline_cache = {}


def get_round_swap_timeline(round_id, sample_stride=20, confident_margin=0.03):
    """Scan an entire round's full video at a coarse stride, using the green-
    trim appearance check at each sample to build a frame -> name_swap-needed
    timeline. A single clip only has ~30-40 frames to sample from, which can
    be too weak/noisy during a clinch (shorts occluded). Scanning the whole
    round gives many more confident samples to anchor on, and ambiguous
    samples simply carry forward the last confident verdict -- so a single
    noisy clinch can't flip the result, only a real, sustained swap can.

    No-ops (returns None) for fights with no identity_marker configured --
    most fights trust the tracker IDs as-is.

    Checks for a cached dataset/<fight>/RoundN/swap_timeline.json sidecar
    first (the same file tools/extract_keypoint_dataset.py writes during
    training-data extraction) before falling back to scanning the raw video.
    This matters beyond just startup speed: the raw video lives under a
    machine-specific Downloads/Desktop path that won't exist on another
    machine, so without the sidecar check, a teammate running this tool on a
    fresh checkout would silently get NO swap correction at all for any
    fight that needs it (the os.path.exists check below just returns None
    rather than crashing -- easy to not notice until names look wrong).
    """
    if round_id in _swap_timeline_cache:
        return _swap_timeline_cache[round_id]

    if IDENTITY_MARKER is None:
        _swap_timeline_cache[round_id] = None
        return None

    sidecar = os.path.join(ROUNDS_DIR, f"Round{round_id}", "swap_timeline.json")
    if os.path.exists(sidecar):
        d = json.load(open(sidecar))
        timeline = {int(k): v for k, v in d["timeline"].items()}
        result = (timeline, d["stride"])
        _swap_timeline_cache[round_id] = result
        return result

    bbox_data = load_sam3d_bboxes_for_round(round_id)
    video_path = os.path.join(VIDEO_FOLDER, f"Round{round_id}.mp4")
    if not bbox_data or not os.path.exists(video_path):
        _swap_timeline_cache[round_id] = None
        return None

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timeline = {}
    last_confident = False
    for fr in range(0, total, sample_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ret, frame = cap.read()
        if not ret:
            continue
        b0 = bbox_data.get(0, {}).get(fr)
        b1 = bbox_data.get(1, {}).get(fr)
        if b0 and b1:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            g0 = green_score(rgb, b0)
            g1 = green_score(rgb, b1)
            if abs(g0 - g1) > confident_margin:
                last_confident = g0 > g1
        timeline[fr] = last_confident
    cap.release()

    result = (timeline, sample_stride)
    _swap_timeline_cache[round_id] = result
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    json.dump({"stride": sample_stride, "timeline": {str(k): v for k, v in timeline.items()}},
              open(sidecar, "w"), indent=2)
    return result


def lookup_name_swap(round_id, frame):
    result = get_round_swap_timeline(round_id)
    if result is None:
        return False
    timeline, stride = result
    nearest = (frame // stride) * stride
    return timeline.get(nearest, False)

# Colours (R,G,B for PIL / hex for tkinter)
COL_IMPACT     = "#00e600"
COL_NOT_IMPACT = "#e63200"
COL_SKIP       = "#cccc00"
COL_NEUTRAL    = "#3a3a3a"
COL_BG         = "#1a1a1a"
COL_TEXT       = "#e0e0e0"
COL_ACCENT     = "#00d7ff"


def load_manifest():
    return json.load(open(MANIFEST_PATH))


def save_manifest(manifest):
    labeled   = sum(1 for c in manifest["clips"] if c["label"] is not None)
    unlabeled = len(manifest["clips"]) - labeled
    impact_clips = [c for c in manifest["clips"] if c["label"] == LABEL_IMPACT]
    manifest["labeled"]   = labeled
    manifest["unlabeled"] = unlabeled
    manifest["impact_frames_marked"]   = sum(1 for c in impact_clips if c.get("impact_frame") is not None)
    manifest["impact_frames_unmarked"] = sum(1 for c in impact_clips if c.get("impact_frame") is None)
    json.dump(manifest, open(MANIFEST_PATH, "w"), indent=2)


class AnnotatorApp:
    def __init__(self, root, queue, manifest, speed=1.0, start_pos=0):
        self.root     = root
        self.queue    = queue        # list of manifest indices to label
        self.manifest = manifest
        self.clips    = manifest["clips"]
        self.speed    = speed

        self.pos      = start_pos    # current position in queue
        self.result   = None         # set by key handler
        self._playing = False
        self._paused  = False
        self._frames  = []
        self._fi      = 0
        self._after_id = None

        self._build_ui()
        self._bind_keys()
        self._load_clip()

    def _maximize_window(self):
        """root.state('zoomed') is Windows-only -- it raises a TclError on
        Linux and is simply unsupported on macOS. -zoomed is the X11
        equivalent and works on most Linux window managers (still wrapped
        in try/except since not all of them honor it); macOS and any
        leftover case fall back to sizing the window to the full screen
        manually, which always works everywhere."""
        if sys.platform == "win32":
            self.root.state("zoomed")
            return
        try:
            self.root.attributes("-zoomed", True)
        except tk.TclError:
            w = self.root.winfo_screenwidth()
            h = self.root.winfo_screenheight()
            self.root.geometry(f"{w}x{h}+0+0")

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("Impact Annotator")
        self.root.configure(bg=COL_BG)
        self._maximize_window()
        self.root.resizable(True, True)

        # Top info bar
        self.info_var = tk.StringVar(value="Loading...")
        info = tk.Label(self.root, textvariable=self.info_var,
                        bg=COL_BG, fg=COL_ACCENT,
                        font=(MONO_FONT, 11, "bold"), anchor="w", padx=8)
        info.pack(fill="x", pady=(6, 0))

        # Sub-info (action / frame range)
        self.sub_var = tk.StringVar(value="")
        sub = tk.Label(self.root, textvariable=self.sub_var,
                       bg=COL_BG, fg=COL_TEXT,
                       font=(MONO_FONT, 10), anchor="w", padx=8)
        sub.pack(fill="x")

        # Marked impact-frame status
        self.impact_frame_var = tk.StringVar(value="")
        impf = tk.Label(self.root, textvariable=self.impact_frame_var,
                        bg=COL_BG, fg="#ffcc00",
                        font=(MONO_FONT, 10, "bold"), anchor="w", padx=8)
        impf.pack(fill="x")

        # Video canvas
        self.canvas = tk.Canvas(self.root, bg="#000000",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self._tk_img = None

        # Transport row: pause/play button + canvas scrub bar
        transport = tk.Frame(self.root, bg=COL_BG)
        transport.pack(fill="x", padx=8, pady=4)

        btn_cfg_t = dict(bg="#444", fg=COL_TEXT, font=(MONO_FONT, 12, "bold"),
                         relief="flat", cursor="hand2", width=3)
        tk.Button(transport, text="◀", command=lambda: self._step_frame(-1),
                  **btn_cfg_t).pack(side="left", padx=(0, 2))
        self.pause_btn = tk.Button(transport, text="⏸",
                                   bg="#444", fg=COL_TEXT,
                                   font=(MONO_FONT, 12, "bold"), width=3,
                                   relief="flat", cursor="hand2",
                                   command=self._toggle_pause)
        self.pause_btn.pack(side="left", padx=(0, 2))
        tk.Button(transport, text="▶", command=lambda: self._step_frame(1),
                  **btn_cfg_t).pack(side="left", padx=(0, 8))

        self.scrub_canvas = tk.Canvas(transport, bg="#333", height=28,
                                      highlightthickness=1,
                                      highlightbackground="#555",
                                      cursor="hand2")
        self.scrub_canvas.pack(side="left", fill="x", expand=True)
        self.scrub_canvas.bind("<Button-1>", self._on_scrub)
        self.scrub_canvas.bind("<B1-Motion>", self._on_scrub)

        # Label buttons
        btn_frame = tk.Frame(self.root, bg=COL_BG)
        btn_frame.pack(fill="x", padx=8, pady=(2, 6))

        self.lbl_status = tk.Label(btn_frame, text="Current: unlabeled",
                                   bg=COL_BG, fg="#888", font=(MONO_FONT, 10))
        self.lbl_status.pack(side="left", padx=6)

        btn_cfg = dict(font=(MONO_FONT, 11, "bold"), width=14,
                       relief="flat", cursor="hand2")
        tk.Button(btn_frame, text="[I] IMPACT",
                  bg=COL_IMPACT, fg="#000",
                  command=lambda: self._label(LABEL_IMPACT),
                  **btn_cfg).pack(side="right", padx=4)
        tk.Button(btn_frame, text="[N] NOT IMPACT",
                  bg=COL_NOT_IMPACT, fg="#fff",
                  command=lambda: self._label(LABEL_NOT_IMPACT),
                  **btn_cfg).pack(side="right", padx=4)
        tk.Button(btn_frame, text="[R] Replay",
                  bg="#444", fg=COL_TEXT,
                  command=self._replay,
                  **btn_cfg).pack(side="right", padx=4)
        tk.Button(btn_frame, text="[S] Skip",
                  bg=COL_SKIP, fg="#000",
                  command=self._skip,
                  **btn_cfg).pack(side="right", padx=4)
        tk.Button(btn_frame, text="[B] Back",
                  bg="#555", fg=COL_TEXT,
                  command=self._prev_clip,
                  **btn_cfg).pack(side="right", padx=4)
        tk.Button(btn_frame, text="[M] Mark Frame",
                  bg="#ff00dc", fg="#000",
                  command=self._mark_impact_frame,
                  **btn_cfg).pack(side="right", padx=4)

        # Bottom hint
        hint = tk.Label(self.root,
                        text="I=impact  N=not_impact  M=mark impact frame  B=back  Space=pause  "
                             "Left/Right=step1  ,/.=step5  click/drag bar=scrub  R=replay  S=skip  Q=quit",
                        bg=COL_BG, fg="#666", font=(MONO_FONT, 9))
        hint.pack(pady=(0, 4))

    def _bind_keys(self):
        self.root.bind("<Key>", self._on_key)
        self.root.focus_set()

    def _on_key(self, event):
        k = event.keysym.lower()
        if k in ("i", "1"):
            self._label(LABEL_IMPACT)
        elif k in ("n", "0"):
            self._label(LABEL_NOT_IMPACT)
        elif k == "m":
            self._mark_impact_frame()
        elif k == "r":
            self._replay()
        elif k == "s":
            self._skip()
        elif k in ("q", "escape"):
            self._quit()
        elif k in ("b", "backspace"):
            self._prev_clip()
        elif k == "space":
            self._toggle_pause()
        elif k == "left":
            self._step_frame(-1)
        elif k == "right":
            self._step_frame(1)
        elif k == "comma":
            self._step_frame(-5)
        elif k == "period":
            self._step_frame(5)

    def _prev_clip(self):
        if self.pos <= 0:
            return
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self._playing = False
        self.pos -= 1
        self._load_clip()

    def _toggle_pause(self):
        if not self._playing:
            return
        self._paused = not self._paused
        self.pause_btn.config(text="▶" if self._paused else "⏸")
        if not self._paused:
            # Resume: restart the loop from current position
            if self._after_id is not None:
                self.root.after_cancel(self._after_id)
            self._play_frame()

    def _step_frame(self, delta):
        if not self._frames:
            return
        if not self._paused:
            self._paused = True
            self.pause_btn.config(text="▶")
        self._fi = max(0, min(len(self._frames) - 1, self._fi + delta))
        self._render_current_frame()

    def _on_scrub(self, event):
        """Click or drag on the scrub bar -> jump straight to that frame."""
        if not self._frames:
            return
        self._paused = True
        self.pause_btn.config(text="▶")
        width = self.scrub_canvas.winfo_width()
        frac  = max(0.0, min(1.0, event.x / max(1, width)))
        self._fi = int(round(frac * (len(self._frames) - 1)))
        self._render_current_frame()

    # ── Clip loading & playback ───────────────────────────────────────────────
    def _load_clip(self):
        if self.pos >= len(self.queue):
            self._finish()
            return

        idx   = self.queue[self.pos]
        entry = self.clips[idx]
        clip_path = os.path.join(CLIPS_DIR, entry["clip"])

        total      = len(self.queue)
        labeled_t  = sum(1 for c in self.clips if c["label"] is not None)
        fighter_name = FIGHTER_NAMES.get(entry['fighter_id'], f"Fighter {entry['fighter_id']}")
        self.info_var.set(
            f"[{self.pos+1}/{total}]  Round {entry['round']}  "
            f"{fighter_name}  —  "
            f"Overall: {labeled_t}/{len(self.clips)} labeled")
        self.sub_var.set(
            f"Action: {entry['action']}   "
            f"frames {entry['window_start']}-{entry['window_end']}")

        existing = entry.get("label")
        if existing:
            col = COL_IMPACT if existing == LABEL_IMPACT else COL_NOT_IMPACT
            self.lbl_status.config(text=f"Current: {existing}", fg=col)
        else:
            self.lbl_status.config(text="Current: unlabeled", fg="#888")
        self._update_impact_frame_label()

        # Load all frames into memory (clips are short: ~1 second)
        self._frames = []
        self._bbox_data = None
        self._name_swap = False
        if not os.path.exists(clip_path):
            self.info_var.set(f"[{self.pos+1}/{total}]  CLIP NOT FOUND: {entry['clip']}")
            self._frames = []
        else:
            cap = cv2.VideoCapture(clip_path)
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_clip_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Compute which frames correspond to the action window (for highlight)
            self._pad_before = 8
            self._win_len    = entry["window_end"] - entry["window_start"]

            # Map clip-local frame index -> global round frame index, for bbox lookup
            s_frame = max(0, entry["window_start"] - self._pad_before)  # 1-indexed, clamped
            self._global_frame_offset = (s_frame - 1) if s_frame > 0 else 0
            self._bbox_data = load_sam3d_bboxes_for_round(entry["round"])

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR -> RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._frames.append(rgb)
            cap.release()

            # Whether fid0/fid1 names need swapping, from the round-wide green
            # -trim timeline (more robust than checking only this clip's own
            # frames, which can be too weak/noisy during a clinch).
            self._name_swap = lookup_name_swap(entry["round"],
                                               self._global_frame_offset)

        self._fi = 0
        self._replay()

    def _replay(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self._fi = 0
        self._paused = False
        self._playing = True
        self.pause_btn.config(text="⏸")
        self._play_frame()

    def _render_current_frame(self):
        """Draw self._frames[self._fi] onto canvas (used for both playback and step)."""
        if not self._frames:
            return
        frame = self._frames[self._fi]
        h, w  = frame.shape[:2]

        in_window = self._pad_before <= self._fi <= self._pad_before + self._win_len
        marked_frame = self.clips[self.queue[self.pos]].get("impact_frame")
        is_marked = marked_frame is not None and marked_frame == self._global_frame_offset + self._fi
        if is_marked:
            frame = frame.copy()
            frame[:6, :] = [255, 0, 220]
            frame[-6:, :] = [255, 0, 220]
            frame[:, :6]  = [255, 0, 220]
            frame[:, -6:] = [255, 0, 220]
        elif in_window:
            frame = frame.copy()
            frame[:4, :] = [200, 200, 0]
            frame[-4:, :] = [200, 200, 0]
            frame[:, :4]  = [200, 200, 0]
            frame[:, -4:] = [200, 200, 0]

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        if nw > 0 and nh > 0:
            img    = Image.fromarray(frame).resize((nw, nh), Image.NEAREST)
            tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, anchor="center",
                                     image=tk_img)
            self._tk_img = tk_img

            # Fighter name labels, positioned over each fighter's SAM3D bbox.
            # Built as a list first (not drawn immediately) so overlapping
            # labels -- e.g. both fighters clinched close together, where
            # their bboxes' top-center points land within a few pixels of
            # each other -- can be detected and pushed apart before drawing,
            # instead of rendering on top of each other into unreadable text.
            if self._bbox_data:
                img_left = cw // 2 - nw // 2
                img_top  = ch // 2 - nh // 2
                global_frame = self._global_frame_offset + self._fi
                labels = []
                for fid in (0, 1):
                    bbox = lookup_bbox_with_fallback(self._bbox_data.get(fid) or {}, global_frame)
                    if not bbox:
                        continue
                    x1, y1, x2, y2 = bbox
                    cx = img_left + ((x1 + x2) / 2) * scale
                    top_y = img_top + y1 * scale
                    label_y = max(16, top_y - 14)
                    show_fid = (1 - fid) if self._name_swap else fid
                    name = FIGHTER_SHORT_NAMES.get(show_fid, f"Fighter {show_fid}")
                    # anchor_x/anchor_y = the fighter's true bbox position (leader
                    # line always points here); label_x/label_y = where the text
                    # is drawn, which collision-avoidance below may nudge away
                    # from the anchor -- the two are tracked separately so a
                    # nudged label still points an (now diagonal) line at the
                    # correct person instead of silently relabeling who it's for.
                    labels.append({"anchor_x": cx, "anchor_y": top_y,
                                   "label_x": cx, "label_y": label_y, "name": name})

                MIN_SEP = 28  # px, below this the two labels visually collide
                if len(labels) == 2 and abs(labels[0]["label_x"] - labels[1]["label_x"]) < MIN_SEP:
                    # Same horizontal spot (clinch range) -- stack vertically.
                    lo, hi = sorted(labels, key=lambda L: L["label_y"])
                    hi["label_y"] = lo["label_y"] - 22
                elif len(labels) == 2 and abs(labels[0]["label_y"] - labels[1]["label_y"]) < MIN_SEP:
                    # Same height, different horizontal position -- nudge apart.
                    left, right = sorted(labels, key=lambda L: L["label_x"])
                    left["label_x"]  -= MIN_SEP / 2
                    right["label_x"] += MIN_SEP / 2

                for L in labels:
                    self.canvas.create_line(L["label_x"], L["label_y"] + 4, L["anchor_x"], L["anchor_y"],
                                            fill="#ff3333", width=2, tags="overlay")
                    for dx, dy, col in ((1, 1, "#000000"), (0, 0, "#ffffff")):
                        self.canvas.create_text(L["label_x"] + dx, L["label_y"] + dy, text=L["name"],
                                                fill=col, font=(MONO_FONT, 13, "bold"),
                                                anchor="s", tags="overlay")

            # Frame counter overlay when paused
            if self._paused:
                win_tag = " [IN WINDOW]" if in_window else ""
                mark_tag = "  *** MARKED IMPACT FRAME ***" if is_marked else ""
                label_txt = f"PAUSED  frame {self._fi+1}/{len(self._frames)}{win_tag}{mark_tag}"
                x, y = cw // 2, 20
                txt_col = "#ff00dc" if is_marked else "#ffff00"
                self.canvas.create_text(x, y, text=label_txt,
                                        fill="#000000", font=(MONO_FONT, 11, "bold"),
                                        anchor="n", tags="overlay")
                self.canvas.create_text(x, y - 1, text=label_txt,
                                        fill=txt_col, font=(MONO_FONT, 11, "bold"),
                                        anchor="n", tags="overlay")

        self._update_scrub()

    def _update_scrub(self):
        """Redraw the canvas scrub bar to reflect self._fi."""
        frac = self._fi / max(1, len(self._frames) - 1) if self._frames else 0.0
        self.scrub_canvas.update_idletasks()
        w = max(1, self.scrub_canvas.winfo_width())
        h = max(1, self.scrub_canvas.winfo_height())
        self.scrub_canvas.delete("all")
        self.scrub_canvas.create_rectangle(0, 0, w, h, fill="#333", outline="")
        filled_w = int(frac * w)
        if filled_w > 0:
            self.scrub_canvas.create_rectangle(0, 0, filled_w, h, fill="#0099dd", outline="")
        px = max(1, min(w - 1, int(frac * w)))
        self.scrub_canvas.create_line(px, 0, px, h, fill="white", width=2)

    def _play_frame(self):
        if not self._playing:
            return
        if not self._frames:
            return

        if self._paused:
            # Stay on current frame; reschedule to keep the loop alive for resume
            delay = max(1, int(1000 / (self._fps * self.speed)))
            self._after_id = self.root.after(delay, self._play_frame)
            return

        self._render_current_frame()

        self._fi += 1
        if self._fi >= len(self._frames):
            self._fi = 0   # loop

        delay = max(1, int(1000 / (self._fps * self.speed)))
        self._after_id = self.root.after(delay, self._play_frame)

    # ── Decision handlers ─────────────────────────────────────────────────────
    def _label(self, label):
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self._playing = False

        idx = self.queue[self.pos]
        self.clips[idx]["label"] = label
        if label == LABEL_NOT_IMPACT:
            # A marked impact frame doesn't apply to a miss -- clear it
            # rather than leave a stale frame number sitting on a clip
            # that's no longer labeled "impact".
            self.clips[idx]["impact_frame"] = None
        save_manifest(self.manifest)

        icon = "HIT" if label == LABEL_IMPACT else "---"
        entry = self.clips[idx]
        print(f"  [{self.pos+1}/{len(self.queue)}] {icon}  {entry['clip']}")

        # Flash background
        col = COL_IMPACT if label == LABEL_IMPACT else COL_NOT_IMPACT
        self._flash(col)

        self.pos += 1
        self.root.after(120, self._load_clip)

    def _mark_impact_frame(self):
        """Mark the currently displayed frame as this clip's exact impact
        frame (global frame number, same numbering as window_start/
        window_end). Independent of labeling -- can be called before or
        after I/N, and pressing M again just overwrites with whatever
        frame is showing now."""
        if not self._frames:
            return
        if self._playing and not self._paused:
            self._paused = True  # freeze on the frame the user was looking at

        global_frame = self._global_frame_offset + self._fi
        idx = self.queue[self.pos]
        self.clips[idx]["impact_frame"] = global_frame
        save_manifest(self.manifest)
        print(f"  [{self.pos+1}/{len(self.queue)}] MARK impact_frame={global_frame}  "
              f"{self.clips[idx]['clip']}")
        self._update_impact_frame_label()
        self._render_current_frame()

    def _update_impact_frame_label(self):
        idx = self.queue[self.pos]
        mf = self.clips[idx].get("impact_frame")
        if mf is None:
            self.impact_frame_var.set("Impact frame: not marked  [M to mark]")
        else:
            self.impact_frame_var.set(f"Impact frame: {mf}  [M to remark]")

    def _skip(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self._playing = False
        entry = self.clips[self.queue[self.pos]]
        print(f"  [{self.pos+1}/{len(self.queue)}] SKP  {entry['clip']}")
        self.pos += 1
        self._load_clip()

    def _quit(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self._playing = False
        self._print_summary()
        self.root.destroy()

    def _finish(self):
        self._print_summary()
        self.root.after(1500, self.root.destroy)

    def _flash(self, colour, n=3):
        orig = self.root.cget("bg")
        def step(i):
            if i >= n * 2:
                self.root.configure(bg=orig)
                return
            self.root.configure(bg=colour if i % 2 == 0 else orig)
            self.root.after(55, lambda: step(i + 1))
        step(0)

    def _print_summary(self):
        labeled   = sum(1 for c in self.clips if c["label"] is not None)
        impact    = sum(1 for c in self.clips if c["label"] == LABEL_IMPACT)
        not_imp   = sum(1 for c in self.clips if c["label"] == LABEL_NOT_IMPACT)
        unlabeled = len(self.clips) - labeled
        marked    = sum(1 for c in self.clips if c["label"] == LABEL_IMPACT and c.get("impact_frame") is not None)
        print(f"\n{'='*50}")
        print(f"  Total labeled : {labeled} / {len(self.clips)}")
        print(f"    IMPACT      : {impact}  ({marked} with an exact frame marked)")
        print(f"    NOT_IMPACT  : {not_imp}")
        print(f"    Unlabeled   : {unlabeled}")
        if impact:
            print(f"\n  Impact frames marked: {marked}/{impact}  "
                  f"(re-run with --needs-impact-frame to fill in the rest)")
        print(f"\n  Manifest saved: {MANIFEST_PATH}")
        if unlabeled == 0:
            print(f"\n  All clips labeled! Run:")
            print(f"    python tools/train_impact_model.py")
        print(f"{'='*50}")


def main():
    global MANIFEST_PATH, CLIPS_DIR, FIGHTER_NAMES, FIGHTER_SHORT_NAMES
    global ROUNDS_DIR, VIDEO_FOLDER, IDENTITY_MARKER

    ap = argparse.ArgumentParser()
    ap.add_argument("--fight",        type=str,   default="lillyella_vs_zoe",
                    choices=fights.all_fight_names(),
                    help="which registered fight to annotate (see dataset/fights.py)")
    ap.add_argument("--round",        type=int,   default=None)
    ap.add_argument("--fighter",      type=int,   default=None)
    ap.add_argument("--action",       type=str,   default=None)
    ap.add_argument("--speed",        type=float, default=1.0)
    ap.add_argument("--show-labeled", action="store_true")
    ap.add_argument("--jump-to",      type=int,   default=None,
                    help="start at this clip number (1-indexed, as shown in the "
                         "[N/496] header), instead of auto-resuming at the first "
                         "unlabeled clip")
    ap.add_argument("--needs-impact-frame", action="store_true",
                    help="review-pass mode: queue is restricted to clips already "
                         "labeled 'impact' that don't have an exact impact_frame "
                         "marked yet, so you can go back through past sessions and "
                         "fill these in without re-finding each one by hand")
    args = ap.parse_args()

    cfg = fights.get_fight(args.fight)
    MANIFEST_PATH   = cfg["manifest_path"]
    CLIPS_DIR       = cfg["clips_dir"]
    FIGHTER_NAMES   = cfg["fighter_names"]
    FIGHTER_SHORT_NAMES = cfg["fighter_short_names"]
    ROUNDS_DIR      = cfg["out_base"]
    VIDEO_FOLDER    = cfg["video_folder"]
    IDENTITY_MARKER = cfg.get("identity_marker")

    if not os.path.exists(MANIFEST_PATH):
        print(f"No manifest found for fight {args.fight!r}: {MANIFEST_PATH}")
        print(f"Run: python dataset/prepare_fight_dataset.py --fight {args.fight}")
        return

    manifest = load_manifest()
    clips    = manifest["clips"]

    # Queue always includes every matching clip (labeled or not), so [B] Back
    # can step into previously-labeled clips to review/correct them. The
    # starting position skips ahead to the first unlabeled one by default,
    # so a normal launch still resumes where you left off.
    queue = []
    for idx, c in enumerate(clips):
        if args.round   is not None and c["round"]      != args.round:   continue
        if args.fighter is not None and c["fighter_id"] != args.fighter: continue
        if args.action  is not None and c["action"]     != args.action:  continue
        if args.needs_impact_frame and not (
                c["label"] == LABEL_IMPACT and c.get("impact_frame") is None):
            continue
        queue.append(idx)

    if not queue:
        if args.needs_impact_frame:
            print("No impact clips are missing a marked frame -- nothing to review.")
        else:
            print("No clips match the filter.")
        return

    if args.jump_to is not None:
        start_pos = max(0, min(len(queue) - 1, args.jump_to - 1))
    elif args.needs_impact_frame:
        # Every clip in this queue already qualifies (label=impact, no marked
        # frame), so there's no "first unlabeled" to skip to -- just start at
        # the top unless --show-labeled asked to land on the end instead.
        start_pos = 0
    else:
        start_pos = 0
        if not args.show_labeled:
            for i, idx in enumerate(queue):
                if clips[idx]["label"] is None:
                    start_pos = i
                    break
            else:
                start_pos = max(0, len(queue) - 1)  # all labeled: land on the last clip

    labeled_total = sum(1 for c in clips if c["label"] is not None)
    print(f"\nAnnotation session: {len(queue)} clips in queue (starting at #{start_pos+1})")
    if args.needs_impact_frame:
        print(f"Review mode: {len(queue)} impact clips missing a marked frame")
    print(f"Already labeled: {labeled_total}/{len(clips)}")
    print("Controls: I=impact  N=not_impact  M=mark frame  B=back  R=replay  S=skip  Q=quit\n")

    # Pre-build the fighter-name swap timeline for every round in the queue,
    # up front, so the first clip loaded from a round doesn't freeze the UI
    # for ~15s mid-session.
    rounds_needed = sorted({clips[idx]["round"] for idx in queue})
    print(f"Building name-overlay timelines for round(s) {rounds_needed} (one-time, ~15s each)...")
    for r in rounds_needed:
        get_round_swap_timeline(r)
    print("Done.\n")

    root = tk.Tk()
    app  = AnnotatorApp(root, queue, manifest, speed=args.speed, start_pos=start_pos)
    root.mainloop()


if __name__ == "__main__":
    main()
