#!/usr/bin/env python3
"""
Video clip annotation tool for impact detection labeling.
Uses tkinter + PIL — works on Windows without a GUI OpenCV build.

Controls (keyboard shortcuts active when window is focused):
    I / 1       -> IMPACT           (punch landed)
    N / 0       -> NOT IMPACT       (missed / blocked)
    Space       -> PAUSE / RESUME
    Left arrow  -> step back 1 frame (while paused)
    Right arrow -> step forward 1 frame (while paused)
    , (comma)   -> step back 5 frames
    . (period)  -> step forward 5 frames
    S           -> SKIP             (leave unlabeled, come back later)
    R           -> REPLAY           current clip from the start
    Q / Esc     -> QUIT             (saves all progress)

Labels are written to manifest.json after every decision.
You can quit and resume at any time.

Usage:
    python tools/annotate_clips.py
    python tools/annotate_clips.py --round 3
    python tools/annotate_clips.py --fighter 0
    python tools/annotate_clips.py --speed 0.5   # slow motion
    python tools/annotate_clips.py --show-labeled # re-review labeled clips
"""

import os
import json
import argparse
import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

MANIFEST_PATH = r"C:\Users\XRIG\Desktop\Impact_Detection_Improve\Impact_Detection\dataset\lillyella_vs_zoe\manifest.json"
CLIPS_DIR     = r"C:\Users\XRIG\Desktop\Impact_Detection_Improve\Impact_Detection\dataset\lillyella_vs_zoe\clips"

LABEL_IMPACT     = "impact"
LABEL_NOT_IMPACT = "not_impact"

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
    manifest["labeled"]   = labeled
    manifest["unlabeled"] = unlabeled
    json.dump(manifest, open(MANIFEST_PATH, "w"), indent=2)


class AnnotatorApp:
    def __init__(self, root, queue, manifest, speed=1.0):
        self.root     = root
        self.queue    = queue        # list of manifest indices to label
        self.manifest = manifest
        self.clips    = manifest["clips"]
        self.speed    = speed

        self.pos      = 0            # current position in queue
        self.result   = None         # set by key handler
        self._playing = False
        self._paused  = False
        self._frames  = []
        self._fi      = 0
        self._after_id = None

        self._build_ui()
        self._bind_keys()
        self._load_clip()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("Impact Annotator")
        self.root.configure(bg=COL_BG)
        self.root.state("zoomed")   # maximized on Windows
        self.root.resizable(True, True)

        # Top info bar
        self.info_var = tk.StringVar(value="Loading...")
        info = tk.Label(self.root, textvariable=self.info_var,
                        bg=COL_BG, fg=COL_ACCENT,
                        font=("Consolas", 11, "bold"), anchor="w", padx=8)
        info.pack(fill="x", pady=(6, 0))

        # Sub-info (action / frame range)
        self.sub_var = tk.StringVar(value="")
        sub = tk.Label(self.root, textvariable=self.sub_var,
                       bg=COL_BG, fg=COL_TEXT,
                       font=("Consolas", 10), anchor="w", padx=8)
        sub.pack(fill="x")

        # Video canvas
        self.canvas = tk.Canvas(self.root, bg="#000000",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self._tk_img = None

        # Progress bar (thin strip)
        self.prog_var = tk.DoubleVar(value=0)
        self.prog_bar = ttk.Progressbar(self.root, variable=self.prog_var,
                                        maximum=100, length=980)
        self.prog_bar.pack(fill="x", padx=8, pady=2)

        # Label buttons
        btn_frame = tk.Frame(self.root, bg=COL_BG)
        btn_frame.pack(fill="x", padx=8, pady=(2, 6))

        self.lbl_status = tk.Label(btn_frame, text="Current: unlabeled",
                                   bg=COL_BG, fg="#888", font=("Consolas", 10))
        self.lbl_status.pack(side="left", padx=6)

        btn_cfg = dict(font=("Consolas", 11, "bold"), width=14,
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

        # Bottom hint
        hint = tk.Label(self.root,
                        text="I=impact  N=not_impact  B=back  Space=pause  Left/Right=step1  ,/.=step5  R=replay  S=skip  Q=quit",
                        bg=COL_BG, fg="#666", font=("Consolas", 9))
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
        if not self._paused:
            # Resume: restart the loop from current position
            if self._after_id is not None:
                self.root.after_cancel(self._after_id)
            self._play_frame()

    def _step_frame(self, delta):
        if not self._frames or not self._paused:
            return
        self._fi = max(0, min(len(self._frames) - 1, self._fi + delta))
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
        self.info_var.set(
            f"[{self.pos+1}/{total}]  Round {entry['round']}  "
            f"Fighter {entry['fighter_id']}  —  "
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

        # Load all frames into memory (clips are short: ~1 second)
        self._frames = []
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
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR -> RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._frames.append(rgb)
            cap.release()

        self._fi = 0
        self._replay()

    def _replay(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self._fi = 0
        self._paused = False
        self._playing = True
        self._play_frame()

    def _render_current_frame(self):
        """Draw self._frames[self._fi] onto canvas (used for both playback and step)."""
        if not self._frames:
            return
        frame = self._frames[self._fi]
        h, w  = frame.shape[:2]

        in_window = self._pad_before <= self._fi <= self._pad_before + self._win_len
        if in_window:
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

            # Frame counter overlay when paused
            if self._paused:
                win_tag = " [IN WINDOW]" if in_window else ""
                label_txt = f"PAUSED  frame {self._fi+1}/{len(self._frames)}{win_tag}"
                x, y = cw // 2, 20
                self.canvas.create_text(x, y, text=label_txt,
                                        fill="#000000", font=("Consolas", 11, "bold"),
                                        anchor="n", tags="overlay")
                self.canvas.create_text(x, y - 1, text=label_txt,
                                        fill="#ffff00", font=("Consolas", 11, "bold"),
                                        anchor="n", tags="overlay")

        pct = 100.0 * self._fi / max(1, len(self._frames) - 1)
        self.prog_var.set(pct)

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
        save_manifest(self.manifest)

        icon = "HIT" if label == LABEL_IMPACT else "---"
        entry = self.clips[idx]
        print(f"  [{self.pos+1}/{len(self.queue)}] {icon}  {entry['clip']}")

        # Flash background
        col = COL_IMPACT if label == LABEL_IMPACT else COL_NOT_IMPACT
        self._flash(col)

        self.pos += 1
        self.root.after(120, self._load_clip)

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
        print(f"\n{'='*50}")
        print(f"  Total labeled : {labeled} / {len(self.clips)}")
        print(f"    IMPACT      : {impact}")
        print(f"    NOT_IMPACT  : {not_imp}")
        print(f"    Unlabeled   : {unlabeled}")
        print(f"\n  Manifest saved: {MANIFEST_PATH}")
        if unlabeled == 0:
            print(f"\n  All clips labeled! Run:")
            print(f"    python tools/train_impact_model.py")
        print(f"{'='*50}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round",        type=int,   default=None)
    ap.add_argument("--fighter",      type=int,   default=None)
    ap.add_argument("--action",       type=str,   default=None)
    ap.add_argument("--speed",        type=float, default=1.0)
    ap.add_argument("--show-labeled", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest()
    clips    = manifest["clips"]

    queue = []
    for idx, c in enumerate(clips):
        if args.round   is not None and c["round"]      != args.round:   continue
        if args.fighter is not None and c["fighter_id"] != args.fighter: continue
        if args.action  is not None and c["action"]     != args.action:  continue
        if not args.show_labeled and c["label"] is not None:             continue
        queue.append(idx)

    if not queue:
        print("No unlabeled clips match the filter.")
        print("Use --show-labeled to re-review already-labeled clips.")
        return

    labeled_total = sum(1 for c in clips if c["label"] is not None)
    print(f"\nAnnotation session: {len(queue)} clips to label")
    print(f"Already labeled: {labeled_total}/{len(clips)}")
    print("Controls: I=impact  N=not_impact  R=replay  S=skip  Q=quit\n")

    root = tk.Tk()
    app  = AnnotatorApp(root, queue, manifest, speed=args.speed)
    root.mainloop()


if __name__ == "__main__":
    main()
