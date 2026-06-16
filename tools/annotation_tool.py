#!/usr/bin/env python3
"""
Boxing Impact Annotation Tool

Run:  python tools/annotation_tool.py
  → file picker opens, choose any fight video
  → navigate frame-by-frame
  → CLICK the spot on the video where the punch lands
  → choose body part + verdict → Enter to save

Keyboard shortcuts:
  ← →            prev / next frame
  Shift+← →      ±10 frames
  Ctrl+← →       ±100 frames
  Space           play / pause
  L / B / M / U  quick verdict (saves immediately after clicking)
  Enter           save current annotation
  D               delete annotation at current frame
  Ctrl+Z          undo last action
  G               jump to frame (dialog)
  Ctrl+S          save JSON

Output:  <video>_gt.json  (same folder as the video)
"""
import os, json, time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import cv2

# ── constants ─────────────────────────────────────────────────────────────────
DISP_W, DISP_H = 960, 540
PANEL_W = 340
MARKER_R = 16                    # impact dot radius (px on display)
BG, BG2, FG, ACC = "#1e1e2e", "#2a2a3e", "#cdd6f4", "#89b4fa"

BODY_PARTS = [
    "Head / Face", "Jaw", "Temple", "Nose / Eye",
    "Body / Torso", "Liver", "Solar Plexus",
    "Shoulder", "Guard (Blocked)", "Back", "Other",
]
VERDICTS = ["LANDED", "BLOCKED", "MISS", "UNCLEAR"]
VRD_CLR  = {"LANDED":"#22c55e","BLOCKED":"#f59e0b","MISS":"#ef4444","UNCLEAR":"#94a3b8"}
QUICK    = {"l":"LANDED","b":"BLOCKED","m":"MISS","u":"UNCLEAR"}


class AnnotationTool:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.configure(bg=BG)
        root.title("Impact Annotation Tool")

        # video
        self.cap        = None
        self.n_frames   = 0
        self.fps        = 25.0
        self.vid_w = self.vid_h = 0
        self.video_path = ""
        self.out_path   = ""
        self.frame_idx  = 0
        self._cache: dict[int, bytes] = {}   # frame_idx → raw JPEG bytes (fast)
        self._cache_max = 120

        # click state
        self.pending_click = None   # (nx, ny) placed but not saved

        # playback
        self.playing    = False
        self._play_job  = None
        self._play_t0   = 0.0       # time of last frame render (for timing compensation)

        # annotation data
        self.annotations: dict[int, dict] = {}
        self._undo_stack: list = []          # list of (frame, prev_ann_or_None)

        # tk vars
        self.sv_body    = tk.StringVar(value=BODY_PARTS[0])
        self.sv_verdict = tk.StringVar(value=VERDICTS[0])
        self.sv_note    = tk.StringVar()
        self.sv_auto    = tk.BooleanVar(value=True)

        self._build_ui()
        root.geometry(f"{DISP_W + PANEL_W + 24}x{DISP_H + 155}")
        root.after(120, self._pick_video)

    # ── file picker ───────────────────────────────────────────────────────────
    def _pick_video(self, reload=False):
        path = filedialog.askopenfilename(
            title="Select fight video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")],
        )
        if not path:
            if not reload:
                self.root.destroy()
            return
        self._load_video(path)

    def _load_video(self, path: str):
        self._stop_play()
        if self.cap:
            self.cap.release()
        self._cache.clear()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open:\n{path}")
            return
        self.video_path = path
        self.n_frames   = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps        = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.vid_w      = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h      = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out_path   = os.path.splitext(path)[0] + "_gt.json"
        self.annotations.clear()
        self._undo_stack.clear()
        self.pending_click = None
        self.slider.config(to=max(0, self.n_frames - 1))
        self.root.title(f"Impact Annotator — {os.path.basename(path)}")
        self.lbl_video.config(text=os.path.basename(path), fg=ACC)
        self._goto(0)

    # ── frame I/O ─────────────────────────────────────────────────────────────
    def _decode(self, idx: int):
        """Return (DISP_W×DISP_H) RGB PIL image, using cache."""
        if idx in self._cache:
            import io
            return Image.open(io.BytesIO(self._cache[idx]))
        if not self.cap:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, bgr = self.cap.read()
        if not ret:
            return None
        # resize with cv2 (fast) then convert
        small = cv2.resize(bgr, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        # cache as JPEG bytes (~30 KB vs ~1.5 MB raw)
        import io
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=85)
        if len(self._cache) >= self._cache_max:
            # evict the frame farthest from current
            victim = max(self._cache, key=lambda k: abs(k - idx))
            del self._cache[victim]
        self._cache[idx] = buf.getvalue()
        return pil

    # ── navigation ────────────────────────────────────────────────────────────
    def _goto(self, idx: int, from_slider=False, fast=False):
        if not self.cap:
            return
        idx = max(0, min(idx, self.n_frames - 1))
        self.frame_idx = idx

        pil = self._decode(idx)
        if pil is None:
            return

        # sync panel ↔ annotation
        if idx in self.annotations:
            a = self.annotations[idx]
            self.pending_click = (a["click_x_norm"], a["click_y_norm"])
            self.sv_body.set(a["body_part"])
            self.sv_verdict.set(a["verdict"])
            self.sv_note.set(a.get("note", ""))
            vrd = a["verdict"]
            self.lbl_status.config(
                text=f"Annotated: {a['body_part']} — {vrd}",
                fg=VRD_CLR.get(vrd, "#94a3b8"))
        else:
            self.pending_click = None
            self.sv_note.set("")
            self.lbl_status.config(
                text="Click on the video to mark impact (saves immediately)",
                fg="#6c7086")

        self._redraw(pil, fast=fast)

        sec = idx / self.fps
        self.lbl_frame.config(
            text=f"Frame {idx:5d} / {self.n_frames-1}    "
                 f"{int(sec//60)}:{sec%60:05.2f}s")
        self.lbl_finfo.config(text=f"Frame: {idx}")
        if not from_slider:
            self.slider.set(idx)

    def _step(self, d: int):
        self._stop_play()
        self._goto(self.frame_idx + d)

    def _jump_dialog(self):
        if not self.cap:
            return
        v = simpledialog.askinteger(
            "Go to frame",
            f"Frame number  (0 – {self.n_frames-1}):",
            parent=self.root, minvalue=0, maxvalue=max(0, self.n_frames-1))
        if v is not None:
            self._goto(v)

    # ── playback ──────────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self.playing:
            self._stop_play()
        else:
            self.playing = True
            self.btn_play.config(text="⏸  Pause", bg="#f59e0b", fg="#000")
            self._play_t0 = time.perf_counter()
            self._play_tick()

    def _play_tick(self):
        if not self.playing:
            return
        nxt = self.frame_idx + 1
        if nxt >= self.n_frames:
            self._stop_play()
            return

        t0 = time.perf_counter()
        self._goto(nxt, fast=True)          # fast=True skips annotation overlays
        elapsed_ms = (time.perf_counter() - t0) * 1000

        target_ms  = 1000.0 / self.fps      # e.g. 40 ms at 25 fps
        delay      = max(1, int(target_ms - elapsed_ms))
        self._play_job = self.root.after(delay, self._play_tick)

    def _stop_play(self):
        self.playing = False
        self.btn_play.config(text="▶  Play", bg="#313244", fg=FG)
        if self._play_job:
            self.root.after_cancel(self._play_job)
            self._play_job = None
        # re-draw current frame with overlays now that we've stopped
        if self.cap:
            pil = self._decode(self.frame_idx)
            if pil:
                self._redraw(pil, fast=False)

    # ── canvas ────────────────────────────────────────────────────────────────
    def _redraw(self, pil: Image.Image, fast=False):
        self.tk_img = ImageTk.PhotoImage(pil)
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        if fast:
            # during playback: just the frame counter, skip all overlays
            c.create_rectangle(0, 0, 220, 24, fill="#000", outline="")
            c.create_text(6, 12, text=f"Frame {self.frame_idx}",
                           fill=FG, anchor="w", font=("Consolas", 10))
            return

        # ── current frame: large crosshair marker ──
        marker = self.pending_click
        if marker:
            nx, ny = marker
            px, py = int(nx * DISP_W), int(ny * DISP_H)

            if self.frame_idx in self.annotations:
                vrd = self.annotations[self.frame_idx]["verdict"]
                clr = VRD_CLR.get(vrd, "#94a3b8")
                bp  = self.annotations[self.frame_idx]["body_part"]
                label = f"{vrd} · {bp}"
            else:
                clr   = "#f38ba8"
                label = "← press Enter to save"

            r = MARKER_R
            # filled circle
            c.create_oval(px-r, py-r, px+r, py+r,
                           fill=clr, outline="#fff", width=2)
            # crosshair arms
            c.create_line(px-r-12, py, px-r-2, py, fill="#fff", width=2)
            c.create_line(px+r+2,  py, px+r+12, py, fill="#fff", width=2)
            c.create_line(px, py-r-12, px, py-r-2,  fill="#fff", width=2)
            c.create_line(px, py+r+2,  px, py+r+12, fill="#fff", width=2)
            # label pill
            lw = len(label) * 7 + 12
            c.create_rectangle(px+r+4, py-10, px+r+4+lw, py+10,
                                fill=clr, outline="")
            c.create_text(px+r+10, py, text=label,
                           fill="#fff", anchor="w", font=("Consolas", 9, "bold"))

        # frame counter overlay
        tag = ""
        if self.frame_idx in self.annotations:
            tag = f"  [{self.annotations[self.frame_idx]['verdict']}]"
        c.create_rectangle(0, 0, 230, 24, fill="#000", outline="")
        c.create_text(6, 12, text=f"Frame {self.frame_idx}{tag}",
                       fill=FG, anchor="w", font=("Consolas", 10))

    def _on_click(self, e):
        """Click on video = immediately save annotation at that point."""
        if not self.cap:
            return
        self._stop_play()
        nx = max(0.0, min(1.0, e.x / DISP_W))
        ny = max(0.0, min(1.0, e.y / DISP_H))
        self.pending_click = (nx, ny)
        self._save_ann()   # save immediately on click

    # ── annotation CRUD ───────────────────────────────────────────────────────
    def _save_ann(self):
        if not self.cap:
            return
        if self.pending_click is None:
            return
        nx, ny = self.pending_click
        prev = self.annotations.get(self.frame_idx)
        self._undo_stack.append((self.frame_idx, prev))
        ann = {
            "frame":        self.frame_idx,
            "time_sec":     round(self.frame_idx / self.fps, 3),
            "click_x_norm": round(nx, 4),
            "click_y_norm": round(ny, 4),
            "pixel_x":      int(nx * self.vid_w),
            "pixel_y":      int(ny * self.vid_h),
            "body_part":    self.sv_body.get(),
            "verdict":      self.sv_verdict.get(),
            "note":         self.sv_note.get().strip(),
        }
        self.annotations[self.frame_idx] = ann
        self._refresh_list()
        vrd = ann["verdict"]
        self.lbl_status.config(
            text=f"Saved: {ann['body_part']} — {vrd}",
            fg=VRD_CLR.get(vrd, "#94a3b8"))
        pil = self._decode(self.frame_idx)
        if pil:
            self._redraw(pil)
        if self.sv_auto.get():
            self._step(1)

    def _del_ann(self):
        if self.frame_idx in self.annotations:
            prev = self.annotations.pop(self.frame_idx)
            self._undo_stack.append((self.frame_idx, prev))
            self.pending_click = None
            self.lbl_status.config(
                text="Click on the video to mark impact (saves immediately)", fg="#6c7086")
            self._refresh_list()
            pil = self._decode(self.frame_idx)
            if pil:
                self._redraw(pil)

    def _undo(self):
        # Priority 1: clear a placed-but-unsaved click on the current frame
        if self.pending_click is not None and self.frame_idx not in self.annotations:
            self.pending_click = None
            self.lbl_status.config(
                text="Click on the video to mark impact (saves immediately)", fg="#6c7086")
            pil = self._decode(self.frame_idx)
            if pil:
                self._redraw(pil)
            return
        # Priority 2: undo the last saved annotation
        if not self._undo_stack:
            return
        frame, prev = self._undo_stack.pop()
        if prev is None:
            self.annotations.pop(frame, None)
        else:
            self.annotations[frame] = prev
        self._refresh_list()
        self._goto(frame)

    def _quick(self, vrd: str):
        self.sv_verdict.set(vrd)
        if self.pending_click is not None or self.frame_idx in self.annotations:
            self._save_ann()

    def _refresh_list(self):
        self.lst.delete(0, tk.END)
        self.lbl_count.config(text=f"{len(self.annotations)} annotated")
        frames = sorted(self.annotations)
        for fr in frames:
            a = self.annotations[fr]
            self.lst.insert(
                tk.END,
                f"  {fr:5d}   {a['verdict']:8s}   {a['body_part'][:15]}")
        for i, fr in enumerate(frames):
            clr = VRD_CLR.get(self.annotations[fr]["verdict"], FG)
            self.lst.itemconfig(i, fg=clr)

    def _on_list_click(self, _):
        sel = self.lst.curselection()
        if sel:
            frames = sorted(self.annotations)
            if sel[0] < len(frames):
                self._goto(frames[sel[0]])

    # ── file I/O ──────────────────────────────────────────────────────────────
    def _save_file(self):
        if not self.annotations:
            messagebox.showinfo("Nothing to save", "No annotations yet.")
            return
        data = {
            "video":            self.video_path,
            "fps":              self.fps,
            "total_frames":     self.n_frames,
            "video_width":      self.vid_w,
            "video_height":     self.vid_h,
            "annotation_count": len(self.annotations),
            "annotations": [self.annotations[k] for k in sorted(self.annotations)],
        }
        with open(self.out_path, "w") as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Saved",
            f"{len(self.annotations)} annotations written\n→ {self.out_path}")

    def _on_close(self):
        if self.annotations:
            ans = messagebox.askyesnocancel("Close", "Save annotations before closing?")
            if ans is None:
                return
            if ans:
                self._save_file()
        if self.cap:
            self.cap.release()
        self.root.destroy()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── top: canvas + right panel ─────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(top, width=DISP_W, height=DISP_H,
                                 bg="#111", cursor="crosshair",
                                 highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=4, pady=4)
        self.canvas.bind("<Button-1>", self._on_click)

        rp = tk.Frame(top, width=PANEL_W, bg=BG2)
        rp.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 4), pady=4)
        rp.pack_propagate(False)
        self._build_panel(rp)

        # ── bottom: frame info + slider + nav ─────────────────────────────────
        bot = tk.Frame(self.root, bg=BG)
        bot.pack(fill=tk.X, padx=4, pady=(0, 4))

        self.lbl_frame = tk.Label(bot, text="No video loaded",
                                   fg=FG, bg=BG, font=("Consolas", 11, "bold"))
        self.lbl_frame.pack()

        self.slider = tk.Scale(
            bot, from_=0, to=1000, orient=tk.HORIZONTAL,
            length=DISP_W + PANEL_W + 8,
            command=lambda v: (self._stop_play(), self._goto(int(float(v)), from_slider=True)),
            bg="#313244", fg=FG, troughcolor="#45475a",
            highlightthickness=0, bd=0, sliderrelief=tk.FLAT,
        )
        self.slider.pack(fill=tk.X)

        nav = tk.Frame(bot, bg=BG)
        nav.pack(pady=3)
        bc = dict(relief=tk.FLAT, font=("Consolas", 10), padx=7, pady=5, cursor="hand2")

        for lbl, d in [("◀◀ -100", -100), ("◀ -10", -10), ("◀", -1)]:
            tk.Button(nav, text=lbl, command=lambda x=d: self._step(x),
                       bg="#313244", fg=FG, **bc).pack(side=tk.LEFT, padx=2)

        self.btn_play = tk.Button(nav, text="▶  Play",
                                   command=self._toggle_play,
                                   bg="#313244", fg=FG,
                                   font=("Consolas", 11, "bold"),
                                   padx=14, pady=5, relief=tk.FLAT,
                                   cursor="hand2")
        self.btn_play.pack(side=tk.LEFT, padx=4)

        for lbl, d in [("▶", 1), ("+10 ▶", 10), ("+100 ▶▶", 100)]:
            tk.Button(nav, text=lbl, command=lambda x=d: self._step(x),
                       bg="#313244", fg=FG, **bc).pack(side=tk.LEFT, padx=2)

        tk.Button(nav, text="Go to frame…", command=self._jump_dialog,
                   bg="#313244", fg=FG, **bc).pack(side=tk.LEFT, padx=8)
        tk.Button(nav, text="Open video…",
                   command=lambda: self._pick_video(reload=True),
                   bg="#313244", fg=ACC, **bc).pack(side=tk.LEFT, padx=2)

        # keybindings
        r = self.root
        for seq, fn in [
            ("<Left>",          lambda e: self._step(-1)),
            ("<Right>",         lambda e: self._step(1)),
            ("<Shift-Left>",    lambda e: self._step(-10)),
            ("<Shift-Right>",   lambda e: self._step(10)),
            ("<Control-Left>",  lambda e: self._step(-100)),
            ("<Control-Right>", lambda e: self._step(100)),
            ("<space>",         lambda e: self._toggle_play()),
            ("<Return>",        lambda e: self._save_ann()),
            ("<d>",             lambda e: self._del_ann()),
            ("<D>",             lambda e: self._del_ann()),
            ("<g>",             lambda e: self._jump_dialog()),
            ("<G>",             lambda e: self._jump_dialog()),
            ("<Control-s>",     lambda e: self._save_file()),
            ("<Control-z>",     lambda e: self._undo()),
            ("<Escape>",        lambda e: (
                setattr(self, 'pending_click', None) or
                self.lbl_status.config(
                    text="Click on the video to mark impact (saves immediately)", fg="#6c7086") or
                self._decode(self.frame_idx) and
                self._redraw(self._decode(self.frame_idx))
            )),
        ]:
            r.bind(seq, fn)
        for k, v in QUICK.items():
            r.bind(f"<{k}>",        lambda e, x=v: self._quick(x))
            r.bind(f"<{k.upper()}>", lambda e, x=v: self._quick(x))

    def _build_panel(self, p):
        head = dict(fg=ACC, bg=BG2, font=("Consolas", 10, "bold"))
        tiny = dict(fg=FG,  bg=BG2, font=("Consolas", 9))
        S    = lambda: ttk.Separator(p).pack(fill=tk.X, padx=8, pady=5)

        # video name
        tk.Label(p, text="VIDEO:", anchor="w", **head).pack(fill=tk.X, padx=8, pady=(10,0))
        self.lbl_video = tk.Label(p, text="(none)", anchor="w", fg="#6c7086",
                                   bg=BG2, font=("Consolas", 9),
                                   wraplength=PANEL_W - 20, justify="left")
        self.lbl_video.pack(fill=tk.X, padx=8)
        S()

        # frame + status
        self.lbl_finfo  = tk.Label(p, text="Frame: —", anchor="w", **head)
        self.lbl_finfo.pack(fill=tk.X, padx=8)
        self.lbl_status = tk.Label(p, text="Click on the video to mark impact (saves immediately)",
                                    anchor="w", fg="#6c7086", bg=BG2,
                                    font=("Consolas", 9), wraplength=PANEL_W - 20,
                                    justify="left")
        self.lbl_status.pack(fill=tk.X, padx=8)
        S()

        # body part
        tk.Label(p, text="BODY PART HIT:", anchor="w", **head).pack(fill=tk.X, padx=8)
        bpf = tk.Frame(p, bg=BG2)
        bpf.pack(fill=tk.X, padx=8, pady=4)
        for i, bp in enumerate(BODY_PARTS):
            tk.Radiobutton(
                bpf, text=bp, variable=self.sv_body, value=bp,
                bg=BG2, fg=FG, selectcolor="#313244",
                activebackground=BG2, activeforeground=ACC,
                font=("Consolas", 9), anchor="w",
            ).grid(row=i // 2, column=i % 2, sticky="w", padx=2, pady=1)
        S()

        # verdict
        tk.Label(p, text="VERDICT   (keys: L  B  M  U)", anchor="w", **head
                 ).pack(fill=tk.X, padx=8)
        vf = tk.Frame(p, bg=BG2)
        vf.pack(fill=tk.X, padx=8, pady=4)
        for v in VERDICTS:
            tk.Radiobutton(
                vf, text=v, variable=self.sv_verdict, value=v,
                bg=BG2, fg=VRD_CLR[v], selectcolor="#313244",
                activebackground=BG2, font=("Consolas", 10, "bold"),
            ).pack(side=tk.LEFT, padx=3)
        S()

        # note + auto-advance
        tk.Label(p, text="NOTE (optional):", anchor="w", **head).pack(fill=tk.X, padx=8)
        tk.Entry(p, textvariable=self.sv_note, bg="#313244", fg=FG,
                  insertbackground=FG, font=("Consolas", 9),
                  relief=tk.FLAT).pack(fill=tk.X, padx=8, pady=4)
        tk.Checkbutton(p, text="Auto-advance after save",
                        variable=self.sv_auto, bg=BG2, fg=FG,
                        selectcolor="#313244", activebackground=BG2,
                        font=("Consolas", 9)).pack(anchor="w", padx=8)
        S()

        # save / delete / undo
        tk.Button(p, text="  RE-SAVE CURRENT FRAME  (Enter)",
                   command=self._save_ann,
                   bg="#22c55e", fg="#fff",
                   font=("Consolas", 10, "bold"),
                   relief=tk.FLAT, pady=8, cursor="hand2",
                   ).pack(fill=tk.X, padx=8, pady=2)

        row2 = tk.Frame(p, bg=BG2)
        row2.pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Button(row2, text="DELETE  (D)",
                   command=self._del_ann, bg="#ef4444", fg="#fff",
                   relief=tk.FLAT, font=("Consolas", 9),
                   pady=4, cursor="hand2").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        tk.Button(row2, text="UNDO  (Ctrl+Z)",
                   command=self._undo, bg="#45475a", fg=FG,
                   relief=tk.FLAT, font=("Consolas", 9),
                   pady=4, cursor="hand2").pack(side=tk.LEFT, fill=tk.X, expand=True)
        S()

        # annotations list
        tk.Label(p, text="ANNOTATIONS:", anchor="w", **head).pack(fill=tk.X, padx=8)
        self.lbl_count = tk.Label(p, text="0 annotated", anchor="w", **tiny)
        self.lbl_count.pack(fill=tk.X, padx=8)
        lf = tk.Frame(p, bg=BG2)
        lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        sb = tk.Scrollbar(lf)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.lst = tk.Listbox(lf, yscrollcommand=sb.set,
                               bg="#313244", fg=FG,
                               selectbackground=ACC, selectforeground="#1e1e2e",
                               font=("Consolas", 9), relief=tk.FLAT, activestyle="none")
        self.lst.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.lst.yview)
        self.lst.bind("<<ListboxSelect>>", self._on_list_click)

        tk.Button(p, text="  Save JSON  (Ctrl+S)",
                   command=self._save_file, bg="#1e66f5", fg="#fff",
                   relief=tk.FLAT, font=("Consolas", 9),
                   pady=6, cursor="hand2").pack(fill=tk.X, padx=8, pady=6)


def main():
    root = tk.Tk()
    root.geometry(f"{DISP_W + PANEL_W + 24}x{DISP_H + 155}")
    app = AnnotationTool(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
