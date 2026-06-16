#!/usr/bin/env python3
"""
Run the trained r3d_18 impact detector on a new video.

Strategy
--------
  1. Train r3d_18 on ALL GT clips.
  2. Load ASFormer action windows from *_full_analysis.json.
  3. Load sam3d.json to get per-frame bounding boxes for each fighter.
     Use the RECEIVER's bbox to crop around the actual target (head/body)
     instead of a fixed frame centre.
  4. For each action window slide at --stride frames within [start-pad, end+pad].
  5. Deduplicate candidates within ±T_HALF of each other.
  6. Extract 21-frame clips, run model, NMS per window.
  7. Render output video with overlays.

Usage
-----
  python tools/detect_new_video.py \\
      --gt-data        outputs/gt_dataset/fight1_gt.npz \\
      --video          @/tmp/fight2_video.txt \\
      --full-analysis  @/tmp/fight2_analysis.txt \\
      --sam3d          @/tmp/fight2_sam3d.txt \\
      [--v9-json       @/tmp/fight2_v9json.txt] \\
      --stride 3  --pad 5  --threshold 0.5  --epochs 40 \\
      --out-dir outputs/fight2_detections
"""

import argparse, json, os, sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

DEV      = "cuda" if torch.cuda.is_available() else "cpu"
SIZE     = 112
T_HALF   = 10
T_FRAMES = 16
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
DISP_W, DISP_H = 1280, 720


def win_to_wsl(path):
    if sys.platform.startswith("linux") and len(path) >= 2 and path[1] == ":":
        return f"/mnt/{path[0].lower()}{path[2:].replace(chr(92), '/')}"
    return path


def resolve_arg(v):
    if v and v.startswith("@"):
        with open(v[1:]) as f:
            v = f.read().strip()
    return win_to_wsl(v)


# ── sam3d bbox lookup ──────────────────────────────────────────────────────────

def build_bbox_lookup(sam3d_path):
    """
    Returns {fighter_id_str: {frame_int: [x1,y1,x2,y2]}}.
    Loads only bbox + frame from the 224 MB sam3d JSON.
    """
    print("Loading sam3d keypoints (building bbox lookup)...", flush=True)
    with open(sam3d_path) as f:
        sam = json.load(f)
    lookup = {}
    for fid, frames in sam.items():
        lookup[fid] = {}
        for entry in frames:
            fr  = entry.get("frame")
            box = entry.get("bbox")
            if fr is not None and box is not None:
                lookup[fid][fr] = box
    sizes = {fid: len(v) for fid, v in lookup.items()}
    print(f"  bbox lookup built: {sizes}", flush=True)
    return lookup


def get_crop_centre(bbox_lookup, fighter_id_str, frame_idx, target, W, H):
    """
    Return (cx, cy) in pixel coords for cropping around the receiver's target zone.
    Falls back to frame centre if bbox not available.
    """
    fid   = str(fighter_id_str)
    table = bbox_lookup.get(fid, {})
    box   = table.get(frame_idx)

    # try ±2 frame neighbour if exact frame missing
    if box is None:
        for delta in range(1, 6):
            box = table.get(frame_idx - delta) or table.get(frame_idx + delta)
            if box: break

    if box is None:
        return W // 2, H // 2   # fallback: frame centre

    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    # head target → top 25% of bbox; body target → middle 50%
    if isinstance(target, str) and target.lower() in ("head", "face"):
        cy = y1 + (y2 - y1) * 0.2
    else:
        cy = y1 + (y2 - y1) * 0.5

    return int(np.clip(cx, 0, W - 1)), int(np.clip(cy, 0, H - 1))


# ── model ──────────────────────────────────────────────────────────────────────

class ClipDS(Dataset):
    def __init__(self, clips, labels, train):
        self.c, self.y, self.train = clips, labels, train

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        clip   = self.c[i]
        jitter = clip.shape[0] - T_FRAMES
        t0     = np.random.randint(0, jitter + 1) if self.train else jitter // 2
        clip   = clip[t0 : t0 + T_FRAMES]
        x = torch.from_numpy(clip).float() / 255.0
        x = x.permute(3, 0, 1, 2)
        if self.train:
            i0 = np.random.randint(0, 128 - SIZE + 1)
            j0 = np.random.randint(0, 128 - SIZE + 1)
            x  = x[:, :, i0:i0+SIZE, j0:j0+SIZE]
            if np.random.rand() < 0.5:
                x = torch.flip(x, [3])
            x = (x * np.random.uniform(0.8, 1.2)).clamp(0, 1)
        else:
            off = (128 - SIZE) // 2
            x   = x[:, :, off:off+SIZE, off:off+SIZE]
        return (x - MEAN) / STD, self.y[i]


def make_model():
    m = torchvision.models.video.r3d_18(weights="KINETICS400_V1")
    for p in m.parameters():
        p.requires_grad = False
    for p in m.layer4.parameters():
        p.requires_grad = True
    m.fc = nn.Linear(512, 1)
    return m.to(DEV)


def train_full(clips, labels, epochs):
    ds    = ClipDS(clips, labels, train=True)
    dl    = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    model = make_model()
    opt   = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if n.startswith("fc")],           "lr": 1e-3},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("fc")],                           "lr": 1e-4},
    ], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(DEV), y.float().to(DEV)
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{epochs}  loss={loss.item():.4f}", flush=True)
    return model


def run_inference(model, clips_np):
    model.eval()
    ds = ClipDS(clips_np, np.zeros(len(clips_np), dtype=np.float32), train=False)
    dl = DataLoader(ds, batch_size=16, num_workers=2)
    scores = []
    with torch.no_grad():
        for x, _ in dl:
            scores += torch.sigmoid(model(x.to(DEV)).squeeze(1)).cpu().tolist()
    return np.array(scores)


# ── clip extraction ────────────────────────────────────────────────────────────

def extract_clip(cap, centre_frame, total_frames, cx, cy, crop_px=320):
    """
    Extract 21-frame clip cropped around (cx, cy) in pixel space.
    """
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    half = crop_px // 2
    x1 = max(0, cx - half);  x2 = min(W, cx + half)
    y1 = max(0, cy - half);  y2 = min(H, cy + half)
    # ensure square crop
    if x2 - x1 < crop_px: x2 = min(W, x1 + crop_px)
    if y2 - y1 < crop_px: y2 = min(H, y1 + crop_px)

    start = max(0, centre_frame - T_HALF)
    end   = min(total_frames - 1, centre_frame + T_HALF)
    idxs  = list(range(start, end + 1))
    while len(idxs) < 21:
        idxs.append(idxs[-1])

    cap.set(cv2.CAP_PROP_POS_FRAMES, idxs[0])
    frames, prev = [], idxs[0]
    for fi in idxs:
        if fi != prev:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            bgr = np.zeros((H, W, 3), dtype=np.uint8)
        prev = fi + 1
        patch = bgr[y1:y2, x1:x2]
        patch = cv2.resize(patch, (128, 128))
        frames.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)   # (21, 128, 128, 3)


# ── candidate building ─────────────────────────────────────────────────────────

def build_candidates(actions, total_frames, video_fps, json_fps, stride, pad):
    ratio = video_fps / json_fps if json_fps > 0 else 1.0
    cands = []
    for ai, a in enumerate(actions):
        ws = max(T_HALF, round(a["window_start"] * ratio) - pad)
        we = min(total_frames - 1 - T_HALF, round(a["window_end"] * ratio) + pad)
        ts = a.get("timestamp_seconds", a.get("frame", 0) / json_fps)
        # receiver = the other fighter
        ft = a.get("fighter_type", "fighter_0")
        receiver_id = "1" if ft.endswith("0") else "0"
        for fr in range(ws, we + 1, stride):
            cands.append({
                "frame":          fr,
                "action_idx":     ai,
                "action":         a.get("action", ""),
                "target":         a.get("target", "Head"),
                "fighter_type":   ft,
                "receiver_id":    receiver_id,
                "window_start":   ws,
                "window_end":     we,
                "timestamp":      ts,
                "asf_confidence": a.get("confidence", 0.0),
            })
    return cands


def deduplicate(cands, radius):
    kept, kept_frames = [], []
    for c in sorted(cands, key=lambda x: (x["action_idx"], x["frame"])):
        if any(abs(c["frame"] - kf) <= radius for kf in kept_frames):
            continue
        kept.append(c)
        kept_frames.append(c["frame"])
    return kept


def nms_per_window(scored_cands):
    from collections import defaultdict
    by_window = defaultdict(list)
    for c in scored_cands:
        by_window[c["action_idx"]].append(c)
    return [max(g, key=lambda x: x["model_score"])
            for _, g in sorted(by_window.items())]


# ── rendering ──────────────────────────────────────────────────────────────────

def put_text_bg(img, text, pos, font_scale=0.7, fg=(255, 255, 255), bg=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    tw, th = cv2.getTextSize(text, font, font_scale, 2)[0]
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 4), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, fg, 2, cv2.LINE_AA)


def draw_score_bar(img, score, x, y, w=160, h=16):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    fill = int(w * score)
    col  = (0, 220, 80) if score >= 0.5 else (60, 120, 200)
    cv2.rectangle(img, (x, y), (x + fill, y + h), col, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 180), 1)


def render_results(video_path, window_results, total_frames, fps,
                   out_path, threshold, v9_hit_frames):
    frame_det = {}
    for det in window_results:
        ctr = det["frame"]
        for fi in range(max(0, ctr - T_HALF), min(total_frames, ctr + T_HALF + 1)):
            if fi not in frame_det or det["model_score"] > frame_det[fi]["model_score"]:
                frame_det[fi] = det

    cap    = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(out_path, fourcc, fps, (DISP_W, DISP_H))

    n_hits  = sum(1 for d in window_results if d["model_score"] >= threshold)
    n_total = len(window_results)

    for fi in range(total_frames):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bgr = cv2.resize(bgr, (DISP_W, DISP_H))

        det = frame_det.get(fi)
        if det is not None:
            score  = det["model_score"]
            age    = abs(fi - det["frame"])
            alpha  = max(0.0, 1.0 - age / T_HALF)
            landed = score >= threshold

            col   = tuple(int(c * alpha) for c in ((0, 220, 80) if landed else (80, 80, 80)))
            thick = max(1, int(8 * alpha))
            cv2.rectangle(bgr, (0, 0), (DISP_W - 1, DISP_H - 1), col, thick)

            if age <= 2:
                lbl    = "LANDED" if landed else "MISSED"
                lc     = (0, 220, 80) if landed else (160, 160, 160)
                put_text_bg(bgr, f"MODEL: {lbl}", (8, 32), font_scale=1.0, fg=lc)
                put_text_bg(bgr, det.get("action", ""), (8, 62), font_scale=0.7)
                tgt = det.get("target", "")
                put_text_bg(bgr, f"target: {tgt}", (8, 88), font_scale=0.65)
                put_text_bg(bgr, det.get("fighter_type", ""), (8, 112), font_scale=0.6)
                draw_score_bar(bgr, score, 8, DISP_H - 44)
                put_text_bg(bgr, f"score {score:.2f}", (175, DISP_H - 28), font_scale=0.65)
                is_v9 = any(abs(det["frame"] - vf) <= T_HALF for vf in v9_hit_frames)
                v9c   = (0, 200, 60) if is_v9 else (100, 100, 100)
                put_text_bg(bgr, "v9:HIT" if is_v9 else "v9:miss",
                            (DISP_W - 130, 32), font_scale=0.75, fg=v9c)

        put_text_bg(bgr, f"hits {n_hits}/{n_total}", (DISP_W - 125, DISP_H - 10), font_scale=0.5)
        put_text_bg(bgr, f"fr {fi}", (DISP_W - 210, DISP_H - 10), font_scale=0.5)
        out.write(bgr)
        if fi % 500 == 0:
            print(f"  rendered {fi}/{total_frames}", flush=True)

    cap.release()
    out.release()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-data",       required=True)
    ap.add_argument("--video",         required=True)
    ap.add_argument("--full-analysis", required=True)
    ap.add_argument("--sam3d",         default=None,
                    help="*_sam3d.json for receiver-bbox cropping")
    ap.add_argument("--v9-json",       default=None)
    ap.add_argument("--stride",    type=int,   default=3)
    ap.add_argument("--pad",       type=int,   default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--epochs",    type=int,   default=40)
    ap.add_argument("--out-dir",   default="outputs/fight2_detections")
    args = ap.parse_args()

    args.video         = resolve_arg(args.video)
    args.full_analysis = resolve_arg(args.full_analysis)
    if args.sam3d:   args.sam3d   = resolve_arg(args.sam3d)
    if args.v9_json: args.v9_json = resolve_arg(args.v9_json)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. train ───────────────────────────────────────────────────────────────
    d      = np.load(args.gt_data, allow_pickle=True)
    clips  = d["clips"]
    labels = d["labels"]
    print(f"Training on {len(labels)} GT clips  pos={labels.sum()}  neg={(labels==0).sum()}")
    print(f"Device: {DEV}  epochs: {args.epochs}")
    model = train_full(clips, labels, args.epochs)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "r3d18_impact_model.pt"))
    print("Model saved.")

    # ── 2. load supporting data ────────────────────────────────────────────────
    with open(args.full_analysis) as f:
        fa = json.load(f)
    actions  = fa["actions"]
    json_fps = float(fa.get("processing_stats", {}).get("fps", 30.0))

    cap          = cv2.VideoCapture(args.video)
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nVideo: {total_frames} frames @ {video_fps:.2f} fps  ({VW}x{VH})")
    print(f"ASFormer actions: {len(actions)}  json_fps={json_fps}")

    # sam3d bbox lookup
    bbox_lookup = {}
    if args.sam3d:
        bbox_lookup = build_bbox_lookup(args.sam3d)

    # v9 hit frames for comparison tags
    v9_hit_frames = set()
    if args.v9_json:
        with open(args.v9_json) as f:
            v9 = json.load(f)
        for imp in v9.get("impacts", []):
            v9_hit_frames.add(round(imp["timestamp_seconds"] * video_fps))
        print(f"v9 hits: {len(v9_hit_frames)}")

    # ── 3. build candidates ────────────────────────────────────────────────────
    cands = build_candidates(actions, total_frames, video_fps, json_fps,
                             stride=args.stride, pad=args.pad)
    print(f"Raw candidates: {len(cands)}")
    cands = deduplicate(cands, radius=T_HALF)
    print(f"After dedup: {len(cands)}")

    # ── 4. extract clips using receiver bbox crops ─────────────────────────────
    print("Extracting clips (receiver-bbox crop)..." if bbox_lookup else "Extracting clips (centre crop)...")
    clip_arr = []
    for c in cands:
        if bbox_lookup:
            cx, cy = get_crop_centre(bbox_lookup, c["receiver_id"],
                                     c["frame"], c["target"], VW, VH)
        else:
            cx, cy = VW // 2, VH // 2
        clip_arr.append(extract_clip(cap, c["frame"], total_frames, cx, cy))
    cap.release()
    clip_arr = np.stack(clip_arr, axis=0)
    print(f"Clips: {clip_arr.shape}")

    # ── 5. inference + NMS ────────────────────────────────────────────────────
    print("Running inference...")
    scores = run_inference(model, clip_arr)
    for i, c in enumerate(cands):
        c["model_score"] = float(scores[i])

    window_results = nms_per_window(cands)
    hits   = [r for r in window_results if r["model_score"] >= args.threshold]
    misses = [r for r in window_results if r["model_score"] <  args.threshold]

    # ── 6. print results ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESULTS  —  {len(hits)} LANDED  /  {len(window_results)} actions  "
          f"(threshold {args.threshold})")
    print(f"{'='*70}")
    print(f"\n{'LANDED':─<70}")
    for r in sorted(hits, key=lambda x: -x["model_score"]):
        is_v9 = any(abs(r["frame"] - vf) <= T_HALF for vf in v9_hit_frames)
        print(f"  t={r['timestamp']:6.1f}s  fr={r['frame']:5d}  "
              f"score={r['model_score']:.3f}  "
              f"{'v9✓' if is_v9 else 'v9✗'}  "
              f"{r['action']:18s}  {r['target']:6s}  {r['fighter_type']}")

    print(f"\n{'MISSED':─<70}")
    for r in sorted(misses, key=lambda x: -x["model_score"]):
        is_v9 = any(abs(r["frame"] - vf) <= T_HALF for vf in v9_hit_frames)
        print(f"  t={r['timestamp']:6.1f}s  fr={r['frame']:5d}  "
              f"score={r['model_score']:.3f}  "
              f"{'v9✓' if is_v9 else 'v9✗'}  "
              f"{r['action']:18s}  {r['target']:6s}")

    # agreement stats
    v9_cnt    = len(v9_hit_frames)
    model_cnt = len(hits)
    both = sum(1 for r in hits
               if any(abs(r["frame"] - vf) <= T_HALF for vf in v9_hit_frames))
    model_only = model_cnt - both
    v9_only    = v9_cnt - both
    print(f"\n{'─'*70}")
    print(f"  v9 hits:          {v9_cnt:3d}")
    print(f"  model hits:       {model_cnt:3d}")
    print(f"  both agree:       {both:3d}")
    print(f"  model-only hits:  {model_only:3d}  ← ASFormer found action, model says LANDED")
    print(f"  v9-only hits:     {v9_only:3d}  ← model disagrees with v9")
    print(f"{'─'*70}")

    # save JSON
    det_path = os.path.join(args.out_dir, "detections.json")
    with open(det_path, "w") as f:
        json.dump({
            "video": args.video, "video_fps": video_fps,
            "total_frames": total_frames, "threshold": args.threshold,
            "n_actions": len(actions), "n_candidates": len(cands),
            "n_landed": len(hits), "n_missed": len(misses),
            "window_results": window_results,
        }, f, indent=2)
    print(f"\nSaved → {det_path}")

    # ── 7. render ─────────────────────────────────────────────────────────────
    print("Rendering video...")
    out_vid = os.path.join(args.out_dir, "detections_video.mp4")
    render_results(args.video, window_results, total_frames, video_fps,
                   out_vid, args.threshold, v9_hit_frames)
    print(f"Video → {out_vid}\nDone.")


if __name__ == "__main__":
    main()
