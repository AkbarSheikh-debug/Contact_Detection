#!/usr/bin/env python3
"""
Sound-Only Boxing Impact Detector
===================================
Detects punch impacts using only the audio track — no keypoints,
no SAM, no ground truth.

How it works
------------
A boxing punch produces two overlapping audio events:
  1. Low-frequency "thud"  (60–400 Hz)  — body/head absorbing the force
  2. High-frequency "slap" (2000–8000 Hz) — leather glove on skin

Both must spike together within a short window for a real impact.
Crowd noise and commentary are broadband/steady-state; they don't
produce the simultaneous low+high transient a punch does.

Pipeline
--------
  1. Load mono 22050 Hz WAV (extracted from video)
  2. Short-time Fourier Transform (512-sample window, 128-sample hop)
  3. Compute spectral flux in 3 frequency bands per frame:
       lo  = 60–400 Hz   (thud)
       mid = 400–2000 Hz (general impact noise)
       hi  = 2000–8000 Hz (slap)
  4. Normalise each band by a 1-second running median (handles
     crowd noise level changes round to round)
  5. Combined onset = 0.4*lo + 0.3*mid + 0.3*hi
  6. Peak-pick with scipy.signal.find_peaks (min height + min distance)
  7. Map audio frame peaks → video frame numbers
  8. Render annotated video with:
       - waveform bar at the bottom of each frame
       - onset envelope overlay (green line)
       - flash + ring + score label at each detected impact
       - running HUD (impact count, timestamp, band scores)
  9. Mux original audio back in with ffmpeg

Usage
-----
    python sound_detector.py
    python sound_detector.py --threshold 0.55 --cooldown 20
    python sound_detector.py --video-in /path/to/video.mp4
"""

import os
import argparse
import json
import subprocess
import wave

import cv2
import numpy as np
from scipy.signal import find_peaks

# ── Paths ──────────────────────────────────────────────────────────────────────
VIDEO_IN  = r"/home/jake/Downloads/sam3d_with_world_coords/3.mp4"
WAV_PATH  = r"/home/jake/Downloads/sam3d_with_world_coords/3.wav"
OUT_DIR   = r"/home/jake/Desktop/HITAI/Contact_Detection/outputs"
OUT_NAME  = "sound.mp4"

FPS = 24.995
SR  = 22050   # sample rate of extracted wav

# ── STFT parameters ────────────────────────────────────────────────────────────
WIN   = 512    # ~23 ms window
HOP   = 128    # ~5.8 ms hop  →  SR/HOP ≈ 172 audio frames per second

# ── Frequency band edges (bin indices at SR=22050, WIN=512) ────────────────────
#   bin k  →  frequency = k * SR / WIN
def hz2bin(hz): return int(round(hz * WIN / SR))

LO_LO,  LO_HI  = hz2bin(60),   hz2bin(400)    # thud
MID_LO, MID_HI = hz2bin(400),  hz2bin(2000)   # general impact
HI_LO,  HI_HI  = hz2bin(2000), hz2bin(8000)   # leather slap

# ── Detection parameters ───────────────────────────────────────────────────────
DEFAULT_THR      = 1.8    # ratio above local noise floor (not a 0-1 scale)
DEFAULT_COOLDOWN = 10     # min frames between detections at video FPS
NORM_WIN_SEC     = 3.0    # baseline window for noise normalisation (seconds)

# ── Colours (BGR) ──────────────────────────────────────────────────────────────
C_FLASH  = (0, 200, 255)   # yellow-white flash
C_RING   = (0, 200, 255)
C_LABEL  = (255, 255, 255)
C_ENV    = (0, 255, 100)   # onset envelope line
C_BAR_LO = (255, 120, 0)   # lo band bar
C_BAR_HI = (50, 200, 255)  # hi band bar
C_HUD    = (15, 15, 20)
FLASH_DUR = 10   # frames


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading & onset detection
# ─────────────────────────────────────────────────────────────────────────────

def ensure_wav(video_path, wav_path):
    if not os.path.exists(wav_path):
        print(f"[sound] extracting audio → {wav_path}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(SR),
             wav_path, "-loglevel", "error"], check=True)
    return wav_path


def load_wav(path):
    with wave.open(path, "rb") as wf:
        sr   = wf.getframerate()
        n    = wf.getnframes()
        raw  = wf.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def compute_onset(x, sr, win=WIN, hop=HOP, norm_win_sec=NORM_WIN_SEC):
    """
    Returns
    -------
    times  : (N,) seconds of each audio frame
    onset  : (N,) combined onset strength as RATIO above local noise floor.
             1.0 = equal to local baseline; 3.0 = 3× the local baseline.
             Threshold at ~1.8 catches transients without swamping in crowd.
    lo_env : (N,) low-band  (thud) ratio
    hi_env : (N,) high-band (slap) ratio

    Key fixes vs the previous version
    ----------------------------------
    - Normalise each band by its local 75th-percentile baseline computed with
      scipy.ndimage.percentile_filter (fast C loop, 3-second window).
      This keeps values as multiples of the local noise floor — a punch that is
      3× the local baseline scores 3.0 regardless of overall crowd volume.
    - No global max division (which was crushing all but one spike to near-zero).
    - RMS energy spike added as a fourth signal.
    - Onset = weighted sum of the four ratio signals; a threshold of ~1.8 means
      the frame is 80% above the local noise baseline.
    """
    from scipy.ndimage import percentile_filter

    hann     = np.hanning(win).astype(np.float32)
    n_frames = 1 + (len(x) - win) // hop
    mags     = np.empty((n_frames, win // 2 + 1), np.float32)
    rms_arr  = np.empty(n_frames, np.float32)

    for i in range(n_frames):
        seg       = x[i * hop: i * hop + win] * hann
        mags[i]   = np.abs(np.fft.rfft(seg))
        rms_arr[i]= float(np.sqrt(np.mean(seg ** 2)))

    # spectral flux per band (positive onset only)
    flux    = np.maximum(0.0, np.diff(mags, axis=0))
    flux    = np.vstack([flux[:1], flux])

    lo_flux  = flux[:, LO_LO:LO_HI].sum(axis=1).astype(np.float32)
    mid_flux = flux[:, MID_LO:MID_HI].sum(axis=1).astype(np.float32)
    hi_flux  = flux[:, HI_LO:HI_HI].sum(axis=1).astype(np.float32)

    # baseline: 75th-percentile filter over NORM_WIN_SEC seconds
    # (faster and more robust than median for suppressing steady crowd noise)
    bsz = max(5, int(norm_win_sec * sr / hop))
    lo_base  = percentile_filter(lo_flux,  75, size=bsz) + 1e-6
    mid_base = percentile_filter(mid_flux, 75, size=bsz) + 1e-6
    hi_base  = percentile_filter(hi_flux,  75, size=bsz) + 1e-6
    rms_base = percentile_filter(rms_arr,  75, size=bsz) + 1e-6

    # ratio above local noise floor
    lo_n  = lo_flux  / lo_base
    mid_n = mid_flux / mid_base
    hi_n  = hi_flux  / hi_base
    rms_n = rms_arr  / rms_base

    # combined: weighted sum — hi-freq (slap) and rms weighted highest
    combined = 0.30 * lo_n + 0.20 * mid_n + 0.35 * hi_n + 0.15 * rms_n

    # light smoothing (3-frame triangular) to avoid single-frame spikes
    kernel   = np.array([0.25, 0.50, 0.25])
    combined = np.convolve(combined, kernel, mode="same")
    lo_n     = np.convolve(lo_n,     kernel, mode="same")
    hi_n     = np.convolve(hi_n,     kernel, mode="same")

    times = (np.arange(n_frames) * hop + win / 2) / sr
    return times, combined, lo_n, hi_n


def detect_impacts(times, onset, threshold, cooldown_frames):
    """Peak-pick onset envelope → list of (time_sec, score) pairs."""
    audio_fps  = len(times) / times[-1]
    cooldown_sec = cooldown_frames / FPS
    min_dist   = max(1, int(cooldown_sec * audio_fps))
    peaks, _   = find_peaks(
        onset,
        height=threshold,
        distance=min_dist,
        prominence=threshold * 0.25,
    )
    impacts = [(float(times[p]), float(onset[p])) for p in peaks]
    return impacts


# ─────────────────────────────────────────────────────────────────────────────
# Video renderer
# ─────────────────────────────────────────────────────────────────────────────

def render(video_path, impacts, onset_data, threshold, out_path):
    """
    Draw for every frame:
      - a thin onset-envelope bar at the bottom
      - band energy bars (lo = orange, hi = cyan) below the envelope
      - flash + expanding ring at each impact frame
      - floating label with score and timestamp
    """
    times, onset, lo_env, hi_env = onset_data
    audio_fps = len(times) / times[-1]   # audio frames per second

    def audio_frame(vid_frame):
        t = vid_frame / FPS
        return int(np.clip(t * audio_fps, 0, len(onset) - 1))

    # pre-build impact lookup: video_frame → score
    imp_by_vframe = {}
    for t_sec, score in impacts:
        vf = int(round(t_sec * FPS))
        if vf not in imp_by_vframe or score > imp_by_vframe[vf]:
            imp_by_vframe[vf] = score

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp    = out_path.replace(".mp4", "_noaudio.mp4")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(
        tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    flash      = None   # (start_frame, score)
    ring_r     = 0
    active_lbl = None   # (start_frame, text, cx, cy)
    n_imp      = 0

    # scrolling onset envelope: show a ±1.5 s window centred on current frame
    ENV_H   = 60    # pixel height of envelope strip at bottom
    ENV_Y   = H - ENV_H
    BAR_H   = 20    # band bars below envelope

    print(f"[sound] rendering {total} frames …")
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        af = audio_frame(fi)

        # ── envelope strip (scrolling window ±90 audio frames) ────────────────
        cv2.rectangle(frame, (0, ENV_Y - BAR_H * 2), (W, H), (10, 10, 10), -1)

        win_half = int(90)
        a_lo = max(0, af - win_half)
        a_hi = min(len(onset), af + win_half)
        seg  = onset[a_lo:a_hi]
        lo_s = lo_env[a_lo:a_hi]
        hi_s = hi_env[a_lo:a_hi]

        if len(seg) > 1:
            xs = np.linspace(0, W - 1, len(seg)).astype(int)
            # scale: display up to 6× noise floor in the strip height
            DISP_MAX = 6.0
            def to_y(v):
                return int(ENV_Y - 1 - np.clip(v / DISP_MAX, 0, 1) * (ENV_H - 4))

            # combined onset (green line)
            pts = np.array([[xi, to_y(v)] for xi, v in zip(xs, seg)],
                           dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, C_ENV, 1, cv2.LINE_AA)
            # threshold line
            thr_y = to_y(threshold)
            cv2.line(frame, (0, thr_y), (W - 1, thr_y), (80, 80, 80), 1)
            cv2.putText(frame, f"{threshold:.1f}x",
                        (W - 50, thr_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
            # lo band bar (orange)
            for xi, v in zip(xs, lo_s):
                bh = int(np.clip(v / DISP_MAX, 0, 1) * BAR_H)
                if bh > 0:
                    cv2.line(frame, (xi, ENV_Y + BAR_H),
                             (xi, ENV_Y + BAR_H - bh), C_BAR_LO, 1)
            # hi band bar (cyan)
            for xi, v in zip(xs, hi_s):
                bh = int(np.clip(v / DISP_MAX, 0, 1) * BAR_H)
                if bh > 0:
                    cv2.line(frame, (xi, ENV_Y + BAR_H * 2),
                             (xi, ENV_Y + BAR_H * 2 - bh), C_BAR_HI, 1)
            # cursor
            mid_x = W // 2
            cv2.line(frame, (mid_x, ENV_Y - ENV_H),
                     (mid_x, H - 1), (200, 200, 200), 1)

        # ── fire impact ────────────────────────────────────────────────────────
        if fi in imp_by_vframe:
            score  = imp_by_vframe[fi]
            n_imp += 1
            flash  = (fi, score)
            ring_r = 4
            cx, cy = W // 2, (ENV_Y - ENV_H) // 2
            active_lbl = (fi, f"IMPACT  {score:.2f}", cx, cy)

        # ── flash ──────────────────────────────────────────────────────────────
        if flash is not None:
            age   = fi - flash[0]
            alpha = max(0.0, 1.0 - age / FLASH_DUR) * 0.35
            if alpha > 0:
                tint = np.zeros_like(frame)
                tint[:, :] = C_FLASH
                cv2.addWeighted(tint, alpha, frame, 1.0, 0, frame)

        # ── expanding ring ─────────────────────────────────────────────────────
        if flash is not None and ring_r > 0:
            age = fi - flash[0]
            r   = 4 + int(age * 13)
            if r < 140:
                cv2.circle(frame, (W // 2, (ENV_Y - ENV_H) // 2),
                           r, C_RING, 2, cv2.LINE_AA)
            else:
                ring_r = 0

        # ── floating label ─────────────────────────────────────────────────────
        if active_lbl is not None:
            lf, txt, cx, cy = active_lbl
            age = fi - lf
            if age <= 20:
                drift_y = cy - age * 2
                alpha   = max(0.0, 1.0 - age / 20)
                overlay = frame.copy()
                cv2.putText(overlay, txt,
                            (cx - 80, drift_y),
                            cv2.FONT_HERSHEY_DUPLEX, 1.1,
                            C_LABEL, 2, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
            else:
                active_lbl = None

        # ── HUD ────────────────────────────────────────────────────────────────
        t_now = fi / FPS
        cv2.rectangle(frame, (0, 0), (480, 30), C_HUD, -1)
        cur_onset = float(onset[af]) if af < len(onset) else 0.0
        cv2.putText(frame,
                    f"sound-only  impacts:{n_imp}  "
                    f"t:{t_now:.1f}s  onset:{cur_onset:.2f}x  thr:{threshold:.1f}x",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 215, 255), 1, cv2.LINE_AA)

        # band labels
        cv2.putText(frame, "lo(thud)", (4, ENV_Y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_BAR_LO, 1)
        cv2.putText(frame, "hi(slap)", (4, ENV_Y + BAR_H + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_BAR_HI, 1)

        writer.write(frame)

    cap.release()
    writer.release()

    # mux original audio
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp, "-i", video_path,
             "-map", "0:v", "-map", "1:a?", "-c:v", "copy",
             "-shortest", out_path, "-loglevel", "error"], check=True)
        os.remove(tmp)
        print(f"[sound] saved with audio: {out_path}")
    except Exception as e:
        print(f"[sound] audio mux failed ({e}); silent video at {tmp}")


# ─────────────────────────────────────────────────────────────────────────────
# Sample extraction — cut candidate impact clips for manual auditioning
# ─────────────────────────────────────────────────────────────────────────────

def write_wav(path, samples, sr):
    s16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(s16.tobytes())


def extract_samples(x, sr, times, onset, out_dir, threshold,
                    cooldown_frames, pad_sec):
    """Cut a short clip around each onset peak so the user can audition and
    label them (punch vs not-punch).  Filenames embed the video timestamp +
    score so each clip can be cross-checked against the video."""
    os.makedirs(out_dir, exist_ok=True)
    # workflow subfolders the user sorts clips into
    keep_dir   = os.path.join(out_dir, "punch")
    reject_dir = os.path.join(out_dir, "not_punch")
    os.makedirs(keep_dir, exist_ok=True)
    os.makedirs(reject_dir, exist_ok=True)

    impacts = detect_impacts(times, onset, threshold, cooldown_frames)
    impacts.sort(key=lambda z: z[0])
    pad = int(pad_sec * sr)

    index = []
    for i, (t, sc) in enumerate(impacts):
        c   = int(t * sr)
        seg = x[max(0, c - pad): c + pad]
        mm  = int(t // 60)
        ss  = t - mm * 60
        vf  = int(round(t * FPS))
        fname = f"{i+1:03d}_t{mm:d}m{ss:05.2f}s_score{sc:.2f}_f{vf}.wav"
        write_wav(os.path.join(out_dir, fname), seg, sr)
        index.append({"id": i + 1, "file": fname, "time_sec": round(t, 3),
                      "timestamp": f"{mm}:{ss:05.2f}", "frame": vf,
                      "onset_score": round(sc, 3)})

    json.dump({"sample_threshold": threshold, "cooldown_frames": cooldown_frames,
               "pad_sec": pad_sec, "n_samples": len(index), "samples": index},
              open(os.path.join(out_dir, "index.json"), "w"), indent=2)

    readme = (
        "BOXING PUNCH SOUND SAMPLES — auditioning workflow\n"
        "=================================================\n\n"
        f"{len(index)} candidate clips were cut at onset peaks >= {threshold:.1f}x "
        "the local noise floor.\n"
        "Each filename = NNN_t<MIN>m<SEC>s_score<onset>_f<videoframe>.wav\n\n"
        "HOW TO USE:\n"
        "  1. Play each clip. Cross-check the timestamp against the video\n"
        "     (open sound.mp4 / 3.mp4 and jump to that time) to confirm.\n"
        "  2. Move clear PUNCH-LANDING clips  -> ./punch/\n"
        "  3. Move clear NON-PUNCH clips (crowd, commentary, glove tap on canvas,\n"
        "     rope, foot shuffle) -> ./not_punch/\n"
        "  4. Leave ambiguous clips where they are (ignored).\n\n"
        "Aim for ~10-20 in each folder. Both classes matter: the not_punch clips\n"
        "are what teaches the detector to STOP firing on every sound.\n"
    )
    open(os.path.join(out_dir, "README.txt"), "w").write(readme)

    print(f"\n[sound] wrote {len(index)} candidate clips to {out_dir}")
    print(f"        sort them into  {keep_dir}/  and  {reject_dir}/")
    print(f"        (filenames carry the video timestamp + onset score)\n")
    for s in index:
        print(f"  {s['id']:3d}.  {s['timestamp']:>9s}  score={s['onset_score']:.2f}"
              f"  f{s['frame']:<5d}  {s['file']}")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-in",  default=VIDEO_IN)
    ap.add_argument("--wav",       default=WAV_PATH)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THR)
    ap.add_argument("--cooldown",  type=float, default=DEFAULT_COOLDOWN,
                    help="min frames between detections")
    ap.add_argument("--out",       default=os.path.join(OUT_DIR, OUT_NAME))
    ap.add_argument("--extract-samples", action="store_true",
                    help="cut candidate impact clips for manual auditioning, then exit")
    ap.add_argument("--sample-dir", default=os.path.join(OUT_DIR, "sound_samples"))
    ap.add_argument("--sample-threshold", type=float, default=1.5,
                    help="onset ratio for candidate clips (lower = more clips)")
    ap.add_argument("--sample-pad", type=float, default=0.20,
                    help="seconds of audio on each side of the onset peak")
    args = ap.parse_args()

    # 1. audio
    ensure_wav(args.video_in, args.wav)
    print(f"[sound] loading {args.wav} …")
    x, sr = load_wav(args.wav)
    print(f"[sound] {len(x)/sr:.1f}s @ {sr}Hz")

    # 2. onset detection
    print("[sound] computing onset envelope …")
    times, onset, lo_env, hi_env = compute_onset(x, sr)
    print(f"[sound] {len(times)} audio frames  "
          f"(audio fps ≈ {len(times)/times[-1]:.0f})")

    # 2b. sample-extraction mode — cut clips and exit
    if args.extract_samples:
        extract_samples(x, sr, times, onset, args.sample_dir,
                        args.sample_threshold, int(args.cooldown), args.sample_pad)
        return

    # 3. peak pick  (cooldown is in FRAMES)
    impacts = detect_impacts(times, onset, args.threshold, args.cooldown)
    print(f"\n[sound] detected {len(impacts)} impacts "
          f"(threshold={args.threshold}, cooldown={args.cooldown}fr)\n")
    for i, (t, sc) in enumerate(impacts):
        m, s = int(t // 60), t % 60
        print(f"  {i+1:3d}.  {m}:{s:05.2f}  score={sc:.3f}  "
              f"frame≈{int(t*FPS)}")

    # 4. save detections JSON
    out_json = args.out.replace(".mp4", ".json")
    json.dump({
        "method": "sound_only",
        "threshold": args.threshold,
        "cooldown_frames": args.cooldown,
        "n_impacts": len(impacts),
        "events": [{"time_sec": round(t, 3),
                    "frame": int(round(t * FPS)),
                    "score": round(sc, 4)}
                   for t, sc in impacts],
    }, open(out_json, "w"), indent=2)
    print(f"\n[sound] JSON saved: {out_json}")

    # 5. render video
    render(args.video_in, impacts,
           (times, onset, lo_env, hi_env),
           args.threshold, args.out)


if __name__ == "__main__":
    main()
