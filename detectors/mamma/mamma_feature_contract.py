#!/usr/bin/env python3
"""
Feature contract between MammaNet and our detector.
===================================================
We define the per-frame feature schema we want from MammaNet so the rest of the pipeline
can be built and tested independently of MAMMA's exact output format. The single adapter
that maps MAMMA's raw 2D-stage output into this schema is the one thing to write against
their real API (see SETUP.md step 3); everything downstream consumes THIS schema.

Schema (one JSON per fight):
{
  "source_video": "...mp4",
  "fps": 24.995,
  "frames": {
     "<frame_int>": {
        "<person_id>": {                 # "0" / "1"
           "landmarks_2d": [[x,y], ...], # dense surface landmarks, image px
           "visibility":   [v, ...],     # 0..1 per landmark
           "uncertainty":  [u, ...],     # >=0 per landmark (lower = better)
           "contact_prob": [c, ...]      # 0..1 per landmark (human-human contact likelihood)
        }, ...
     }, ...
  }
}
"""
import json
import numpy as np


def load_mamma_features(path):
    """Load a mamma_features.json into {frame:int -> {pid:int -> dict of np arrays}}."""
    raw = json.load(open(path))
    out = {}
    for f_str, persons in raw.get("frames", {}).items():
        f = int(f_str)
        out[f] = {}
        for pid_str, d in persons.items():
            out[f][int(pid_str)] = {
                "landmarks_2d": np.asarray(d["landmarks_2d"], float),
                "visibility":   np.asarray(d.get("visibility", []), float),
                "uncertainty":  np.asarray(d.get("uncertainty", []), float),
                "contact_prob": np.asarray(d.get("contact_prob", []), float),
            }
    return out


def best_contact_score(feat_frame, striker_id, receiver_id, top_k=8):
    """A simple, learnable-free contact score for a single frame: the mean of the
    highest receiver-surface contact probabilities, weighted by visibility. This is a
    drop-in replacement for v9's hand-made wrist-gap once MammaNet features exist.
    Returns (score in 0..1, None) or (0.0, None) if data missing."""
    rp = feat_frame.get(receiver_id)
    if rp is None or rp["contact_prob"].size == 0:
        return 0.0
    cp = rp["contact_prob"]
    vis = rp["visibility"] if rp["visibility"].size == cp.size else np.ones_like(cp)
    w = cp * vis
    if w.size == 0:
        return 0.0
    k = min(top_k, w.size)
    return float(np.sort(w)[-k:].mean())


def schema_example():
    """Tiny synthetic example so downstream code can be unit-tested with no GPU."""
    return {
        "source_video": "example.mp4", "fps": 25.0,
        "frames": {
            "100": {
                "0": {"landmarks_2d": [[10, 20], [12, 22]], "visibility": [1.0, 0.9],
                      "uncertainty": [0.1, 0.2], "contact_prob": [0.8, 0.1]},
                "1": {"landmarks_2d": [[60, 40], [62, 42]], "visibility": [1.0, 0.8],
                      "uncertainty": [0.1, 0.3], "contact_prob": [0.7, 0.2]},
            }
        },
    }


if __name__ == "__main__":
    # self-test against the synthetic example (no GPU / no MAMMA needed)
    import tempfile, os
    ex = schema_example()
    p = os.path.join(tempfile.gettempdir(), "_mamma_example.json")
    json.dump(ex, open(p, "w"))
    feats = load_mamma_features(p)
    s = best_contact_score(feats[100], striker_id=0, receiver_id=1)
    print(f"[contract] loaded {len(feats)} frame(s); example contact score = {s:.3f}")
    print("[contract] schema OK")
