#!/usr/bin/env python3
"""Single build_model() entry point spanning the original tools/keypoint_model.py
architectures (tcn, gru) and the two new SVTAS/BRT-inspired ones (asformer, brt),
so the comparison script can train all of them through one interface."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + r"\..")
from keypoint_model import build_model as _build_tcn_gru  # noqa: E402

from asformer_binary import ImpactASFormer  # noqa: E402
from brt_binary import ImpactBRT  # noqa: E402

MODEL_NAMES = ("tcn", "gru", "asformer", "brt")


def build_model(name, num_features, **kwargs):
    if name in ("tcn", "gru"):
        return _build_tcn_gru(name, num_features, **kwargs)
    if name == "asformer":
        return ImpactASFormer(num_features, **kwargs)
    if name == "brt":
        return ImpactBRT(num_features, **kwargs)
    raise ValueError(f"Unknown model {name!r} (expected one of {MODEL_NAMES})")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
