#!/usr/bin/env python3
"""
ASFormer-inspired binary impact/not_impact classifier.

ASFormer itself (see D:\\SVTAS\\SVTAS) is a per-frame multi-class TEMPORAL
ACTION SEGMENTATION model -- a different problem shape than ours (one label
per already-segmented clip, not a label per frame). This adapts its core
building-block idea -- a dilated-conv layer (local receptive field, like our
existing ImpactTCN) combined with a self-attention layer (global receptive
field, which the pure-conv TCN doesn't have) per block, stacked with
increasing dilation -- then collapses the whole sequence to one prediction
via masked average pooling + a small classifier head, instead of ASFormer's
multi-stage per-frame decoder (which doesn't apply when there's only one
label for the whole clip).

Kept deliberately small (~3 blocks, 32-dim) given the dataset has only a few
hundred effective training clips per leave-one-round-out fold -- a full-size
ASFormer (64-dim, 7+ layers) would almost certainly overfit here.
"""
import torch
import torch.nn as nn


class DilatedConvAttnBlock(nn.Module):
    """One block = dilated Conv1d (local context) -> self-attention (global
    context) -> feed-forward, each with a residual + LayerNorm, mirroring
    ASFormer's per-layer structure (dilated conv THEN attention)."""

    def __init__(self, dim, dilation, n_heads, dropout):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask):
        # x: [B, T, D]
        h = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm1(x + self.dropout(h))
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm2(x + self.dropout(attn_out))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class ImpactASFormer(nn.Module):
    """
    Input:  x    [B, T, F]   padded per-frame feature sequence
            mask [B, T]      1.0 = real frame, 0.0 = padding
    Output: logit [B]
    """

    def __init__(self, num_features, dim=32, n_layers=3, n_heads=2, dropout=0.3, fc_dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(num_features, dim)
        self.blocks = nn.ModuleList([
            DilatedConvAttnBlock(dim, dilation=2 ** i, n_heads=n_heads, dropout=dropout)
            for i in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(dim, 16), nn.ReLU(inplace=True), nn.Dropout(fc_dropout), nn.Linear(16, 1),
        )

    def forward(self, x, mask):
        key_padding_mask = mask == 0  # True where padded -> ignored by attention
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h, key_padding_mask)
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.classifier(pooled).squeeze(-1)
