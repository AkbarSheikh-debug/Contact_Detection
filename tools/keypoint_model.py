#!/usr/bin/env python3
"""
Model architectures for the keypoint-based impact / not_impact classifier.

ImpactTCN (primary): small dilated Conv1d stack, inspired by
C:\\Users\\XRIG\\Desktop\\Secuirty_AI\\Module_3's TemporalCNN, but downsized
(~25K params vs Module_3's ~140K+ conv block) given this dataset has only
~370 effective training rows per leave-one-round-out fold, and uses masked
average pooling (not nn.AdaptiveAvgPool1d) so the repeat-last-frame padding
doesn't dilute the pooled feature.

ImpactGRU (fallback): bidirectional GRU with pack_padded_sequence, to try if
the TCN's fixed local receptive field misses longer-range dependencies (e.g.
a slow feint before a fast final strike).
"""
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class ImpactTCN(nn.Module):
    """
    Input:  x    [B, T, F]   padded per-frame feature sequence
            mask [B, T]      1.0 = real frame, 0.0 = padding
    Output: logit [B]        raw logit, feed to BCEWithLogitsLoss
    """

    def __init__(self, num_features, dropout=0.3, fc_dropout=0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 48, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm1d(48), nn.ReLU(inplace=True), nn.Dropout(dropout),

            nn.Conv1d(48, 64, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout),

            nn.Conv1d(64, 48, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(48), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(48, 32), nn.ReLU(inplace=True), nn.Dropout(fc_dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, mask):
        # x: [B, T, F] -> [B, F, T] for Conv1d
        h = self.conv(x.permute(0, 2, 1))          # [B, 48, T]
        m = mask.unsqueeze(1)                       # [B, 1, T]
        pooled = (h * m).sum(dim=2) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.classifier(pooled).squeeze(-1)  # [B]


class ImpactGRU(nn.Module):
    """
    Input:  x    [B, T, F]
            mask [B, T]
    Output: logit [B]
    """

    def __init__(self, num_features, hidden_size=32, dropout=0.3, fc_dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features, hidden_size=hidden_size,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32), nn.ReLU(inplace=True), nn.Dropout(fc_dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
        packed = rnn_utils.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)          # h_n: [2, B, hidden]
        h = torch.cat([h_n[0], h_n[1]], dim=-1)  # [B, hidden*2] (fwd+bwd final states)
        h = self.dropout(h)
        return self.classifier(h).squeeze(-1)


def build_model(name, num_features, **kwargs):
    if name == "tcn":
        return ImpactTCN(num_features, **kwargs)
    if name == "gru":
        return ImpactGRU(num_features, **kwargs)
    raise ValueError(f"Unknown model name: {name!r} (expected 'tcn' or 'gru')")
