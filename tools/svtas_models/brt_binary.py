#!/usr/bin/env python3
"""
BRT-inspired binary impact/not_impact classifier.

BRT (Block Recurrent Transformer, see D:\\SVTAS\\SVTAS) is a streaming variant
of ASFormer: it processes a long sequence in fixed-size blocks instead of
needing the whole thing upfront, carrying a recurrent memory state forward
from block to block so later blocks still have context from earlier ones.

Our clips are short (<=41 frames) so streaming isn't actually necessary for
runtime reasons -- this is included because the user asked to evaluate BRT's
*architecture* (block-local attention + a carried recurrent state) against
ASFormer's *global* attention and the TCN's *local convolutional* receptive
field, as a 3-way comparison of how much sequence-modeling style matters on
this dataset, not because streaming inference is needed here.

Each block: a tiny self-attention block (block-local context) that also
attends to a single "memory token" carrying the running summary of all
previous blocks; after the block, the memory is updated via a GRUCell from
the block's own pooled output. The final prediction uses BOTH the
mask-pooled features over the whole sequence and the final memory state
(which is itself a compressed summary of the whole sequence, just built up
incrementally rather than attended to all at once like ASFormer).
"""
import torch
import torch.nn as nn


class BlockRecurrentLayer(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mem_gate = nn.GRUCell(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, block, mem, key_padding_mask):
        """block: [B, Tb, D]   mem: [B, D] (recurrent state from the previous
        block)   key_padding_mask: [B, Tb] True=padding."""
        mem_tok = mem.unsqueeze(1)  # [B, 1, D] -- prepended as extra context
        seq = torch.cat([mem_tok, block], dim=1)
        pad_mem = torch.zeros(block.shape[0], 1, dtype=torch.bool, device=block.device)
        kpm = torch.cat([pad_mem, key_padding_mask], dim=1)

        attn_out, _ = self.attn(seq, seq, seq, key_padding_mask=kpm)
        seq = self.norm1(seq + self.dropout(attn_out))
        seq = self.norm2(seq + self.dropout(self.ff(seq)))
        block_out = seq[:, 1:, :]  # drop the memory-token position from the output

        real = (~key_padding_mask).float().unsqueeze(-1)  # [B, Tb, 1]
        pooled_block = (block_out * real).sum(1) / real.sum(1).clamp(min=1.0)
        new_mem = self.mem_gate(pooled_block, mem)
        return block_out, new_mem


class ImpactBRT(nn.Module):
    """
    Input:  x    [B, T, F]
            mask [B, T]
    Output: logit [B]
    """

    def __init__(self, num_features, dim=32, n_heads=2, block_size=8, n_layers=2,
                 dropout=0.3, fc_dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(num_features, dim)
        self.block_size = block_size
        self.dim = dim
        self.layers = nn.ModuleList([
            BlockRecurrentLayer(dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, 16), nn.ReLU(inplace=True), nn.Dropout(fc_dropout), nn.Linear(16, 1),
        )

    def forward(self, x, mask):
        B, T, _ = x.shape
        h = self.input_proj(x)
        mem = [torch.zeros(B, self.dim, device=x.device) for _ in self.layers]

        outputs = []
        for s in range(0, T, self.block_size):
            e = min(T, s + self.block_size)
            cur = h[:, s:e, :]
            kpm = mask[:, s:e] == 0
            for li, layer in enumerate(self.layers):
                cur, mem[li] = layer(cur, mem[li], kpm)
            outputs.append(cur)

        full = torch.cat(outputs, dim=1)  # [B, T, dim]
        m = mask.unsqueeze(-1)
        pooled = (full * m).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        final_mem = mem[-1]
        combined = torch.cat([pooled, final_mem], dim=-1)
        return self.classifier(combined).squeeze(-1)
