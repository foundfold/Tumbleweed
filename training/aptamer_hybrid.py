"""Hybrid MLM + contrastive model — SimCSE-style multi-task.

Shared encoder backbone, two heads:
  - MLM head: per-token vocab projection (cross-entropy on 15%-masked positions)
  - Contrastive head: L2-normalized embed projection (InfoNCE on triplets)

Single forward pass returns both (logits, projection). Loss is a weighted sum.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from aptamer_dataset import N_TOKENS, PAD_ID
from aptamer_encoder import SinusoidalPosEnc, info_nce_loss
from aptamer_mlm import mlm_loss


class AptamerHybridModel(nn.Module):
    def __init__(self,
                 d_model: int = 384,
                 nhead: int = 6,
                 num_layers: int = 8,
                 dim_ff: int = 1536,
                 dropout: float = 0.1,
                 embed_dim: int = 192,
                 max_len: int = 128,
                 grad_checkpoint: bool = False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.embed = nn.Embedding(N_TOKENS, d_model, padding_idx=PAD_ID)
        self.pos = SinusoidalPosEnc(d_model, max_len=max_len + 16)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # MLM head
        self.mlm_dense = nn.Linear(d_model, d_model)
        self.mlm_norm = nn.LayerNorm(d_model)
        self.mlm_bias = nn.Parameter(torch.zeros(N_TOKENS))

        # Contrastive projection head
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, embed_dim),
        )

    def _layer_fwd(self, layer, x, key_padding_mask):
        return layer(x, src_key_padding_mask=key_padding_mask)

    def _encode_hidden(self, input_ids):
        pad_mask = (input_ids == PAD_ID)
        x = self.embed(input_ids)
        x = self.pos(x)
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._layer_fwd, layer, x, pad_mask, use_reentrant=False)
            else:
                x = self._layer_fwd(layer, x, pad_mask)
        x = self.norm(x)
        return x, pad_mask

    def encode(self, input_ids):
        """Pooled hidden state for downstream eval (matches AptamerEncoder.encode)."""
        h, pad_mask = self._encode_hidden(input_ids)
        not_pad = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * not_pad).sum(dim=1) / not_pad.sum(dim=1).clamp_min(1e-6)
        return pooled

    def forward(self, input_ids, want_mlm: bool = True, want_proj: bool = True):
        """Single forward pass; returns (logits, projection) dict.

        - logits     : (B, L, V) only if want_mlm
        - projection : (B, embed_dim) L2-normalized, only if want_proj
        """
        h, pad_mask = self._encode_hidden(input_ids)
        out = {}
        if want_mlm:
            x = F.gelu(self.mlm_dense(h))
            x = self.mlm_norm(x)
            logits = x @ self.embed.weight.t() + self.mlm_bias
            out['logits'] = logits
        if want_proj:
            not_pad = (~pad_mask).float().unsqueeze(-1)
            pooled = (h * not_pad).sum(dim=1) / not_pad.sum(dim=1).clamp_min(1e-6)
            z = self.proj(pooled)
            z = F.normalize(z, dim=-1)
            out['projection'] = z
        return out


def hybrid_loss(logits, label_ids, active_mask,
                anc, pos, neg,
                alpha: float = 1.0, beta: float = 0.5, tau: float = 0.2):
    """alpha * MLM_CE + beta * InfoNCE. Tracked as (total, mlm, contrast)."""
    l_mlm = mlm_loss(logits, label_ids, active_mask)
    l_con = info_nce_loss(anc, pos, neg, temperature=tau)
    return alpha * l_mlm + beta * l_con, l_mlm.detach(), l_con.detach()


if __name__ == '__main__':
    m = AptamerHybridModel()
    n = sum(p.numel() for p in m.parameters())
    print(f'AptamerHybridModel: {n/1e6:.2f}M params')
    fake = torch.randint(0, N_TOKENS - 1, (4, 32))
    out = m(fake)
    print(f'  logits: {out["logits"].shape}  proj: {out["projection"].shape}')
