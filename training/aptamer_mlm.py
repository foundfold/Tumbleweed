"""MLM head + loss wrapping the AptamerEncoder backbone.

For the head-to-head ablation, the encoder is IDENTICAL to the contrastive
trainer's. Only differences:
  - drop the L2-normalized projection head (used for cosine-sim contrastive)
  - add a per-token vocab-projection head: d_model → N_TOKENS
  - loss is cross-entropy over the masked positions only

This mirrors RaptScore / InstructNA / AptaBERT: BERT-style 15% masking, predict
the original token at masked positions.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from aptamer_dataset import N_TOKENS, PAD_ID, MASK_ID
from aptamer_encoder import SinusoidalPosEnc


class AptamerMLMModel(nn.Module):
    """Encoder backbone + MLM head.

    Shares the same internal transformer layout as AptamerEncoder so that
    swapping the trainer (contrastive vs MLM) does not change anything else
    in the head-to-head.
    """

    def __init__(self,
                 d_model: int = 384,
                 nhead: int = 6,
                 num_layers: int = 8,
                 dim_ff: int = 1536,
                 dropout: float = 0.1,
                 max_len: int = 128,
                 grad_checkpoint: bool = False,
                 # the following exist only to absorb extra config keys (so
                 # we can share the same YAML between contrastive + MLM):
                 embed_dim: int | None = None):
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
        # MLM head: 2-layer MLP → vocab. Tied to embedding for parameter savings
        # and per-BERT convention (output projection shares weights with input
        # embedding).
        self.mlm_dense = nn.Linear(d_model, d_model)
        self.mlm_norm = nn.LayerNorm(d_model)
        self.mlm_bias = nn.Parameter(torch.zeros(N_TOKENS))
        # weight is the transpose of self.embed.weight via mlm_logits

    def _layer_fwd(self, layer, x, key_padding_mask):
        return layer(x, src_key_padding_mask=key_padding_mask)

    def encode(self, input_ids):
        """Pooled encoder hidden state — same shape and meaning as
        AptamerEncoder.encode(). Used by downstream benchmark eval to keep
        contrastive vs MLM apples-to-apples (same feature extractor in both).
        """
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
        not_pad = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * not_pad).sum(dim=1) / not_pad.sum(dim=1).clamp_min(1e-6)
        return pooled

    def forward(self, input_ids):
        """Returns logits over N_TOKENS at every position (B, L, V).

        Loss is computed externally over the active mask positions only.
        """
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
        x = F.gelu(self.mlm_dense(x))
        x = self.mlm_norm(x)
        # Tied projection: logits = x @ embed.weight.T + bias
        logits = x @ self.embed.weight.t() + self.mlm_bias
        return logits


def mlm_loss(logits: torch.Tensor, label_ids: torch.Tensor,
             active_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the active (masked) positions only.

    logits      : (B, L, V)
    label_ids   : (B, L)   original tokens
    active_mask : (B, L)   bool — True where loss should be computed
    """
    V = logits.size(-1)
    # Gather only the active positions for efficiency
    active = active_mask.bool()
    flat_logits = logits[active]                       # (M, V)
    flat_labels = label_ids[active]                    # (M,)
    if flat_logits.numel() == 0:
        return logits.sum() * 0.0  # no active positions; return tracked zero
    return F.cross_entropy(flat_logits, flat_labels)


if __name__ == '__main__':
    m = AptamerMLMModel()
    n = sum(p.numel() for p in m.parameters())
    print(f'AptamerMLMModel: {n/1e6:.2f}M params')
    fake = torch.randint(0, N_TOKENS - 1, (4, 32))
    fake_labels = fake.clone()
    fake_active = torch.zeros_like(fake, dtype=torch.bool)
    fake_active[:, ::3] = True
    out = m(fake)
    print(f'  logits: {out.shape}')
    loss = mlm_loss(out, fake_labels, fake_active)
    print(f'  loss (random init): {loss.item():.3f}  (random baseline ≈ ln(6) = 1.79)')
