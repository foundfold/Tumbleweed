"""Transformer encoder + projection head for aptamer contrastive training (production).

Architecture (configurable via YAML):
  - Token embedding: 5 tokens (A/C/G/U/PAD) → d_model
  - Sinusoidal positional encoding
  - N transformer encoder layers (pre-norm, GELU)
  - Mean-pool over non-PAD tokens
  - 2-layer MLP projection head → embed_dim
  - L2-normalize for cosine-similarity InfoNCE

Production additions vs v1 scaffold:
  - Optional gradient checkpointing on the transformer stack (saves ~40% memory
    for ~30% compute overhead — important for d_model >= 512 + L >= 256 on A100).
  - Configurable from YAML.
  - Symmetric multi-positive InfoNCE (CALM-style) supporting the case where the
    same parent binder produces many late-round reads.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aptamer_dataset import N_TOKENS, PAD_ID


class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CNNFrontEnd(nn.Module):
    """Learnable-motif convolutional front-end (v15).

    Stacks 1-3 Conv1d layers over the embedded nucleotide sequence so the model
    has explicit motif-detector capacity before attention. Residual + LayerNorm
    so an untrained CNN is approximately a no-op (gradients flow around it).

    Args:
        d_model: same as encoder's embedding dim (in == out, residual lives).
        kernels: list of kernel sizes per layer (e.g. [5, 7] for two layers).
        dropout: post-activation dropout per conv layer.
    """
    def __init__(self, d_model: int, kernels=(5, 7), dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2)
            for k in kernels
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, d) → conv1d expects (B, d, L)
        residual = x
        h = x.transpose(1, 2)
        for conv in self.convs:
            h = F.gelu(conv(h))
            h = self.dropout(h)
        h = h.transpose(1, 2)
        return self.norm(residual + h)


class AptamerEncoder(nn.Module):
    def __init__(self,
                 d_model: int = 384,
                 nhead: int = 6,
                 num_layers: int = 8,
                 dim_ff: int = 1536,
                 embed_dim: int = 192,
                 dropout: float = 0.1,
                 max_len: int = 128,
                 grad_checkpoint: bool = False,
                 regression_head: bool = False,
                 cnn_kernels=None,    # v15: list of conv kernel sizes, e.g. [5, 7]; None = no CNN front-end
                 # ─── Target protein conditioning (ESM-2 embedding) ───────────
                 # `target_cond_mode` selects how the per-target ESM-2 embedding
                 # enters the encoder. Three options for paper-1 ablation:
                 #   None             — no conditioning (baseline)
                 #   'context_token'  — project target → d_model, prepend as token 0
                 #   'cross_attn'     — dedicated cross-attention layer; aptamer
                 #                      tokens attend to projected target embed
                 #   'film'           — FiLM (scale+shift) modulation per encoder
                 #                      layer derived from target embed
                 target_cond_mode: Optional[str] = None,
                 target_embed_dim: int = 1280,  # ESM-2 650M default
                 pool: str = 'mean',  # 'mean' (default) or 'cls' (pos 0 = chemistry token)
                 target_cond_dropout: float = 0.0,  # prob of replacing target with null embed at train time
                 gated_context: bool = False,  # learned scalar gate on context_token magnitude
                 mlm_head: bool = False,  # per-token vocab head for hybrid MLM aux loss (v8+)
                 kd_head: bool = False,  # regression head for measured-Kd supervision (v12+)
                 kmer_token_dim: int = 0,  # v18: if >0, accept per-seq kmer features and prepend as 2nd context token
                 ):
        super().__init__()
        assert pool in ('mean', 'cls'), pool
        self.pool = pool
        self.target_cond_dropout = target_cond_dropout
        self.gated_context = gated_context
        self.has_mlm_head = mlm_head
        self.has_kd_head = kd_head
        self.grad_checkpoint = grad_checkpoint
        self.embed = nn.Embedding(N_TOKENS, d_model, padding_idx=PAD_ID)
        self.pos = SinusoidalPosEnc(d_model, max_len=max_len + 16)
        # CNN motif front-end (v15). None → identity; list of kernel sizes →
        # stacked conv layers acting as learnable motif detectors over the
        # embedded nucleotide track.
        self.cnn_front = CNNFrontEnd(d_model, tuple(cnn_kernels), dropout=dropout) \
            if cnn_kernels else None
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, embed_dim),
        )
        self.embed_dim = embed_dim
        # ─── Optional regression head: predict round_frac per sequence ────────
        # 2-layer MLP on the pooled (pre-projection) encoder hidden. Predicts
        # a scalar in [0, 1] via sigmoid. Used as an auxiliary regression loss
        # alongside InfoNCE so the model directly learns the affinity gradient
        # of SELEX rounds instead of treating rounds as binary positive/negative.
        self.regression_head = regression_head
        if regression_head:
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        # ─── Target conditioning module ───────────────────────────────────────
        self.target_cond_mode = target_cond_mode
        self.target_embed_dim = target_embed_dim
        if target_cond_mode == 'context_token':
            # Project ESM-2 → d_model; prepend as a single "context token" before
            # the aptamer sequence. Self-attention sees both; pool over aptamer
            # positions only (we mask out the context token at pooling time).
            self.tgt_proj = nn.Sequential(
                nn.Linear(target_embed_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            # A learned "null target" embedding for sequences without target
            # (synthetic R0, naive libraries, unknown-target SELEX reads)
            self.null_target = nn.Parameter(torch.randn(target_embed_dim) * 0.02)
        elif target_cond_mode in ('cross_attn', 'cross_attn_per_residue'):
            # Dedicated cross-attention block: each aptamer hidden state attends
            # to the target embedding (projected to d_model). Applied AFTER the
            # standard transformer stack so we don't add complexity per-layer.
            # In `cross_attn` mode: KV is (B, 1, d_model) from mean-pooled target.
            # In `cross_attn_per_residue`: KV is (B, L_target, d_model) from per-
            # residue ESM-2 embeddings, with key_padding_mask for variable lengths.
            self.tgt_proj = nn.Sequential(
                nn.Linear(target_embed_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead, dropout=dropout,
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(d_model)
            self.null_target = nn.Parameter(torch.randn(target_embed_dim) * 0.02)
        elif target_cond_mode == 'film':
            # FiLM: per-layer scale + shift modulation derived from target embed.
            # 2 × num_layers learnable (scale, shift) vectors of size d_model.
            self.film_gen = nn.Sequential(
                nn.Linear(target_embed_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2 * d_model * num_layers),
            )
            self.null_target = nn.Parameter(torch.randn(target_embed_dim) * 0.02)
        elif target_cond_mode is not None:
            raise ValueError(f'unknown target_cond_mode: {target_cond_mode}')

        # Learned scalar gate on context-token magnitude. Init at logit=-2 →
        # sigmoid(-2)≈0.12 so the model starts with weak target conditioning
        # and can grow it back if needed.
        if self.gated_context and target_cond_mode == 'context_token':
            self.context_gate = nn.Parameter(torch.tensor(-2.0))

        # MLM head: per-token vocab projection from per-position encoder hidden.
        # Forward pass via forward_mlm_logits() uses target_emb=None so the
        # MLM loss can't be satisfied via target conditioning shortcut.
        if mlm_head:
            self.mlm_dense = nn.Linear(d_model, d_model)
            self.mlm_norm = nn.LayerNorm(d_model)
            self.mlm_decoder = nn.Linear(d_model, N_TOKENS)

        # Kd regression head: predicts normalized log10(Kd_nM) in [0, 1] from
        # the pooled (target-conditioned) hidden. Trained only on rows with
        # measured Kd labels — typically a tiny fraction of the contrastive
        # corpus. See train_contrastive.py's Kd-batch sampler.
        if kd_head:
            self.kd_head_mlp = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        # v18: kmer-token. When kmer_token_dim > 0, the forward path accepts a
        # per-sequence kmer feature vector (computed deterministically outside
        # the model by kmer_features.build_kmer_features) and prepends it as a
        # second context token immediately after the target context_token.
        # Layout (when both target context_token and kmer_token enabled):
        #   pos 0: target_ESM2 projected
        #   pos 1: kmer features projected
        #   pos 2+: chemistry token + aptamer sequence
        # Pooling skips positions 0 and 1.
        self.kmer_token_dim = kmer_token_dim
        if kmer_token_dim > 0:
            self.kmer_proj = nn.Sequential(
                nn.Linear(kmer_token_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )

    def _layer_fwd(self, layer, x, mask):
        return layer(x, src_key_padding_mask=mask)

    def _resolve_target(self, target_emb: Optional[torch.Tensor], batch_size: int,
                        device: torch.device) -> Optional[torch.Tensor]:
        """If target_emb is None but conditioning is enabled, broadcast the learned
        null-target embedding to the batch. Else pass through.
        Returns shape (B, target_embed_dim) or (B, L_target, target_embed_dim)
        depending on the input, or None if no conditioning."""
        if self.target_cond_mode is None:
            return None
        if target_emb is None:
            return self.null_target.unsqueeze(0).expand(batch_size, -1)
        return target_emb

    def encode(self, ids, target_emb: Optional[torch.Tensor] = None,
               target_mask: Optional[torch.Tensor] = None,
               kmer_feats: Optional[torch.Tensor] = None):
        """Pooled encoder hidden state (pre-projection head). Optionally conditioned
        on a target protein embedding (e.g. ESM-2 mean-pool).

        Args:
            ids: (B, L) LongTensor of aptamer token ids
            target_emb: (B, target_embed_dim) FloatTensor of target ESM-2 features,
                        or None (falls back to learned null embedding when
                        target_cond_mode is set).
        """
        B = ids.size(0)
        mask = (ids == PAD_ID)
        x = self.embed(ids)
        x = self.pos(x)
        # CNN motif front-end (v15): learnable motif detectors before attention
        if self.cnn_front is not None:
            x = self.cnn_front(x)
        # Training-time target dropout: with p=target_cond_dropout, replace
        # target_emb with the null embed so the encoder learns a strong
        # sequence-only path that doesn't depend on target conditioning.
        if (self.training and self.target_cond_dropout > 0.0
                and target_emb is not None
                and torch.rand((), device=x.device).item() < self.target_cond_dropout):
            target_emb = None
        tgt = self._resolve_target(target_emb, B, x.device)

        # Per-layer FiLM scales/shifts, precomputed from target embed
        film_params = None
        if self.target_cond_mode == 'film' and tgt is not None:
            film_params = self.film_gen(tgt).view(B, len(self.layers), 2, -1)
            # → (B, num_layers, 2 [scale, shift], d_model)

        # Context-token mode: prepend a projected-target token to the sequence
        n_prepended = 0
        if self.target_cond_mode == 'context_token' and tgt is not None:
            ctx_tok = self.tgt_proj(tgt).unsqueeze(1)  # (B, 1, d_model)
            if self.gated_context:
                ctx_tok = torch.sigmoid(self.context_gate) * ctx_tok
            x = torch.cat([ctx_tok, x], dim=1)
            mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
            n_prepended += 1
        # v18: kmer-token prepended after target token, before chemistry token
        if self.kmer_token_dim > 0 and kmer_feats is not None:
            kmer_tok = self.kmer_proj(kmer_feats).unsqueeze(1)  # (B, 1, d_model)
            x = torch.cat([kmer_tok, x], dim=1)
            mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
            n_prepended += 1

        for li, layer in enumerate(self.layers):
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._layer_fwd, layer, x, mask, use_reentrant=False)
            else:
                x = self._layer_fwd(layer, x, mask)
            if film_params is not None:
                scale = 1.0 + film_params[:, li, 0].unsqueeze(1)  # (B, 1, d_model)
                shift = film_params[:, li, 1].unsqueeze(1)
                x = x * scale + shift

        x = self.norm(x)

        # Cross-attention conditioning happens after the stack
        if self.target_cond_mode in ('cross_attn', 'cross_attn_per_residue') and tgt is not None:
            if tgt.dim() == 2:
                # Mean-pool conditioning: (B, D) → (B, 1, d_model), no padding mask
                kv = self.tgt_proj(tgt).unsqueeze(1)
                kv_mask = None
            else:
                # Per-residue conditioning: (B, L_target, D) → (B, L_target, d_model)
                kv = self.tgt_proj(tgt)
                kv_mask = target_mask  # (B, L_target) True = pad position
            attn_out, _ = self.cross_attn(query=x, key=kv, value=kv,
                                          key_padding_mask=kv_mask)
            x = self.cross_attn_norm(x + attn_out)

        # Pool. For context_token mode, the first position IS the target context —
        # exclude it so the aptamer representation isn't averaged with context.
        # Also skip the kmer-token if it was prepended (n_prepended accounts for both).
        if n_prepended > 0:
            x_apt = x[:, n_prepended:]
            mask_apt = mask[:, n_prepended:]
        else:
            x_apt = x
            mask_apt = mask
        if self.pool == 'cls':
            # pos 0 of x_apt is the chemistry token ([RNA] or [DNA])
            pooled = x_apt[:, 0]
        else:
            not_pad = (~mask_apt).float().unsqueeze(-1)
            pooled = (x_apt * not_pad).sum(dim=1) / not_pad.sum(dim=1).clamp_min(1e-6)
        return pooled

    def forward(self, ids, target_emb: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None,
                kmer_feats: Optional[torch.Tensor] = None):
        """Projected + L2-normalized embedding for contrastive training."""
        pooled = self.encode(ids, target_emb=target_emb, target_mask=target_mask,
                              kmer_feats=kmer_feats)
        z = self.proj(pooled)
        return F.normalize(z, dim=-1)

    def forward_kd(self, ids, target_emb: Optional[torch.Tensor] = None,
                   target_mask: Optional[torch.Tensor] = None):
        """Predicted normalized log10 Kd in [0, 1] from (sequence, target).
        Returns (B,) scalar tensor. Trained against measured Kd labels via MSE.
        """
        assert self.has_kd_head, 'kd_head=False; rebuild encoder with kd_head=True'
        pooled = self.encode(ids, target_emb=target_emb, target_mask=target_mask)
        return torch.sigmoid(self.kd_head_mlp(pooled).squeeze(-1))

    def forward_mlm_logits(self, ids):
        """BERT-style per-token logits. ALWAYS runs with target_emb=None so the
        MLM loss cannot be satisfied via target-conditioning shortcut — the
        encoder must rely on sequence content alone to reconstruct masks.
        Returns: (B, L, N_TOKENS) logits — only positions corresponding to
        the original input length (chemistry token + nucleotides + PAD)."""
        assert self.has_mlm_head, 'mlm_head=False; rebuild encoder with mlm_head=True'
        B = ids.size(0)
        mask = (ids == PAD_ID)
        x = self.embed(ids)
        x = self.pos(x)
        if self.cnn_front is not None:
            x = self.cnn_front(x)
        # No target conditioning — sequence-only path.
        for layer in self.layers:
            x = self._layer_fwd(layer, x, mask)
        x = self.norm(x)
        h = self.mlm_norm(F.gelu(self.mlm_dense(x)))
        logits = self.mlm_decoder(h)
        return logits

    def forward_with_reg(self, ids, target_emb: Optional[torch.Tensor] = None,
                         target_mask: Optional[torch.Tensor] = None,
                         kmer_feats: Optional[torch.Tensor] = None):
        """Returns (L2-normalized projection, sigmoid-bounded scalar round_frac).
        Only valid when regression_head=True.
        """
        pooled = self.encode(ids, target_emb=target_emb, target_mask=target_mask,
                              kmer_feats=kmer_feats)
        z = F.normalize(self.proj(pooled), dim=-1)
        r = torch.sigmoid(self.reg_head(pooled).squeeze(-1))  # (B,)
        return z, r


def info_nce_loss(anchors: torch.Tensor,
                  positives: torch.Tensor,
                  negatives: torch.Tensor,
                  temperature: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE with explicit negative pool (round-0 reads) + in-batch negatives.

    For anchor i:
      - the true positive is positives[i]
      - false positives are positives[j] for j != i (other anchors' positives)
      - explicit negatives are the round-0 pool (negatives[k])
    Computes loss in both directions (anchor → positive and positive → anchor)
    and averages.
    """
    B = anchors.size(0)
    device = anchors.device
    pos_logits = anchors @ positives.t() / temperature        # (B, B)
    neg_logits = anchors @ negatives.t() / temperature        # (B, N_neg)
    pos2_logits = positives @ anchors.t() / temperature
    neg2_logits = positives @ negatives.t() / temperature

    targets = torch.arange(B, device=device)
    loss_a = F.cross_entropy(torch.cat([pos_logits, neg_logits], dim=1), targets)
    loss_p = F.cross_entropy(torch.cat([pos2_logits, neg2_logits], dim=1), targets)
    return 0.5 * (loss_a + loss_p)


if __name__ == '__main__':
    enc = AptamerEncoder(grad_checkpoint=True)
    n = sum(p.numel() for p in enc.parameters())
    print(f'AptamerEncoder: {n/1e6:.2f}M params')
    fake_ids = torch.randint(0, N_TOKENS, (8, 64))
    z = enc(fake_ids)
    print(f'forward: {z.shape}, norm={z.norm(dim=-1).mean():.3f}')
