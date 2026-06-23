"""Tumbleweed-Hybrid: joint masked-diffusion denoiser + contrastive scorer.

Architecture (from-scratch, ~50M params @ d=768/L=8):

  Inputs:
    input_ids      (B, L)        masked aptamer sequence (some positions = MASK_ID)
    target_emb     (B, 1280)     frozen ESM-2 mean-pool of target protein
    t              (B,)          mask ratio in [0,1] — equivalently the diffusion timestep
                                 (t=1 → fully masked / r0; t=0 → clean / r_max)
  Outputs:
    logits         (B, L, V)     per-position vocab distribution (denoiser)
    proj           (B, embed_dim) L2-normed sequence embedding (contrastive)

Conditioning:
  - target_emb is projected to d_model and PREPENDED as a context token (position 0
    after chemistry token, so layout: [CHEM][TGT_PROJ][CLEAN][...])
  - t is sinusoidally embedded and ADDED to all positions (channel-broadcast)
  - chemistry is already encoded in input_ids (RNA_TOK_ID / DNA_TOK_ID at position 0)

Training (see train_diffusion_hybrid.py):
  for batch (clean_seq, target_emb, selex_round, R_max, chem):
      t = 1 - selex_round / R_max      # round-derived noise level
      mask_mask = random positions with prob t
      noisy = clean_seq.where(~mask_mask, MASK_ID)
      logits, proj = model(noisy, target_emb, t)
      L_denoise   = CE(logits[mask_mask], clean_seq[mask_mask]) * (1/t).clamp(1/300, 1)
      L_contrast  = InfoNCE(proj, target_proj)   # target_emb projected through small head
      L = L_contrast + λ_diff * L_denoise

Sampling (in-silico SELEX):
  Start from x_T = all MASK, t=1. Iteratively:
    1. predict logits, sample top-K confident positions to unmask
    2. mask remaining; advance t
  At inference, can add classifier guidance: gradient from contrastive head ∂score/∂logits
  steers unmasking toward higher binder score.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aptamer_dataset import N_TOKENS, PAD_ID, MASK_ID
from aptamer_encoder import SinusoidalPosEnc, CNNFrontEnd

# Token-id → base for in-model kmer decoding (matches train_contrastive ID2BASE).
_ID2BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 8: 'T'}


def _ids_to_seqs(ids: torch.Tensor) -> list[str]:
    """Decode (B, L) input_ids → A/C/G/T strings (U→T), dropping non-base tokens."""
    out = []
    for row in ids.detach().cpu().numpy():
        out.append(''.join(_ID2BASE[v] for v in row if v in _ID2BASE).replace('U', 'T'))
    return out


class TimeEmbed(nn.Module):
    """Sinusoidal embedding of continuous t∈[0,1], projected to d_model."""
    def __init__(self, d_model: int, n_freqs: int = 64):
        super().__init__()
        self.n_freqs = n_freqs
        self.proj = nn.Sequential(
            nn.Linear(2 * n_freqs, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) → (B, d_model)
        freqs = torch.exp(torch.linspace(0, math.log(1000.), self.n_freqs,
                                          device=t.device, dtype=t.dtype))
        ang = t.unsqueeze(-1) * freqs.unsqueeze(0)         # (B, n_freqs)
        sincos = torch.cat([ang.sin(), ang.cos()], dim=-1)  # (B, 2*n_freqs)
        return self.proj(sincos)                            # (B, d_model)


class AptamerDiffusionHybrid(nn.Module):
    """Joint contrastive + masked-discrete-diffusion model.

    Defaults sized for ~50M params (8 layers × 768 hidden = ~47M).
    Adjust d_model/num_layers to scale.
    """
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 8,
        dim_ff: int = 3072,
        dropout: float = 0.1,
        embed_dim: int = 256,                 # contrastive projection dim
        max_len: int = 128,
        target_embed_dim: int = 1280,         # ESM-2 650M
        use_target_token: bool = True,        # prepend projected target as position-0 token
        grad_checkpoint: bool = False,
        cnn_kernels=None,                      # v15-style motif CNN front-end on input embeds (helps RNA)
        kmer_token_dim: int = 0,               # v18-style kmer features added to pooled contrastive vec (helps DNA)
        target_pool: str = 'mean',             # 'mean' = legacy single-vec ESM-2; 'attn' = learnable-query
                                               #   attention over PER-RESIDUE ESM-2 (epitope-aware, v4). When 'attn'
                                               #   the model expects target_emb shape (B, P, D) + target_mask (B, P).
        attn_pool_heads: int = 8,              # heads for the epitope attention-pool
        target_film: bool = False,             # FiLM: per-layer γ/β from the target vector modulate the
                                               #   denoiser trunk so the denoise loss MUST flow through the
                                               #   target pathway (fixes denoiser ignoring the prepend token)
        target_xattn: bool = False,            # v6: per-layer CROSS-ATTENTION from sequence tokens (queries)
                                               #   to PER-RESIDUE target ESM-2 (K,V). Richer than FiLM (each
                                               #   seq position attends to specific epitope residues) and
                                               #   richer than attn-pool (no collapse to one vector). Expects
                                               #   target_emb (B,P,D) + target_mask (B,P). Zero-init gate →
                                               #   residual-safe no-op at start.
        trifp_dim: int = 0,                    # v9: dimension of the TriFP prediction-fingerprint (predicted
                                               #   log10 Kd over a fixed reference aptamer panel = a dense,
                                               #   BINDING-aware target rep). When >0, it is projected to
                                               #   d_model and ADDED (via a zero-init gate) to the target rep
                                               #   that feeds BOTH the prepend token and the FiLM source — so
                                               #   the denoiser is conditioned on "what kind of aptamers bind
                                               #   this target," not just raw ESM-2. Zero gate → no-op at start.
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_len = max_len
        self.use_target_token = use_target_token
        self.grad_checkpoint = grad_checkpoint
        self.kmer_token_dim = kmer_token_dim
        self.target_pool = target_pool
        self.target_film = target_film
        self.target_xattn = target_xattn
        self.trifp_dim = trifp_dim
        self.nhead = nhead

        # Epitope-aware attention-pool: a learnable query attends over the protein's per-residue
        # ESM-2 embeddings → one (D,) conditioning vector. Replaces the fixed mean-pool that
        # can't discriminate same-family targets (CXCL5 vs IL-8 cos 0.972). Pools BEFORE the
        # existing target_proj / target_contrast heads, so the rest of the model is unchanged.
        if target_pool == 'attn':
            self.pool_query = nn.Parameter(torch.randn(1, 1, target_embed_dim) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                target_embed_dim, attn_pool_heads, dropout=dropout, batch_first=True)
            self.pool_norm = nn.LayerNorm(target_embed_dim)

        # Token + positional
        self.embed = nn.Embedding(N_TOKENS, d_model, padding_idx=PAD_ID)
        # +1 position for the prepended target token
        self.pos = SinusoidalPosEnc(d_model, max_len=max_len + 16)

        # CNN motif front-end (v15). Runs inside _trunk, so the denoiser sees it on
        # MASKED input while the contrastive path sees it on CLEAN input (t=0) via a
        # separate forward — no train/inference distribution shift (the v2 bug).
        # Residual+LayerNorm → ≈no-op when untrained. None → disabled.
        self.cnn_front = CNNFrontEnd(d_model, tuple(cnn_kernels), dropout=dropout) \
            if cnn_kernels else None

        # Target projection (ESM-2 → d_model)
        self.target_proj = nn.Sequential(
            nn.Linear(target_embed_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.null_target = nn.Parameter(torch.randn(target_embed_dim) * 0.02)

        # FiLM conditioning: the projected target rep → per-layer (γ, β). Modulates the trunk
        # hidden states after every transformer layer (h ← γ⊙h + β), forcing the denoiser output
        # to depend on the target. Zero-init → γ=1, β=0 at start (residual-safe no-op).
        if target_film:
            self.film = nn.Linear(d_model, 2 * num_layers * d_model)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

        # TriFP prediction-fingerprint conditioning (v9): project the |panel|-dim predicted-Kd
        # vector → d_model and ADD it to the target rep via a zero-init scalar gate (residual-safe
        # no-op at start, same pattern as the xattn gate). Because the sum feeds tgt_rep, the
        # binding-aware signal modulates the denoiser through BOTH the prepend token and FiLM.
        if trifp_dim > 0:
            self.trifp_proj = nn.Sequential(
                nn.Linear(trifp_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.trifp_gate = nn.Parameter(torch.zeros(1))

        # Cross-attention conditioning (v6): per-layer MHA where the SEQUENCE attends to the
        # protein's PER-RESIDUE ESM-2 (projected to d_model). Each layer gets a learnable scalar
        # gate, zero-init → the cross-attn output is a no-op at start (residual-safe, like FiLM).
        # Inspired by the AR→Diffusion KV-injection in NVIDIA Cosmos: inject context as K,V into
        # every diffusion-branch attention block rather than as a single prepended token.
        if target_xattn:
            self.xattn_kv = nn.Linear(target_embed_dim, d_model)
            self.xattn = nn.ModuleList([
                nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ])
            self.xattn_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
            self.xattn_gate = nn.Parameter(torch.zeros(num_layers))

        # Time embedding
        self.time_embed = TimeEmbed(d_model)

        # Transformer blocks (pre-norm for stability with denoising)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Diffusion denoiser head (per-position vocab logits)
        self.denoise_dense = nn.Linear(d_model, d_model)
        self.denoise_norm = nn.LayerNorm(d_model)
        self.denoise_bias = nn.Parameter(torch.zeros(N_TOKENS))

        # kmer features (v18). Computed from the CLEAN sequence and ADDED to the pooled
        # contrastive vector only — it never enters the transformer or the denoiser head,
        # so it cannot leak clean-sequence info into masked-token prediction.
        if kmer_token_dim > 0:
            self.kmer_proj = nn.Sequential(
                nn.Linear(kmer_token_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )

        # Contrastive projection head (pooled seq embedding)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, embed_dim),
        )
        # Target embedding projection for contrastive (separate from token-prepend proj)
        self.target_contrast = nn.Sequential(
            nn.Linear(target_embed_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, embed_dim),
        )

    def _pool_target(self, target_emb: torch.Tensor | None,
                     target_mask: torch.Tensor | None = None) -> torch.Tensor | None:
        """Collapse the raw target conditioning to a single (B, target_embed_dim) vector.

        - 'mean' pool (legacy): target_emb is already (B, D) → returned unchanged.
        - 'attn' pool (epitope-aware): target_emb is (B, P, D) per-residue ESM-2; a learnable
          query attends over the P residues (ignoring padding via target_mask) → (B, D).
        Tolerates a (B, D) input even in 'attn' mode (e.g. a pre-pooled probe) by passing through.
        Returns None when target_emb is None (null-conditioning path handled downstream)."""
        if target_emb is None:
            return None
        if target_emb.dim() == 2:
            return target_emb
        if self.target_pool == 'attn':
            # (B, P, D) → attention-pool. target_mask: True = valid residue; MHA wants True = ignore.
            q = self.pool_query.expand(target_emb.size(0), -1, -1).to(dtype=target_emb.dtype)
            key_pad = (~target_mask) if target_mask is not None else None
            pooled, _ = self.pool_attn(q, target_emb, target_emb, key_padding_mask=key_pad,
                                       need_weights=False)
            return self.pool_norm(pooled.squeeze(1))   # (B, D)
        # Per-residue input but no learnable pool (e.g. cross-attn mode): masked MEAN over residues
        # → a single (B, D) vector for the prepend token / FiLM source / contrastive head. The
        # full per-residue tensor is consumed separately by the cross-attention path in _trunk.
        if target_mask is not None:
            m = target_mask.unsqueeze(-1).to(target_emb.dtype)
            return (target_emb * m).sum(1) / m.sum(1).clamp_min(1e-6)
        return target_emb.mean(1)

    def _make_target_token(self, target_emb: torch.Tensor | None, B: int, device, dtype):
        if target_emb is None:
            target_emb = self.null_target.unsqueeze(0).expand(B, -1).to(device=device, dtype=dtype)
        return self.target_proj(target_emb)   # (B, d_model)

    def _layer_fwd(self, layer, x, key_padding_mask):
        return layer(x, src_key_padding_mask=key_padding_mask)

    def _trunk(self, ids: torch.Tensor, target_emb: torch.Tensor | None, t: torch.Tensor,
               xattn_kv: torch.Tensor | None = None, xattn_mask: torch.Tensor | None = None,
               trifp_fp: torch.Tensor | None = None):
        """Shared embed → CNN → pos → time → target-token → transformer → norm trunk.
        Returns (x_seq, seq_pad) with the prepended target token stripped.
        The CNN runs on whatever `ids` are passed; callers route MASKED ids here for the
        denoiser and CLEAN ids (t=0) for the contrastive path so the CNN never has a
        train/inference distribution shift (the v2 bug)."""
        B, L = ids.shape
        device, dtype = ids.device, self.embed.weight.dtype

        x = self.embed(ids)                                          # (B, L, d)
        if self.cnn_front is not None:
            x = self.cnn_front(x)
        x = self.pos(x)

        t_emb = self.time_embed(t).unsqueeze(1)                       # (B, 1, d)
        x = x + t_emb

        # Target rep (also the FiLM source). Always computed when a target token or FiLM is used.
        tgt_rep = None
        if self.use_target_token or self.target_film:
            tgt_rep = self._make_target_token(target_emb, B, device, dtype)   # (B, d)
            if self.trifp_dim > 0 and trifp_fp is not None:
                tgt_rep = tgt_rep + self.trifp_gate * self.trifp_proj(trifp_fp)

        if self.use_target_token:
            x = torch.cat([tgt_rep.unsqueeze(1), x], dim=1)           # (B, L+1, d)
            pad_mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=device),
                (ids == PAD_ID),
            ], dim=1)
        else:
            pad_mask = (ids == PAD_ID)

        film_g = film_b = None
        if self.target_film:
            gb = self.film(tgt_rep).view(B, self.num_layers, 2, self.d_model)
            film_g = 1.0 + gb[:, :, 0]                                # (B, num_layers, d) → γ≈1 at init
            film_b = gb[:, :, 1]                                       # β≈0 at init

        xattn_pad = (~xattn_mask) if (self.target_xattn and xattn_mask is not None) else None
        do_xattn = self.target_xattn and xattn_kv is not None

        for i, layer in enumerate(self.layers):
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._layer_fwd, layer, x, pad_mask, use_reentrant=False)
            else:
                x = self._layer_fwd(layer, x, pad_mask)
            if self.target_film:
                x = film_g[:, i].unsqueeze(1) * x + film_b[:, i].unsqueeze(1)
            if do_xattn:
                q = self.xattn_norm[i](x)
                ca, _ = self.xattn[i](q, xattn_kv, xattn_kv,
                                      key_padding_mask=xattn_pad, need_weights=False)
                x = x + self.xattn_gate[i] * ca
        x = self.norm(x)

        if self.use_target_token:
            x_seq = x[:, 1:]
            seq_pad = (ids == PAD_ID)
        else:
            x_seq = x
            seq_pad = pad_mask
        return x_seq, seq_pad

    def forward(
        self,
        input_ids: torch.Tensor,              # (B, L) MASKED ids (denoiser input)
        target_emb: torch.Tensor | None,      # (B, D) mean-pool OR (B, P, D) per-residue (attn pool); None → null
        t: torch.Tensor,                       # (B,) ∈ [0,1]
        want_denoise: bool = True,
        want_proj: bool = True,
        kmer_feats: torch.Tensor | None = None,   # (B, kmer_token_dim) from CLEAN seq; proj head only
        clean_ids: torch.Tensor | None = None,    # (B, L) CLEAN ids for the contrastive trunk
        target_mask: torch.Tensor | None = None,  # (B, P) True=valid residue, for 'attn' target_pool
        trifp_fp: torch.Tensor | None = None,     # (B, trifp_dim) TriFP prediction-fingerprint (v9)
    ):
        out = {}
        x_seq_d = seq_pad_d = None

        # Collapse target conditioning to one (B, D) vector up front (mean = passthrough,
        # attn = epitope attention-pool over per-residue ESM-2). Used by both heads.
        tgt_vec = self._pool_target(target_emb, target_mask)

        # Cross-attention KV (v6): project per-residue ESM-2 → d_model once; reused by every
        # trunk call. None when no per-residue target is supplied (null path / mean-pool input).
        xattn_kv = xattn_mask = None
        if self.target_xattn and target_emb is not None and target_emb.dim() == 3:
            xattn_kv = self.xattn_kv(target_emb)        # (B, P, d)
            xattn_mask = target_mask

        # Denoiser: trunk on the MASKED input at its sampled noise level t.
        if want_denoise:
            x_seq_d, seq_pad_d = self._trunk(input_ids, tgt_vec, t, xattn_kv, xattn_mask, trifp_fp)
            h = self.denoise_dense(x_seq_d)
            h = F.gelu(h)
            h = self.denoise_norm(h)
            logits = h @ self.embed.weight.T + self.denoise_bias       # (B, L, V)
            out['logits'] = logits

        # Contrastive pooled vector. Three modes:
        #  - clean_ids given (v3): SEPARATE trunk on CLEAN seq at t=0 → CNN-consistent.
        #  - else reuse the noisy denoiser forward at round-t (v1): the round-noise on the
        #    contrastive representation is a beneficial SELEX-matched augmentation.
        #  - encode(): want_denoise=False, input_ids already clean, t supplied → fresh trunk.
        if want_proj:
            if clean_ids is not None:
                t0 = torch.zeros(clean_ids.size(0), device=clean_ids.device, dtype=t.dtype)
                xs, sp = self._trunk(clean_ids, tgt_vec, t0, xattn_kv, xattn_mask, trifp_fp)
            elif x_seq_d is not None:
                xs, sp = x_seq_d, seq_pad_d
            else:
                xs, sp = self._trunk(input_ids, tgt_vec, t, xattn_kv, xattn_mask, trifp_fp)
            not_pad = (~sp).float().unsqueeze(-1)
            pooled = (xs * not_pad).sum(dim=1) / not_pad.sum(dim=1).clamp_min(1e-6)
            if self.kmer_token_dim > 0 and kmer_feats is not None:
                pooled = pooled + self.kmer_proj(kmer_feats)
            # pre-projection mean-pooled TRUNK hidden. Trained by the denoising objective, so it
            # is a valid representation even when the contrastive proj_head is ablated off
            # (lam_contrast=0) and never trained. KdBench ranks the nc generator off this.
            out['pooled'] = pooled
            proj = F.normalize(self.proj_head(pooled), dim=-1)
            out['proj'] = proj
            if tgt_vec is not None:
                out['target_proj'] = F.normalize(self.target_contrast(tgt_vec), dim=-1)

        return out

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, target_emb: torch.Tensor | None = None,
               kmer_feats: torch.Tensor | None = None,
               target_mask: torch.Tensor | None = None,
               representation: str = 'proj'):
        """Inference helper for kdbench scoring (matches AptamerEncoder.encode signature).

        If the model uses kmer features and none are supplied, they are computed from
        input_ids here so callers (e.g. the LOO probe) need no extra plumbing.

        representation='proj' (default) returns the L2-normalized contrastive projection
        (valid only when the model was trained with the contrastive term). 'pooled' returns
        the pre-projection mean-pooled trunk hidden, which is trained by the denoising
        objective and is the correct ranker for contrastive-free (lam_contrast=0) models.
        """
        if self.kmer_token_dim > 0 and kmer_feats is None:
            from kmer_features import build_kmer_features
            feats = build_kmer_features(_ids_to_seqs(input_ids))
            kmer_feats = torch.as_tensor(feats, dtype=self.embed.weight.dtype,
                                         device=input_ids.device)
        t = torch.zeros(input_ids.size(0), device=input_ids.device)
        out = self.forward(input_ids, target_emb, t, want_denoise=False, want_proj=True,
                           kmer_feats=kmer_feats, target_mask=target_mask)
        return out['pooled'] if representation == 'pooled' else out['proj']


# ─── Loss helpers ─────────────────────────────────────────────────────────────

def denoise_loss(
    logits: torch.Tensor,        # (B, L, V)
    targets: torch.Tensor,       # (B, L) ground-truth token ids
    mask_positions: torch.Tensor,  # (B, L) bool — True where token was masked
    t: torch.Tensor,             # (B,) mask ratio used per batch
    weight_clamp: tuple[float, float] = (1/300, 1.0),
) -> torch.Tensor:
    """EvoFlow-style weighted CE: per-sample weight = 1/mask_ratio, clamped."""
    B, L, V = logits.shape
    # Per-sample weights
    w = (1.0 / t.clamp_min(1e-4)).clamp(weight_clamp[0], weight_clamp[1])  # (B,)
    # Token-level CE on masked positions only
    logp = F.log_softmax(logits, dim=-1)                                    # (B, L, V)
    nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)                # (B, L)
    nll = nll * mask_positions.float()                                       # zero out unmasked
    nll_per_sample = nll.sum(-1) / mask_positions.float().sum(-1).clamp_min(1.0)  # (B,)
    return (w * nll_per_sample).mean()


def contrast_loss(
    seq_proj: torch.Tensor,        # (B, embed_dim) L2-normed
    target_proj: torch.Tensor,     # (B, embed_dim) L2-normed
    temperature: float = 0.07,
    same_target_mask: torch.Tensor | None = None,   # (B, B) bool, True where rows share a target
) -> torch.Tensor:
    """InfoNCE: within-batch (sequence, target) pairs are positive; off-diagonals neg.

    Same loss family as aptamer_encoder.info_nce_loss but written as cross-batch
    bidirectional symmetry.

    `same_target_mask`: when several rows in a batch share the SAME target (unavoidable when
    a few targets dominate the corpus, e.g. the 6-target SELEX set), their off-diagonal entries
    are FALSE negatives — standard InfoNCE then cannot drop below ln(n_distinct_targets) and is
    actively frustrated. Masking those entries to -inf (supervised-InfoNCE) removes the false
    negatives; the diagonal (true positive) is always kept.
    """
    logits = seq_proj @ target_proj.T / temperature                          # (B, B)
    if same_target_mask is not None:
        B = seq_proj.size(0)
        off_diag_same = same_target_mask & ~torch.eye(B, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(off_diag_same, float('-inf'))
    labels = torch.arange(seq_proj.size(0), device=seq_proj.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def round_to_t(round_idx: torch.Tensor, max_round: torch.Tensor) -> torch.Tensor:
    """Map (round_idx, max_round) → t ∈ [0,1].
       r=0       → t=1.0   (fully noisy)
       r=R_max   → t=0.0   (clean binder)
    """
    return (1.0 - round_idx.float() / max_round.float().clamp_min(1.0)).clamp(0.0, 1.0)


# ─── Param count ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B, L = 4, 64
    ids = torch.randint(0, N_TOKENS, (B, L))
    t = torch.rand(B)

    # mean-pool (legacy)
    m = AptamerDiffusionHybrid()
    n = sum(p.numel() for p in m.parameters())
    print(f'[mean] params: {n/1e6:.1f}M')
    out = m(ids, torch.randn(B, 1280), t)
    print(f'  logits {out["logits"].shape}  proj {out["proj"].shape}  target_proj {out["target_proj"].shape}')

    # attn-pool (epitope-aware, v4): per-residue (B, P, D) + mask
    ma = AptamerDiffusionHybrid(target_pool='attn')
    na = sum(p.numel() for p in ma.parameters())
    print(f'[attn] params: {na/1e6:.1f}M  (+{(na-n)/1e6:.1f}M for attention-pool)')
    P = 50
    res = torch.randn(B, P, 1280)
    mask = torch.ones(B, P, dtype=torch.bool); mask[0, 30:] = False  # ragged protein
    out = ma(ids, res, t, target_mask=mask)
    print(f'  logits {out["logits"].shape}  proj {out["proj"].shape}  target_proj {out["target_proj"].shape}')
    # null path still works
    out0 = ma(ids, None, t)
    print(f'  null target ok: logits {out0["logits"].shape}')

    # FiLM (v5): mean-pool + per-layer γ/β into the denoiser trunk
    mf = AptamerDiffusionHybrid(target_film=True)
    nf = sum(p.numel() for p in mf.parameters())
    print(f'[film] params: {nf/1e6:.1f}M  (+{(nf-n)/1e6:.1f}M for FiLM)')
    # At init (zero-init film → γ=1,β=0) the denoiser logits must equal a no-FiLM forward
    mf.eval()
    with torch.no_grad():
        tgt = torch.randn(B, 1280)
        lf = mf(ids, tgt, t, want_proj=False)['logits']
        # toggle film off and compare
        mf.target_film = False
        lo = mf(ids, tgt, t, want_proj=False)['logits']
        mf.target_film = True
    print(f'  FiLM no-op at init: max|Δlogits|={ (lf-lo).abs().max().item():.2e} (expect ~0)')
    outn = mf(ids, None, t)
    print(f'  FiLM null target ok: logits {outn["logits"].shape}')

    # cross-attention (v6): per-residue (B,P,D) + mask, per-layer seq→target cross-attn
    mx = AptamerDiffusionHybrid(target_xattn=True)
    nx = sum(p.numel() for p in mx.parameters())
    print(f'[xattn] params: {nx/1e6:.1f}M  (+{(nx-n)/1e6:.1f}M for cross-attn)')
    P = 50
    res = torch.randn(B, P, 1280)
    mask = torch.ones(B, P, dtype=torch.bool); mask[0, 30:] = False  # ragged protein
    mx.eval()
    with torch.no_grad():
        lx = mx(ids, res, t, want_proj=False, target_mask=mask)['logits']
        mx.target_xattn = False
        lo2 = mx(ids, res, t, want_proj=False, target_mask=mask)['logits']
        mx.target_xattn = True
    print(f'  xattn no-op at init (zero gate): max|Δlogits|={ (lx-lo2).abs().max().item():.2e} (expect ~0)')
    outx = mx(ids, res, t, target_mask=mask)
    print(f'  logits {outx["logits"].shape}  proj {outx["proj"].shape}  target_proj {outx["target_proj"].shape}')
    out0x = mx(ids, None, t)
    print(f'  xattn null target ok: logits {out0x["logits"].shape}')

    # TriFP-fingerprint conditioning (v9): mean-pool + FiLM + a panel-dim fingerprint fused into
    # the target rep via a zero-init gate. At init the gate=0 → must equal the no-fingerprint forward.
    TRIFP_DIM = 812
    mt = AptamerDiffusionHybrid(target_film=True, trifp_dim=TRIFP_DIM)
    nt = sum(p.numel() for p in mt.parameters())
    print(f'[trifp] params: {nt/1e6:.1f}M  (+{(nt-nf)/1e6:.1f}M for TriFP-FiLM over v5_film)')
    mt.eval()
    with torch.no_grad():
        tgt = torch.randn(B, 1280); fp = torch.randn(B, TRIFP_DIM)
        l_fp = mt(ids, tgt, t, want_proj=False, trifp_fp=fp)['logits']
        l_no = mt(ids, tgt, t, want_proj=False, trifp_fp=None)['logits']
    print(f'  TriFP no-op at init (zero gate): max|Δlogits|={ (l_fp-l_no).abs().max().item():.2e} (expect ~0)')
    outt = mt(ids, tgt, t, trifp_fp=fp)
    print(f'  logits {outt["logits"].shape}  proj {outt["proj"].shape}  target_proj {outt["target_proj"].shape}')
    out0t = mt(ids, None, t, trifp_fp=fp)
    print(f'  TriFP null target ok: logits {out0t["logits"].shape}')
