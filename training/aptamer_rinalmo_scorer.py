"""AptamerRinalmoScorer: v10 heads on a FROZEN RiNALMo-giga backbone.

Motivation
----------
RiNALMo-giga (650M RNA masked-LM) is our strongest RNA sequence encoder, but raw
RiNALMo + RidgeCV is a generic regressor with NO chemistry awareness, NO target
conditioning, and NO round/enrichment signal. v10's *heads* add exactly those:

  - chemistry token       ([RNA] / [DNA]) — RiNALMo is RNA-only, so DNA is folded
                          T→U upstream; the chem token lets the head still flag the
                          modality (carries over to the DNA Kd lane).
  - ESM-2 FiLM target     per-block (γ,β) generated from the mean-pool ESM-2 target
    conditioning          embedding modulate the sequence rep → the score depends on
                          *which protein* we are ranking against.
  - contrastive round     InfoNCE between the (FiLM-modulated) sequence rep and the
    teacher               ESM-2 target rep, with the supervised same-target mask
                          (drop within-batch false negatives) + round-derived
                          weighting (late SELEX rounds = enriched binders).

This is the SCORER-FIRST variant: RiNALMo is FROZEN and its pooled embedding is
PRECOMPUTED once (see scripts/precompute_rinalmo_embeddings.py). All trainable
params live in the heads (~a few M), so the checkpoint is single-digit MB and head
training is fast. RiNALMo's 650M weights are NEVER trained and NOT in the ckpt —
load the public weights at embed time.

Inputs (training, on the cached pooled embeddings):
    rinalmo_emb   (B, rinalmo_dim)   frozen RiNALMo mean-pool of the aptamer
    target_emb    (B, target_dim)    frozen ESM-2 mean-pool of the target protein
    chem          (B,)               0 = RNA, 1 = DNA
    t             (B,)               round-derived ∈ [0,1] (1=r0 noisy, 0=r_max clean)
Outputs:
    proj          (B, embed_dim)     L2-normed aptamer embedding (contrastive)
    target_proj   (B, embed_dim)     L2-normed target embedding (contrastive)

The contrastive heads + loss helpers are the SAME family as
aptamer_diffusion_hybrid (proj_head / target_contrast / contrast_loss / round_to_t),
so the Kd-ranking probe scores this model identically: cos(proj, target_proj).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the exact InfoNCE + round→t helpers so the "round teacher" is identical to v10
from aptamer_diffusion_hybrid import contrast_loss, round_to_t  # noqa: F401


class TimeEmbed(nn.Module):
    """Sinusoidal embedding of continuous t∈[0,1] → d_model (round-awareness)."""
    def __init__(self, d_model: int, n_freqs: int = 64):
        super().__init__()
        self.n_freqs = n_freqs
        self.proj = nn.Sequential(
            nn.Linear(2 * n_freqs, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        import math
        freqs = torch.exp(torch.linspace(0, math.log(1000.), self.n_freqs,
                                          device=t.device, dtype=t.dtype))
        ang = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return self.proj(torch.cat([ang.sin(), ang.cos()], dim=-1))


class AptamerRinalmoScorer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        embed_dim: int = 256,
        rinalmo_dim: int = 1280,          # RiNALMo-giga hidden
        target_embed_dim: int = 1280,     # ESM-2 650M mean-pool
        n_film_blocks: int = 2,           # FiLM-modulated residual dense blocks on the seq rep
        dropout: float = 0.1,
        target_film: bool = True,         # FiLM target conditioning (the v5_film win, ported)
        use_chem: bool = True,            # [RNA]/[DNA] chemistry embedding (v10)
        use_time: bool = True,            # round-derived t conditioning (round teacher)
    ):
        super().__init__()
        self.d_model = d_model
        self.n_film_blocks = n_film_blocks
        self.target_film = target_film
        self.use_chem = use_chem
        self.use_time = use_time

        # frozen-RiNALMo pooled vec → d_model adapter
        self.seq_proj = nn.Sequential(
            nn.Linear(rinalmo_dim, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )
        if use_chem:
            self.chem_embed = nn.Embedding(2, d_model)   # 0=RNA, 1=DNA
            nn.init.normal_(self.chem_embed.weight, std=0.02)
        if use_time:
            self.time_embed = TimeEmbed(d_model)

        # FiLM source: ESM-2 target → d_model
        self.target_proj = nn.Sequential(
            nn.Linear(target_embed_dim, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )
        self.null_target = nn.Parameter(torch.randn(target_embed_dim) * 0.02)

        # residual dense blocks modulated by per-block FiLM (γ,β) from the target rep
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            ) for _ in range(n_film_blocks)
        ])
        self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_film_blocks)])
        if target_film:
            self.film = nn.Linear(d_model, 2 * n_film_blocks * d_model)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)   # zero-init → γ=1, β=0 (residual-safe no-op)

        # contrastive heads (same shape/role as aptamer_diffusion_hybrid)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, embed_dim),
        )
        self.target_contrast = nn.Sequential(
            nn.Linear(target_embed_dim, d_model), nn.GELU(), nn.Linear(d_model, embed_dim),
        )

    def _seq_rep(self, rinalmo_emb, target_emb, chem, t):
        B = rinalmo_emb.size(0)
        device, dtype = rinalmo_emb.device, self.seq_proj[0].weight.dtype

        # target rep (FiLM source); null vector when no target supplied
        if target_emb is None:
            target_emb = self.null_target.unsqueeze(0).expand(B, -1).to(device=device, dtype=dtype)
        tgt_rep = self.target_proj(target_emb)            # (B, d)

        h = self.seq_proj(rinalmo_emb)                    # (B, d)
        if self.use_chem and chem is not None:
            h = h + self.chem_embed(chem)
        if self.use_time and t is not None:
            h = h + self.time_embed(t)

        film_g = film_b = None
        if self.target_film:
            gb = self.film(tgt_rep).view(B, self.n_film_blocks, 2, self.d_model)
            film_g = 1.0 + gb[:, :, 0]
            film_b = gb[:, :, 1]

        for i, (blk, ln) in enumerate(zip(self.blocks, self.block_norms)):
            h = h + blk(ln(h))                            # pre-norm residual block
            if self.target_film:
                h = film_g[:, i] * h + film_b[:, i]       # per-block FiLM
        return h, target_emb

    def forward(self, rinalmo_emb, target_emb, chem=None, t=None):
        h, tgt_full = self._seq_rep(rinalmo_emb, target_emb, chem, t)
        out = {'proj': F.normalize(self.proj_head(h), dim=-1)}
        if tgt_full is not None:
            out['target_proj'] = F.normalize(self.target_contrast(tgt_full), dim=-1)
        return out

    @torch.no_grad()
    def encode(self, rinalmo_emb, target_emb=None, chem=None, t=None):
        """Inference helper for Kd-ranking probes: returns the L2-normed aptamer proj.
        t defaults to 0 (clean / r_max binder)."""
        if t is None:
            t = torch.zeros(rinalmo_emb.size(0), device=rinalmo_emb.device)
        return self.forward(rinalmo_emb, target_emb, chem=chem, t=t)['proj']


if __name__ == '__main__':
    B = 8
    m = AptamerRinalmoScorer()
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'AptamerRinalmoScorer trainable params: {n_train/1e6:.2f}M (RiNALMo 650M frozen, not counted)')

    rin = torch.randn(B, 1280)
    tgt = torch.randn(B, 1280)
    chem = torch.randint(0, 2, (B,))
    t = torch.rand(B)
    out = m(rin, tgt, chem, t)
    print(f'  proj {out["proj"].shape}  target_proj {out["target_proj"].shape}')

    # FiLM no-op at init: zero-init film → the modulated rep equals the un-modulated rep
    m.eval()
    with torch.no_grad():
        h1, _ = m._seq_rep(rin, tgt, chem, t)
        m.target_film = False
        h0, _ = m._seq_rep(rin, tgt, chem, t)
        m.target_film = True
    print(f'  FiLM no-op at init: max|Δ|={ (h1-h0).abs().max().item():.2e} (expect ~0)')

    # null-target path
    out0 = m(rin, None, chem, t)
    print(f'  null target ok: proj {out0["proj"].shape}, has target_proj={"target_proj" in out0}')
