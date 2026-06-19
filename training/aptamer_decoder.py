"""Autoregressive decoder that maps an AptamerEncoder pooled embedding back to
sequence tokens. Used for latent-BO generation:

    1. Encode a known binder → embedding e
    2. Optimize e in latent space (HC-HEBO etc.) toward higher "binder score"
    3. Decode the optimized e* → generated aptamer sequence

Architecture: 2-layer transformer decoder. Cross-attention queries the encoder's
pooled embedding (projected to d_model_dec). Self-attention is causal over the
generated tokens. Vocab is A/C/G/U + special BOS/EOS/PAD tokens.

Tokens:
    0..3 = A/C/G/U (same IDs as encoder for consistency)
    4    = PAD
    5    = MASK (unused by decoder, kept for vocab alignment)
    6    = BOS  (start-of-sequence, decoder-only)
    7    = EOS  (end-of-sequence, decoder-only)

We extend the encoder's 6-token vocab to 8 here. The encoder ignores BOS/EOS;
the decoder needs them for autoregressive boundary signals.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder vocab is 6 (A/C/G/U/PAD/MASK). We add BOS/EOS for the decoder.
DEC_VOCAB = 8
BOS_ID = 6
EOS_ID = 7
ENC_PAD_ID = 4
ENC_VOCAB = 6


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


class AptamerDecoder(nn.Module):
    """Autoregressive decoder conditioned on the encoder's pooled embedding.

    Args:
        encoder_d_model: hidden dim of the pretrained encoder (384 for 14M)
        d_model: hidden dim of the decoder (default 256 — smaller than encoder)
        nhead: decoder attention heads (default 4)
        num_layers: decoder transformer layers (default 2)
        dim_ff: feedforward dim (default 1024)
        max_len: max sequence length (default 128, matches encoder)
        dropout: dropout rate (default 0.1)
    """

    def __init__(self,
                 encoder_d_model: int = 384,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_ff: int = 1024,
                 max_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Project encoder embedding to decoder d_model
        self.context_proj = nn.Linear(encoder_d_model, d_model)
        # Token embedding (8 tokens: ACGU + PAD + MASK + BOS + EOS)
        self.tok_embed = nn.Embedding(DEC_VOCAB, d_model, padding_idx=ENC_PAD_ID)
        self.pos = SinusoidalPosEnc(d_model, max_len=max_len + 16)
        # Standard transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, DEC_VOCAB)

    def forward(self, encoder_emb: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced training forward.
        Args:
            encoder_emb: [B, encoder_d_model] pooled encoder hidden
            decoder_input_ids: [B, L] target tokens shifted right (BOS-prefixed)
        Returns:
            logits: [B, L, DEC_VOCAB]
        """
        B, L = decoder_input_ids.shape
        # Context: single-token KV from the projected encoder embedding
        ctx = self.context_proj(encoder_emb).unsqueeze(1)  # [B, 1, d_model]
        # Embed decoder tokens
        x = self.tok_embed(decoder_input_ids)
        x = self.pos(x)
        # Causal mask
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        # Standard transformer decoder: self-attention over x with causal mask;
        # cross-attention over the single-token context
        out = self.decoder(tgt=x, memory=ctx, tgt_mask=causal_mask)
        out = self.norm(out)
        logits = self.head(out)  # [B, L, DEC_VOCAB]
        return logits

    @torch.no_grad()
    def generate(self, encoder_emb: torch.Tensor, max_len: int | None = None,
                 temperature: float = 1.0, sample: bool = True,
                 force_length: int | None = None) -> torch.Tensor:
        """Autoregressive generation from a given encoder embedding.
        Args:
            encoder_emb: [B, encoder_d_model]
            max_len: generation cap (default self.max_len)
            temperature: sampling temperature (1.0 = no scaling)
            sample: True = multinomial sample, False = greedy argmax
            force_length: if set, suppresses EOS for the first force_length content tokens
                          then forces EOS. Output sequences will have exactly force_length nt.
        Returns:
            token_ids: [B, generated_len] including BOS at index 0
        """
        max_len = max_len or self.max_len
        B = encoder_emb.shape[0]
        device = encoder_emb.device
        ctx = self.context_proj(encoder_emb).unsqueeze(1)
        # Start with BOS
        tokens = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        # When force_length set, generate exactly force_length content tokens then 1 EOS
        target_total = force_length + 2 if force_length is not None else max_len
        for step in range(target_total - 1):
            L = tokens.size(1)
            x = self.tok_embed(tokens)
            x = self.pos(x)
            causal_mask = torch.triu(
                torch.full((L, L), float('-inf'), device=device), diagonal=1)
            out = self.decoder(tgt=x, memory=ctx, tgt_mask=causal_mask)
            out = self.norm(out)
            logits = self.head(out[:, -1])  # [B, V]
            # Always forbid BOS, MASK, PAD in content positions
            logits[:, BOS_ID] = float('-inf')
            logits[:, 5] = float('-inf')  # MASK
            logits[:, ENC_PAD_ID] = float('-inf')
            if force_length is not None:
                pos_in_content = step  # 0-indexed content position after BOS
                if pos_in_content < force_length:
                    logits[:, EOS_ID] = float('-inf')  # forbid early EOS
                else:
                    # At exactly force_length content tokens: force EOS
                    forced = torch.full((B,), EOS_ID, dtype=torch.long, device=device)
                    forced = torch.where(finished, torch.full_like(forced, ENC_PAD_ID), forced)
                    finished = finished | (forced == EOS_ID)
                    tokens = torch.cat([tokens, forced.unsqueeze(1)], dim=1)
                    break
            if sample:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tok = logits.argmax(-1)
            # Once a row has emitted EOS, freeze it
            next_tok = torch.where(finished, torch.full_like(next_tok, ENC_PAD_ID), next_tok)
            finished = finished | (next_tok == EOS_ID)
            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            if finished.all():
                break
        return tokens


    @torch.no_grad()
    def beam_search(self, encoder_emb: torch.Tensor, beam_width: int = 10,
                    force_length: int | None = None, max_len: int | None = None,
                    length_penalty: float = 0.0) -> torch.Tensor:
        """Beam search decoding. Returns top-1 beam per input.

        Step 0: single live beam per input expands to W candidates (avoids NaN
        from -inf parent scores).
        Subsequent steps: standard W×V → top-W expansion.
        Always forbids BOS/MASK tokens in next-token logits. EOS forbidden
        before pos_in_content == force_length, forced at it.
        """
        max_len = max_len or self.max_len
        target_total = force_length + 2 if force_length is not None else max_len
        B = encoder_emb.shape[0]
        W = beam_width
        device = encoder_emb.device

        # ----- Step 0: expand from single beam per input -----
        ctx_single = self.context_proj(encoder_emb).unsqueeze(1)  # [B, 1, d]
        tokens_s = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        x = self.pos(self.tok_embed(tokens_s))
        out = self.decoder(tgt=x, memory=ctx_single,
                           tgt_mask=torch.zeros(1, 1, device=device))
        logits = self.head(self.norm(out[:, -1]))  # [B, V]
        logits[:, BOS_ID] = float('-inf')
        logits[:, 5] = float('-inf')  # MASK
        logits[:, ENC_PAD_ID] = float('-inf')
        if force_length is not None and 0 < force_length:
            logits[:, EOS_ID] = float('-inf')
        log_probs = F.log_softmax(logits, dim=-1)  # [B, V]
        V = log_probs.size(-1)
        # Step 0 can yield at most V unique candidates. If W > V, allow
        # duplicates by repeating the top-V picks to fill the W slots
        # (subsequent steps will diverge them via stochastic decode dynamics).
        k0 = min(W, V)
        top_lp_k0, top_idx_k0 = log_probs.topk(k0, dim=-1)  # [B, k0]
        if W > V:
            # Pad by repeating from top picks
            pad_lp = top_lp_k0[:, :W - V]
            pad_idx = top_idx_k0[:, :W - V]
            top_lp = torch.cat([top_lp_k0, pad_lp], dim=-1)
            top_idx = torch.cat([top_idx_k0, pad_idx], dim=-1)
        else:
            top_lp, top_idx = top_lp_k0, top_idx_k0
        tokens = torch.cat([
            tokens_s.unsqueeze(1).expand(-1, W, -1),
            top_idx.unsqueeze(-1),
        ], dim=-1).reshape(B * W, -1)
        scores = top_lp.reshape(-1)
        finished = (top_idx.reshape(-1) == EOS_ID)
        # Replicate context per beam from here on
        ctx_beam = ctx_single.unsqueeze(1).expand(-1, W, -1, -1).reshape(B * W, 1, -1).contiguous()

        # ----- Steps 1 .. target_total-2: standard beam expansion -----
        for step in range(1, target_total - 1):
            L = tokens.size(1)
            x = self.pos(self.tok_embed(tokens))
            causal_mask = torch.triu(
                torch.full((L, L), float('-inf'), device=device), diagonal=1)
            out = self.decoder(tgt=x, memory=ctx_beam, tgt_mask=causal_mask)
            logits = self.head(self.norm(out[:, -1]))  # [B*W, V]
            logits[:, BOS_ID] = float('-inf')
            logits[:, 5] = float('-inf')  # MASK
            logits[:, ENC_PAD_ID] = float('-inf')  # PAD not a content token
            if force_length is not None:
                pos_in_content = step
                if pos_in_content < force_length:
                    logits[:, EOS_ID] = float('-inf')
                else:
                    # Force EOS for live beams; PAD for already-finished
                    forced = torch.full((B * W,), EOS_ID, dtype=torch.long, device=device)
                    forced = torch.where(finished, torch.full_like(forced, ENC_PAD_ID), forced)
                    tokens = torch.cat([tokens, forced.unsqueeze(1)], dim=1)
                    break
            log_probs = F.log_softmax(logits, dim=-1)
            V = log_probs.size(-1)
            # Finished beams continue with PAD log_prob=0, all others -inf
            fin_mask = finished.unsqueeze(-1)
            fin_lp = torch.full_like(log_probs, float('-inf'))
            fin_lp[:, ENC_PAD_ID] = 0.0
            log_probs = torch.where(fin_mask, fin_lp, log_probs)
            # Per-input top-W from W*V candidates
            cand = (scores.unsqueeze(-1) + log_probs).view(B, W * V)
            # Replace any NaN (shouldn't happen now) with -inf for safety
            cand = torch.nan_to_num(cand, nan=float('-inf'))
            top_scores, top_idx = cand.topk(W, dim=-1)
            parent = top_idx // V
            next_tok = top_idx % V
            old = tokens.view(B, W, -1)
            new = torch.gather(old, 1, parent.unsqueeze(-1).expand(-1, -1, old.size(-1)))
            tokens = torch.cat([new, next_tok.unsqueeze(-1)], dim=-1).view(B * W, -1)
            scores = top_scores.view(-1)
            fin_old = finished.view(B, W)
            fin_new = torch.gather(fin_old, 1, parent) | (next_tok == EOS_ID)
            finished = fin_new.view(-1)
            if finished.all():
                break

        # Pick top-1 beam per input (with optional length penalty)
        s = scores.view(B, W)
        if length_penalty > 0:
            lens = (tokens != ENC_PAD_ID).sum(dim=-1).float().view(B, W)
            lp = ((5.0 + lens) / 6.0) ** length_penalty
            s = s / lp
        best = s.argmax(dim=-1)
        tg = tokens.view(B, W, -1)
        return tg[torch.arange(B, device=device), best]


def decoder_loss(logits: torch.Tensor, labels: torch.Tensor, pad_id: int = ENC_PAD_ID) -> torch.Tensor:
    """CE loss over decoder predictions, ignoring PAD positions.
    Args:
        logits: [B, L, V]
        labels: [B, L]  — already shifted to next-token target
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
    )


def prepare_decoder_io(seq_ids: torch.Tensor, pad_id: int = ENC_PAD_ID,
                       bos: int = BOS_ID, eos: int = EOS_ID) -> tuple[torch.Tensor, torch.Tensor]:
    """Given encoder-token sequences, build (decoder_input, decoder_target).
    decoder_input = [BOS, t0, t1, ..., t_{L-2}]
    decoder_target = [t0, t1, ..., t_{L-1}]
    But we mark EOS at the position AFTER the last non-pad token so the decoder
    learns to stop.

    Args:
        seq_ids: [B, L] from encoder.encode() — ACGU + PAD trailing
    Returns:
        (dec_in [B, L], dec_target [B, L])
    """
    B, L = seq_ids.shape
    device = seq_ids.device
    # Decoder target = seq_ids with EOS inserted at first PAD position
    dec_target = seq_ids.clone()
    # Find first PAD per row; that's where EOS goes
    not_pad = (seq_ids != pad_id)
    has_pad = (~not_pad).any(dim=1)
    # Position of first PAD per row, or L if no pad
    pad_pos = torch.where(has_pad, (~not_pad).int().argmax(dim=1), torch.tensor(L, device=device))
    # Insert EOS at pad_pos (clamped to L-1 so we have somewhere to put it)
    safe_pos = torch.clamp(pad_pos, max=L - 1)
    batch_idx = torch.arange(B, device=device)
    dec_target[batch_idx, safe_pos] = eos
    # Decoder input = [BOS, dec_target[:-1]]
    dec_in = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    dec_in[:, 0] = bos
    dec_in[:, 1:] = dec_target[:, :-1]
    return dec_in, dec_target


if __name__ == '__main__':
    # Smoke test
    dec = AptamerDecoder()
    n = sum(p.numel() for p in dec.parameters())
    print(f'AptamerDecoder: {n/1e6:.2f}M params')
    fake_emb = torch.randn(4, 384)
    fake_ids = torch.tensor([
        [0, 1, 2, 3, 4, 4, 4, 4],
        [0, 1, 2, 0, 1, 2, 4, 4],
        [0, 0, 1, 1, 2, 3, 0, 4],
        [3, 2, 1, 0, 4, 4, 4, 4],
    ])
    dec_in, dec_tgt = prepare_decoder_io(fake_ids)
    print(f'dec_in[0]:     {dec_in[0].tolist()}')
    print(f'dec_target[0]: {dec_tgt[0].tolist()}')
    logits = dec(fake_emb, dec_in)
    loss = decoder_loss(logits, dec_tgt)
    print(f'logits: {logits.shape}  loss: {loss.item():.3f}  (random init ≈ ln({DEC_VOCAB})={math.log(DEC_VOCAB):.3f})')
    gen = dec.generate(fake_emb, max_len=16, sample=False)
    print(f'greedy gen[0]: {gen[0].tolist()}')
