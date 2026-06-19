"""In-silico SELEX sampler for AptamerDiffusionHybrid.

Iterative denoising: start from fully-masked sequence, iteratively unmask
top-K most-confident positions until clean. Conditional on target ESM-2 +
chemistry token.

Simplified P2-style sampling — predict probabilities, unmask top-K, repeat.
Steps = L (one position unmasked per step) by default; can be K-per-step for speed.

Usage:
  python3 training/sample_diffusion_hybrid.py CKPT --target_uniprot P21980 \
    --chemistry RNA --n_samples 100 --length 64
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_dataset import N_TOKENS, PAD_ID, MASK_ID, RNA_TOK_ID, DNA_TOK_ID, T_ID
from aptamer_diffusion_hybrid import AptamerDiffusionHybrid


TOKEN2BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 8: 'T'}
DISALLOWED_FOR_OUTPUT = {PAD_ID, MASK_ID, RNA_TOK_ID, DNA_TOK_ID}


@torch.no_grad()
def sample_p2(
    model: AptamerDiffusionHybrid,
    target_emb: torch.Tensor,         # (1, 1280)
    chemistry: str,                    # 'DNA' or 'RNA'
    length: int = 64,
    n_samples: int = 100,
    n_steps: int = 32,
    temperature: float = 1.0,
    device: str = 'cuda',
) -> list[str]:
    """Generate n_samples sequences via iterative denoising.

    Schedule: at step i (0..n_steps-1), t = 1 - i/n_steps. Unmask top-K most-
    confident positions, where K = ceil(length / n_steps) per step.
    """
    chem_id = RNA_TOK_ID if chemistry == 'RNA' else DNA_TOK_ID
    L = length + 1   # +1 for chemistry token at position 0

    # Initialize: chem_token at pos 0, MASK_ID at all content positions
    ids = torch.full((n_samples, L), MASK_ID, dtype=torch.long, device=device)
    ids[:, 0] = chem_id

    target_emb_batch = target_emb.expand(n_samples, -1)
    K_per_step = max(1, length // n_steps)

    for i in range(n_steps):
        t = torch.tensor([1.0 - i / n_steps] * n_samples, device=device)
        out = model(ids, target_emb_batch, t, want_denoise=True, want_proj=False)
        logits = out['logits']                                              # (B, L, V)
        # Forbid outputting reserved tokens (PAD/MASK/[RNA]/[DNA])
        for tok in DISALLOWED_FOR_OUTPUT:
            logits[:, :, tok] = -1e9
        # Temperature
        probs = torch.softmax(logits / temperature, dim=-1)
        # For each sample, find positions still masked
        masked = (ids == MASK_ID)
        # Confidence: max prob per position
        conf = probs.max(dim=-1).values                                     # (B, L)
        conf_masked = conf.masked_fill(~masked, -1.0)                       # only consider masked positions
        # Unmask top-K positions per sample
        for b in range(n_samples):
            n_mask = int(masked[b].sum())
            if n_mask == 0:
                continue
            k = min(K_per_step, n_mask)
            top_idx = torch.topk(conf_masked[b], k=k).indices
            # Sample tokens at those positions
            for pos in top_idx:
                token_probs = probs[b, pos]
                token = torch.multinomial(token_probs, 1).item()
                ids[b, pos] = token

    # Decode to strings (skip pos 0 = chemistry token)
    seqs = []
    for row in ids[:, 1:]:
        chars = []
        for v in row:
            b = TOKEN2BASE.get(int(v))
            if b is not None:
                chars.append(b)
        s = ''.join(chars)
        if chemistry == 'RNA':
            s = s.replace('T', 'U')  # keep RNA representation
        else:
            s = s.replace('U', 'T')
        seqs.append(s)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', type=Path)
    ap.add_argument('--target_uniprot', required=True)
    ap.add_argument('--chemistry', default='RNA', choices=['DNA', 'RNA'])
    ap.add_argument('--n_samples', type=int, default=100)
    ap.add_argument('--length', type=int, default=64)
    ap.add_argument('--n_steps', type=int, default=32)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--out', type=Path, default=Path('/tmp/diffusion_samples.fasta'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck['config']
    model = AptamerDiffusionHybrid(**cfg['model']).to(device).eval()
    model.load_state_dict(ck['model'])

    # Load target embedding
    emb_df = pd.read_parquet(Path(cfg['data']['target_embeddings_parquet']).expanduser())
    row = emb_df[emb_df['uniprot_id'].astype(str) == args.target_uniprot]
    if len(row) == 0:
        raise SystemExit(f'No target ESM-2 found for {args.target_uniprot}')
    target_emb = torch.tensor(row.iloc[0]['embedding'], dtype=torch.float32,
                                device=device).unsqueeze(0)

    print(f'sampling {args.n_samples} sequences of length {args.length} for {args.target_uniprot} ({args.chemistry})...')
    seqs = sample_p2(
        model, target_emb, args.chemistry,
        length=args.length, n_samples=args.n_samples, n_steps=args.n_steps,
        temperature=args.temperature, device=device,
    )

    with open(args.out, 'w') as f:
        for i, s in enumerate(seqs):
            f.write(f'>gen_{i:04d}\n{s}\n')
    print(f'saved {len(seqs)} sequences → {args.out}')
    print('first 3:')
    for s in seqs[:3]:
        print(f'  {s}')


if __name__ == '__main__':
    main()
