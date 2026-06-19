"""Score each of the 171 SPR-labeled RaptScore sequences with Tumbleweed and
compute Pearson r against measured Relative Activity per dataset. This is the
head-to-head we need against RaptScore's published 0.65 / 0.78 / 0.65.

Approach (zero-shot, no probe fitting):
  For each test sequence s in Dataset X (A/B/C):
    1. Build a "binder reference pool" = top-K highest-relative-activity OTHER
       sequences from Dataset X's Freq/Enr-selected set (Tables 4–6). Held-out.
    2. score(s) = mean cosine similarity between embed(s) and embed(reference).
    3. Pearson r over the full Freq/Enr set per dataset.

This is leave-one-out "binder-similarity" — the model's geometric prediction of
how binder-like s is, using OTHER known binders as the anchor.

Usage:
  python3 eval_raptscore_pearson.py <ckpt_path>
  python3 eval_raptscore_pearson.py ~/Desktop/autoRNA_data/tumbleweed/training_runs/tumbleweed_v1/ckpt_step30000.pt
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_dataset import encode

SPR_PATH = Path(__file__).resolve().parent.parent / 'data_refs/raptscore_spr_eval.parquet'
TOP_K = 5   # how many highest-activity OTHER sequences to anchor against per dataset


@torch.no_grad()
def encode_batch(model, seqs, max_len, device, batch=64):
    out = []
    for i in range(0, len(seqs), batch):
        ids = torch.tensor(np.stack([encode(s, max_len) for s in seqs[i:i+batch]]),
                           device=device)
        out.append(model.encode(ids).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    device = (torch.device('cuda') if args.device == 'auto' and torch.cuda.is_available()
              else torch.device('mps') if args.device == 'auto' and torch.backends.mps.is_available()
              else torch.device(args.device) if args.device != 'auto'
              else torch.device('cpu'))
    print(f'device: {device}')
    print(f'checkpoint: {args.checkpoint}')

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck.get('config', {})
    model_cfg = dict(cfg.get('model', {}))
    model_cfg.pop('grad_checkpoint', None)
    model_cfg.setdefault('max_len', args.max_len)

    if 'encoder' in ck:
        from aptamer_encoder import AptamerEncoder
        model = AptamerEncoder(**model_cfg).to(device).eval()
        model.load_state_dict(ck['encoder'])
    else:
        from aptamer_mlm import AptamerMLMModel
        model_cfg.pop('embed_dim', None)
        model = AptamerMLMModel(**model_cfg).to(device).eval()
        model.load_state_dict(ck['model'])
    print(f'  loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

    df = pd.read_parquet(SPR_PATH)
    df = df[df['selection_method'] == 'Frequency/Enrichment'].copy()  # apples-to-apples with RaptScore Table 1
    df['sequence'] = df['sequence'].str.upper().str.replace('T', 'U', regex=False)

    print(f'  scoring {len(df)} sequences across {df["dataset"].nunique()} datasets')
    t0 = time.time()
    embeds = encode_batch(model, df['sequence'].tolist(), args.max_len, device)
    embeds = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9)
    df['_idx'] = range(len(df))

    out = {}
    print()
    print(f'{"Dataset":<8} {"n":>4}  {"Pearson r":>10}  {"published":>10}  {"Δ":>7}')
    print('-' * 50)
    PUBLISHED = {'A': 0.65, 'B': 0.78, 'C': 0.65}
    for ds in ['A', 'B', 'C']:
        sub = df[df['dataset'] == ds].sort_values('relative_activity', ascending=False).reset_index(drop=True)
        n = len(sub)
        if n < 6: continue
        sub_embeds = embeds[sub['_idx'].values]
        sims = sub_embeds @ sub_embeds.T

        scores = []
        for i in range(n):
            ranked = [(j, sub.loc[j, 'relative_activity']) for j in range(n) if j != i]
            ranked.sort(key=lambda x: -x[1])
            ref_idx = [j for j, _ in ranked[:TOP_K]]
            scores.append(sims[i, ref_idx].mean())
        scores = np.array(scores)
        r, _ = pearsonr(scores, sub['relative_activity'].values)
        delta = r - PUBLISHED[ds]
        marker = '✓' if delta > 0 else ' '
        out[f'pearson_{ds}'] = float(r)
        print(f'  {ds:<6} {n:>4}  {r:+10.3f}  {PUBLISHED[ds]:+10.3f}  {delta:+7.3f}  {marker}')

    print(f'\n  ({time.time()-t0:.1f}s)')
    out['ckpt'] = str(args.checkpoint)
    return out


if __name__ == '__main__':
    main()
