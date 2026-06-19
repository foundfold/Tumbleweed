"""InstructNA panel benchmark (#91 external comparator): does Tumbleweed-Hybrid's target-conditioned
denoising likelihood rank InstructNA's REAL SPR-validated binders above its weak/non-binders?

InstructNA (Zhang et al. 2026, Nat Comp Sci, doi:10.1038/s43588-026-00965-3) released NO model weights —
the only public artifact is 40 ssDNA aptamers (20 vs LOX1, 20 vs CXCL5; 10 InstructNA-generated "G*" +
10 SELEX "T*" per target) with SPR Kd labels from Fig 2 + Supp Tables 3-6. So unlike the EvoFlow
comparator (NLL on identical seqs), this is a discrimination benchmark on InstructNA's OWN designs:
  - 12 strong binders (Kd <= 100 nM, numeric kd_nm) vs 28 weak_or_none.
We score each sequence with the SAME non-circular metric as RecoveryBench: target-conditioned masked-
diffusion pseudo-NLL at the low-t (converged) regime, conditioned on the target's ESM-2 embedding.

These are ssDNA HT-SELEX aptamers -> chemistry = DNA (Tumbleweed prepends the [DNA] token in encode()).

Readouts (per target + pooled):
  - AUROC(strong vs weak_or_none)        -- can the Hybrid separate real binders from junk?
  - Spearman(-NLL, kd_nm) over the strong binders (lower Kd = tighter; expect monotone w/ score).

Scores the curated panel: data_refs/instructna_lox1_cxcl5_kd.parquet
  (cols: name,sequence,length,target,source,kd_nm,kd_class,random_region,source_note).

Usage (Thunder, Hybrid env):
  python3 scripts/eval_instructna_kd.py \
    --ckpt ~/Desktop/autoRNA_data/tumbleweed/training_runs/tumbleweed_60m_diffusion_v1_multifam/ckpt_step15000.pt \
    --panel ~/Desktop/autoRNA_data/tumbleweed/instructna_benchmark/processed/instructna_lox1_cxcl5_kd.parquet \
    --t_levels 0.1,0.15,0.2 \
    --out_csv ~/Tumbleweed/data_refs/instructna_hybrid_scores.csv

UniProt map (target name -> accession in the ckpt's target_protein_embeddings.parquet):
  LOX1 = P78380 (OLR1), CXCL5 = P42830.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from aptamer_dataset import encode, PAD_ID, MASK_ID, RNA_TOK_ID, DNA_TOK_ID  # noqa: E402
from aptamer_diffusion_hybrid import AptamerDiffusionHybrid  # noqa: E402

NEVER_MASK = (PAD_ID, RNA_TOK_ID, DNA_TOK_ID)
N_MASK_REPS = 8  # panel is tiny (40 seqs) so we can afford more masks for a stable score
UNIPROT = {'LOX1': 'P78380', 'CXCL5': 'P42830'}


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    all_s = np.concatenate([pos, neg])
    r = rankdata(all_s)
    rp = r[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.no_grad()
def pseudo_nll(model, ids, target_emb, t_levels, device, batch=128, gen=None) -> np.ndarray:
    """Per-sequence mean masked-token NLL averaged over t_levels x N_MASK_REPS, conditioned on target."""
    maskable = torch.ones_like(ids, dtype=torch.bool)
    for nm in NEVER_MASK:
        maskable &= (ids != nm)
    N = ids.size(0)
    acc = torch.zeros(N, device=device)
    cnt = torch.zeros(N, device=device)
    for t in t_levels:
        for _ in range(N_MASK_REPS):
            r = torch.rand(ids.shape, generator=gen, device=device)
            mpos = maskable & (r < t)
            empty = mpos.sum(-1) == 0
            if empty.any():
                for ri in torch.nonzero(empty, as_tuple=False).flatten():
                    cand = torch.nonzero(maskable[ri], as_tuple=False).flatten()
                    if len(cand):
                        mpos[ri, cand[torch.randint(len(cand), (1,), generator=gen, device=device)]] = True
            for b0 in range(0, N, batch):
                chunk = ids[b0:b0 + batch]
                mp = mpos[b0:b0 + batch]
                noisy = chunk.clone()
                noisy[mp] = MASK_ID
                temb = target_emb.expand(chunk.size(0), -1)
                tt = torch.full((chunk.size(0),), t, device=device)
                out = model(noisy, temb, tt, want_denoise=True, want_proj=False)
                logp = F.log_softmax(out['logits'], dim=-1)
                nll = -logp.gather(-1, chunk.unsqueeze(-1)).squeeze(-1) * mp.float()
                acc[b0:b0 + chunk.size(0)] += nll.sum(-1)
                cnt[b0:b0 + chunk.size(0)] += mp.float().sum(-1)
    return (acc / cnt.clamp_min(1.0)).cpu().numpy()


def main():
    from scipy.stats import spearmanr
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--panel', type=Path, required=True)
    ap.add_argument('--t_levels', type=str, default='0.1,0.15,0.2')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_csv', type=Path, required=True)
    args = ap.parse_args()
    t_levels = [float(x) for x in args.t_levels.split(',')]
    print(f't levels = {t_levels}  reps/level = {N_MASK_REPS}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = torch.Generator(device=device).manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck['config']
    model = AptamerDiffusionHybrid(**cfg['model']).to(device).eval()
    model.load_state_dict(ck['model'])
    max_len = cfg['model']['max_len']
    emb_df = pd.read_parquet(Path(cfg['data']['target_embeddings_parquet']).expanduser())

    def emb(uid):
        row = emb_df[emb_df['uniprot_id'].astype(str) == uid]
        if not len(row):
            raise SystemExit(f'no ESM-2 for {uid}')
        return torch.tensor(row.iloc[0]['embedding'], dtype=torch.float32, device=device).unsqueeze(0)

    df = pd.read_parquet(args.panel)
    df['score'] = np.nan  # -NLL, higher = model favors it
    rows = []
    for target, g in df.groupby('target', sort=False):
        uid = UNIPROT[target]
        te = emb(uid)
        ids = torch.as_tensor(
            np.stack([encode(s, max_len, chemistry='DNA') for s in g['sequence']]),
            dtype=torch.long, device=device)
        s = -pseudo_nll(model, ids, te, t_levels, device, gen=gen)
        df.loc[g.index, 'score'] = s
        strong = g['kd_class'].values == 'strong'
        au = auroc(s[strong], s[~strong]) if strong.any() and (~strong).any() else float('nan')
        # Spearman over strong binders: lower Kd should mean higher score (-NLL) -> negative rho expected
        kd = g['kd_nm'].values[strong].astype(float)
        rho = spearmanr(kd, s[strong]).correlation if strong.sum() > 2 else float('nan')
        rows.append(dict(target=target, uniprot=uid, n=len(g), n_strong=int(strong.sum()),
                         auroc_strong_vs_weak=au, spearman_kd_vs_score=rho))
        print(f'{target:<6} {uid}: n={len(g)} strong={int(strong.sum())} '
              f'AUROC(strong vs weak)={au:.3f}  Spearman(Kd,score)={rho:.3f}')

    # pooled AUROC across both targets (scores are -NLL, comparable across targets at same t-regime)
    pooled_strong = df['kd_class'].values == 'strong'
    pooled_au = auroc(df['score'].values[pooled_strong], df['score'].values[~pooled_strong])
    print(f'\npooled AUROC(strong vs weak, both targets) = {pooled_au:.3f}  (0.5=chance)')

    res = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    summ = args.out_csv.with_name(args.out_csv.stem + '_summary.csv')
    res.to_csv(summ, index=False)
    print(f'\nwrote per-seq scores -> {args.out_csv}')
    print(f'wrote summary        -> {summ}')


if __name__ == '__main__':
    main()
