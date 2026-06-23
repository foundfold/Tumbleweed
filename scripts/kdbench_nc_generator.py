#!/usr/bin/env python3
"""Recompute ONLY the generator column of Tumbleweed-KdBench for the contrastive-free
(nc) headline, then merge it into data_refs/v2_loo_per_target.csv.

Why a separate script: the original probe ranks the generator via its contrastive
.encode() projection head (out['proj']). The nc model (lam_contrast=0) never trains that
head, so proj is random. We instead rank off the pre-projection mean-pooled TRUNK hidden
(encode(representation='pooled')), which IS trained by the denoising objective. The other
four KdBench rankers (v10, v15, RiNALMo, TriFP) are unchanged models, so their columns in
v2_loo_per_target.csv are left exactly as-is; only the generator column is replaced.

Protocol matches probe_v2_loo_kd_full_comparison.py exactly: per held-out target, RidgeCV
on all-other-target embeddings, predict held-out, per-chemistry (DNA/RNA, n>=MIN_N_CHEM)
Spearman rho between predicted and measured log10 Kd.

Run ON THE NODE (needs torch + GPU + the nc checkpoint).
Inputs : data_refs/aptamer_kd_all_unified_v2.parquet, data_refs/target_protein_embeddings.parquet,
         the nc checkpoint (arg 1), data_refs/v2_loo_per_target.csv (to merge into)
Output : data_refs/v2_loo_per_target.csv  (generator column replaced with the nc values)
         data_refs/kdbench_nc_generator_per_panel.csv  (standalone per-panel nc result)
Usage  : python3 scripts/kdbench_nc_generator.py <nc_ckpt.pt>
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
# training/ ONLY on the path: a stale scripts/aptamer_diffusion_hybrid.py (no target_film)
# would otherwise shadow the real model class.
sys.path.insert(0, str(ROOT / 'training'))
from aptamer_dataset import encode  # noqa: E402
from aptamer_diffusion_hybrid import AptamerDiffusionHybrid  # noqa: E402

KD = ROOT / 'data_refs' / 'aptamer_kd_all_unified_v2.parquet'
ESM2 = ROOT / 'data_refs' / 'target_protein_embeddings.parquet'
OUT_CSV = ROOT / 'data_refs' / 'v2_loo_per_target.csv'
PANEL_CSV = ROOT / 'data_refs' / 'kdbench_nc_generator_per_panel.csv'
MIN_N_TOTAL = 5
MIN_N_CHEM = 3
ALPHAS = [0.01, 0.1, 1, 10, 100]


def load_nc(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = dict(ck['config']['model'])
    model = AptamerDiffusionHybrid(**model_cfg).to(device).eval()
    model.load_state_dict(ck['model'], strict=True)
    return model, int(model_cfg.get('max_len', 128))


@torch.no_grad()
def embed_pooled(model, df, embs_z, uid_to_idx, device, max_len):
    """Mean-pooled TRUNK representation per aptamer (NOT the contrastive proj head)."""
    encs = []
    for _, r in df.iterrows():
        ids = encode(r['sequence'], max_len, chemistry=r['chemistry_norm'])
        ids_t = torch.from_numpy(ids[None]).to(device)
        tgt = torch.from_numpy(embs_z[uid_to_idx[r['target_uniprot_id']]][None]).to(device)
        pooled = model.encode(ids_t, target_emb=tgt, representation='pooled')
        encs.append(pooled[0].float().cpu().numpy())
    return np.stack(encs)


def main():
    ckpt = Path(sys.argv[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kd = pd.read_parquet(KD)
    emb = pd.read_parquet(ESM2)
    have = set(emb['uniprot_id'].dropna().astype(str)) - {'None', 'nan', ''}
    kd = kd[kd['target_uniprot_id'].isin(have)].reset_index(drop=True)
    counts = kd['target_uniprot_id'].value_counts()
    eligible = counts[counts >= MIN_N_TOTAL].index.tolist()
    kd = kd[kd['target_uniprot_id'].isin(eligible)].reset_index(drop=True)
    print(f'eligible: {len(eligible)} targets, {len(kd)} sequences')

    embs_all = np.asarray([np.asarray(e, dtype=np.float32) for e in emb['embedding']])
    mu, sd = embs_all.mean(0, keepdims=True), embs_all.std(0, keepdims=True) + 1e-6
    embs_z = ((embs_all - mu) / sd).astype(np.float32)
    uid_to_idx = {u: i for i, u in enumerate(emb['uniprot_id'].astype(str))}

    model, max_len = load_nc(ckpt, device)
    print('embedding KdBench aptamers with nc generator (pooled trunk)...')
    X = embed_pooled(model, kd, embs_z, uid_to_idx, device, max_len)
    print(f'  shape {X.shape}')

    y = kd['log10_kd_nm'].values
    tgt = kd['target_uniprot_id'].values
    chem = kd['chemistry_norm'].values

    rows = []
    for ho in eligible:
        tr, te = tgt != ho, tgt == ho
        m = RidgeCV(alphas=ALPHAS, cv=3).fit(X[tr], y[tr])
        for ch in ['DNA', 'RNA']:
            idx = (chem == ch) & te
            n = int(idx.sum())
            if n < MIN_N_CHEM:
                continue
            rho, _ = spearmanr(m.predict(X[idx]), y[idx])
            rows.append({'target_uniprot': ho, 'chem': ch, 'n': n, 'generator_nc_rho': float(rho)})

    res = pd.DataFrame(rows)
    res.to_csv(PANEL_CSV, index=False)
    print(f'wrote {PANEL_CSV}  ({len(res)} panels)')
    print(f'  mean nc generator rho = {res["generator_nc_rho"].mean():.4f}  '
          f'(median {res["generator_nc_rho"].median():.4f}, '
          f'win-rate {(res["generator_nc_rho"] > 0).mean():.2f})')

    # merge into the leaderboard CSV: replace the generator column on (target,chem) keys
    lb = pd.read_csv(OUT_CSV)
    key = ['target_uniprot', 'chem']
    lb = lb.merge(res[key + ['generator_nc_rho']], on=key, how='left')
    # keep the historical column name the manuscript/forest plot read, but as the nc values
    lb['hybridfilm_rho'] = lb['generator_nc_rho']
    lb = lb.drop(columns=['generator_nc_rho'])
    lb.to_csv(OUT_CSV, index=False)
    print(f'updated {OUT_CSV}: hybridfilm_rho column now holds nc-generator (pooled) values')


if __name__ == '__main__':
    main()
