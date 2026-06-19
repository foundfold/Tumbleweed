"""DECISIVE held-out conditioning test for the v2_kdbench retrain.

LOX1 (P78380) and CXCL5 (P42830) are NOT in the training corpus (neither the 6 SELEX targets
nor the 193 Kd-bench targets — verified by scope_kdbench_retrain.py: in_kd_pool=False). So scoring
the InstructNA panel under their OWN target embedding vs a ZERO vector vs the FARTHEST bank protein
is a true ZERO-SHOT generalization test of target conditioning:

  AUROC(strong vs weak | OWN)   >>   AUROC(... | ZERO)  ==>  conditioning generalizes to unseen targets
  own ~= zero                                            ==>  still target-inert (data didn't fix it)

Same masked-diffusion pseudo-NLL scorer as RecoveryBench / eval_instructna_kd.py, at low-t.
Embeddings come from the ckpt config (so a v2 ckpt automatically uses the CENTERED bank).

Usage (Thunder):
  python3 scripts/eval_instructna_conditioning.py \
    --ckpt ~/Desktop/autoRNA_data/tumbleweed/training_runs/tumbleweed_60m_diffusion_v2_kdbench/ckpt_step20000.pt \
    --panel ~/Desktop/autoRNA_data/tumbleweed/instructna_benchmark/processed/instructna_lox1_cxcl5_kd.parquet \
    --out_csv ~/Tumbleweed/data_refs/instructna_conditioning_v2.csv

UniProt map: LOX1 = P78380 (OLR1), CXCL5 = P42830.
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
N_MASK_REPS = 8
UNIPROT = {'LOX1': 'P78380', 'CXCL5': 'P42830'}


def auroc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    rp = r[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.no_grad()
def pseudo_nll(model, ids, target_emb, t_levels, device, batch=128, gen=None, target_mask=None):
    maskable = torch.ones_like(ids, dtype=torch.bool)
    for nm in NEVER_MASK:
        maskable &= (ids != nm)
    per_residue = target_emb.dim() == 3   # (1,P,D) per-residue vs (1,D) mean-pool
    N = ids.size(0)
    acc = torch.zeros(N, device=device); cnt = torch.zeros(N, device=device)
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
                chunk = ids[b0:b0 + batch]; mp = mpos[b0:b0 + batch]
                noisy = chunk.clone(); noisy[mp] = MASK_ID
                if per_residue:
                    temb = target_emb.expand(chunk.size(0), -1, -1)
                    tmask = target_mask.expand(chunk.size(0), -1) if target_mask is not None else None
                else:
                    temb = target_emb.expand(chunk.size(0), -1)
                    tmask = None
                tt = torch.full((chunk.size(0),), t, device=device)
                out = model(noisy, temb, tt, want_denoise=True, want_proj=False, target_mask=tmask)
                logp = F.log_softmax(out['logits'], dim=-1)
                nll = -logp.gather(-1, chunk.unsqueeze(-1)).squeeze(-1) * mp.float()
                acc[b0:b0 + chunk.size(0)] += nll.sum(-1)
                cnt[b0:b0 + chunk.size(0)] += mp.float().sum(-1)
    return (acc / cnt.clamp_min(1.0)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--panel', type=Path, required=True)
    ap.add_argument('--t_levels', type=str, default='0.1,0.15,0.2')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_csv', type=Path, required=True)
    args = ap.parse_args()
    t_levels = [float(x) for x in args.t_levels.split(',')]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = torch.Generator(device=device).manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck['config']
    model = AptamerDiffusionHybrid(**cfg['model']).to(device).eval()
    model.load_state_dict(ck['model'])
    max_len = cfg['model']['max_len']

    per_residue = cfg['model'].get('target_pool', 'mean') == 'attn' or cfg['model'].get('target_xattn', False)
    emb_df = pd.read_parquet(Path(cfg['data']['target_embeddings_parquet']).expanduser())
    emb_df['uniprot_id'] = emb_df['uniprot_id'].astype(str)
    uids = emb_df['uniprot_id'].tolist()

    if per_residue:
        # Per-residue bank: 'residue_embeddings' rows are (P,1280). Conditions are (1,P,D)+mask;
        # farthest target by per-protein MEAN-pool cosine.
        res_by_uid = {u: np.stack([np.asarray(x, dtype=np.float32) for x in r])
                      for u, r in zip(emb_df['uniprot_id'], emb_df['residue_embeddings'])}
        meanpool = np.stack([res_by_uid[u].mean(0) for u in uids])
        mp_unit = meanpool / (np.linalg.norm(meanpool, axis=1, keepdims=True) + 1e-9)
        D = meanpool.shape[1]

        def _emb3(arr):
            te = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
            return te, torch.ones(1, te.size(1), dtype=torch.bool, device=device)

        def own(uid):
            return _emb3(res_by_uid[uid])
        zero_emb = (torch.zeros(1, 1, D, device=device),
                    torch.ones(1, 1, dtype=torch.bool, device=device))

        def far(uid):
            cos = mp_unit @ mp_unit[uids.index(uid)]
            return _emb3(res_by_uid[uids[int(np.argmin(cos))]])
    else:
        bank = np.stack(emb_df['embedding'].apply(lambda v: np.asarray(v, dtype=np.float32)))
        bank_unit = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-9)
        zero_emb = torch.zeros(1, bank.shape[1], dtype=torch.float32, device=device)

        def own(uid):
            return torch.tensor(emb_df[emb_df['uniprot_id'] == uid].iloc[0]['embedding'],
                                dtype=torch.float32, device=device).unsqueeze(0)

        def far(uid):
            cos = bank_unit @ bank_unit[uids.index(uid)]
            return torch.tensor(bank[int(np.argmin(cos))], dtype=torch.float32, device=device).unsqueeze(0)

    df = pd.read_parquet(args.panel)
    rows = []
    for target, g in df.groupby('target', sort=False):
        uid = UNIPROT[target]
        ids = torch.as_tensor(np.stack([encode(s, max_len, chemistry='DNA') for s in g['sequence']]),
                              dtype=torch.long, device=device)
        strong = g['kd_class'].values == 'strong'
        conds = {'own': own(uid), 'zero': zero_emb, 'far': far(uid)}
        au = {}
        for c, ce in conds.items():
            e, m = ce if isinstance(ce, tuple) else (ce, None)
            s = -pseudo_nll(model, ids, e, t_levels, device, gen=gen, target_mask=m)
            au[c] = auroc(s[strong], s[~strong]) if strong.any() and (~strong).any() else float('nan')
        rows.append(dict(target=target, uniprot=uid, n=len(g), n_strong=int(strong.sum()),
                         auroc_own=au['own'], auroc_zero=au['zero'], auroc_far=au['far'],
                         own_minus_zero=au['own'] - au['zero'], own_minus_far=au['own'] - au['far']))
        print(f'{target:<6} {uid}: own={au["own"]:.3f} zero={au["zero"]:.3f} far={au["far"]:.3f}  '
              f'own-zero={au["own"]-au["zero"]:+.3f}')

    res = pd.DataFrame(rows)
    res.to_csv(args.out_csv, index=False)
    print(f'\nmean own-zero = {res["own_minus_zero"].mean():+.3f}   (>0 => conditioning generalizes to held-out targets)')
    print(f'mean own-far  = {res["own_minus_far"].mean():+.3f}')
    print(f'wrote {args.out_csv}')


if __name__ == '__main__':
    main()
