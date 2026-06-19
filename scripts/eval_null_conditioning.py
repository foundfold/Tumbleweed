"""Null-conditioning control: does the target embedding do ANY work, vs a weak sibling-shuffle?

The sibling target-shuffle (eval_target_shuffle.py) gave mean delta(own-wrong) ~ 0, but the 5 RNA
targets' ESM-2 embeddings are near-collinear (cosine 0.71-0.96) so "wrong sibling" is a weak perturbation.
This is the decisive null test: score AUROC(winner vs naive) per target under
  - OWN target embedding
  - ZERO vector            (no protein info at all)
  - MEAN embedding         (generic "average protein" over the whole 298-target bank)
  - FARTHEST real target   (the bank protein with lowest cosine to this target -> a maximally wrong cond.)
If own ~= zero ~= mean ~= farthest, the denoiser ignores the target embedding => conditioning is inert.
If own > {zero, mean, farthest}, the conditioning carries real target-specific signal.

Scores data_refs/recovery_seqs.parquet. Same scorer as RecoveryBench (low-t masked-diffusion pseudo-NLL).

Usage (Thunder, Hybrid env):
  python3 scripts/eval_null_conditioning.py \
    --ckpt ~/Desktop/autoRNA_data/tumbleweed/training_runs/tumbleweed_60m_diffusion_v1_multifam/ckpt_step15000.pt \
    --seqs ~/Tumbleweed/data_refs/recovery_seqs.parquet \
    --t_levels 0.1,0.15,0.2 --out_csv ~/Tumbleweed/data_refs/null_conditioning.csv
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
N_MASK_REPS = 6


def auroc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    rp = r[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.no_grad()
def pseudo_nll(model, ids, target_emb, t_levels, device, batch=256, gen=None, target_mask=None):
    maskable = torch.ones_like(ids, dtype=torch.bool)
    for nm in NEVER_MASK:
        maskable &= (ids != nm)
    N = ids.size(0)
    per_residue = target_emb.dim() == 3   # (1,P,D) per-residue vs (1,D) mean-pool
    acc = torch.zeros(N, device=device); cnt = torch.zeros(N, device=device)
    for t in t_levels:
        for _ in range(N_MASK_REPS):
            r = torch.rand(ids.shape, generator=gen, device=device)
            mpos = maskable & (r < t)
            for b0 in range(0, N, batch):
                chunk = ids[b0:b0 + batch]; mp = mpos[b0:b0 + batch]
                noisy = chunk.clone(); noisy[mp] = MASK_ID
                tmask = None
                if per_residue:
                    temb = target_emb.expand(chunk.size(0), -1, -1)
                    if target_mask is not None:
                        tmask = target_mask.expand(chunk.size(0), -1)
                else:
                    temb = target_emb.expand(chunk.size(0), -1)
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
    ap.add_argument('--seqs', type=Path, required=True)
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
    uid_list = emb_df['uniprot_id'].tolist()

    if per_residue:
        # Per-residue bank: each row 'residue_embeddings' is (P,1280). Conditions are (1,P,D)+mask.
        # Farthest target chosen by per-protein MEAN-pool cosine (a protein-level distance).
        res_by_uid = {u: np.stack([np.asarray(x, dtype=np.float32) for x in r])
                      for u, r in zip(emb_df['uniprot_id'], emb_df['residue_embeddings'])}
        meanpool = np.stack([res_by_uid[u].mean(0) for u in uid_list])          # (N, D)
        mp_unit = meanpool / (np.linalg.norm(meanpool, axis=1, keepdims=True) + 1e-9)
        all_res_mean = np.concatenate([res_by_uid[u] for u in uid_list]).mean(0)  # generic residue (D,)

        def _emb3(arr):  # (P,D) -> (1,P,D) tensor + (1,P) all-valid mask
            te = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
            return te, torch.ones(1, te.size(1), dtype=torch.bool, device=device)

        def own_emb(uid):
            return _emb3(res_by_uid[uid])

        mean_emb = _emb3(all_res_mean[None, :])    # 1-residue "average protein"
        zero_emb = (torch.zeros(1, 1, meanpool.shape[1], device=device),
                    torch.ones(1, 1, dtype=torch.bool, device=device))

        def farthest_emb(uid):
            v = mp_unit[uid_list.index(uid)]
            cos = mp_unit @ v
            far = int(np.argmin(cos))
            te, tm = _emb3(res_by_uid[uid_list[far]])
            return (te, tm), uid_list[far], float(cos[far])
    else:
        bank = np.stack(emb_df['embedding'].apply(lambda v: np.asarray(v, dtype=np.float32)))  # (298, D)
        bank_unit = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-9)
        mean_emb = torch.tensor(bank.mean(0), dtype=torch.float32, device=device).unsqueeze(0)
        zero_emb = torch.zeros_like(mean_emb)

        def own_emb(uid):
            row = emb_df[emb_df['uniprot_id'] == uid]
            return torch.tensor(row.iloc[0]['embedding'], dtype=torch.float32, device=device).unsqueeze(0)

        def farthest_emb(uid):
            v = bank_unit[uid_list.index(uid)]
            cos = bank_unit @ v
            far = int(np.argmin(cos))
            return torch.tensor(bank[far], dtype=torch.float32, device=device).unsqueeze(0), uid_list[far], float(cos[far])

    df = pd.read_parquet(args.seqs)
    rows = []
    for (name, uid, chem), g in df.groupby(['target', 'uniprot', 'chem'], sort=False):
        win = g[g['set'] == 'winner']['sequence'].tolist()
        nai = g[g['set'] == 'naive']['sequence'].tolist()
        to_ids = lambda S: torch.as_tensor(
            np.stack([encode(s, max_len, chemistry=chem) for s in S]), dtype=torch.long, device=device)
        wid, nid = to_ids(win), to_ids(nai)
        far_e, far_uid, far_cos = farthest_emb(uid)
        conds = {'own': own_emb(uid), 'zero': zero_emb, 'mean': mean_emb, 'far': far_e}
        au = {}
        for cname, ce in conds.items():
            emb, mask = ce if isinstance(ce, tuple) else (ce, None)
            au[cname] = auroc(-pseudo_nll(model, wid, emb, t_levels, device, gen=gen, target_mask=mask),
                              -pseudo_nll(model, nid, emb, t_levels, device, gen=gen, target_mask=mask))
        rows.append(dict(target=name, uniprot=uid, far_uid=far_uid, far_cos=round(far_cos, 3),
                         auroc_own=au['own'], auroc_zero=au['zero'], auroc_mean=au['mean'],
                         auroc_far=au['far'], own_minus_far=au['own'] - au['far'],
                         own_minus_zero=au['own'] - au['zero']))
        print(f'{name:<7} own={au["own"]:.3f} zero={au["zero"]:.3f} mean={au["mean"]:.3f} '
              f'far={au["far"]:.3f} (far={far_uid} cos={far_cos:.2f})  own-far={au["own"]-au["far"]:+.3f}')

    res = pd.DataFrame(rows)
    res.to_csv(args.out_csv, index=False)
    print(f'\nmean own-far  = {res["own_minus_far"].mean():+.3f}')
    print(f'mean own-zero = {res["own_minus_zero"].mean():+.3f}   (>0 => conditioning carries real signal)')
    print(f'wrote {args.out_csv}')


if __name__ == '__main__':
    main()
