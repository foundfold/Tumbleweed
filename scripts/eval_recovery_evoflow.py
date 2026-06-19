"""RecoveryBench for EvoFlow-RNA (external comparator for generation benchmark #91).

EvoFlow-RNA (Patel et al. 2025, Atom Bioworks/Duke) = RiNALMo backbone + masked-discrete-diffusion
finetune. It is the published instantiation of "RiNALMo + diffusion head", but UNCONDITIONAL:
no target protein, no SELEX-round noise schedule, RNA-only. So it gives us exactly the two
non-conditional RecoveryBench readouts to compare against Tumbleweed-Hybrid:
  1. AUROC(winner vs random)  — does the model rank real binders above composition-matched junk?
  2. AUROC(winner vs naive)   — does it capture SELEX ENRICHMENT?
There is NO target-conditioning matrix (EvoFlow cannot condition on a protein).

Apples-to-apples: scores the SAME sequences Tumbleweed scored. Feed it the parquet dumped by
eval_recovery_likelihood.py --dump_seqs (columns: target,uniprot,chem,set,sequence).

Per-sequence score := -pseudo_NLL, pseudo_NLL = mean over low mask-ratios t and several random
masks of the per-(masked-token) cross-entropy under EvoFlow's denoiser logits. Same low-t regime
(converged SELEX rounds ~ small t) used for the Tumbleweed --t_levels run.

Source: github.com/AtomBio/evoflow-rna ; weights zenodo.org/records/15009560 (mini-v1.ckpt, 33M).

Usage (evoflow-rna conda env, cwd = repo root so load_from_pretrained finds configs/lm/evoflow.yaml):
  cd ~/Tumbleweed/evoflow-rna
  python ~/Tumbleweed/scripts/eval_recovery_evoflow.py \
    --ckpt weights/mini-v1.ckpt \
    --seqs ~/Tumbleweed/data_refs/recovery_seqs.parquet \
    --t_levels 0.1,0.15,0.2 --out_csv ~/Tumbleweed/data_refs/recovery_evoflow_lowt.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

T_LEVELS = (0.1, 0.15, 0.2)
N_MASK_REPS = 4


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    all_s = np.concatenate([pos, neg])
    r = rankdata(all_s)
    rp = r[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.no_grad()
def pseudo_nll(model, alphabet, seqs: list[str], device, batch=128, gen=None) -> np.ndarray:
    """Per-sequence mean masked-token NLL under EvoFlow, averaged over T_LEVELS x N_MASK_REPS."""
    # tokenize (CLS + seq + EOS + PAD), special tokens are the low indices
    ids = torch.tensor(alphabet.batch_tokenize(seqs), dtype=torch.long, device=device)
    special = {alphabet.cls_idx, alphabet.eos_idx, alphabet.pad_idx,
               alphabet.unk_idx, alphabet.mask_idx}
    maskable = torch.ones_like(ids, dtype=torch.bool)
    for sp in special:
        maskable &= (ids != sp)

    N = ids.size(0)
    acc = torch.zeros(N, device=device)
    cnt = torch.zeros(N, device=device)
    for t in T_LEVELS:
        for _ in range(N_MASK_REPS):
            r = torch.rand(ids.shape, generator=gen, device=device)
            mpos = maskable & (r < t)
            # guarantee >=1 masked token per row (else NLL undefined for that draw)
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
                noisy[mp] = alphabet.mask_idx
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(noisy)  # (b,L,V); flash-attn requires half precision
                if isinstance(logits, tuple):
                    logits = logits[0]
                logp = F.log_softmax(logits.float(), dim=-1)
                nll = -logp.gather(-1, chunk.unsqueeze(-1)).squeeze(-1) * mp.float()
                acc[b0:b0 + chunk.size(0)] += nll.sum(-1)
                cnt[b0:b0 + chunk.size(0)] += mp.float().sum(-1)
    return (acc / cnt.clamp_min(1.0)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default='weights/mini-v1.ckpt')
    ap.add_argument('--seqs', type=Path, required=True)
    ap.add_argument('--t_levels', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_csv', type=Path, required=True)
    args = ap.parse_args()

    if args.t_levels:
        global T_LEVELS
        T_LEVELS = tuple(float(x) for x in args.t_levels.split(','))
    print(f'mask-ratio levels (t) = {T_LEVELS}  reps/level = {N_MASK_REPS}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = torch.Generator(device=device).manual_seed(args.seed)

    from src.ncrna.tasks.lm.drnafm import EvoFlow  # noqa: E402
    model = EvoFlow.load_from_pretrained(args.ckpt).to(device).eval()
    alphabet = model.alphabet

    df = pd.read_parquet(args.seqs)
    rows = []
    for (name, uid), g in df.groupby(['target', 'uniprot'], sort=False):
        win = g[g['set'] == 'winner']['sequence'].tolist()
        nai = g[g['set'] == 'naive']['sequence'].tolist()
        rnd = g[g['set'] == 'random']['sequence'].tolist()
        s_win = -pseudo_nll(model, alphabet, win, device, gen=gen)
        s_nai = -pseudo_nll(model, alphabet, nai, device, gen=gen)
        s_rnd = -pseudo_nll(model, alphabet, rnd, device, gen=gen)
        rows.append(dict(
            target=name, uniprot=uid, n=len(win),
            mean_nll_winner=float(-s_win.mean()),
            mean_nll_naive=float(-s_nai.mean()),
            mean_nll_random=float(-s_rnd.mean()),
            AUROC_winner_vs_random=auroc(s_win, s_rnd),
            AUROC_winner_vs_naive=auroc(s_win, s_nai),
        ))
        print(f'{name:<7} {uid}: nll win={-s_win.mean():.3f} naive={-s_nai.mean():.3f} '
              f'rand={-s_rnd.mean():.3f} | AUROC vs_rand={rows[-1]["AUROC_winner_vs_random"]:.3f} '
              f'vs_naive={rows[-1]["AUROC_winner_vs_naive"]:.3f}')

    res = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print('\n=== EvoFlow-RNA RecoveryBench: likelihood on REAL sequences (lower NLL = favored) ===')
    print(res.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\n  mean AUROC(winner vs random) = {res["AUROC_winner_vs_random"].mean():.4f}  (0.5=chance)')
    print(f'  mean AUROC(winner vs naive)  = {res["AUROC_winner_vs_naive"].mean():.4f}')

    res.to_csv(args.out_csv, index=False)
    print(f'\nwrote {args.out_csv}')


if __name__ == '__main__':
    main()
