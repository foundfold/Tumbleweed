"""v10: TriFP-guided generation from a Tumbleweed-Hybrid masked-diffusion checkpoint.

The denoiser tells us WHO the target is (via FiLM/token conditioning); TriFP tells us WHETHER a
candidate binds. v10 fuses them at SAMPLING time (no retraining): generate target-conditioned
candidates by masked-diffusion, then hill-climb in sequence space using the denoiser as the
proposal and the TriFP GBDT (predicted log10 Kd, lower = stronger) as a gradient-free energy.
Because TriFP needs a complete sequence (kmer features), guidance is applied as remask+re-denoise
proposals scored on the COMPLETE candidate — not on partial/masked states.

The decisive read is the TRADE-OFF, not "did TriFP go down" (it must, since we optimise it):
  - trifp  : mean predicted log10 Kd over the pool   (lower = better binders)
  - pnll   : mean denoiser low-t pseudo-NLL          (lower = more plausible/in-distribution)
  - div    : mean pairwise normalised Hamming        (higher = more diverse, not mode-collapsed)
A useful guidance signal lowers trifp WITHOUT blowing up pnll or collapsing diversity.

Inputs:
  --ckpt        a trained Hybrid ckpt (default use case: the v5_film winner)
  --trifp       data_refs/trifp_gbdt.joblib  (GBDT + z-score stats from build_trifp_fingerprint_bank.py)
  --raw_emb     data_refs/target_protein_embeddings.parquet  (RAW ESM-2 for TriFP z-scoring)
  conditioning bank comes from the CKPT config (cfg.data.target_embeddings_parquet) so a v5_film
  ckpt automatically uses its CENTERED mean-pool bank.

Usage (Thunder, Hybrid env):
  python3 scripts/sample_trifp_guided.py \
    --ckpt ~/Desktop/autoRNA_data/tumbleweed/training_runs/tumbleweed_60m_diffusion_v5_film/ckpt_step20000.pt \
    --trifp ~/Tumbleweed/data_refs/trifp_gbdt.joblib \
    --raw_emb ~/Tumbleweed/data_refs/target_protein_embeddings.parquet \
    --targets P78380,P42830,P31371,P09874 --chem DNA \
    --n 64 --gen_len 40 --steps 16 --remask_rounds 12 --remask_frac 0.15 \
    --out_csv ~/Tumbleweed/data_refs/trifp_guided_v5_film.csv
"""
from __future__ import annotations
import argparse, sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from aptamer_dataset import PAD_ID, MASK_ID, RNA_TOK_ID, DNA_TOK_ID  # noqa: E402
from aptamer_diffusion_hybrid import AptamerDiffusionHybrid, _ID2BASE  # noqa: E402

BASE_RNA = [0, 1, 2, 3]      # A C G U
BASE_DNA = [0, 1, 2, 8]      # A C G T


def kmer_features(seqs, ks):
    feats = []
    for k in ks:
        rows = []
        for s in seqs:
            s = s.upper().replace('U', 'T')
            c = Counter(s[i:i + k] for i in range(len(s) - k + 1)
                        if all(ch in 'ACGT' for ch in s[i:i + k]))
            rows.append([c.get(''.join(t), 0) for t in product('ACGT', repeat=k)])
        mat = np.asarray(rows, dtype=np.float32)
        feats.append(mat / (mat.sum(1, keepdims=True) + 1e-9))
    return np.concatenate(feats, axis=1)


def ids_to_seqs(ids):
    out = []
    for row in ids.tolist():
        out.append(''.join(_ID2BASE[v] for v in row[1:] if v in _ID2BASE).replace('U', 'T'))
    return out


def trifp_score(seqs, esm_z_target, critic):
    """Predicted log10 Kd for each seq paired with the target (lower = stronger)."""
    X_k = kmer_features(seqs, critic['ks'])
    X = np.concatenate([X_k, np.tile(esm_z_target, (len(seqs), 1))], axis=1)
    return critic['model'].predict(X).astype(np.float32)


@torch.no_grad()
def mdm_sample(model, target_emb, chem, gen_len, n, steps, t_hi, device, gen, init_ids=None,
               n_masked_init=None, temperature=0.0, cond_t=None):
    """Confidence-scheduled masked-diffusion decode. If init_ids given, remask n_masked_init
    random base positions of each row and re-denoise only those (guidance proposal).

    temperature=0 -> greedy argmax decode (deterministic; n parallel chains from the same all-MASK
    start collapse to one identical sequence). temperature>0 -> multinomial token sampling with
    Gumbel-noised confidence ordering, so the n chains diverge and we draw from p(seq|target)
    instead of taking its mode.

    cond_t: if set, the model's conditioning-t INPUT is held at this value at every step instead
    of following the anneal schedule to 0. Training used t = 1 - round/r_max, so cond_t = 1 - r/r_max
    makes the denoiser emulate round r's reconstruction distribution (round-conditioned generation:
    high cond_t -> early-round diverse/weak, low cond_t -> late-round converged/strong). The unmask
    SCHEDULE (how many positions get filled) still runs normally so we always emit full sequences."""
    chem_id = DNA_TOK_ID if chem == 'DNA' else RNA_TOK_ID
    forbid = [PAD_ID, MASK_ID, RNA_TOK_ID, DNA_TOK_ID] + ([3] if chem == 'DNA' else [8])
    L = 1 + gen_len
    if init_ids is None:
        ids = torch.full((n, L), MASK_ID, dtype=torch.long, device=device)
        ids[:, 0] = chem_id
        masked = torch.ones(n, gen_len, dtype=torch.bool, device=device)
    else:
        ids = init_ids.clone()
        masked = torch.zeros(n, gen_len, dtype=torch.bool, device=device)
        for r in range(n):
            pos = torch.randperm(gen_len, generator=gen, device=device)[:n_masked_init]
            masked[r, pos] = True
            ids[r, 1 + pos] = MASK_ID
    ts = torch.linspace(t_hi, 0.0, steps + 1, device=device)
    for s in range(steps):
        t_next = ts[s + 1]
        t_in = float(cond_t) if cond_t is not None else float(ts[s])
        out = model(ids, target_emb.expand(n, -1), torch.full((n,), t_in, device=device),
                    want_denoise=True, want_proj=False)
        logits = out['logits'][:, 1:, :].clone()      # (n, gen_len, V), strip chem position
        logits[..., forbid] = float('-inf')
        if temperature > 0:
            prob = F.softmax(logits / temperature, dim=-1)
            pred = torch.multinomial(prob.reshape(-1, prob.shape[-1]), 1,
                                     generator=gen).reshape(n, gen_len)
            conf = prob.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            u = torch.rand(conf.shape, generator=gen, device=device).clamp_(1e-9, 1.0)
            conf = conf + temperature * (-torch.log(-torch.log(u)))   # Gumbel-noised unmask order
        else:
            prob = F.softmax(logits, dim=-1)
            conf, pred = prob.max(dim=-1)               # (n, gen_len) greedy
        conf = conf.masked_fill(~masked, -1e9)          # only consider still-masked positions
        target_masked = int(round(gen_len * float(t_next)))
        for r in range(n):
            still = int(masked[r].sum())
            k_unmask = still - target_masked if s < steps - 1 else still
            if k_unmask <= 0:
                continue
            order = torch.topk(conf[r], k=min(k_unmask, still)).indices
            ids[r, 1 + order] = pred[r, order]
            masked[r, order] = False
    # fill any leftover masked with argmax
    if masked.any():
        t_in = float(cond_t) if cond_t is not None else 0.0
        out = model(ids, target_emb.expand(n, -1), torch.full((n,), t_in, device=device),
                    want_denoise=True, want_proj=False)
        logits = out['logits'][:, 1:, :].clone(); logits[..., forbid] = float('-inf')
        if temperature > 0:
            p = F.softmax(logits / temperature, dim=-1)
            pred = torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=gen).reshape(n, gen_len)
        else:
            pred = logits.argmax(-1)
        ids[:, 1:][masked] = pred[masked]
    return ids


@torch.no_grad()
def pseudo_nll(model, ids, target_emb, device, t_levels=(0.1, 0.15, 0.2), reps=4, gen=None):
    n, L = ids.shape
    maskable = torch.ones(n, L, dtype=torch.bool, device=device)
    for nm in (PAD_ID, RNA_TOK_ID, DNA_TOK_ID):
        maskable &= (ids != nm)
    acc = torch.zeros(n, device=device); cnt = torch.zeros(n, device=device)
    for t in t_levels:
        for _ in range(reps):
            r = torch.rand(ids.shape, generator=gen, device=device)
            mp = maskable & (r < t)
            noisy = ids.clone(); noisy[mp] = MASK_ID
            out = model(noisy, target_emb.expand(n, -1), torch.full((n,), t, device=device),
                        want_denoise=True, want_proj=False)
            logp = F.log_softmax(out['logits'], dim=-1)
            nll = -logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1) * mp.float()
            acc += nll.sum(-1); cnt += mp.float().sum(-1)
    return (acc / cnt.clamp_min(1.0)).cpu().numpy()


def diversity(seqs):
    if len(seqs) < 2:
        return 0.0
    arr = np.array([list(s) for s in seqs])
    n = len(seqs); tot = 0.0; cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += np.mean(arr[i] != arr[j]); cnt += 1
    return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--trifp', type=Path, required=True)
    ap.add_argument('--raw_emb', type=Path, required=True)
    ap.add_argument('--targets', type=str, required=True, help='comma-sep uniprot ids')
    ap.add_argument('--chem', type=str, default='DNA', choices=['DNA', 'RNA'])
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--gen_len', type=int, default=40)
    ap.add_argument('--steps', type=int, default=16)
    ap.add_argument('--t_hi', type=float, default=0.9)
    ap.add_argument('--remask_rounds', type=int, default=12)
    ap.add_argument('--remask_frac', type=float, default=0.15)
    ap.add_argument('--remask_steps', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='0=greedy argmax decode; >0=multinomial token sampling + Gumbel unmask order')
    ap.add_argument('--out_csv', type=Path, required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = torch.Generator(device=device).manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck['config']
    model = AptamerDiffusionHybrid(**cfg['model']).to(device).eval()
    model.load_state_dict(ck['model'])

    cond = pd.read_parquet(Path(cfg['data']['target_embeddings_parquet']).expanduser())
    cond['uniprot_id'] = cond['uniprot_id'].astype(str)
    raw = pd.read_parquet(args.raw_emb.expanduser()); raw['uniprot_id'] = raw['uniprot_id'].astype(str)
    critic = joblib.load(args.trifp)
    mu, sd = critic['esm_mu'], critic['esm_sd']

    n_mask = max(1, int(round(args.gen_len * args.remask_frac)))
    rows = []
    for uid in args.targets.split(','):
        cr = cond[cond['uniprot_id'] == uid]
        rr = raw[raw['uniprot_id'] == uid]
        if not len(cr) or not len(rr):
            print(f'{uid}: MISSING emb (cond={len(cr)} raw={len(rr)}) — skip'); continue
        temb = torch.tensor(np.asarray(cr.iloc[0]['embedding'], dtype=np.float32),
                            device=device).unsqueeze(0)
        esm_z = ((np.asarray(rr.iloc[0]['embedding'], dtype=np.float32)[None] - mu) / sd).astype(np.float32)[0]

        ids = mdm_sample(model, temb, args.chem, args.gen_len, args.n, args.steps, args.t_hi,
                         device, gen, temperature=args.temperature)
        seqs = ids_to_seqs(ids)
        s_base = trifp_score(seqs, esm_z, critic)
        pnll_base = pseudo_nll(model, ids, temb, device, gen=gen)
        best_ids = ids.clone(); best_s = s_base.copy()

        for _ in range(args.remask_rounds):
            prop = mdm_sample(model, temb, args.chem, args.gen_len, args.n, args.remask_steps,
                              args.t_hi, device, gen, init_ids=best_ids, n_masked_init=n_mask,
                              temperature=args.temperature)
            ps = trifp_score(ids_to_seqs(prop), esm_z, critic)
            imp = ps < best_s                              # lower log10 Kd = better
            best_ids[imp] = prop[imp]; best_s[imp] = ps[imp]

        seqs_g = ids_to_seqs(best_ids)
        pnll_g = pseudo_nll(model, best_ids, temb, device, gen=gen)
        rows.append(dict(
            uniprot=uid, n=args.n,
            trifp_base=float(np.mean(s_base)), trifp_guided=float(np.mean(best_s)),
            trifp_delta=float(np.mean(best_s) - np.mean(s_base)),
            pnll_base=float(np.mean(pnll_base)), pnll_guided=float(np.mean(pnll_g)),
            pnll_delta=float(np.mean(pnll_g) - np.mean(pnll_base)),
            div_base=round(diversity(seqs), 4), div_guided=round(diversity(seqs_g), 4)))
        print(f'{uid}: trifp {np.mean(s_base):.3f}->{np.mean(best_s):.3f} '
              f'(Δ{np.mean(best_s)-np.mean(s_base):+.3f})  '
              f'pnll {np.mean(pnll_base):.3f}->{np.mean(pnll_g):.3f}  '
              f'div {diversity(seqs):.3f}->{diversity(seqs_g):.3f}')

    res = pd.DataFrame(rows)
    res.to_csv(args.out_csv, index=False)
    if len(res):
        print(f'\nmean trifp Δ = {res["trifp_delta"].mean():+.3f} (lower=better binders)')
        print(f'mean pnll  Δ = {res["pnll_delta"].mean():+.3f} (near 0 = plausibility preserved)')
    print(f'wrote {args.out_csv}')


if __name__ == '__main__':
    main()
