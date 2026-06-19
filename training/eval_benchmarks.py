"""End-to-end benchmark eval — runs the full suite on a trained encoder.

Used in two contexts:
  1. Inline during training (called from train_contrastive.py / train_mlm.py)
  2. Offline on saved checkpoints (CLI: python3 eval_benchmarks.py <ckpt>)

Five benchmarks (the head-to-head numbers from the paper's ablation table):
  A. **A4 DeepAptamer enrichment**: linear probe Pearson r on enrichment_frequency
     regression. Test split = last 30% of held-out reads. Bar to beat: best general
     RNA FM is 0.066 Pearson (Mach-1/EVA/RNA-FM/etc).
  B. **AptaTrans/Li 2014 API classification**: logistic probe AUROC on the 580/145
     train/test pair split. Bar to beat: AptaTrans's 0.921 AUROC.
  C. **InstructNA LOX1/CXCL5 panel R@1/R@10/Spearman**: retrieve panel members vs
     R0 distractors; Spearman ρ between mean-sim-to-strong-binders and -Kd.
  D. **UTexas DB Kd regression**: linear probe Pearson on log10(Kd) over the 1,112
     entries with measured Kd. Random 80/20 split.
  E. **Held-out test split retrieval**: anchor = test seq, gallery = test + R0
     distractors, R@1/R@10. Tests OOD generalization.

To keep contrastive (L2-normalized projection) and MLM (no projection) apples-to-
apples, all benchmarks use the encoder's POOLED HIDDEN STATE (via .encode()) as
the feature representation. This is the layer immediately before the head split.

The function takes an `encode_fn(ids) -> [B, d]` so it works with any model.

Usage:
  python3 eval_benchmarks.py /path/to/ckpt_step10000.pt
  python3 eval_benchmarks.py /path/to/ckpt --device mps
  python3 eval_benchmarks.py /path/to/ckpt --report-only-keys r_at_1,a4_pearson
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_dataset import encode


# Paths to benchmark parquets (resolves home for Thunder + Mac transparently).
BENCH = {
    'a4_deepaptamer':    Path.home() / 'Desktop/autoRNA_data/tumbleweed/deepaptamer/processed/deepaptamer_combined.parquet',
    'aptatrans_li':      Path.home() / 'Desktop/autoRNA_data/tumbleweed/aptatrans_li/processed/aptatrans_li_pairs.parquet',
    'instructna_panel':  Path.home() / 'Desktop/autoRNA_data/tumbleweed/instructna_benchmark/processed/instructna_lox1_cxcl5_kd.parquet',
    'utexas_db':         Path.home() / 'Desktop/autoRNA_data/tumbleweed/utexas_db/processed/utexas_aptamer_db.parquet',
    'syn_r0_lox1':       Path.home() / 'Desktop/autoRNA_data/tumbleweed/synthetic_r0/InstructNA_LOX1/synthetic_r0.parquet',
    'aptamer_splits':    Path('/home/ubuntu/Tumbleweed/data_refs/aptamer_splits.parquet'),
    'raptscore_spr':     Path(__file__).resolve().parent.parent / 'data_refs/raptscore_spr_eval.parquet',
}


@torch.no_grad()
def _embed(encode_fn, seqs: list[str], max_len: int, device, batch: int = 256) -> np.ndarray:
    """Apply encode_fn to a list of sequences in mini-batches; return CPU numpy."""
    out = []
    for i in range(0, len(seqs), batch):
        ids = torch.tensor(np.stack([encode(s, max_len) for s in seqs[i:i+batch]]),
                           device=device)
        emb = encode_fn(ids).float().cpu().numpy()
        out.append(emb)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1))


def bench_a4(encode_fn, max_len: int, device, n_subsample: int = 30_000) -> dict:
    """A. Linear probe Pearson r on DeepAptamer enrichment frequency."""
    p = BENCH['a4_deepaptamer']
    if not p.exists(): return {'a4_pearson': None, 'a4_n': 0, 'a4_skip': 'parquet missing'}
    df = pd.read_parquet(p, columns=['aptamer_sequence_RNA', 'enrichment_frequency'])
    df = df.dropna()
    if len(df) > n_subsample:
        df = df.sample(n_subsample, random_state=0).reset_index(drop=True)
    seqs = df['aptamer_sequence_RNA'].tolist()
    y = np.log10(df['enrichment_frequency'].values.clip(min=1e-10))
    X = _embed(encode_fn, seqs, max_len, device)
    n_train = int(len(X) * 0.7)
    reg = Ridge(alpha=1.0).fit(X[:n_train], y[:n_train])
    pred = reg.predict(X[n_train:])
    rho, _ = pearsonr(pred, y[n_train:])
    return {'a4_pearson': float(rho), 'a4_n_test': int(len(X) - n_train)}


def bench_li(encode_fn, max_len: int, device) -> dict:
    """B. Logistic probe AUROC on Li 2014 aptamer-protein binding pairs."""
    p = BENCH['aptatrans_li']
    if not p.exists(): return {'li_auroc': None, 'li_skip': 'parquet missing'}
    df = pd.read_parquet(p, columns=['aptamer_sequence_RNA', 'label_binary', 'split'])
    train_df = df[df['split'] == 'training']
    test_df  = df[df['split'] == 'test']
    Xtr = _embed(encode_fn, train_df['aptamer_sequence_RNA'].tolist(), max_len, device)
    Xte = _embed(encode_fn, test_df['aptamer_sequence_RNA'].tolist(),  max_len, device)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, train_df['label_binary'].values)
    pred = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(test_df['label_binary'].values, pred)
    return {'li_auroc': float(auc), 'li_n_test': int(len(Xte))}


def bench_instructna_panel(encode_fn, max_len: int, device,
                            n_distractors: int = 1000) -> dict:
    """C. R@1/R@10 + Kd Spearman on the LOX1+CXCL5 panel."""
    p = BENCH['instructna_panel']
    r0_p = BENCH['syn_r0_lox1']
    if not p.exists(): return {'panel_skip': 'parquet missing'}
    panel = pd.read_parquet(p)
    r0 = pd.read_parquet(r0_p).head(n_distractors)
    panel_emb = _embed(encode_fn, panel['sequence'].tolist(), max_len, device)
    r0_emb = _embed(encode_fn, r0['sequence'].tolist(), max_len, device)

    # Normalize for cosine sim
    pe = panel_emb / (np.linalg.norm(panel_emb, axis=1, keepdims=True) + 1e-9)
    re = r0_emb / (np.linalg.norm(r0_emb, axis=1, keepdims=True) + 1e-9)
    all_e = np.concatenate([pe, re], axis=0)
    sims = pe @ all_e.T
    np.fill_diagonal(sims, -1e9)
    topk = np.argsort(-sims, axis=1)[:, :10]
    is_panel = topk < len(pe)
    r_at_1  = float(is_panel[:, 0].mean())
    r_at_10 = float(is_panel.any(axis=1).mean())

    strong_idx = (panel['kd_class'] == 'strong').values
    if strong_idx.sum() >= 2:
        strong_emb = pe[strong_idx]
        panel_to_strong = (pe @ strong_emb.T).mean(axis=1)
        kd_known_mask = panel['kd_nm'].notna().values
        if kd_known_mask.sum() >= 5:
            rho, _ = spearmanr(panel_to_strong[kd_known_mask],
                               -panel.loc[kd_known_mask, 'kd_nm'].values)
        else:
            rho = float('nan')
    else:
        rho = float('nan')

    return {
        'panel_r_at_1': r_at_1, 'panel_r_at_10': r_at_10,
        'panel_kd_spearman': float(rho) if not np.isnan(rho) else None,
        'panel_n': len(pe),
    }


def bench_utexas(encode_fn, max_len: int, device) -> dict:
    """D. Linear probe Pearson on log10(Kd_nM) over UTexas DB entries with Kd."""
    p = BENCH['utexas_db']
    if not p.exists(): return {'utexas_pearson': None, 'utexas_skip': 'parquet missing'}
    df = pd.read_parquet(p, columns=['aptamer_sequence_clean', 'kd_nm'])
    df = df.dropna()
    df = df[(df['kd_nm'] > 0) & (df['kd_nm'] < 1e8)]
    if len(df) < 50:
        return {'utexas_pearson': None, 'utexas_skip': 'too few labels'}
    seqs = df['aptamer_sequence_clean'].tolist()
    y = np.log10(df['kd_nm'].values)
    X = _embed(encode_fn, seqs, max_len, device)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_train = int(len(X) * 0.8)
    tr, te = idx[:n_train], idx[n_train:]
    reg = Ridge(alpha=1.0).fit(X[tr], y[tr])
    pred = reg.predict(X[te])
    rho, _ = pearsonr(pred, y[te])
    return {'utexas_pearson': float(rho), 'utexas_n_test': int(len(te))}


def bench_test_retrieval(encode_fn, max_len: int, device,
                          n_test_max: int = 2000, n_dist: int = 5000) -> dict:
    """E. Retrieval R@1/R@10 over the held-out test split sequences as anchors,
    test sequences as gallery, R0 reads as distractors."""
    sp = BENCH['aptamer_splits']
    r0_p = BENCH['syn_r0_lox1']
    if not sp.exists(): return {'test_skip': 'splits parquet missing'}
    splits = pd.read_parquet(sp, columns=['sequence', 'split', 'cluster_rep'])
    test_seqs = splits[splits['split'] == 'test']
    if len(test_seqs) > n_test_max:
        test_seqs = test_seqs.sample(n_test_max, random_state=0)
    r0 = pd.read_parquet(r0_p).head(n_dist)
    test_emb = _embed(encode_fn, test_seqs['sequence'].astype(str).tolist(),
                      max_len, device)
    r0_emb = _embed(encode_fn, r0['sequence'].astype(str).tolist(), max_len, device)
    te = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-9)
    re = r0_emb / (np.linalg.norm(r0_emb, axis=1, keepdims=True) + 1e-9)
    all_e = np.concatenate([te, re], axis=0)
    sims = te @ all_e.T
    np.fill_diagonal(sims, -1e9)
    topk = np.argsort(-sims, axis=1)[:, :10]
    is_test = topk < len(te)
    return {
        'test_r_at_1': float(is_test[:, 0].mean()),
        'test_r_at_10': float(is_test.any(axis=1).mean()),
        'test_n_anchor': int(len(te)),
        'test_n_dist': int(len(re)),
    }


@torch.no_grad()
def bench_raptscore_pll(mlm_fn, max_len: int, device) -> dict:
    """F. RaptScore-style Pseudo-Log-Likelihood on the 171 SPR-labeled aptamers
    from RaptScore 2026 Supp Tables 4-9.

    For each sequence: mask one nucleotide at a time (skip first/last for boundary
    effects, mimicking RaptScore's "start at 2nd token from each end" rule),
    sum log p(true token | masked context). Higher PLL = more "natural" sequence
    under the MLM. Pearson r vs Relative Activity reports per-dataset correlation.

    RaptScore paper (NAR 2026, gkaf1480) reports r = 0.65 / 0.78 / 0.65 on
    Datasets A / B / C with their DNABERT-3 + continual-PT model. Beating those
    using our MLM here = direct head-to-head win at matched scoring method.

    `mlm_fn(ids) -> Tensor[B, L, V]` returns vocab logits. None / skip for
    contrastive-only models.
    """
    if mlm_fn is None:
        return {'pll_skip': 'no mlm_fn provided (contrastive-only model)'}
    p = BENCH['raptscore_spr']
    if not p.exists():
        return {'pll_skip': f'{p} missing'}

    from aptamer_dataset import PAD_ID, MASK_ID
    df = pd.read_parquet(p, columns=['sequence', 'dataset', 'relative_activity', 'selection_method'])
    df = df.dropna(subset=['sequence', 'relative_activity'])

    plls = []
    for seq in df['sequence']:
        seq_str = str(seq).upper().replace('T', 'U')
        ids = encode(seq_str, max_len)            # numpy [max_len]
        L = int((ids != PAD_ID).sum())
        if L < 5:
            plls.append(float('nan'))
            continue
        positions = list(range(1, L - 1))         # skip first/last
        masked = np.tile(ids, (len(positions), 1))
        for i, pos in enumerate(positions):
            masked[i, pos] = MASK_ID
        masked_t = torch.from_numpy(masked).long().to(device)
        logits = mlm_fn(masked_t).float()         # [num_pos, max_len, V]
        log_probs = torch.log_softmax(logits, dim=-1)
        pos_t = torch.tensor(positions, device=device)
        true_tokens = torch.tensor([int(ids[p]) for p in positions], device=device)
        lp_at_mask = log_probs[torch.arange(len(positions), device=device), pos_t, true_tokens]
        plls.append(float(lp_at_mask.sum().item()))

    df = df.assign(pll=plls)
    out = {}
    for ds in ['A', 'B', 'C']:
        # Compare only on Freq/Enrichment-selected (matches what RaptScore reports)
        sub = df[(df['dataset'] == ds) & (df['selection_method'] == 'Frequency/Enrichment')].dropna(subset=['pll'])
        if len(sub) < 5:
            out[f'pll_pearson_{ds}'] = None
            out[f'pll_n_{ds}'] = int(len(sub))
            continue
        r, _ = pearsonr(sub['pll'], sub['relative_activity'])
        out[f'pll_pearson_{ds}'] = float(r)
        out[f'pll_n_{ds}'] = int(len(sub))
    return out


def run_all(encode_fn, max_len: int, device, mlm_fn=None) -> dict:
    out = {}
    t0 = time.time()
    out.update(bench_instructna_panel(encode_fn, max_len, device))
    out.update(bench_a4(encode_fn, max_len, device))
    out.update(bench_li(encode_fn, max_len, device))
    out.update(bench_utexas(encode_fn, max_len, device))
    out.update(bench_test_retrieval(encode_fn, max_len, device))
    out.update(bench_raptscore_pll(mlm_fn, max_len, device))
    out['eval_seconds'] = round(time.time() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--max-len', type=int, default=128)
    args = ap.parse_args()

    if args.device == 'auto':
        device = (torch.device('cuda') if torch.cuda.is_available()
                  else torch.device('mps') if torch.backends.mps.is_available()
                  else torch.device('cpu'))
    else:
        device = torch.device(args.device)
    print(f'device: {device}')

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck.get('config', {})
    model_cfg = dict(cfg.get('model', {}))
    model_cfg.setdefault('max_len', args.max_len)

    # Detect contrastive vs MLM by which state-dict key is present
    if 'encoder' in ck:
        from aptamer_encoder import AptamerEncoder
        model_cfg.pop('grad_checkpoint', None)
        model = AptamerEncoder(**model_cfg).to(device)
        model.load_state_dict(ck['encoder'])
        kind = 'contrastive'
    else:
        from aptamer_mlm import AptamerMLMModel
        model_cfg.pop('embed_dim', None)
        model_cfg.pop('grad_checkpoint', None)
        model = AptamerMLMModel(**model_cfg).to(device)
        model.load_state_dict(ck['model'])
        kind = 'mlm'
    model.eval()
    print(f'kind: {kind}, params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    encode_fn = lambda ids: model.encode(ids)
    results = run_all(encode_fn, max_len=cfg.get('data', {}).get('max_len', args.max_len), device=device)
    results['kind'] = kind
    results['ckpt'] = str(args.checkpoint)
    results['step'] = ck.get('step', None)
    print(json.dumps(results, indent=2))

    out = args.checkpoint.parent / f'{args.checkpoint.stem}_bench.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'→ {out}')


if __name__ == '__main__':
    main()
