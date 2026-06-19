"""Per-benchmark / per-target evaluation on the held-out splits.

For each `benchmark_id` in aptamer_splits_v2.parquet, compute:
  - Pearson r (where kd_value is present + continuous)
  - Target-retrieval R@1 / R@10: for each anchor, rank a gallery of (all OTHER
    test seqs + R0 distractors) by cosine similarity, count fraction of the top-K
    that share the anchor's `target_protein`. This is the metric that tells us
    "did corpus expansion teach the model to cluster sequences by their target?"
  - Mean cosine-similarity-to-top-K within-benchmark binder score (legacy)

Then aggregates per `target_protein` — useful for the corpus-expansion benchmarks
(`corpus_v2_{parp1,mecp2,snca,hiv1_rt}`) where each target has exactly one
benchmark, but the per-target view is what we report.

Uses target_protein_embeddings.parquet for target conditioning where the model
supports it (target_cond_mode != None in the checkpoint config).

Usage:
  python3 eval_per_benchmark.py <ckpt_path>
  python3 eval_per_benchmark.py ~/Desktop/.../ckpt_step5000.pt --output report.json
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
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptamer_dataset import encode

# Z-score normalization stats (must match what dataset does at load time).
# Computed once and cached.
_TARGET_EMB_CACHE: dict = {}


def load_target_embeddings(parquet_path: Path | None) -> tuple[dict, int, bool]:
    """Returns (target_protein → embedding numpy, embed_dim, is_per_residue).
    Mean-pool: z-scored. Per-residue: raw (z-score not applied per-residue)."""
    if parquet_path is None or not Path(parquet_path).exists():
        return {}, 0, False
    te = pd.read_parquet(parquet_path)
    is_pr = 'residue_embeddings' in te.columns
    if is_pr:
        out = {}
        dim = 0
        for _, row in te.iterrows():
            inner = [np.asarray(r, dtype=np.float32) for r in row['residue_embeddings']]
            out[row['target_protein']] = np.stack(inner, axis=0)
            dim = int(row['embedding_dim'])
        return out, dim, True
    # Mean-pool z-scored
    embs = np.asarray([np.asarray(e, dtype=np.float32) for e in te['embedding']])
    mu = embs.mean(axis=0, keepdims=True)
    sd = embs.std(axis=0, keepdims=True) + 1e-6
    embs_z = (embs - mu) / sd
    out = {}
    for i, row in te.reset_index(drop=True).iterrows():
        out[row['target_protein']] = embs_z[i].astype(np.float32)
    return out, int(te['embedding_dim'].iloc[0]), False


@torch.no_grad()
def encode_batch(model, seqs: list[str], target_embs: list[np.ndarray] | None,
                  max_len: int, device, batch: int = 64) -> np.ndarray:
    """Encode a batch of sequences with optional per-sample target embeddings."""
    is_per_residue = target_embs is not None and target_embs and target_embs[0].ndim == 2
    out = []
    for i in range(0, len(seqs), batch):
        chunk_seqs = seqs[i:i+batch]
        ids = torch.tensor(np.stack([encode(s, max_len) for s in chunk_seqs]), device=device)
        kwargs = {}
        if target_embs is not None:
            chunk_tes = target_embs[i:i+batch]
            if is_per_residue:
                Lmax = max(e.shape[0] for e in chunk_tes)
                D = chunk_tes[0].shape[1]
                padded = np.zeros((len(chunk_tes), Lmax, D), dtype=np.float32)
                mask = np.ones((len(chunk_tes), Lmax), dtype=bool)
                for k, e in enumerate(chunk_tes):
                    L = e.shape[0]
                    padded[k, :L] = e
                    mask[k, :L] = False
                kwargs['target_emb'] = torch.from_numpy(padded).to(device)
                kwargs['target_mask'] = torch.from_numpy(mask).to(device)
            else:
                kwargs['target_emb'] = torch.from_numpy(np.stack(chunk_tes)).to(device)
        emb = model.encode(ids, **kwargs)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def per_benchmark_eval(model, df: pd.DataFrame, target_emb_map: dict,
                       embed_dim: int, is_per_residue: bool,
                       max_len: int, device, top_k: int = 5) -> list[dict]:
    """For each benchmark_id, compute:
      - n: number of test sequences
      - pearson_r: Pearson between embedding-based score and kd_value (where avail)
      - retrieval_R@1, R@10: leave-one-out top-K retrieval within benchmark
      - target_protein, organism: paper-level metadata
    """
    rows = []
    null_emb = (np.zeros((1, max(1, embed_dim)), dtype=np.float32) if is_per_residue
                else np.zeros(max(1, embed_dim), dtype=np.float32))

    for bench_id, sub in df.groupby('benchmark_id'):
        n = len(sub)
        if n < 3:
            continue
        seqs = sub['sequence'].astype(str).str.upper().tolist()
        # Per-sample target embedding (zero vector if target_protein unknown)
        target_embs = None
        if target_emb_map:
            target_embs = []
            for tp in sub['target_protein']:
                emb = target_emb_map.get(tp, null_emb)
                target_embs.append(emb)
        embeds = encode_batch(model, seqs, target_embs, max_len, device)
        # L2-normalize
        embeds = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9)

        # Score = mean cosine similarity to top-K other binders within this benchmark
        sims = embeds @ embeds.T
        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            # Top-K other binders by kd_value descending (or by themselves if no kd)
            if sub['kd_value'].notna().any():
                kd_sorted = sub.iloc[
                    [j for j in range(n) if j != i]
                ].sort_values('kd_value', ascending=False).index
                ref_local_idx = [sub.index.get_loc(idx) for idx in kd_sorted[:top_k]]
            else:
                ref_local_idx = [j for j in range(n) if j != i][:top_k]
            scores[i] = float(sims[i, ref_local_idx].mean())

        # Pearson r (only if kd_value continuous)
        kd_arr = pd.to_numeric(sub['kd_value'], errors='coerce').values
        kd_mask = ~np.isnan(kd_arr)
        if kd_mask.sum() >= 5 and len(np.unique(kd_arr[kd_mask])) >= 3:
            r, p = pearsonr(scores[kd_mask], kd_arr[kd_mask])
            pearson_r = float(r)
            pearson_p = float(p)
        else:
            pearson_r = None
            pearson_p = None

        # NB: proper target-retrieval R@K is computed by cross_benchmark_target_retrieval()
        # below (over all benchmarks + R0 distractors in one shared gallery). The
        # old in-benchmark "R@K" block was trivial (r_at_1=1 for every row) and has
        # been removed — see cross_benchmark_target_retrieval for the real metric.

        target_protein = sub['target_protein'].mode().iloc[0] if len(sub['target_protein'].mode()) else None
        rows.append(dict(
            benchmark_id=bench_id,
            n=int(n),
            target_protein=target_protein,
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            mean_score=float(scores.mean()),
            score_std=float(scores.std()),
        ))
    return rows


def cross_benchmark_target_retrieval(model, df: pd.DataFrame, target_emb_map: dict,
                                      embed_dim: int, is_per_residue: bool,
                                      max_len: int, device,
                                      r0_path: Path | str | None = None,
                                      n_distractors: int = 2000,
                                      ks: tuple[int, ...] = (1, 5, 10),
                                      max_anchors_per_bench: int = 500) -> list[dict]:
    """Real target-retrieval R@K with a cross-benchmark gallery + R0 distractors.

    For each anchor (a test seq with a known target_protein):
      gallery = (all OTHER test seqs across all benchmarks) + n_distractors R0 seqs
      rank gallery by cosine similarity (with target conditioning per the anchor's target)
      R@K = fraction of top-K that share the anchor's target_protein

    Aggregated per `benchmark_id`. Random baseline = (n_same_in_gallery) / (n_total_gallery).

    For very large benchmarks (corpus_v2_* with 3K rows), we subsample anchors to
    `max_anchors_per_bench` to keep wall time tractable.
    """
    work = df[df['target_protein'].notna() & (df['target_protein'] != 'unknown')].copy()
    work = work[work['target_protein'].astype(str) != 'sparkseq_unknown']
    work = work.reset_index(drop=True)
    if len(work) < 5:
        return []

    null_emb = (np.zeros((1, max(1, embed_dim)), dtype=np.float32) if is_per_residue
                else np.zeros(max(1, embed_dim), dtype=np.float32))

    # 1. Embed all test seqs (conditioned on their target_protein)
    seqs = work['sequence'].astype(str).str.upper().tolist()
    targets = work['target_protein'].astype(str).tolist()
    target_embs = None
    if target_emb_map:
        target_embs = [target_emb_map.get(tp, null_emb) for tp in targets]
    print(f'  embedding {len(seqs):,} test seqs across {len(set(targets))} targets ...')
    test_embeds = encode_batch(model, seqs, target_embs, max_len, device, batch=128)
    test_embeds = test_embeds / (np.linalg.norm(test_embeds, axis=1, keepdims=True) + 1e-9)

    # 2. Embed R0 distractors (no target conditioning — null target embedding)
    r0_path = r0_path or (Path.home() / 'Desktop/autoRNA_data/tumbleweed/synthetic_r0/InstructNA_LOX1/synthetic_r0.parquet')
    r0_path = Path(r0_path)
    r0_embeds = np.zeros((0, test_embeds.shape[1]), dtype=np.float32)
    if r0_path.exists():
        r0_df = pd.read_parquet(r0_path).head(n_distractors)
        r0_seqs = r0_df['sequence'].astype(str).str.upper().tolist()
        r0_target_embs = [null_emb] * len(r0_seqs) if target_emb_map else None
        print(f'  embedding {len(r0_seqs):,} R0 distractors ...')
        r0_embeds = encode_batch(model, r0_seqs, r0_target_embs, max_len, device, batch=128)
        r0_embeds = r0_embeds / (np.linalg.norm(r0_embeds, axis=1, keepdims=True) + 1e-9)

    gallery_embeds = np.concatenate([test_embeds, r0_embeds], axis=0)
    gallery_targets = np.asarray(targets + ['__R0__'] * len(r0_embeds), dtype=object)
    n_gallery = len(gallery_embeds)
    print(f'  gallery: {len(test_embeds):,} test seqs + {len(r0_embeds):,} R0 = {n_gallery:,}')

    # 3. For each benchmark, R@K for its anchors against the gallery
    rng = np.random.default_rng(0)
    kmax = max(ks)
    rows = []
    for bench_id, sub in work.groupby('benchmark_id'):
        anchor_idx = sub.index.tolist()
        if len(anchor_idx) > max_anchors_per_bench:
            anchor_idx = list(rng.choice(anchor_idx, size=max_anchors_per_bench, replace=False))
        anchor_target = sub['target_protein'].mode().iloc[0] if len(sub['target_protein'].mode()) else None
        if anchor_target is None:
            continue
        # n_same_in_gallery (excluding self for each anchor): same target_protein in test_embeds
        n_same_total = int((gallery_targets == anchor_target).sum())
        # Random baseline (per-anchor effective): (n_same - 1) / (n_gallery - 1)
        baseline = (n_same_total - 1) / max(1, n_gallery - 1)
        recall_counts = {k: 0 for k in ks}
        n_anchors = len(anchor_idx)
        for ai in anchor_idx:
            sims = gallery_embeds @ test_embeds[ai]
            sims[ai] = -np.inf  # exclude self
            top = np.argpartition(-sims, kmax)[:kmax]
            top = top[np.argsort(-sims[top])]
            same = (gallery_targets[top] == anchor_target)
            for k in ks:
                if same[:k].any():
                    recall_counts[k] += 1
        rows.append(dict(
            benchmark_id=bench_id,
            target_protein=anchor_target,
            n_anchors=int(n_anchors),
            n_same_in_gallery=int(n_same_total),
            random_baseline=baseline,
            **{f'R@{k}': recall_counts[k] / n_anchors for k in ks},
        ))
    return rows


def corpus_v2_retrieval(model, df: pd.DataFrame, target_emb_map: dict,
                        embed_dim: int, is_per_residue: bool,
                        max_len: int, device,
                        ks: tuple[int, ...] = (1, 5, 10)) -> list[dict]:
    """Cross-target retrieval over the corpus_v2_* hold-out pool.

    The four corpus_v2 benchmarks each contribute ~3K test seqs from a single
    target (PARP1, MECP2, SNCA, HIV1_RT). For each anchor we rank ALL other
    pool seqs by cosine similarity and check: of the top-K, what fraction
    share the anchor's target_protein? Random baseline = (n_same - 1) /
    (n_total - 1) ≈ 0.25 in a balanced pool.

    Returns one row per target with R@K and same-target AUROC.
    """
    pool = df[df['benchmark_id'].astype(str).str.startswith('corpus_v2_')].copy()
    if pool.empty:
        return []
    pool = pool.reset_index(drop=True)
    targets = np.asarray(pool['target_protein'].astype(str), dtype=object)
    seqs = pool['sequence'].astype(str).str.upper().tolist()
    null_emb = (np.zeros((1, max(1, embed_dim)), dtype=np.float32) if is_per_residue
                else np.zeros(max(1, embed_dim), dtype=np.float32))
    target_embs = None
    if target_emb_map:
        target_embs = [target_emb_map.get(tp, null_emb) for tp in targets]

    n = len(pool)
    print(f'  corpus_v2 retrieval pool: {n:,} seqs across {len(set(targets))} targets')
    embeds = encode_batch(model, seqs, target_embs, max_len, device, batch=128)
    embeds = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9)

    # Compute cosine sims in row-chunks to keep peak memory bounded.
    rows: list[dict] = []
    chunk = 512
    kmax = max(ks)
    # Per-anchor: store top-kmax indices (excluding self) and full AUROC stats.
    topk_indices = np.zeros((n, kmax), dtype=np.int32)
    # For AUROC we need ranking of all gallery items per anchor; do it incrementally
    # via per-target hits-at-rank counts to avoid storing full N×N.
    same_target_mask_global = np.zeros((n, n), dtype=bool)  # n=12k => 144MB bool, ok
    for tgt in set(targets):
        idx = np.where(targets == tgt)[0]
        same_target_mask_global[np.ix_(idx, idx)] = True

    auroc_sums: dict[str, list[float]] = {t: [] for t in set(targets)}
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = embeds[start:end] @ embeds.T  # (chunk, n)
        for i_local, i_global in enumerate(range(start, end)):
            row = block[i_local].copy()
            row[i_global] = -np.inf  # exclude self
            # top-kmax indices, descending
            top = np.argpartition(-row, kmax)[:kmax]
            top = top[np.argsort(-row[top])]
            topk_indices[i_global] = top
            # AUROC: rank gallery (self removed) by cosine; positives = same target.
            keep = np.ones(n, dtype=bool)
            keep[i_global] = False
            scores_k = row[keep]
            labels_k = same_target_mask_global[i_global][keep]
            n_pos = int(labels_k.sum())
            n_neg = scores_k.size - n_pos
            if n_pos > 0 and n_neg > 0:
                # Mann-Whitney U expects ascending ranks (rank 1 = lowest score).
                order_k = np.argsort(scores_k)
                ranks_k = np.empty(scores_k.size, dtype=np.float64)
                ranks_k[order_k] = np.arange(1, scores_k.size + 1)
                sum_ranks_pos = ranks_k[labels_k].sum()
                auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
                auroc_sums[targets[i_global]].append(float(auroc))

    # Per-target metrics from topk_indices + auroc_sums
    for tgt in sorted(set(targets)):
        anchor_idx = np.where(targets == tgt)[0]
        n_anchors = len(anchor_idx)
        n_same_total = n_anchors  # includes self; for R@K denominator use anchors only
        recall_by_k = {}
        for k in ks:
            top = topk_indices[anchor_idx, :k]  # (n_anchors, k)
            same = (targets[top] == tgt).astype(np.float32)
            recall_by_k[k] = float(same.mean())
        # Random baseline: (n_same_total - 1) / (n - 1)
        baseline = (n_same_total - 1) / (n - 1)
        auroc_mean = float(np.mean(auroc_sums[tgt])) if auroc_sums[tgt] else None
        rows.append(dict(
            target_protein=tgt,
            n_anchors=int(n_anchors),
            pool_size=int(n),
            random_baseline=baseline,
            **{f'R@{k}': recall_by_k[k] for k in ks},
            auroc_same_target=auroc_mean,
        ))
    return rows


def categorize_benchmark(target_protein: str | None) -> str:
    """In-distribution = training-corpus target; zero-shot = held-out.
    Mirrors target_registry.csv role='training' as of 2026-05-23 (clean rebuild)."""
    IN_DIST = {
        # Original 14
        'FGF9', 'IL1RL1', 'TGM2', 'LOX1', 'CXCL5',
        'PTK7', 'NRP1', 'CDCP1', 'ITGA3', 'PTPRD_F_S',
        'PARP1', 'MECP2', 'SNCA', 'HIV1_RT',
        # Clean rebuild additions: DeepAptamer untangle
        'BCMA', 'CTGF', 'DKK1',
        # Clean rebuild additions: Tier1 per-study
        'ANXA2', 'RpoA', 'F13A1', 'NDM_1', 'eIF4A_Oryza_sativa',
    }
    if target_protein in IN_DIST:
        return 'in_distribution'
    if target_protein in ('unknown', None, 'sparkseq_unknown'):
        return 'unconditional'
    return 'zero_shot'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--splits', type=Path,
                    default=Path(__file__).parent.parent / 'data_refs/aptamer_splits_v2.parquet')
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()

    device = (torch.device('cuda') if args.device == 'auto' and torch.cuda.is_available()
              else torch.device('mps') if args.device == 'auto' and torch.backends.mps.is_available()
              else torch.device(args.device) if args.device != 'auto'
              else torch.device('cpu'))
    print(f'device: {device}')
    print(f'checkpoint: {args.checkpoint}')

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck.get('config', {})
    target_emb_parquet = cfg.get('data', {}).get('target_embeddings_parquet')
    target_emb_map, embed_dim, is_per_residue = load_target_embeddings(
        Path(target_emb_parquet) if target_emb_parquet else None)
    if target_emb_map:
        print(f'  loaded {len(target_emb_map)} target embeddings (dim={embed_dim}, '
              f'per_residue={is_per_residue})')

    from aptamer_encoder import AptamerEncoder
    model_cfg = dict(cfg.get('model', {}))
    model_cfg.pop('grad_checkpoint', None)
    model_cfg.setdefault('max_len', args.max_len)
    model = AptamerEncoder(**model_cfg).to(device).eval()
    model.load_state_dict(ck['encoder'])
    print(f'  loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, '
          f'cond_mode={model.target_cond_mode}')

    df = pd.read_parquet(args.splits)
    df = df[df['split'] == 'test'].copy()
    print(f'  loaded {len(df):,} test sequences across {df["benchmark_id"].nunique()} benchmarks')

    t0 = time.time()
    rows = per_benchmark_eval(
        model, df, target_emb_map, embed_dim, is_per_residue, args.max_len, device)
    print(f'  evaluated in {time.time()-t0:.1f}s')

    out = pd.DataFrame(rows)
    out['category'] = out['target_protein'].apply(categorize_benchmark)
    out = out.sort_values(['category', 'pearson_r' if 'pearson_r' in out.columns else 'n'],
                          ascending=[True, False], na_position='last')

    # Per-benchmark table
    print(f'\n{"Benchmark":<40} {"Target":<20} {"Cat":<14} {"n":>6} {"Pearson r":>10}')
    print('-' * 100)
    for _, r in out.iterrows():
        pr = f'{r["pearson_r"]:+.3f}' if pd.notna(r["pearson_r"]) else '   —  '
        cat = r['category'][:14]
        print(f'  {r["benchmark_id"]:<38} {str(r["target_protein"])[:18]:<20} {cat:<14} '
              f'{r["n"]:>6} {pr:>10}')

    # Summary by category
    print(f'\n=== Summary by category ===')
    for cat, sub in out.groupby('category'):
        valid = sub[sub['pearson_r'].notna()]
        if len(valid):
            print(f'  {cat:<16}: n={len(sub)} benchmarks; '
                  f'pearson_r mean={valid["pearson_r"].mean():+.3f} '
                  f'(over {len(valid)} with kd labels)')
        else:
            print(f'  {cat:<16}: n={len(sub)} benchmarks; no kd labels')

    # Cross-benchmark target-retrieval R@K (real R@K, with R0 distractors).
    print(f'\n=== Cross-benchmark target-retrieval R@K (all benchmarks, shared gallery + R0 distractors) ===')
    t0 = time.time()
    xb_rows = cross_benchmark_target_retrieval(
        model, df, target_emb_map, embed_dim, is_per_residue, args.max_len, device)
    if xb_rows:
        print(f'  evaluated in {time.time()-t0:.1f}s\n')
        print(f'  {"Benchmark":<30} {"Target":<18} {"Cat":<14} {"n":>5} {"baseline":>9} {"R@1":>7} {"R@5":>7} {"R@10":>7}')
        print('  ' + '-' * 105)
        # Sort by category then R@1 desc
        for r in xb_rows:
            r['category'] = categorize_benchmark(r['target_protein'])
        for r in sorted(xb_rows, key=lambda x: (x['category'], -x['R@1'])):
            print(f'  {r["benchmark_id"]:<30} {r["target_protein"]:<18} {r["category"][:14]:<14} '
                  f'{r["n_anchors"]:>5} {r["random_baseline"]:>9.4f} '
                  f'{r["R@1"]:>7.3f} {r["R@5"]:>7.3f} {r["R@10"]:>7.3f}')

        # Per-target rollup
        print(f'\n=== Per-target rollup (avg R@K across benchmarks for that target) ===')
        by_target: dict = {}
        for r in xb_rows:
            tp = r['target_protein']
            by_target.setdefault(tp, []).append(r)
        target_rows = []
        for tp, rows_t in by_target.items():
            cat = categorize_benchmark(tp)
            n_anchors = sum(r['n_anchors'] for r in rows_t)
            avg_r1 = float(np.mean([r['R@1'] for r in rows_t]))
            avg_r5 = float(np.mean([r['R@5'] for r in rows_t]))
            avg_r10 = float(np.mean([r['R@10'] for r in rows_t]))
            avg_baseline = float(np.mean([r['random_baseline'] for r in rows_t]))
            target_rows.append(dict(
                target_protein=tp, category=cat, n_benchmarks=len(rows_t),
                n_anchors=n_anchors, random_baseline=avg_baseline,
                **{'R@1': avg_r1, 'R@5': avg_r5, 'R@10': avg_r10}
            ))
        print(f'  {"Target":<22} {"Cat":<14} {"#bench":>6} {"n":>5} {"baseline":>9} {"R@1":>7} {"R@5":>7} {"R@10":>7}')
        print('  ' + '-' * 95)
        for r in sorted(target_rows, key=lambda x: (x['category'], -x['R@1'])):
            print(f'  {r["target_protein"]:<22} {r["category"][:14]:<14} '
                  f'{r["n_benchmarks"]:>6} {r["n_anchors"]:>5} {r["random_baseline"]:>9.4f} '
                  f'{r["R@1"]:>7.3f} {r["R@5"]:>7.3f} {r["R@10"]:>7.3f}')
        # Category macro
        for cat, sub in pd.DataFrame(target_rows).groupby('category'):
            print(f'  --- {cat} macro: R@1={sub["R@1"].mean():.3f}  R@5={sub["R@5"].mean():.3f}  R@10={sub["R@10"].mean():.3f}  '
                  f'(baseline {sub["random_baseline"].mean():.4f}, {len(sub)} targets)')
    else:
        target_rows = []
        print('  no benchmarks with target_protein found')

    # Per-target cross-target retrieval on corpus_v2_* pool (PARP1/MECP2/SNCA/HIV1_RT).
    print(f'\n=== Corpus-v2 cross-target retrieval (within-pool same-target) ===')
    t0 = time.time()
    corpus_rows = corpus_v2_retrieval(
        model, df, target_emb_map, embed_dim, is_per_residue, args.max_len, device)
    if corpus_rows:
        print(f'  evaluated in {time.time()-t0:.1f}s\n')
        print(f'  {"Target":<12} {"n":>5} {"baseline":>9} '
              f'{"R@1":>7} {"R@5":>7} {"R@10":>7} {"AUROC":>7}')
        print('  ' + '-' * 70)
        for r in corpus_rows:
            auroc = f'{r["auroc_same_target"]:.3f}' if r['auroc_same_target'] is not None else '  —  '
            print(f'  {r["target_protein"]:<12} {r["n_anchors"]:>5} '
                  f'{r["random_baseline"]:>9.3f} '
                  f'{r["R@1"]:>7.3f} {r["R@5"]:>7.3f} {r["R@10"]:>7.3f} {auroc:>7}')
        # Macro-average row
        macro = lambda k: np.mean([r[k] for r in corpus_rows])
        auroc_vals = [r['auroc_same_target'] for r in corpus_rows if r['auroc_same_target'] is not None]
        auroc_macro = f'{np.mean(auroc_vals):.3f}' if auroc_vals else '  —  '
        print('  ' + '-' * 70)
        print(f'  {"MACRO":<12} {"":>5} {macro("random_baseline"):>9.3f} '
              f'{macro("R@1"):>7.3f} {macro("R@5"):>7.3f} {macro("R@10"):>7.3f} {auroc_macro:>7}')
    else:
        print('  no corpus_v2_* rows found in splits')

    if args.output:
        payload = {
            'per_benchmark': out.to_dict(orient='records'),
            'cross_benchmark_retrieval': xb_rows,
            'per_target_rollup': target_rows,
            'corpus_v2_retrieval': corpus_rows,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
        print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
