"""Kmer feature extraction for aptamer sequences.

Computes deterministic kmer feature vectors per sequence. Used by v18+ encoders
to inject explicit motif-composition signal alongside the learned transformer
representation.

Two feature families:
  - **standard_kmers**: contiguous k-mers, normalized counts. k ∈ {3,4,5}.
                       4³ + 4⁴ + 4⁵ = 64 + 256 + 1024 = 1344 features.
  - **gapped_kmers**:   gkm-SVM-style patterns of total length l with k
                       informative positions. For (k=3, l=4): one gap, 4 gap
                       positions × 4³ patterns = 256 features.
                       For (k=3, l=5): two gaps, C(5,3)=10 × 4³ = 640 features.

Defaults: standard k=3,4 + gapped (k=3, l=4) = 64+256+256 = 576-d vector.
Larger defaults blow up the projection layer; keep modest unless ablating.

Usage:
    from kmer_features import build_kmer_features
    feats = build_kmer_features(['ACGTACG', 'GGGCATC'], chemistry='DNA')
    # → np.ndarray (N, 576)
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations, product
from typing import Iterable

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Vocabulary (uppercase A/C/G/T; RNA U folded to T for hashability)
# ──────────────────────────────────────────────────────────────────────────
BASES = 'ACGT'
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}


def _normalize_seq(s: str) -> str:
    return s.upper().replace('U', 'T')


# ──────────────────────────────────────────────────────────────────────────
# Standard k-mers
# ──────────────────────────────────────────────────────────────────────────

def _kmer_vec(seq: str, k: int) -> np.ndarray:
    """Normalized k-mer count vector of length 4^k."""
    s = _normalize_seq(seq)
    counts = Counter()
    for i in range(len(s) - k + 1):
        sub = s[i:i + k]
        if all(c in BASES for c in sub):
            counts[sub] += 1
    vec = np.array(
        [counts.get(''.join(t), 0) for t in product(BASES, repeat=k)],
        dtype=np.float32,
    )
    total = vec.sum()
    return vec / total if total > 0 else vec


def standard_kmers(seqs: list[str], ks=(3, 4)) -> np.ndarray:
    """Stack normalized k-mer count vectors for k ∈ ks. (N, sum(4^k))."""
    feats = []
    for k in ks:
        feats.append(np.stack([_kmer_vec(s, k) for s in seqs]))
    return np.concatenate(feats, axis=1)


# ──────────────────────────────────────────────────────────────────────────
# Gapped k-mers (gkm-SVM style)
# ──────────────────────────────────────────────────────────────────────────

def _gapped_patterns(k: int, l: int) -> list[tuple[int, ...]]:
    """All ways to pick k informative positions out of l total."""
    return list(combinations(range(l), k))


def _gapped_kmer_vec(seq: str, k: int, l: int) -> np.ndarray:
    """For total length l with k informative positions, count each (positions, kmer)
    combination. Returns vector of length C(l,k) × 4^k."""
    s = _normalize_seq(seq)
    patterns = _gapped_patterns(k, l)
    n_pat = len(patterns)
    n_kmer = 4 ** k
    out = np.zeros(n_pat * n_kmer, dtype=np.float32)
    for i in range(len(s) - l + 1):
        window = s[i:i + l]
        if not all(c in BASES for c in window):
            continue
        for p_idx, positions in enumerate(patterns):
            kmer_idx = 0
            for pos in positions:
                kmer_idx = kmer_idx * 4 + BASE_TO_IDX[window[pos]]
            out[p_idx * n_kmer + kmer_idx] += 1
    total = out.sum()
    return out / total if total > 0 else out


def gapped_kmers(seqs: list[str], k: int = 3, l: int = 4) -> np.ndarray:
    """Stack gapped k-mer feature vectors. (N, C(l,k) * 4^k)."""
    return np.stack([_gapped_kmer_vec(s, k, l) for s in seqs])


# ──────────────────────────────────────────────────────────────────────────
# Combined builder (the convenience entrypoint for v18+)
# ──────────────────────────────────────────────────────────────────────────

def build_kmer_features(
    seqs: list[str],
    standard_ks: tuple[int, ...] = (3, 4, 5),
    gapped_configs: tuple[tuple[int, int], ...] = ((3, 4),),
    chemistry: str | None = None,    # reserved for chemistry-conditional features
) -> np.ndarray:
    """Concat feature vector per sequence: standard k-mers + gapped k-mers.

    Default: k=3,4,5 standard (1344) + gapped (k=3, l=4) (256) = 1600 features.
    """
    feats = [standard_kmers(seqs, ks=standard_ks)]
    for (k, l) in gapped_configs:
        feats.append(gapped_kmers(seqs, k=k, l=l))
    return np.concatenate(feats, axis=1)


def kmer_feature_dim(
    standard_ks: tuple[int, ...] = (3, 4, 5),
    gapped_configs: tuple[tuple[int, int], ...] = ((3, 4),),
) -> int:
    """Dimensionality of build_kmer_features output (for projection layer sizing)."""
    d = sum(4 ** k for k in standard_ks)
    for (k, l) in gapped_configs:
        from math import comb
        d += comb(l, k) * (4 ** k)
    return d


# ──────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    seqs = [
        'GGCATCTGACCTCTGTGCTGCT',
        'ATACCAGCTTATTCAATTATG',
        'CACGGCATGGTGGGCGTCGTG',
    ]
    feats = build_kmer_features(seqs)
    print(f'feats shape: {feats.shape}')
    print(f'expected dim: {kmer_feature_dim()}')
    print(f'first 8 values of seq 0: {feats[0][:8]}')
    print(f'standard k=3,4,5: {sum(4**k for k in (3,4,5))} features')
    print(f'gapped (k=3, l=4): {4 * 64} features')
    print(f'total: {sum(4**k for k in (3,4,5)) + 4 * 64}')
