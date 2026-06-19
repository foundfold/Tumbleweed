"""Streaming PyTorch dataset for contrastive aptamer pretraining (production version).

Design points:
  - **Streaming, not in-memory.** The corpus is ~1.1B reads (Tier-1 + Jolma 2013) —
    we never load it all. Each per-source parquet is opened with pyarrow.ParquetFile
    and read in batches. A shuffle buffer keeps things non-correlated.
  - **Source-weighted sampling.** Each YAML source has a `weight` scalar; reads are
    drawn proportionally. UTexas DB (kd_filter applied) gets 3× weight, DeepAptamer
    2×, synthetic R0 0.3× (lacks PCR biases vs empirical R0).
  - **Round-filtered.** Each source declares `round_filter` (e.g. `>= 3` for Jolma
    late-round, `== 0` for naive). The filter is a tiny eval'd expression in pandas.
  - **Leakage exclusion.** test-set sequences from data_refs/aptamer_splits.parquet
    are excluded from pretraining via a sequence-hash bloom filter (fast in-memory
    membership check at ingest).
  - **Tokenization.** 4-letter A/C/G/U (T→U), single-token PAD. Variable length
    padded to `max_len` per config (default 128).

Class API:
  ContrastiveAptamerIterableDataset.iter_pairs() yields dicts with:
    'anchor_ids':   LongTensor[max_len]  — late-round / enriched sequence
    'positive_ids': LongTensor[max_len]  — another late-round from same source
    'study_id':     int                  — per-source ID for cross-study sampling

A separate r0 pool is maintained for the in-batch contrastive negatives. The
training loop draws negatives via `dataset.sample_negatives(batch_size)`.
"""
from __future__ import annotations
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ALPHABET = list('ACGU')
TOKEN2ID = {b: i for i, b in enumerate(ALPHABET)}
PAD_ID = len(ALPHABET)               # = 4
MASK_ID = PAD_ID + 1                  # = 5, used only by MLM
# ─── Chemistry tokens (added 2026-05-25 for DNA/RNA dual-chemistry support) ───
# Prepended at position 0 of every sequence to signal chemistry context.
RNA_TOK_ID = PAD_ID + 2              # = 6
DNA_TOK_ID = PAD_ID + 3              # = 7
# ─── Separate T token (added 2026-05-28 to break chemistry collapse) ─────────
# Previously DNA's T was folded into U at encode time, so the only chemistry
# signal was the prepended [DNA]/[RNA] token (1 of ~80 positions). With T as
# its own token, chemistry differs at every T position in the sequence — the
# pool can't dilute it away and target conditioning can't override it.
T_ID = PAD_ID + 4                     # = 8 (DNA-specific thymine)
N_TOKENS = len(ALPHABET) + 5          # = 9 (A/C/G/U/PAD/MASK/[RNA]/[DNA]/T)
TOKEN2ID['T'] = T_ID
NUC_IDS = list(range(len(ALPHABET))) + [T_ID]  # for MLM random-token replacement


def encode(seq: str, max_len: int, chemistry: str | None = None) -> np.ndarray:
    """Encode RNA or DNA sequence as token IDs.

    Args:
        seq: nucleic acid sequence. If chemistry='RNA', any T is folded to U
             (RNA biologically has no T). If chemistry='DNA' or None, T is
             preserved as its own token (T_ID=8).
        max_len: total output length (includes chemistry token at pos 0 if set)
        chemistry: 'RNA' / 'DNA' / None. If set, prepend the matching chemistry
                   token at position 0, content shifts to positions 1+.
    """
    s = seq.upper()
    if chemistry == 'RNA':
        s = s.replace('T', 'U')  # RNA has no T; defensive in case sources spell it ACGT
    ids: list[int] = []
    if chemistry == 'RNA':
        ids.append(RNA_TOK_ID)
    elif chemistry == 'DNA':
        ids.append(DNA_TOK_ID)
    elif chemistry is not None:
        raise ValueError(f'unknown chemistry: {chemistry!r}')
    content_budget = max_len - len(ids)
    ids += [TOKEN2ID.get(c, PAD_ID) for c in s[:content_budget]]
    ids += [PAD_ID] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64)


def expand_user(p):
    return Path(os.path.expanduser(p)) if isinstance(p, str) else p


def build_alias(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vose's alias method — preprocess a length-N weighted distribution into
    two length-N tables so single-sample draws are O(1). Replaces the
    O(N)-per-call cost of `np.random.choice(N, p=probs)` which was bottlenecking
    big corpus parquets at scale.

    Returns (prob_table, alias_table). To sample one index from probs:
        i = rng.integers(N)
        return i if rng.random() < prob_table[i] else alias_table[i]
    """
    n = len(probs)
    p = np.asarray(probs, dtype=np.float64) * n  # scale so mean = 1
    prob_t = np.zeros(n, dtype=np.float64)
    alias_t = np.zeros(n, dtype=np.int64)
    small = [int(i) for i in np.where(p < 1.0)[0]]
    large = [int(i) for i in np.where(p >= 1.0)[0]]
    while small and large:
        s = small.pop()
        l = large.pop()
        prob_t[s] = p[s]
        alias_t[s] = l
        p[l] -= (1.0 - p[s])
        (small if p[l] < 1.0 else large).append(l)
    # Float drift can leave a few stragglers in either list; they get prob 1
    # (always picked, alias self).
    for i in small + large:
        prob_t[i] = 1.0
        alias_t[i] = i
    return prob_t, alias_t


def alias_sample(prob_t: np.ndarray, alias_t: np.ndarray, rng: np.random.Generator) -> int:
    n = len(prob_t)
    i = int(rng.integers(n))
    return i if rng.random() < prob_t[i] else int(alias_t[i])


@dataclass
class SourceSpec:
    name: str
    parquet: Path
    class_: str                # 'r0' or 'r_late'
    weight: float
    sequence_col: str = 'sequence'
    round_filter: Optional[str] = None      # e.g. '== 0', '>= 3', '>= 1 | isna'
    selex_round_override: Optional[int] = None
    kd_filter_lt_nm: Optional[float] = None
    tag: Optional[str] = None  # curriculum tag (e.g. 'easy_neg', 'mid_neg', 'hard_neg')
    # ─── RaptScore-style read-count weighting (NAR 2026, Fig 1C settings) ──────
    # When True, sample sequences with prob ∝ log(count)+1 (instead of uniformly
    # over unique seqs). Closes the duplicate-handling gap — RaptScore showed
    # keep-duplicates / log-weighted beats remove-duplicates. Requires the
    # parquet to have a `count_col` column. Only supported for eager-loaded
    # sources (small enough to hold the count array in memory).
    weight_by_count: bool = False
    count_col: str = 'count'
    # ─── Sliding-round-window curriculum + regression target ───────────────────
    # round_frac in [0, 1] — this source's normalized enrichment "rank".
    #   - synthetic R0  → 0.0
    #   - RaptScore A R3 of 6-round study → 3/6 = 0.5
    #   - RaptScore A R6 (most enriched)  → 6/6 = 1.0
    #   - curated binders (UTexas, DeepAptamer) → 1.0
    # Doubles as the regression-head target (predict round_frac per sequence).
    round_frac: Optional[float] = None
    # ─── Motif-family-aware sampling (downweight dominant families) ────────────
    # When `family_weighted=True`, sampling probability per sequence is divided by
    # its family's size raised to `family_weight_power`. Prevents one dominant
    # motif from overwhelming the InfoNCE objective (e.g. SPARK-seq PTK7 Family-1
    # is 97% of PTK7 entries; without correction the model collapses to that motif).
    # Combines multiplicatively with `weight_by_count` if both are enabled.
    family_weighted: bool = False
    family_col: str = 'motif_family'
    family_weight_power: float = 0.5  # 0.0 = no correction, 0.5 = 1/sqrt(size), 1.0 = 1/size
    # ─── Target-protein conditioning ───────────────────────────────────────────
    # If set, this source's sequences are conditioned on the given target protein
    # at training time. The trainer looks up the ESM-2 embedding from
    # `target_embeddings_parquet` and passes it through to the encoder. Sources
    # without a target (synthetic R0, generic naive libraries) leave this as None
    # and the model uses a zero vector / learned null target.
    target_protein: Optional[str] = None
    # ─── Chemistry token ('RNA' / 'DNA') ───────────────────────────────────────
    # Prepended to every sampled sequence as a chemistry-class token at position 0
    # (encoder side only — decoder doesn't see this). Default 'RNA' preserves
    # backward compatibility with all existing SELEX RNA training data. Set to
    # 'DNA' for DNA aptamer panels + their synthetic R0 partners.
    chemistry: str = 'RNA'
    # ─── Library identity for per-paper R0 matching ────────────────────────────
    # When the sampler picks negatives for an anchor from library X, it prefers
    # r0 sources with the same library_id (controls SELEX scaffold). Defaults to
    # the source `name` if not set, so each paper is its own library by default.
    library_id: Optional[str] = None


def parse_round_filter(df: pd.DataFrame, expr: Optional[str]) -> pd.DataFrame:
    """Apply a round filter expression like '== 0' or '>= 1 | isna' to df['selex_round']."""
    if expr is None:
        return df
    sr = df['selex_round']
    # Special token 'isna' allows null-rounds to also count
    if '|' in expr or 'isna' in expr:
        # Split on | and OR-combine
        parts = [p.strip() for p in expr.split('|')]
        mask = pd.Series(False, index=df.index)
        for part in parts:
            if part == 'isna':
                mask = mask | sr.isna()
            else:
                mask = mask | sr.fillna(-9999).pipe(lambda s: eval(f's {part}', {'s': s}))
        return df[mask]
    return df[sr.fillna(-9999).pipe(lambda s: eval(f's {expr}', {'s': s}))]


class ContrastiveAptamerDataset:
    """Memory-light, source-weighted streaming dataset.

    NOT a torch.utils.data.IterableDataset directly — we implement our own
    iter_pairs() loop so we can tightly control source mixing + the in-batch
    r0 negative pool. The train loop calls collate() to produce batches.
    """

    def __init__(self, sources: list[dict], max_len: int = 128,
                 test_exclusion_parquet: Optional[Path] = None,
                 shuffle_buffer: int = 50_000,
                 seed: int = 0,
                 target_embeddings_parquet: Optional[Path] = None):
        self.max_len = max_len
        self.shuffle_buffer = shuffle_buffer
        self.rng = np.random.default_rng(seed)

        # ─── Target protein embeddings (for ESM-2 conditioned training) ──────
        # Loaded once at startup. Stored as a single contiguous TABLE
        # (n_targets+1, [L_max,] D) so the dataloader only ships per-sample
        # INDEX integers across IPC (saves ~95% of bandwidth vs shipping raw
        # arrays per batch). Index 0 = the null/zero target embedding (used for
        # samples whose source has no target_protein declared).
        #
        # Mean-pool table: shape (n_targets+1, D), z-score normalized.
        # Per-residue table: shape (n_targets+1, L_max, D) zero-padded;
        #   companion mask table shape (n_targets+1, L_max), True = pad.
        self._target_protein_to_idx: dict[str, int] = {}
        self._target_emb_table: np.ndarray | None = None
        self._target_mask_table: np.ndarray | None = None  # only for per-residue
        self._target_embed_dim: int = 0
        self._target_embed_is_per_residue: bool = False
        if target_embeddings_parquet is not None and Path(target_embeddings_parquet).exists():
            te = pd.read_parquet(target_embeddings_parquet)
            self._target_embed_is_per_residue = 'residue_embeddings' in te.columns
            n = len(te)
            d = int(te['embedding_dim'].iloc[0])
            self._target_embed_dim = d
            if self._target_embed_is_per_residue:
                # Build padded (n+1, L_max, D) table with mask
                inners = []
                for _, row in te.reset_index(drop=True).iterrows():
                    inner = [np.asarray(r, dtype=np.float32) for r in row['residue_embeddings']]
                    inners.append(np.stack(inner, axis=0))  # (L_i, D)
                L_max = max(arr.shape[0] for arr in inners)
                # Slot 0 = null target. Padded to L_max with all-zero entries and mask=True
                table = np.zeros((n + 1, L_max, d), dtype=np.float32)
                mask = np.ones((n + 1, L_max), dtype=bool)  # True = pad
                # Null target slot keeps all-zero embedding; mask first position valid so attn
                # has at least one position to attend to (rest are masked).
                mask[0, 0] = False
                for i, row in te.reset_index(drop=True).iterrows():
                    arr = inners[i]
                    L = arr.shape[0]
                    table[i + 1, :L] = arr
                    mask[i + 1, :L] = False
                    self._target_protein_to_idx[row['target_protein']] = i + 1
                self._target_emb_table = table
                self._target_mask_table = mask
                print(f'  loaded per-residue target table: shape={table.shape}, '
                      f'L_max={L_max}, {n} proteins')
            else:
                # Mean-pool z-scored (n+1, D)
                embs = np.asarray([np.asarray(e, dtype=np.float32) for e in te['embedding']])
                mu = embs.mean(axis=0, keepdims=True)
                sd = embs.std(axis=0, keepdims=True) + 1e-6
                embs_z = ((embs - mu) / sd).astype(np.float32)
                table = np.zeros((n + 1, d), dtype=np.float32)
                table[1:] = embs_z  # slot 0 = null = zeros
                for i, row in te.reset_index(drop=True).iterrows():
                    self._target_protein_to_idx[row['target_protein']] = i + 1
                self._target_emb_table = table
                print(f'  loaded mean-pool target table: shape={table.shape}, '
                      f'{n} proteins z-scored')

        # Build exclusion set: sequences that are in our test split — never use
        # them as pretraining anchors / positives.
        self.exclude_hashes: set[int] = set()
        if test_exclusion_parquet and Path(test_exclusion_parquet).exists():
            split = pd.read_parquet(test_exclusion_parquet,
                                    columns=['sequence', 'split'])
            test_seqs = split.loc[split['split'] == 'test', 'sequence'].tolist()
            self.exclude_hashes = {hash(s) for s in test_seqs}
            print(f'  excluding {len(self.exclude_hashes):,} test-set sequences from pretraining')

        # Parse source specs into class-bucketed lists.
        self.r0_specs: list[SourceSpec] = []
        self.late_specs: list[SourceSpec] = []
        for d in sources:
            rf = d.get('round_frac')
            if rf is None:
                rf = 0.0 if d['class'] == 'r0' else 1.0
            spec = SourceSpec(
                name=d['name'], parquet=expand_user(d['parquet']),
                class_=d['class'], weight=float(d['weight']),
                sequence_col=d.get('sequence_col', 'sequence'),
                round_filter=d.get('round_filter'),
                selex_round_override=d.get('selex_round_override'),
                kd_filter_lt_nm=d.get('kd_filter_lt_nm'),
                tag=d.get('tag'),
                weight_by_count=d.get('weight_by_count', False),
                count_col=d.get('count_col', 'count'),
                round_frac=float(rf),
                family_weighted=d.get('family_weighted', False),
                family_col=d.get('family_col', 'motif_family'),
                family_weight_power=float(d.get('family_weight_power', 0.5)),
                target_protein=d.get('target_protein'),
                chemistry=d.get('chemistry', 'RNA'),
                library_id=d.get('library_id') or d['name'],
            )
            if not spec.parquet.exists():
                print(f'  miss {spec.name}: {spec.parquet}')
                continue
            (self.r0_specs if spec.class_ == 'r0' else self.late_specs).append(spec)
            print(f'  src {spec.class_:<6} {spec.name:<20} weight={spec.weight}  '
                  f'round_frac={spec.round_frac:.2f}  tag={spec.tag or "—"}  filter={spec.round_filter or "—"}')

        if not self.r0_specs:
            raise ValueError('no r0 sources configured')
        if not self.late_specs:
            raise ValueError('no r_late sources configured')

        self.late_weights = np.asarray([s.weight for s in self.late_specs])
        self.late_weights /= self.late_weights.sum()
        self.r0_weights = np.asarray([s.weight for s in self.r0_specs])
        self.r0_weights /= self.r0_weights.sum()

        # Curriculum support: active subset of r0 specs (defaults to all).
        self._active_r0_idx = list(range(len(self.r0_specs)))
        self._active_r0_weights = self.r0_weights.copy()

        # ─── Sliding-round-window curriculum state ────────────────────────────
        # When set (by set_active_round_frac_window), these override the
        # default r0/late split — positives drawn from sources whose round_frac
        # is in [pos_lo, pos_hi], negatives from [neg_lo, neg_hi]. Sources are
        # indexed in the unified pool self._all_specs.
        self._all_specs: list[SourceSpec] = self.r0_specs + self.late_specs
        self._sliding_active = False
        self._slide_pos_idx: list[int] = []
        self._slide_neg_idx: list[int] = []
        self._slide_pos_weights: np.ndarray = np.zeros(0)
        self._slide_neg_weights: np.ndarray = np.zeros(0)
        self._slide_window: tuple = ()  # (pos_lo, pos_hi, neg_lo, neg_hi) for logging

        # Eager-load the small sources (UTexas, InstructNA, RaptRanker single files)
        # into in-memory numpy arrays; stream the big ones (tier1_combined,
        # jolma2013_combined) batch by batch.
        self._inmem: dict[str, np.ndarray] = {}
        self._inmem_probs: dict[str, np.ndarray] = {}  # log(count)+1 normalized; only set when weight_by_count=True
        self._inmem_alias: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # Vose alias tables for O(1) weighted sampling
        self._streamers: dict[str, list] = {}     # spec.name -> [pyarrow ParquetFile, batch iter, buffer]
        for spec in self.r0_specs + self.late_specs:
            size_mb = spec.parquet.stat().st_size / 1e6
            if size_mb < 200:
                self._inmem[spec.name] = self._eager_load(spec)
            else:
                self._streamers[spec.name] = self._open_stream(spec)
                print(f'  stream {spec.name}: {size_mb:.0f} MB')

        # Drop empty sources (loaded 0 seqs — e.g., all filtered by test-set
        # exclusion or kd_filter). Keeps the sampler from div-by-zero crashes.
        empty_names = [n for n, arr in self._inmem.items() if len(arr) == 0]
        if empty_names:
            print(f'  dropping {len(empty_names)} empty sources: {empty_names}')
            for n in empty_names:
                del self._inmem[n]
                self._inmem_alias.pop(n, None)
                self._inmem_probs.pop(n, None)
            self.r0_specs = [s for s in self.r0_specs if s.name not in empty_names]
            self.late_specs = [s for s in self.late_specs if s.name not in empty_names]
            # Recompute weights + _all_specs after dropping
            if self.r0_specs:
                self.r0_weights = np.asarray([s.weight for s in self.r0_specs])
                self.r0_weights /= self.r0_weights.sum()
                self._active_r0_idx = list(range(len(self.r0_specs)))
                self._active_r0_weights = self.r0_weights.copy()
            if self.late_specs:
                self.late_weights = np.asarray([s.weight for s in self.late_specs])
                self.late_weights /= self.late_weights.sum()
            self._all_specs = self.r0_specs + self.late_specs

    def _get_target_idx(self, target_protein: Optional[str]) -> int:
        """Lookup table index for a target_protein. Returns 0 (null target slot)
        if no target known. The dataloader ships these tiny int indices instead
        of full embedding tensors → ~95% reduction in worker-IPC bandwidth."""
        if target_protein is None:
            return 0
        return self._target_protein_to_idx.get(target_protein, 0)

    def _eager_load(self, spec: SourceSpec) -> np.ndarray:
        cols = [spec.sequence_col]
        if spec.round_filter is not None:
            cols.append('selex_round')
        if spec.kd_filter_lt_nm is not None:
            cols.append('kd_nm')
        if spec.weight_by_count:
            cols.append(spec.count_col)
        if spec.family_weighted:
            cols.append(spec.family_col)
        df = pd.read_parquet(spec.parquet, columns=[c for c in cols if c])
        if spec.round_filter is not None:
            df = parse_round_filter(df, spec.round_filter)
        if spec.kd_filter_lt_nm is not None:
            df = df[df['kd_nm'].fillna(1e9) < spec.kd_filter_lt_nm]
        df = df.dropna(subset=[spec.sequence_col])
        # Test-set leakage exclusion (vectorized — much faster than per-row hash)
        if self.exclude_hashes:
            seqs_arr = df[spec.sequence_col].astype(str).to_numpy()
            keep = np.fromiter((hash(s) not in self.exclude_hashes for s in seqs_arr),
                               dtype=bool, count=len(seqs_arr))
            df = df.loc[keep].reset_index(drop=True)
        seqs = df[spec.sequence_col].astype(str).to_numpy()
        # Build per-sequence sampling weights — multiplicative combination of
        # log(count) weights and 1/family_size^p weights when enabled.
        weights = np.ones(len(seqs), dtype=np.float64)
        info_parts = []
        if spec.weight_by_count and spec.count_col in df.columns:
            counts = df[spec.count_col].fillna(1).astype(float).to_numpy()
            counts = np.clip(counts, 1, None)
            weights *= (np.log(counts) + 1.0)
            info_parts.append(f'log-count (max={int(counts.max()):,})')
        if spec.family_weighted and spec.family_col in df.columns:
            fam = df[spec.family_col].fillna('__UNGROUPED__').astype(str).to_numpy()
            fam_size = pd.Series(fam).value_counts().to_dict()
            sizes = np.asarray([fam_size[f] for f in fam], dtype=np.float64)
            weights *= 1.0 / (sizes ** spec.family_weight_power)
            n_fams = len(fam_size)
            top_fam = max(fam_size.items(), key=lambda kv: kv[1])
            info_parts.append(f'family-weighted (p={spec.family_weight_power}, '
                              f'{n_fams} fams, largest={top_fam[0]}@{top_fam[1]})')
        if info_parts:
            probs = weights / weights.sum()
            self._inmem_probs[spec.name] = probs
            self._inmem_alias[spec.name] = build_alias(probs)
            print(f'  load  {spec.name}: {len(seqs):,} seqs in memory  ({"; ".join(info_parts)})')
        else:
            print(f'  load  {spec.name}: {len(seqs):,} seqs in memory')
        return seqs

    def _open_stream(self, spec: SourceSpec):
        pf = pq.ParquetFile(spec.parquet)
        cols = [spec.sequence_col]
        if spec.round_filter is not None:
            cols.append('selex_round')
        return {'pf': pf, 'cols': cols, 'spec': spec,
                'buffer': [], 'batch_iter': None}

    def _refill_buffer(self, name: str):
        s = self._streamers[name]
        if s['batch_iter'] is None:
            s['batch_iter'] = s['pf'].iter_batches(batch_size=10_000, columns=s['cols'])
        try:
            batch = next(s['batch_iter'])
        except StopIteration:
            # Restart from the top of the file
            s['batch_iter'] = s['pf'].iter_batches(batch_size=10_000, columns=s['cols'])
            batch = next(s['batch_iter'])
        df = batch.to_pandas()
        if s['spec'].round_filter is not None:
            df = parse_round_filter(df, s['spec'].round_filter)
        seqs = df[s['spec'].sequence_col].dropna().astype(str).tolist()
        seqs = [seq for seq in seqs if hash(seq) not in self.exclude_hashes]
        self.rng.shuffle(seqs)
        s['buffer'].extend(seqs)

    def sample_seq(self, spec: SourceSpec) -> str:
        if spec.name in self._inmem:
            arr = self._inmem[spec.name]
            alias = self._inmem_alias.get(spec.name)
            if alias is not None:
                # O(1) weighted sample via Vose's alias method
                prob_t, alias_t = alias
                i = alias_sample(prob_t, alias_t, self.rng)
            else:
                i = int(self.rng.integers(len(arr)))
            return arr[i]
        # Streaming: top up buffer when low (uniform sampling; not weighted)
        s = self._streamers[spec.name]
        while len(s['buffer']) < 1000:
            self._refill_buffer(spec.name)
        return s['buffer'].pop()

    def set_active_neg_tags(self, active_tags):
        """Curriculum hook: restrict negative sampling to specs whose `tag` is in
        active_tags. Pass None or [] to clear restriction (all r0 specs active).
        Renormalizes weights over the active subset.
        """
        if not active_tags:
            self._active_r0_idx = list(range(len(self.r0_specs)))
        else:
            self._active_r0_idx = [
                i for i, s in enumerate(self.r0_specs)
                if (s.tag or s.class_) in active_tags
            ]
        if not self._active_r0_idx:
            raise ValueError(f'no r0 specs match curriculum tags {active_tags}')
        w = np.asarray([self.r0_specs[i].weight for i in self._active_r0_idx])
        self._active_r0_weights = w / w.sum()
        names = [self.r0_specs[i].name for i in self._active_r0_idx]
        print(f'  curriculum: r0 active = {names}  (tags={active_tags})')
        # Disable the sliding-window path if it was active
        self._sliding_active = False

    def set_active_round_frac_window(self, pos_lo: float, pos_hi: float,
                                     neg_lo: float, neg_hi: float):
        """Sliding-round-window curriculum: restrict positives to sources whose
        round_frac ∈ [pos_lo, pos_hi] and negatives to [neg_lo, neg_hi]. Cuts
        across the r0/r_late dichotomy — operates on the unified self._all_specs
        pool. Pos and neg ranges may overlap with each other only by error
        (we don't enforce — caller is responsible).
        """
        pos = [(i, s) for i, s in enumerate(self._all_specs)
               if pos_lo <= s.round_frac <= pos_hi]
        neg = [(i, s) for i, s in enumerate(self._all_specs)
               if neg_lo <= s.round_frac <= neg_hi]
        if not pos:
            raise ValueError(f'no sources with round_frac in pos [{pos_lo}, {pos_hi}]')
        if not neg:
            raise ValueError(f'no sources with round_frac in neg [{neg_lo}, {neg_hi}]')
        self._slide_pos_idx = [i for i, _ in pos]
        self._slide_neg_idx = [i for i, _ in neg]
        wp = np.asarray([s.weight for _, s in pos]); self._slide_pos_weights = wp / wp.sum()
        wn = np.asarray([s.weight for _, s in neg]); self._slide_neg_weights = wn / wn.sum()
        self._sliding_active = True
        self._slide_window = (pos_lo, pos_hi, neg_lo, neg_hi)
        pos_names = [self._all_specs[i].name for i in self._slide_pos_idx]
        neg_names = [self._all_specs[i].name for i in self._slide_neg_idx]
        print(f'  window: pos [{pos_lo:.2f},{pos_hi:.2f}] = {pos_names}')
        print(f'  window: neg [{neg_lo:.2f},{neg_hi:.2f}] = {neg_names}')

    def sample_one(self, class_: str, library_id: str | None = None) -> tuple[str, int, float, int, str, str]:
        """Returns (sequence, source_index_within_class_or_pool, round_frac,
        target_idx, chemistry, library_id).

        target_idx is an INT into self._target_emb_table (0 = null target).
        chemistry/library_id come from the source SourceSpec — used by encode()
        and by library-matched negative sampling.

        If `library_id` is provided AND class_ == 'r0', preferentially sample
        from r0 sources whose library_id matches (per-paper R0 matching).
        Falls back to weighted-uniform across all r0 if no match exists.
        """
        if self._sliding_active:
            if class_ == 'r0':
                k = int(self.rng.choice(len(self._slide_neg_idx), p=self._slide_neg_weights))
                i = self._slide_neg_idx[k]
            else:
                k = int(self.rng.choice(len(self._slide_pos_idx), p=self._slide_pos_weights))
                i = self._slide_pos_idx[k]
            spec = self._all_specs[i]
            return (self.sample_seq(spec), i, spec.round_frac,
                    self._get_target_idx(spec.target_protein), spec.chemistry, spec.library_id)
        if class_ == 'r0':
            # Library-matched R0 sampling: if anchor's library has a matching r0 source, prefer it
            if library_id is not None:
                matching_idx = [j for j, sp in enumerate(self.r0_specs) if sp.library_id == library_id]
                if matching_idx:
                    # Match: sample uniformly weighted among matching r0 sources
                    weights = np.asarray([self.r0_specs[j].weight for j in matching_idx])
                    weights /= weights.sum()
                    chosen = int(self.rng.choice(len(matching_idx), p=weights))
                    i = matching_idx[chosen]
                    spec = self.r0_specs[i]
                    return (self.sample_seq(spec), i, spec.round_frac,
                            self._get_target_idx(spec.target_protein), spec.chemistry, spec.library_id)
            # Fall through: standard cross-library R0 sampling
            k = int(self.rng.choice(len(self._active_r0_idx),
                                    p=self._active_r0_weights))
            i = self._active_r0_idx[k]
            spec = self.r0_specs[i]
            return (self.sample_seq(spec), i, spec.round_frac,
                    self._get_target_idx(spec.target_protein), spec.chemistry, spec.library_id)
        i = int(self.rng.choice(len(self.late_specs), p=self.late_weights))
        spec = self.late_specs[i]
        return (self.sample_seq(spec), i, spec.round_frac,
                self._get_target_idx(spec.target_protein), spec.chemistry, spec.library_id)

    def __iter__(self) -> Iterator[dict]:
        while True:
            anchor, src_i, anc_rf, anc_ti, anc_chem, anc_lib = self.sample_one('r_late')
            positive, _, pos_rf, pos_ti, pos_chem, _ = self.sample_one('r_late')
            yield dict(
                anchor_ids=encode(anchor, self.max_len, chemistry=anc_chem),
                positive_ids=encode(positive, self.max_len, chemistry=pos_chem),
                source_id=src_i,
                anchor_round_frac=np.float32(anc_rf),
                positive_round_frac=np.float32(pos_rf),
                anchor_target_idx=int(anc_ti),
                positive_target_idx=int(pos_ti),
                anchor_library_id=anc_lib,
                anchor_chemistry=anc_chem,
            )

    def sample_negatives(self, n: int, anchor_library_ids: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (ids[n, max_len], round_frac[n], target_idx[n]) for negs.

        If `anchor_library_ids` is provided, samples are library-matched 1-to-1
        with the anchors (per-paper R0 contrast). If None, samples are drawn
        from the global R0 pool with the standard weighted-uniform distribution.
        """
        ids = np.zeros((n, self.max_len), dtype=np.int64)
        rfs = np.zeros(n, dtype=np.float32)
        tis = np.zeros(n, dtype=np.int64)
        for k in range(n):
            lib = anchor_library_ids[k] if anchor_library_ids else None
            seq, _, rf, ti, neg_chem, _ = self.sample_one('r0', library_id=lib)
            ids[k] = encode(seq, self.max_len, chemistry=neg_chem)
            rfs[k] = rf
            tis[k] = ti
        return ids, rfs, tis

    def collate(self, batch: list[dict]) -> dict:
        """Returns batch tensors. Target embeddings are NOT included — only the
        per-sample target_idx integers. The trainer indexes into its GPU-resident
        target embedding table by these indices, saving ~95% IPC bandwidth."""
        import torch
        b = len(batch)
        anchor_ids = torch.from_numpy(np.stack([x['anchor_ids'] for x in batch]))
        positive_ids = torch.from_numpy(np.stack([x['positive_ids'] for x in batch]))
        # Library-matched R0 sampling (per-paper R0 — same SELEX scaffold as anchor)
        anchor_libs = [x['anchor_library_id'] for x in batch]
        neg_ids_np, neg_rf_np, neg_ti_np = self.sample_negatives(b, anchor_library_ids=anchor_libs)
        negative_ids = torch.from_numpy(neg_ids_np)
        source_id = torch.tensor([x['source_id'] for x in batch], dtype=torch.long)
        anchor_rf = torch.tensor([x['anchor_round_frac'] for x in batch], dtype=torch.float32)
        positive_rf = torch.tensor([x['positive_round_frac'] for x in batch], dtype=torch.float32)
        negative_rf = torch.from_numpy(neg_rf_np)
        anchor_ti = torch.tensor([x['anchor_target_idx'] for x in batch], dtype=torch.long)
        positive_ti = torch.tensor([x['positive_target_idx'] for x in batch], dtype=torch.long)
        negative_ti = torch.from_numpy(neg_ti_np)
        return dict(
            anchor_ids=anchor_ids, positive_ids=positive_ids,
            negative_ids=negative_ids, source_id=source_id,
            anchor_round_frac=anchor_rf,
            positive_round_frac=positive_rf,
            negative_round_frac=negative_rf,
            anchor_target_idx=anchor_ti,
            positive_target_idx=positive_ti,
            negative_target_idx=negative_ti,
        )


# ============================================================ MLM DATASET ===
class MLMAptamerDataset:
    """Streaming dataset for the MLM baseline (RaptScore / InstructNA / AptaBERT
    style). Treats all sources as a single pool — round info is intentionally
    discarded. This is the apples-to-apples baseline against our contrastive run:
    same data, same encoder, only the pretraining objective differs.

    Yields per-step `{input_ids, label_ids, mask}`:
      input_ids :  token ids with 15% positions corrupted (BERT-style: 80% →
                   MASK, 10% → random nucleotide, 10% → unchanged)
      label_ids :  the original token at each position (used by the LM head)
      mask      :  boolean mask flagging which positions are "active" for the loss

    BERT-recipe per Devlin 2018:
      - For each non-pad position, with probability 15%:
        - 80%: replace with MASK_ID
        - 10%: replace with a uniform random nucleotide
        - 10%: keep original (acts as a regularizer)
      - Loss is computed ONLY over the 15% selected positions (the "active mask"),
        not over the 85% kept-as-is.

    Source mixing: ignores `class_` from the spec (no r0 vs r_late distinction).
    Uses the per-source `weight` to bias sampling.
    """

    def __init__(self, sources: list[dict], max_len: int = 128,
                 test_exclusion_parquet: Optional[Path] = None,
                 mask_prob: float = 0.15,
                 seed: int = 0):
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.rng = np.random.default_rng(seed)

        self.exclude_hashes: set[int] = set()
        if test_exclusion_parquet and Path(test_exclusion_parquet).exists():
            split = pd.read_parquet(test_exclusion_parquet, columns=['sequence', 'split'])
            test_seqs = split.loc[split['split'] == 'test', 'sequence'].tolist()
            self.exclude_hashes = {hash(s) for s in test_seqs}
            print(f'  excluding {len(self.exclude_hashes):,} test-set sequences from MLM pretraining')

        # Reuse SourceSpec semantics but drop the class distinction.
        self.specs: list[SourceSpec] = []
        for d in sources:
            spec = SourceSpec(
                name=d['name'], parquet=expand_user(d['parquet']),
                class_=d.get('class', 'r_late'),  # not used; kept for compatibility
                weight=float(d['weight']),
                sequence_col=d.get('sequence_col', 'sequence'),
                round_filter=d.get('round_filter'),
                selex_round_override=d.get('selex_round_override'),
                kd_filter_lt_nm=d.get('kd_filter_lt_nm'),
                weight_by_count=d.get('weight_by_count', False),
                count_col=d.get('count_col', 'count'),
                chemistry=d.get('chemistry', 'RNA'),
                library_id=d.get('library_id') or d['name'],
            )
            if not spec.parquet.exists():
                print(f'  miss {spec.name}: {spec.parquet}')
                continue
            self.specs.append(spec)
            print(f'  src {spec.name:<20} weight={spec.weight}  '
                  f'filter={spec.round_filter or "—"}')
        if not self.specs:
            raise ValueError('no sources configured for MLM')

        self.weights = np.asarray([s.weight for s in self.specs])
        self.weights /= self.weights.sum()

        # Same eager-load / stream split as the contrastive class.
        self._inmem: dict[str, np.ndarray] = {}
        self._inmem_probs: dict[str, np.ndarray] = {}
        self._inmem_alias: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._streamers: dict[str, dict] = {}
        for spec in self.specs:
            size_mb = spec.parquet.stat().st_size / 1e6
            if size_mb < 200:
                self._inmem[spec.name] = self._eager_load(spec)
            else:
                self._streamers[spec.name] = self._open_stream(spec)
                print(f'  stream {spec.name}: {size_mb:.0f} MB')

    # --- methods identical in spirit to the contrastive class --------------
    def _eager_load(self, spec: SourceSpec) -> np.ndarray:
        cols = [spec.sequence_col]
        if spec.round_filter is not None: cols.append('selex_round')
        if spec.kd_filter_lt_nm is not None: cols.append('kd_nm')
        if spec.weight_by_count: cols.append(spec.count_col)
        df = pd.read_parquet(spec.parquet, columns=[c for c in cols if c])
        if spec.round_filter is not None:
            df = parse_round_filter(df, spec.round_filter)
        if spec.kd_filter_lt_nm is not None:
            df = df[df['kd_nm'].fillna(1e9) < spec.kd_filter_lt_nm]
        df = df.dropna(subset=[spec.sequence_col])
        if self.exclude_hashes:
            seqs_arr = df[spec.sequence_col].astype(str).to_numpy()
            keep = np.fromiter((hash(s) not in self.exclude_hashes for s in seqs_arr),
                               dtype=bool, count=len(seqs_arr))
            df = df.loc[keep].reset_index(drop=True)
        seqs = df[spec.sequence_col].astype(str).to_numpy()
        if spec.weight_by_count and spec.count_col in df.columns:
            counts = df[spec.count_col].fillna(1).astype(float).to_numpy()
            counts = np.clip(counts, 1, None)
            log_w = np.log(counts) + 1.0
            probs = log_w / log_w.sum()
            self._inmem_probs[spec.name] = probs
            self._inmem_alias[spec.name] = build_alias(probs)
            print(f'  load  {spec.name}: {len(seqs):,} seqs in memory  '
                  f'(log-count weighted; max-count={int(counts.max()):,})')
        else:
            print(f'  load  {spec.name}: {len(seqs):,} seqs in memory')
        return seqs

    def _open_stream(self, spec: SourceSpec):
        pf = pq.ParquetFile(spec.parquet)
        cols = [spec.sequence_col]
        if spec.round_filter is not None: cols.append('selex_round')
        return {'pf': pf, 'cols': cols, 'spec': spec, 'buffer': [], 'batch_iter': None}

    def _refill(self, name):
        s = self._streamers[name]
        if s['batch_iter'] is None:
            s['batch_iter'] = s['pf'].iter_batches(batch_size=10_000, columns=s['cols'])
        try: batch = next(s['batch_iter'])
        except StopIteration:
            s['batch_iter'] = s['pf'].iter_batches(batch_size=10_000, columns=s['cols'])
            batch = next(s['batch_iter'])
        df = batch.to_pandas()
        if s['spec'].round_filter is not None:
            df = parse_round_filter(df, s['spec'].round_filter)
        seqs = df[s['spec'].sequence_col].dropna().astype(str).tolist()
        seqs = [x for x in seqs if hash(x) not in self.exclude_hashes]
        self.rng.shuffle(seqs)
        s['buffer'].extend(seqs)

    def sample_seq(self) -> str:
        i = int(self.rng.choice(len(self.specs), p=self.weights))
        spec = self.specs[i]
        if spec.name in self._inmem:
            arr = self._inmem[spec.name]
            alias = self._inmem_alias.get(spec.name)
            if alias is not None:
                prob_t, alias_t = alias
                j = alias_sample(prob_t, alias_t, self.rng)
            else:
                j = int(self.rng.integers(len(arr)))
            return arr[j]
        s = self._streamers[spec.name]
        while len(s['buffer']) < 1000:
            self._refill(spec.name)
        return s['buffer'].pop()

    def __iter__(self):
        while True:
            seq = self.sample_seq()
            ids = encode(seq, self.max_len)
            input_ids, label_ids, active_mask = _bert_mask(ids, self.mask_prob, self.rng)
            yield dict(input_ids=input_ids, label_ids=label_ids, active_mask=active_mask)

    def collate(self, batch: list[dict]) -> dict:
        import torch
        input_ids = torch.from_numpy(np.stack([x['input_ids'] for x in batch]))
        label_ids = torch.from_numpy(np.stack([x['label_ids'] for x in batch]))
        active_mask = torch.from_numpy(np.stack([x['active_mask'] for x in batch]))
        return dict(input_ids=input_ids, label_ids=label_ids, active_mask=active_mask)


def _bert_mask(ids: np.ndarray, p: float, rng: np.random.Generator):
    """BERT-style masking. Returns (input_ids, label_ids, active_mask).

    For each non-pad position: with probability p,
      80%: replace with MASK_ID
      10%: replace with a uniform random nucleotide
      10%: keep original
    Loss is computed over `active_mask=True` positions only.
    """
    n = len(ids)
    not_pad = ids != PAD_ID
    # Decide which positions to mark "active" (subject to corruption + loss)
    rand = rng.random(n)
    active = not_pad & (rand < p)

    # Of the active ones, 80/10/10 split
    sub_rand = rng.random(n)
    is_mask    = active & (sub_rand < 0.8)
    is_rand    = active & (sub_rand >= 0.8) & (sub_rand < 0.9)
    # is_keep is implicit (the rest of active)

    input_ids = ids.copy()
    input_ids[is_mask] = MASK_ID
    rand_nucs = rng.integers(len(NUC_IDS), size=int(is_rand.sum()))
    input_ids[is_rand] = rand_nucs

    label_ids = ids.copy()  # original tokens; loss only computed where active=True
    return input_ids, label_ids, active
