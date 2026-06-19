# Tumbleweed benchmarks

Two public benchmarks for target-conditional aptamer modeling, released alongside the
Tumbleweed paper. Both are fully computational; no new wet-lab data.

- **Tumbleweed-KdBench** — a curated, citable corpus of measured aptamers (sequence,
  chemistry, protein target, K_D, primary-literature citation). Affinity-ranking task:
  leave-one-target-out Spearman correlation between predicted score and measured K_D.
- **Tumbleweed-RecoveryBench** — given a SELEX target, rank that target's enriched
  "winner" sequences above composition-matched random sequences (AUROC).

---

## `kdbench/`

### `kdbench_aptamers.csv`  (911 aptamers)
One row per measured aptamer.

| column | meaning |
|---|---|
| `aptamer_id` | internal id (may be blank for some rows) |
| `sequence` | aptamer nucleotide sequence |
| `chemistry` | `DNA` (653) or `RNA` (258) |
| `target` | protein target (canonical name) |
| `target_uniprot_id` | UniProt accession of the target |
| `kd_nm` | dissociation constant K_D, nM |
| `log10_kd_nm` | log10 of K_D (nM) |
| `pubmed_id` | PubMed ID of the originating study (blank if unknown) |
| `doi` | DOI, when resolvable from the PMID |
| `citation` | formatted primary-literature citation (blank if no PMID) |

635 / 911 rows carry a PubMed ID and a resolved citation; the remaining 276 are
included with their measured data even though we could not attach a source ID.

### `references.csv` / `references.bib`
The 232 unique primary references behind the cited rows, resolved from PubMed via NCBI
E-utilities (`esummary`). `.bib` is the same set as BibTeX.

**Provenance note.** Some K_D values were located through aggregator databases
(AptaDB, AptamerBase). Those aggregators were used **only as a PubMed-ID / citation
finder** — every citation here points to the original primary paper, not the
aggregator. Rows whose original PMID could not be recovered are kept but left uncited.

---

## `recoverybench/`

### `recoverybench_sequences.parquet`
Winner and composition-matched random sequences for the 5 RecoveryBench targets
(FGF9, IL1RL1, PARP1, MECP2, SNCA).

---

## `tables/`

- `recoverybench_per_target.csv` — per-target winner-vs-random AUROC for Tumbleweed
  (target-matched) vs released EvoFlow-RNA (33M, unconditional) and a RiNALMo-MLM
  baseline. Tumbleweed mean 0.943.
- `kdbench_stability.csv` — KdBench affinity-ranking metrics (mean/median/trimmed
  Spearman, win rate, t-stat) across 47 leave-one-target-out panels for every method.
  No method ranks held-out affinity above chance.

---

## SELEX data sources (RecoveryBench)

The RecoveryBench sequences derive from published SELEX datasets. We do not re-host raw
reads; cite the originating study / accession:

| Targets | Source | Accession |
|---|---|---|
| FGF9 (P31371), IL1RL1 (Q01638) | RaptScore (NAR 2026) supplemental tables | — (published supplemental) |
| PARP1 (P09874), MECP2 (P51608) | RAPID-SELEX | ENA `PRJNA1122221` / GEO `GSE269538` |
| SNCA / α-synuclein (P37840) | 2'-F-pyrimidine RNA N35 SELEX (NAR 2024) | ENA `PRJEB70964`, doi:10.1093/nar/gkae544 |

---

## Reproducing the release

`scripts/build_benchmark_release.py` (in the Tumbleweed repo) regenerates every file in
this directory from the curated corpus, resolving citations live from PubMed.
