# Tumbleweed

A target-conditional DNA and RNA aptamer generator, released with two open benchmarks
for target-conditional aptamer modeling.

Tumbleweed is a masked-discrete-diffusion model that is both **chemistry-aware** and
**target-conditional**. A leading chemistry token (`[RNA]`/`[DNA]`) lets a single model
generate and score both RNA and DNA aptamers, while the protein target is injected into
every transformer layer through ESM-2 embeddings and feature-wise linear modulation
(FiLM).

This repository contains the model and training code, the two benchmarks, and the
scripts that reproduce the paper's figures and benchmark evaluations.

## Layout

- `training/` — model definition, training, sampling, and evaluation code, plus the
  full set of experiment configs (`training/configs/`).
- `benchmark_release/` — the two public benchmarks and their documentation
  (see `benchmark_release/README.md`):
  - **Tumbleweed-RecoveryBench** — rank true SELEX winners above composition-matched
    random sequences (AUROC).
  - **Tumbleweed-KdBench** — rank held-out aptamer affinity under leave-one-target-out
    transfer (Spearman ρ); released as an open *negative* benchmark.
- `scripts/` — figure generators (`fig*.py`), the benchmark builder
  (`build_benchmark_release.py`), and the baseline/eval scripts
  (`eval_*.py`, `sample_trifp_guided.py`).

## Benchmarks

See [`benchmark_release/README.md`](benchmark_release/README.md) for the full data
dictionary, provenance notes, and SELEX data accessions. Raw SELEX reads are not
re-hosted; the benchmark points to the originating studies and accessions.

## License

- **Code** (everything under `training/` and `scripts/`): MIT (see `LICENSE`).
- **Benchmark data** (`benchmark_release/`): CC-BY-4.0 (see `benchmark_release/LICENSE`).
