"""Tabulate the final-eval bench metrics for all completed runs.

Reads ~/Desktop/autoRNA_data/tumbleweed/training_runs/*/eval.jsonl, picks the
LAST eval line in each, prints a single comparison table.

Usage:
  python3 bench_summary.py                # all runs
  python3 bench_summary.py mlm_v1 curriculum_30min   # specific run_ids
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

RUN_ROOT = Path.home() / 'Desktop/autoRNA_data/tumbleweed/training_runs'

METRICS = [
    ('panel_r_at_1',     'pR@1'),
    ('panel_r_at_10',    'pR@10'),
    ('panel_kd_spearman', 'ρ'),
    ('a4_pearson',       'A4 r'),
    ('li_auroc',         'Li AUC'),
    ('utexas_pearson',   'UTex r'),
    ('test_r_at_1',      'tR@1'),
    # RaptScore-style PLL on the 171 SPR-labeled sequences (their Tables 4-6 → A, B, C).
    # MLM/Hybrid models only; contrastive runs report nan.
    # Their published bar: r = 0.65 / 0.78 / 0.65 for A / B / C.
    ('pll_pearson_A',    'PLL_A'),
    ('pll_pearson_B',    'PLL_B'),
    ('pll_pearson_C',    'PLL_C'),
]


def last_eval(run_dir):
    f = run_dir / 'eval.jsonl'
    if not f.exists():
        return None
    last = None
    for line in f.read_text().splitlines():
        if line.strip(): last = json.loads(line)
    return last


def fmt(v):
    if v is None: return '  —  '
    if isinstance(v, float): return f'{v:+.3f}' if v < 0 else f' {v:.3f}'
    return str(v)


def main():
    wanted = sys.argv[1:] if len(sys.argv) > 1 else None
    runs = sorted(d for d in RUN_ROOT.iterdir() if d.is_dir())
    if wanted:
        runs = [d for d in runs if d.name in wanted or any(w in d.name for w in wanted)]
    if not runs:
        print('no runs found')
        return

    header = f'{"run":<38} {"step":>6}  '
    header += '  '.join(f'{lbl:>6}' for _, lbl in METRICS)
    print(header)
    print('-' * len(header))
    for d in runs:
        e = last_eval(d)
        if e is None:
            print(f'{d.name:<38} {"-":>6}  (no eval.jsonl)')
            continue
        cells = '  '.join(fmt(e.get(k)) for k, _ in METRICS)
        print(f'{d.name:<38} {e.get("step", "?"):>6}  {cells}')


if __name__ == '__main__':
    main()
