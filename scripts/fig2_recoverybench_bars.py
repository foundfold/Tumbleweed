"""Fig 2 — Tumbleweed-RecoveryBench: per-target winner-vs-random AUROC.

Grouped bars over the 5 RecoveryBench targets (FGF9, IL1RL1, PARP1, MECP2, SNCA) comparing:
  - Tumbleweed (headline v7_film_cnn, MATCHED on EvoFlow's identical sequence set)
  - EvoFlow-RNA (33M public ckpt, RiNALMo-150M-derived, RNA-only / unconditional)
  - RiNALMo-MLM baseline

AUROC = AUROC(winner vs composition-matched random), low-t (0.1/0.15/0.2), N_MASK_REPS=4.
0.5 = chance. Tumbleweed is scored on EvoFlow's exact seqs so the margin is not a set artifact.

Sources (data_refs/):
  recovery_likelihood_v7_film_cnn_matchedseqs_lowt.csv  (Tumbleweed, matched)
  recovery_evoflow_lowt.csv                             (EvoFlow-RNA)
  recovery_rinalmo_lowt.csv                             (RiNALMo-MLM)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / 'data_refs'
OUT = ROOT / 'research' / 'manuscript' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ['FGF9', 'IL1RL1', 'PARP1', 'MECP2', 'SNCA']
SERIES = [
    ('recovery_likelihood_v7_film_cnn_matchedseqs_lowt.csv', 'Tumbleweed (matched)', '#1b6ca8'),
    ('recovery_evoflow_lowt.csv', 'EvoFlow-RNA (33M, uncond.)', '#b0b0b0'),
    ('recovery_rinalmo_lowt.csv', 'RiNALMo-MLM', '#d98a3d'),
]
COL = 'AUROC_winner_vs_random'


def load(fname: str) -> dict:
    df = pd.read_csv(DR / fname).set_index('target')
    return {t: float(df.loc[t, COL]) for t in TARGETS}


def main():
    data = {lab: load(f) for f, lab, _ in SERIES}
    x = np.arange(len(TARGETS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, (_, lab, color) in enumerate(SERIES):
        vals = [data[lab][t] for t in TARGETS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=color,
                      edgecolor='white', linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=7.5)
    means = {lab: np.mean([data[lab][t] for t in TARGETS]) for _, lab, _ in SERIES}
    ax.axhline(0.5, ls='--', lw=1, color='#555', zorder=0)
    ax.text(len(TARGETS) - 0.5, 0.515, 'chance', fontsize=8, color='#555', ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS)
    ax.set_ylabel('AUROC (winner vs. random)')
    ax.set_ylim(0.4, 1.05)
    ax.set_title('Tumbleweed-RecoveryBench: ranking true SELEX winners above random')
    handles_labels = [f'{lab}  (mean {means[lab]:.3f})' for _, lab, _ in SERIES]
    ax.legend(ax.containers, handles_labels, loc='lower center', ncol=3,
              fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.22))
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig2_recoverybench.{ext}', dpi=300, bbox_inches='tight')
    print('wrote', OUT / 'fig2_recoverybench.png')
    print('means:', {k: round(v, 4) for k, v in means.items()})


if __name__ == '__main__':
    main()
