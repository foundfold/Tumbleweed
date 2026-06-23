"""Fig 4 — Tumbleweed-KdBench: draw-the-negative forest plot.

The KdBench result is a CLEAN NEGATIVE: no method ranks aptamer affinity above chance. This
figure SHOWS that instead of tabulating it — for each of the 5 benchmarked methods, plot the
mean per-panel LOO Spearman rho over the 47 rankable panels with a 95% CI (mean ± 1.96*SE).
Every CI crosses 0; every t-statistic is ~1. A method that beat chance would have its whole
interval to the right of the dashed 0 line — none do.

We annotate each row with its t-stat (mean / (sd/sqrt(k))). The hybrid-film point is the
nominal leader but its CI still spans 0, and its edge is small-n spike-inflated (see §S1).

Source: data_refs/v2_loo_per_target.csv
  cols: target_uniprot, chem, n, v10_rho, v15_rho, rinalmo_rho, trifp_rho, hybridfilm_rho
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PER = ROOT / 'data_refs' / 'v2_loo_per_target.csv'
OUT = ROOT / 'manuscript' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# (column, display label) — TW-* = rankers WE built (so reviewers see these are all Tumbleweed
# attempts); RiNALMo is the external baseline, left unbranded. Descriptive arch names, no v10/v15
# tags. Ordered top→bottom by nominal t-stat.
METHODS = [
    ('hybridfilm_rho', 'TW-Generator (likelihood)'),
    ('trifp_rho', 'TW-TriFP (kmer⊕ESM-2→GBDT)'),
    ('rinalmo_rho', 'RiNALMo-650M → Ridge'),
    ('v10_rho', 'TW-Contrastive scorer'),
    ('v15_rho', 'TW-Contrastive +CNN'),
]
HIGHLIGHT = 'TW-Generator (likelihood)'  # nominal leader (t=1.27), still spans 0


def stats(rho: np.ndarray) -> tuple:
    r = rho[~np.isnan(rho)]
    k = len(r)
    m = r.mean()
    sd = r.std(ddof=1)
    se = sd / np.sqrt(k)
    t = m / se if se > 0 else np.nan
    return m, 1.96 * se, t, k


def main():
    df = pd.read_csv(PER)
    rows = [(lab, *stats(df[col].values.astype(float))) for col, lab in METHODS]
    # plot bottom→top = reverse so first listed is on top
    rows = rows[::-1]
    ys = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.axvline(0, ls='--', lw=1.2, color='#555', zorder=1)
    for y, (lab, m, ci, t, k) in zip(ys, rows):
        color = '#1b6ca8' if lab == HIGHLIGHT else '#444'
        ax.errorbar(m, y, xerr=ci, fmt='o', color=color, ecolor=color,
                    elinewidth=1.6, capsize=4, markersize=7, zorder=3)
        ax.annotate(f't={t:+.2f}', (m, y), textcoords='offset points', xytext=(0, 9),
                    ha='center', va='bottom', fontsize=7.5, color=color)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, *_ in rows], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_xlim(-0.3, 0.55)
    ax.set_xlabel('mean LOO Spearman ρ (95% CI over 47 panels)')
    ax.set_title('Tumbleweed-KdBench: no method ranks affinity above chance', pad=14)
    ax.annotate('chance (ρ=0)', (0, len(rows) - 0.35), ha='center', va='bottom',
                fontsize=8, color='#555')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig4_kdbench_forest.{ext}', dpi=300, bbox_inches='tight')
    print('wrote', OUT / 'fig4_kdbench_forest.png')
    for lab, m, ci, t, k in rows[::-1]:
        print(f'  {lab:24s} mean={m:+.3f}  95%CI=±{ci:.3f}  t={t:+.2f}  k={k}')


if __name__ == '__main__':
    main()
