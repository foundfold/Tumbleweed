"""Fig 3 — Tumbleweed target conditioning: within-domain works, cross-domain collapses.

Two panels, side by side. The juxtaposition IS the argument:

 (a) WITHIN-DOMAIN. For each of the 5 RecoveryBench targets whose SELEX family IS in
     training, swapping in the target's own ESM-2 conditioning vs a zeroed embedding lifts
     winner-vs-random AUROC by +0.15..+0.24 (own_minus_zero). Conditioning is real and
     target-specific when the family is seen.

 (b) CROSS-DOMAIN (fair LOO). Drop the held-out target's SELEX family from training but KEEP
     its ESM-2 conditioning at score time (fair zero-shot, comparable to an unconditional
     EvoFlow). All 5 collapse to chance: mean AUROC 0.503, below EvoFlow's 0.521. The
     conditioning signal does not generalize to an unseen target family — the ~229-target
     label-diversity ceiling, not the architecture.

Sources (data_refs/):
  (a) null_conditioning_tumbleweed_60m_diffusion_v7_film_cnn_mst2.csv
        cols: target, auroc_own, auroc_zero, own_minus_zero  (mouse ST2 fix)
  (b) recovery_loo_{FGF9,IL1RL1,PARP1,MECP2,SNCA}_mst2_lowt.csv
        col: AUROC_winner_vs_random
  EvoFlow reference: recovery_evoflow_lowt.csv mean AUROC_winner_vs_random (= 0.521)
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
BLUE = '#1b6ca8'
GREY = '#b0b0b0'
RED = '#c0392b'


def load_panel_a() -> tuple:
    df = pd.read_csv(DR / 'null_conditioning_tumbleweed_60m_diffusion_v8_film_cnn_nc_mst2.csv').set_index('target')
    own = [float(df.loc[t, 'auroc_own']) for t in TARGETS]
    zero = [float(df.loc[t, 'auroc_zero']) for t in TARGETS]
    return own, zero


def load_panel_b() -> list:
    vals = []
    for t in TARGETS:
        df = pd.read_csv(DR / f'recovery_nc_loo_{t}_mst2_lowt.csv')
        vals.append(float(df['AUROC_winner_vs_random'].iloc[0]))
    return vals


def evoflow_per_target() -> list:
    df = pd.read_csv(DR / 'recovery_evoflow_lowt.csv').set_index('target')
    return [float(df.loc[t, 'AUROC_winner_vs_random']) for t in TARGETS]


def main():
    own, zero = load_panel_a()
    loo = load_panel_b()
    evo_pt = evoflow_per_target()
    x = np.arange(len(TARGETS))
    w = 0.38

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # --- Panel A: within-domain own vs zero conditioning ---
    bA1 = axA.bar(x - w / 2, own, w, label='target ESM-2 conditioning', color=BLUE,
                  edgecolor='white', linewidth=0.6)
    bA2 = axA.bar(x + w / 2, zero, w, label='zeroed conditioning', color=GREY,
                  edgecolor='white', linewidth=0.6)
    for bars in (bA1, bA2):
        for b in bars:
            axA.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                     f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=7)
    axA.set_xticks(x)
    axA.set_xticklabels(TARGETS)
    axA.set_ylabel('AUROC (winner vs. random)')
    axA.set_ylim(0.4, 1.05)
    lifts = np.array(own) - np.array(zero)
    axA.set_title(f'(a) within-domain: conditioning works\nown−zero lift +{lifts.mean():.2f} mean',
                  fontsize=10)
    axA.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.30),
               fontsize=8, frameon=False)

    # --- Panel B: cross-domain fair LOO — Tumbleweed (zero-shot) vs EvoFlow ---
    bB1 = axB.bar(x - w / 2, loo, w, label='Tumbleweed (LOO zero-shot)', color=BLUE,
                  edgecolor='white', linewidth=0.6)
    bB2 = axB.bar(x + w / 2, evo_pt, w, label='EvoFlow-RNA (33M)', color=RED,
                  edgecolor='white', linewidth=0.6)
    for bars in (bB1, bB2):
        for b in bars:
            axB.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                     f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=7)
    axB.set_xticks(x)
    axB.set_xticklabels(TARGETS)
    axB.set_ylim(0.4, 1.05)
    axB.set_title(f'(b) cross-domain LOO: conditioning does not transfer\n'
                  f'Tumbleweed {np.mean(loo):.3f} vs EvoFlow {np.mean(evo_pt):.3f}',
                  fontsize=10)
    axB.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.30),
               fontsize=8, frameon=False)

    for ax in (axA, axB):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    fig.suptitle('Target conditioning generalizes within SELEX families, not across them', y=1.02,
                 fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig3_conditioning_ab.{ext}', dpi=300, bbox_inches='tight')
    print('wrote', OUT / 'fig3_conditioning_ab.png')
    print('panel A own :', {t: round(v, 3) for t, v in zip(TARGETS, own)})
    print('panel A zero:', {t: round(v, 3) for t, v in zip(TARGETS, zero)})
    print('panel A lift mean:', round(lifts.mean(), 3))
    print('panel B LOO :', {t: round(v, 3) for t, v in zip(TARGETS, loo)})
    print('panel B evo :', {t: round(v, 3) for t, v in zip(TARGETS, evo_pt)})
    print(f'panel B mean: {np.mean(loo):.3f}  EvoFlow mean: {np.mean(evo_pt):.3f}')


if __name__ == '__main__':
    main()
