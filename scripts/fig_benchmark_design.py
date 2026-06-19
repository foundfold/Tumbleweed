"""Fig 3 — Benchmark design schematic: how Tumbleweed-RecoveryBench and
Tumbleweed-KdBench are constructed and scored.

A schematic (fixed geometry, no data file) of the two evaluation protocols, NOT a
results plot — the numbers live in Fig 4 / Fig 6 and Tables 1-2.

Panel A (RecoveryBench): for each target, the model scores true SELEX winners and
composition-matched random sequences by low-mask pseudo-NLL; the metric is
AUROC(winner ranked above random), 0.5 = chance. It asks whether the generator
assigns higher likelihood to real binders than to shuffled-composition decoys.

Panel B (KdBench): one panel = one (target x chemistry) with >=4 aptamers of
measured K_D. The target's family is held entirely out of training (leave-one-
target-out); the held-out aptamers are then scored and the predicted order is
compared to the measured K_D order by Spearman rho, aggregated over 47 panels.

Run to regenerate the PNG/PDF.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'research' / 'manuscript' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

BLUE = '#1b6ca8'
LBLUE = '#d6e6f2'
ORANGE = '#d98a3d'
LOR = '#f5e3cf'
GREY = '#666'
LGREY = '#ededed'
GREEN = '#3d8a5a'
LGREEN = '#d9ecdf'
NONBIND = '#bdbdbd'      # grey = composition-matched random / non-binder
WINNER = '#3d8a5a'       # green = true SELEX winner
NUC = [BLUE, GREEN, ORANGE, '#7d5ba6']
# binder-dot palette shared with the training figure (Fig 2): colored = binder
BINDER_COLORS = ['#c0392b', '#127a6b', '#b9770e', '#b03a8c']
MASK = '#111111'         # black = a masked (hidden) position


def box(ax, x, y, w, h, label, face, edge, fontsize=8.3, weight='normal', tc='black'):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.04',
                       linewidth=1.3, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(p)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fontsize,
                weight=weight, color=tc, zorder=3)
    return (x + w / 2, y + h / 2)


def arrow(ax, p0, p1, color=GREY, style='-|>', lw=1.7, rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13, lw=lw, color=color,
                        connectionstyle=f'arc3,rad={rad}', zorder=1)
    ax.add_patch(a)


def strip(ax, x, y, w, seed, tint=None, ncells=8, h=0.17):
    """One sequence as a row of nucleotide cells. tint overrides the per-cell palette
    with a single colour (grey = matched-random decoy)."""
    rng = np.random.default_rng(seed)
    cell = w / ncells
    for i in range(ncells):
        c = tint if tint is not None else NUC[rng.integers(0, 4)]
        ax.add_patch(Rectangle((x + i * cell, y), cell * 0.86, h, facecolor=c,
                               edgecolor='white', linewidth=0.5, zorder=3))
    ax.add_patch(Rectangle((x - 0.006, y - 0.006), w + 0.004, h + 0.012, fill=False,
                           edgecolor=GREY, linewidth=0.7, zorder=4))


def masked_strip(ax, x, y, w, seed, ncells=10, nmask=2, h=0.17):
    """A sequence scored at a low mask ratio: nmask of ncells are hidden (black 'M'),
    the rest revealed. Illustrates the 'low mask' label on the scoring box."""
    rng = np.random.default_rng(seed)
    cell = w / ncells
    mask_idx = set(rng.choice(ncells, size=nmask, replace=False).tolist())
    for i in range(ncells):
        if i in mask_idx:
            c, letter = MASK, 'M'
        else:
            c, letter = NUC[rng.integers(0, 4)], ''
        ax.add_patch(Rectangle((x + i * cell, y), cell * 0.86, h, facecolor=c,
                               edgecolor='white', linewidth=0.5, zorder=3))
        if letter:
            ax.text(x + i * cell + cell * 0.43, y + h / 2, letter, ha='center',
                    va='center', fontsize=5.8, weight='bold', color='white', zorder=4)
    ax.add_patch(Rectangle((x - 0.006, y - 0.006), w + 0.004, h + 0.012, fill=False,
                           edgecolor=GREY, linewidth=0.7, zorder=4))


def pool(ax, cx, cy, r, frac_binder, seed, ndots=22):
    """A SELEX pool drawn in the training-figure dot motif: a circle of small sequences,
    colored = binder, grey = non-binder. frac_binder sets the colored fraction (1.0 =
    fully enriched winner pool, 0.0 = grey decoy pool)."""
    rng = np.random.default_rng(seed)
    ax.add_patch(Circle((cx, cy), r, facecolor='white', edgecolor=GREY, linewidth=1.4,
                        zorder=2))
    n_col = round(frac_binder * ndots)
    cols = [BINDER_COLORS[i % 4] for i in range(n_col)] + [NONBIND] * (ndots - n_col)
    rng.shuffle(cols)
    rad = r * 0.80 * np.sqrt(rng.random(ndots))          # area-uniform scatter
    ang = 2 * np.pi * rng.random(ndots)
    for x, y, c in zip(cx + rad * np.cos(ang), cy + rad * np.sin(ang), cols):
        edge = '#888' if c == NONBIND else 'white'
        ax.add_patch(Circle((x, y), 0.05, facecolor=c, edgecolor=edge, linewidth=0.4,
                            zorder=3))


def main():
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    # divider between the two panels
    ax.plot([5.15, 5.15], [0.25, 4.05], color='#cccccc', lw=1.1, ls=(0, (4, 3)), zorder=0)

    # ============================ PANEL A — RecoveryBench ============================
    ax.text(2.5, 4.22, '(A)  Tumbleweed-RecoveryBench', ha='center', va='center',
            fontsize=10.5, weight='bold', color=BLUE)
    ax.text(2.5, 3.97, 'rank true SELEX winners above composition-matched random',
            ha='center', va='center', fontsize=8.0, color=GREY, style='italic')
    ax.text(2.5, 3.78, 'in-domain: the target family is in training', ha='center',
            va='center', fontsize=7.4, color=GREEN, weight='bold', style='italic')

    # winner pool (enriched SELEX, colored binders) and matched-random pool (grey decoys),
    # drawn in the same dot motif as the training figure (colored = binder, grey = non-binder)
    ax.text(1.05, 3.52, 'SELEX winners', ha='center', va='center', fontsize=7.8,
            color=GREEN, weight='bold')
    pool(ax, 1.05, 2.98, 0.42, frac_binder=1.0, seed=11)
    ax.text(1.05, 2.28, 'composition-\nmatched random', ha='center', va='center',
            fontsize=7.8, color=GREY, weight='bold')
    pool(ax, 1.05, 1.66, 0.42, frac_binder=0.0, seed=41)

    # low-mask glyph: a scored sequence with a few bases hidden (black M), illustrating
    # the "(low mask)" label on the box; the pseudo-NLL mechanism itself is in the text
    masked_strip(ax, 2.63, 3.24, 1.14, seed=7, ncells=10, nmask=2)
    ax.text(3.20, 3.49, 'low mask: hide a few bases', ha='center', va='center',
            fontsize=6.4, color=GREY, style='italic')

    # scoring box (centred between the two pools and the score ladder)
    box(ax, 2.45, 2.26, 1.5, 0.80, 'Tumbleweed\npseudo-NLL\n(low mask)', LBLUE, BLUE,
        fontsize=7.8, weight='bold')
    arrow(ax, (1.50, 2.86), (2.45, 2.86), color=GREEN, rad=-0.10)
    arrow(ax, (1.50, 1.78), (2.45, 2.50), color=GREY, rad=0.10)

    # score ladder: each scored sequence is a bar, winners (green) rank above random (grey)
    ax.text(4.28, 3.34, 'score', ha='center', va='center', fontsize=7.2, color=GREY)
    for i in range(3):                                   # winners on top
        ax.add_patch(Rectangle((4.08, 3.10 - i * 0.135), 0.40, 0.10, facecolor=WINNER,
                               edgecolor='white', linewidth=0.5, zorder=3))
    for i in range(3):                                   # random below
        ax.add_patch(Rectangle((4.08, 2.50 - i * 0.135), 0.40, 0.10, facecolor=NONBIND,
                               edgecolor='white', linewidth=0.5, zorder=3))
    arrow(ax, (3.95, 2.72), (4.04, 2.72), color=BLUE)    # into the gap between groups
    arrow(ax, (4.28, 2.18), (4.28, 1.96), color=GREEN)   # ladder → AUROC, clear of bars
    box(ax, 3.50, 1.30, 1.56, 0.62, 'AUROC\n(winner > random)\n0.5 = chance', LGREEN,
        GREEN, fontsize=7.4, weight='bold')

    # name the 5 SELEX targets + what actually ships, so the panel is concrete and honest
    ax.text(2.5, 0.66, '5 SELEX targets:  FGF9 · IL1RL1 · PARP1 · MECP2 · SNCA',
            ha='center', va='center', fontsize=7.4, color=GREY, style='italic')
    ax.text(2.5, 0.45, '400 winner / 400 naive / 400 random per target',
            ha='center', va='center', fontsize=6.6, color=GREY, style='italic')

    # ============================ PANEL B — KdBench ============================
    ax.text(7.65, 4.22, '(B)  Tumbleweed-KdBench', ha='center', va='center',
            fontsize=10.5, weight='bold', color=BLUE)
    ax.text(7.65, 3.95, 'rank held-out aptamers by measured affinity (leave-one-target-out)',
            ha='center', va='center', fontsize=8.0, color=GREY, style='italic')

    # one panel = 1 target x chemistry, >=4 aptamers each with a measured Kd
    box(ax, 5.42, 2.40, 2.30, 1.42, '', '#fbfbfb', GREY)
    ax.text(6.57, 3.66, 'one panel = 1 target × chemistry', ha='center', va='center',
            fontsize=7.0, color=GREY, style='italic')
    kds = ['1.2 nM', '8 nM', '40 nM', '210 nM']
    for i, kd in enumerate(kds):
        sy = 3.34 - i * 0.27
        strip(ax, 5.58, sy, 0.92, seed=70 + i)
        ax.text(6.62, sy + 0.085, f'$K_D$ {kd}', ha='left', va='center',
                fontsize=6.8, color='#333')

    # linear pipeline: hold the family out, then score the held-out aptamers
    box(ax, 7.95, 2.98, 1.95, 0.66, 'hold target family\nout of training',
        LOR, ORANGE, fontsize=7.4, weight='bold')
    arrow(ax, (7.72, 3.31), (7.95, 3.31), color=ORANGE)
    box(ax, 7.95, 2.05, 1.95, 0.62, 'score held-out\naptamers', LBLUE, BLUE,
        fontsize=7.6, weight='bold')
    arrow(ax, (8.925, 2.98), (8.925, 2.67), color=BLUE)

    # Spearman rho compares the predicted order (from scoring) to the measured Kd order
    box(ax, 6.00, 1.08, 2.30, 0.66, 'Spearman ρ\n(predicted vs measured $K_D$)', LGREEN,
        GREEN, fontsize=7.5, weight='bold')
    arrow(ax, (8.40, 2.05), (7.78, 1.74), color=BLUE, rad=0.14)   # predicted order
    arrow(ax, (6.25, 2.40), (6.62, 1.74), color=GREY, rad=-0.10)  # measured order
    ax.text(8.55, 1.96, 'predicted', ha='left', va='center', fontsize=6.2,
            color=BLUE, style='italic')
    ax.text(5.72, 2.06, 'measured', ha='center', va='center', fontsize=6.2,
            color=GREY, style='italic')
    ax.text(7.15, 0.78, 'aggregate ρ over 47 leave-one-target-out panels',
            ha='center', va='center', fontsize=7.6, color=GREY, style='italic')
    ax.text(7.15, 0.55, '911 measured aptamers (653 DNA / 258 RNA)',
            ha='center', va='center', fontsize=6.6, color=GREY, style='italic')

    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig_benchmark_design.{ext}', dpi=300, bbox_inches='tight')
    print('wrote', OUT / 'fig_benchmark_design.png')


if __name__ == '__main__':
    main()
