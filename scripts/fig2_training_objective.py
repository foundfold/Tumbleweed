"""Fig 2 — Tumbleweed training objective: SELEX round sets the diffusion noise level.

A schematic (fixed geometry, no data file) of the training recipe, NOT the model weights.
The defining idea: a sequence observed at SELEX round k of a family with max round R is
assigned diffusion timestep t = 1 - k/R. The diverse round-0 pool enters nearly fully
masked (t≈1); the converged winner enters clean (t≈0). Denoising therefore replays SELEX
enrichment, and ancestral sampling all-masked -> clean is an in-silico selection trajectory.

Top band: four SELEX-round pools drawn as circles holding many small sequences
(grey = non-binder, colored = binder); as rounds enrich, the grey washes out and the
colored binders take over. A left-to-right "enrichment" arrow runs across them.
Under each pool: the representative individual sequence the model trains on at that round,
drawn as a multicolored nucleotide strip with BLACK masks over a fraction t of positions
(round 0 nearly all masked, round R clean).
Bottom: the sampling arrow (all-masked -> winner = in-silico SELEX, same direction as
enrichment) and the joint loss L = L_contrast + 0.5 * L_denoise with its two heads.

Run to regenerate the PNG/PDF.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'manuscript' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

BLUE = '#1b6ca8'
LBLUE = '#d6e6f2'
ORANGE = '#d98a3d'
LOR = '#f5e3cf'
GREY = '#666'
NONBIND = '#bdbdbd'   # grey = non-binding sequence in the pool
MASK = '#111111'      # black = masked position on an individual sequence
GREEN = '#3d8a5a'
LGREEN = '#d9ecdf'
# Two distinct palettes so the strips don't read as pool dots and vice versa.
NUC_COLORS = [BLUE, GREEN, ORANGE, '#7d5ba6']               # revealed nucleotide cells
BINDER_COLORS = ['#c0392b', '#127a6b', '#b9770e', '#b03a8c']   # pool-dot binders (circles)
NUC = ['A', 'C', 'G', 'U']                 # base letter drawn on each revealed cell


def box(ax, x, y, w, h, label, face, edge, fontsize=9, weight='normal', tc='black'):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.05',
                       linewidth=1.3, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(p)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fontsize,
                weight=weight, color=tc, zorder=3)
    return (x + w / 2, y + h / 2)


def arrow(ax, p0, p1, color='black', style='-|>', lw=1.8, rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=15, lw=lw, color=color,
                        connectionstyle=f'arc3,rad={rad}', zorder=1)
    ax.add_patch(a)


def pool(ax, cx, cy, r, t, seed, ndots):
    """Draw a SELEX-round pool as a circle of small sequences. Both the population size
    and the non-binder (grey) fraction shrink as selection enriches: round 0 (t≈1) is a
    packed, diverse, mostly-grey library; each round washes most sequences away, leaving
    round R (t≈0) a sparse pool of a few colored binders."""
    rng = np.random.default_rng(seed)
    ax.add_patch(Circle((cx, cy), r, facecolor='white', edgecolor=GREY, linewidth=1.4,
                        zorder=2))
    n_grey = round((0.18 + 0.72 * t) * ndots)        # grey fraction shrinks with t
    cols = [NONBIND] * n_grey + [BINDER_COLORS[i % 4] for i in range(ndots - n_grey)]
    rng.shuffle(cols)
    # scatter dots uniformly inside the circle (sqrt for area-uniform radius)
    rad = r * 0.80 * np.sqrt(rng.random(ndots))
    ang = 2 * np.pi * rng.random(ndots)
    xs, ys = cx + rad * np.cos(ang), cy + rad * np.sin(ang)
    for x, y, c in zip(xs, ys, cols):
        edge = '#888' if c == NONBIND else 'white'
        ax.add_patch(Circle((x, y), 0.052, facecolor=c, edgecolor=edge, linewidth=0.4,
                            zorder=3))


def masked_seq(ax, cx, y, t, seed, ncells=9, cell=0.135, h=0.24):
    """One individual training sequence: each cell is one nucleotide. Revealed cells show
    their base letter (A/C/G/U) on a colored tile; masked cells show 'M' on a BLACK tile.
    Each position is independently masked with probability t (per-sequence per-position),
    so t=1 is fully masked and t=0 is clean."""
    rng = np.random.default_rng(seed)
    bases = rng.integers(0, 4, size=ncells)          # the sequence's true bases
    masked = rng.random(ncells) < t                   # per-position mask draw
    x0 = cx - ncells * cell / 2
    for i in range(ncells):
        if masked[i]:
            col, letter = MASK, 'M'
        else:
            col, letter = NUC_COLORS[bases[i]], NUC[bases[i]]
        ax.add_patch(Rectangle((x0 + i * cell, y), cell * 0.9, h, facecolor=col,
                               edgecolor='white', linewidth=0.5, zorder=3))
        ax.text(x0 + i * cell + cell * 0.45, y + h / 2, letter, ha='center', va='center',
                fontsize=6.2, weight='bold', color='white', zorder=4)
    ax.add_patch(Rectangle((x0 - 0.008, y - 0.008), ncells * cell + 0.006, h + 0.016,
                           fill=False, edgecolor=GREY, linewidth=0.8, zorder=5))


def main():
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    ax.set_xlim(0, 8.0)
    ax.set_ylim(2.3, 6.0)
    ax.axis('off')

    ax.text(4.0, 5.78, 'Training objective: the SELEX round sets the diffusion noise level',
            ha='center', va='center', fontsize=11.5, weight='bold', color=BLUE)
    ax.text(4.0, 5.5, 'each strip is one aptamer from that round '
            '(one cell = one base:  letter = revealed, black M = masked)',
            ha='center', va='center', fontsize=7.6, color=GREY, style='italic')

    # ---------------- top: SELEX-round pools + per-round masked sequence ----------------
    rounds = [(0, '0'), (3, '3'), (6, '6'), (9, 'R')]
    R = 9
    xs = [1.45, 3.30, 5.15, 7.00]
    ndots = [46, 26, 14, 7]          # population washes out: packed library → few binders
    pool_cy, pool_r = 4.55, 0.55

    for (k, lab), x, nd in zip(rounds, xs, ndots):
        t = 1 - k / R
        pool(ax, x, pool_cy, pool_r, t, seed=100 + k, ndots=nd)
        ax.text(x, 3.84, f'round {lab}', ha='center', va='center', fontsize=8.5,
                color='#333', weight='bold')
        ax.text(x, 3.66, f't = {t:.2f}', ha='center', va='center', fontsize=8.5,
                color=BLUE, weight='bold')
        masked_seq(ax, x, 3.20, t, seed=200 + k)

    # tag the strip row so it reads as a single aptamer, not the whole pool
    ax.text(0.78, 3.32, 'aptamer', ha='right', va='center', fontsize=7.4,
            color=GREY, style='italic')

    # mapping caption
    ax.text(4.0, 2.92, r'round $k$  →  timestep  $t = 1 - k/R$    '
            '(round 0: diverse & masked   →   round R: enriched & clean)',
            ha='center', va='center', fontsize=8.6, color='#333')

    # ---------------- generation note + sampling arrow (all-masked → clean winner) ----------------
    # long left-to-right orange arrow: same direction as SELEX enrichment / in-silico selection
    arrow(ax, (0.95, 2.80), (7.05, 2.80), color=ORANGE, style='-|>', lw=2.4)
    ax.text(4.0, 2.62, 'the model denoises a masked sequence into a clean aptamer for '
            'generation', ha='center', va='center', fontsize=8.4,
            color=ORANGE, style='italic')

    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig2_training_objective.{ext}', dpi=300, bbox_inches='tight')
    print('wrote', OUT / 'fig2_training_objective.png')


if __name__ == '__main__':
    main()
