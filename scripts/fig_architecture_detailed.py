"""Detailed Tumbleweed-Gen architecture block diagram (matplotlib).

A faithful, presentation-quality rendering of the real forward pass in
training/aptamer_diffusion_hybrid.py (config v7_film_cnn, 77.5M params):

  - one shared generative trunk run TWICE: a denoise pass on MASKED ids at the
    SELEX-round timestep, and a contrastive pass on CLEAN ids at t=0;
  - a frozen ESM-2 target branch (mean-pool + mean-center) that enters the trunk
    ONLY through per-layer FiLM (gamma, beta), zero-initialized;
  - two heads (weighted-CE denoise; InfoNCE contrastive) combined as
    L = L_contrast + 0.5 * L_denoise.

This is a SCHEMATIC from fixed geometry (no data file). Run to regenerate PNG/PDF.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
PURPLE = '#7d5ba6'


def box(ax, x, y, w, h, label, face, edge, fontsize=8.5, weight='normal', tc='black',
        ls='-', lw=1.3):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.05',
                       linewidth=lw, edgecolor=edge, facecolor=face, zorder=2,
                       linestyle=ls)
    ax.add_patch(p)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fontsize,
                weight=weight, color=tc, zorder=3)
    return (x + w / 2, y + h / 2)


def arrow(ax, p0, p1, color=GREY, style='-|>', lw=1.6, rad=0.0, ls='-'):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13, lw=lw, color=color,
                        connectionstyle=f'arc3,rad={rad}', zorder=1, linestyle=ls)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(10.4, 12.0))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(2.3, 13.5)
    ax.axis('off')

    ax.text(5.2, 13.25, 'Tumbleweed-Gen architecture', ha='center', va='center',
            fontsize=15, weight='bold', color=BLUE)
    ax.text(5.2, 12.92, '77.5M parameters   ·   d = 768   ·   8 layers   ·   12 heads   '
            '·   dim_ff = 3072', ha='center', va='center', fontsize=9.5, color=GREY)

    # ===================== TRUNK COLUMN (left) =====================
    tx, tw = 0.7, 3.7
    box(ax, tx, 11.95, tw, 0.55, 'aptamer sequence   [RNA]/[DNA] + bases\n'
        '(chemistry token at position 0)', LGREY, GREY, fontsize=8.3)
    box(ax, tx, 11.15, tw, 0.5, 'token embedding   (B, L, 768)', LBLUE, BLUE)
    box(ax, tx, 10.35, tw, 0.5, 'depthwise-CNN motif front-end   (k = 5, 7)', LBLUE, BLUE)
    box(ax, tx, 9.60, tw, 0.45, '+ positional encoding', LBLUE, BLUE, fontsize=8.2)
    box(ax, tx, 8.80, tw, 0.5, '+ timestep embed   t = round_to_t(k, R)\n'
        '(SELEX round → diffusion noise level)', LBLUE, BLUE, fontsize=7.8)
    trunk_c = box(ax, tx, 7.15, tw, 1.1,
                  'Transformer trunk   ×8\n(target token prepended @ pos 0)\n'
                  r'FiLM modulation each layer:  $x \leftarrow \gamma\odot x + \beta$',
                  LBLUE, BLUE, fontsize=8.6, weight='bold')

    for y0, y1 in [(11.95, 11.65), (11.15, 10.85), (10.35, 10.05), (9.60, 9.30),
                   (8.80, 8.25)]:
        arrow(ax, (tx + tw / 2, y0), (tx + tw / 2, y1), color=BLUE)

    # ===================== CONDITIONING BRANCH (right) =====================
    cx, cw = 6.55, 3.1
    box(ax, cx, 11.95, cw, 0.55, 'protein target', LOR, ORANGE, weight='bold')
    box(ax, cx, 11.15, cw, 0.5, 'ESM-2 650M   (frozen)', LOR, ORANGE)
    box(ax, cx, 10.30, cw, 0.55, 'mean-pool over residues\n+ mean-center across target bank',
        LOR, ORANGE, fontsize=7.9)
    tgt_vec = box(ax, cx + 0.35, 9.55, cw - 0.7, 0.45, 'tgt_vec   (B, 1280)', LOR, ORANGE,
                  fontsize=8.2)
    for y0, y1 in [(11.95, 11.65), (11.15, 10.85), (10.30, 10.00)]:
        arrow(ax, (cx + cw / 2, y0), (cx + cw / 2, y1), color=ORANGE)

    # two projection heads off tgt_vec. The upper box (tgt_rep) is narrowed so a
    # clear vertical lane opens on its right for the tgt_vec -> target_contrast
    # arrow to pass straight down without crossing the box.
    tgt_rep = box(ax, cx - 0.1, 8.55, 2.55, 0.6,
                  'target_proj → tgt_rep (B, 768)\n(FiLM source + prepend token)',
                  ORANGE, ORANGE, fontsize=7.9, tc='white', weight='bold')
    tgt_con = box(ax, cx - 0.1, 7.55, cw + 0.2, 0.55,
                  'target_contrast → target_proj (B, 256)\nL2-normalized',
                  LOR, ORANGE, fontsize=7.9)
    arrow(ax, (7.6, 9.55), (7.6, 9.15), color=ORANGE)
    arrow(ax, (9.2, 9.55), (9.2, 8.1), color=ORANGE)

    # FiLM injection into the trunk (gamma, beta)
    arrow(ax, (cx - 0.1, 8.85), (tx + tw, 7.85), color=ORANGE, lw=2.2, rad=-0.18)
    ax.text(5.3, 8.98, 'FiLM (γ, β)\nper layer', ha='center', va='center', fontsize=8,
            color=ORANGE, weight='bold', style='italic')

    # ===================== "RUN TWICE" banner =====================
    # Caption sits at the LEFT, above the line, to keep the central corridor clear
    # for the trunk->heads arrows (otherwise they would cross the text).
    ax.plot([0.5, 9.9], [6.78, 6.78], color=GREY, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.5, 6.97, 'the trunk is run TWICE per step',
            ha='left', va='center', fontsize=8.6, color=GREY, style='italic',
            weight='bold')

    # ===================== TWO PASSES =====================
    # (A) denoise — left
    ax.text(2.55, 6.25, '(A)  DENOISE pass', ha='center', va='center', fontsize=9,
            color=BLUE, weight='bold')
    ax.text(2.55, 6.03, 'masked ids,  t = round-t', ha='center', va='center',
            fontsize=7.8, color=GREY, style='italic')
    dh = box(ax, 0.7, 5.15, 3.7, 0.62,
             'dense → GELU → norm\nlogits = h·Eᵀ + bias   (B, L, V)', LBLUE, BLUE,
             fontsize=8.2)
    ld = box(ax, 0.65, 4.05, 3.8, 0.72,
             'L_denoise\nweighted CE on masked positions\n(EvoFlow-style, weight 1/t, clamp [1/300, 1])',
             LGREEN, GREEN, fontsize=7.8)

    # (B) contrastive — right
    ax.text(7.55, 6.25, '(B)  CONTRASTIVE pass', ha='center', va='center', fontsize=9,
            color=ORANGE, weight='bold')
    ax.text(7.55, 6.03, 'clean ids,  t = 0', ha='center', va='center', fontsize=7.8,
            color=GREY, style='italic')
    ch = box(ax, 5.7, 5.15, 3.7, 0.62,
             'mean-pool over L → proj_head\nL2-norm = proj   (B, 256)', LBLUE, BLUE,
             fontsize=8.2)
    lc = box(ax, 5.65, 4.05, 3.8, 0.72,
             'L_contrast\nInfoNCE(proj, target_proj)\nsame-target = false negatives',
             LGREEN, GREEN, fontsize=7.8)

    # trunk → both heads. Offset in x so neither arrow runs through the (A)/(B)
    # pass labels (centered at x=2.55 / 7.55) or the banner caption.
    arrow(ax, (3.85, 7.15), (3.85, 5.77), color=BLUE)
    arrow(ax, (4.15, 7.15), (7.0, 5.77), color=ORANGE, rad=-0.08)
    arrow(ax, (2.55, 5.15), (2.55, 4.77), color=BLUE)
    arrow(ax, (7.55, 5.15), (7.55, 4.77), color=ORANGE)
    # target_proj feeds the InfoNCE
    arrow(ax, (cx + cw + 0.1, 7.82), (9.45, 4.77), color=ORANGE, rad=-0.25)

    # ===================== JOINT LOSS =====================
    jl = box(ax, 2.95, 2.95, 4.5, 0.7,
             r'$L \;=\; L_{\mathrm{contrast}} \;+\; 0.5 \cdot L_{\mathrm{denoise}}$',
             '#cfe3f2', BLUE, fontsize=13, weight='bold')
    arrow(ax, (2.55, 4.05), (4.4, 3.65), color=GREEN, rad=0.0)
    arrow(ax, (7.55, 4.05), (6.0, 3.65), color=GREEN, rad=0.0)

    # ===================== legend =====================
    lx = 0.7
    for i, (c, fc, lab) in enumerate([
        (BLUE, LBLUE, 'generative trunk / heads'),
        (ORANGE, LOR, 'frozen ESM-2 target conditioning'),
        (GREEN, LGREEN, 'training loss')]):
        yy = 2.55 - i * 0.0  # single row
        box(ax, lx + i * 3.25, 2.45, 0.28, 0.18, '', fc, c, lw=1.1)
        ax.text(lx + i * 3.25 + 0.38, 2.54, lab, ha='left', va='center', fontsize=7.6,
                color=GREY)

    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig_architecture_detailed.{ext}', dpi=300, bbox_inches='tight')
    print('wrote', OUT / 'fig_architecture_detailed.png')


if __name__ == '__main__':
    main()
