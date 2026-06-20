#!/usr/bin/env python3
"""
build_paper.py — Convert the Tumbleweed manuscript to Word (.docx) or PDF.

Adapted from the Dogcatcher2 build script. Reads tumbleweed_paper.md, strips the
internal "Figure Plan" section and HTML comments, inlines each figure image with its
legend after the matching results heading, and converts via pandoc with citeproc.

Figures are PNGs in ./figures; citations resolve from references.bib using nar.csl
(Nucleic Acids Research numeric style, for NAR Genomics & Bioinformatics).

PDF engine: this machine has no LaTeX, so the default PDF engine is weasyprint
(pandoc --pdf-engine=weasyprint). Override with --engine if xelatex is installed.

Usage:
    python build_paper.py              # generates .docx (default)
    python build_paper.py --pdf        # generates .pdf (weasyprint)
    python build_paper.py --both       # generates both
    python build_paper.py --pdf --engine xelatex
"""

import os
import re
import shutil
import argparse
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.dirname(SCRIPT_DIR)
# Figures live alongside the manuscript so this directory is self-contained.
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
MANUSCRIPT = os.path.join(SCRIPT_DIR, 'tumbleweed_paper.md')
BIBLIOGRAPHY = os.path.join(SCRIPT_DIR, 'references.bib')
CSL = os.path.join(SCRIPT_DIR, 'nar.csl')

# (results-section heading, image filename in FIGURES_DIR, legend heading)
FIGURE_MAP = [
    ('### Tumbleweed is a chemistry-aware, target-conditional masked-diffusion model (Fig 1)',
     'fig_architecture_detailed.png', '### Figure 1'),
    ('### Training objective',
     'fig2_training_objective.png', '### Figure 2'),
    ('### Tumbleweed-RecoveryBench',
     'fig_benchmark_design.png', '### Figure 3'),
    ('### Tumbleweed recovers SELEX winners above the released unconditional baseline (Fig 4)',
     'fig2_recoverybench.png', '### Figure 4'),
    ('### Target conditioning generalizes within SELEX families but not across them (Fig 5)',
     'fig3_conditioning_ab.png', '### Figure 5'),
    ('### No method ranks held-out aptamer affinity above chance (Table 2)',
     'fig4_kdbench_forest.png', '### Figure 6'),
]

# Supporting-information figures, inlined at a placeholder token (no legend extraction).
SI_FIGURES = {}


def read_manuscript():
    with open(MANUSCRIPT, 'r') as f:
        return f.read()


def strip_internal_sections(text):
    """Remove the Figure Plan section (internal notes) and all HTML comments."""
    match = re.search(r'^## Figure Plan', text, re.MULTILINE)
    if match:
        text = text[:match.start()].rstrip()
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def insert_figures(text, inline_images=True):
    """Insert each figure's image (optional) and legend after its results heading."""
    # Pull the legends out of the Figure Legends section.
    legends = {}
    for _, _, legend_heading in FIGURE_MAP:
        pattern = re.escape(legend_heading) + r'\n(.*?)(?=\n### |\n---|\n## |\Z)'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            legends[legend_heading] = m.group(1).strip()

    # Remove the whole Figure Legends section (legends are inlined instead).
    text = re.sub(r'\n## Figure Legends\n.*?(?=\n## |\Z)', '\n', text, flags=re.DOTALL)

    for results_heading, filename, legend_heading in reversed(FIGURE_MAP):
        m = re.search(re.escape(results_heading), text)
        if not m:
            continue
        rest = text[m.end():]
        nxt = re.search(r'\n(### |## |---)', rest)
        insert_pos = m.end() + nxt.start() if nxt else len(text)

        caption = legends.get(legend_heading, '')
        if inline_images:
            img_path = os.path.join(FIGURES_DIR, filename)
            block = f'\n\n![]({img_path}){{width=6.5in}}\n\n{caption}\n\n'
        else:
            block = f'\n\n{caption}\n\n'
        text = text[:insert_pos] + block + text[insert_pos:]

    # Inline supporting-information figures at their placeholder tokens.
    for token, filename in SI_FIGURES.items():
        if inline_images:
            img_path = os.path.join(FIGURES_DIR, filename)
            repl = f'![]({img_path}){{width=6in}}'
        else:
            repl = ''
        text = text.replace(token, repl)

    return text


def build(output_format='docx', engine='weasyprint', inline_images=True):
    print(f"Building {output_format} (engine={engine})...")
    text = read_manuscript()
    text = strip_internal_sections(text)
    text = insert_figures(text, inline_images=inline_images)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False,
                                     dir=SCRIPT_DIR) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    try:
        cite_args = []
        if os.path.exists(BIBLIOGRAPHY):
            cite_args = ['--citeproc', '--bibliography', BIBLIOGRAPHY]
            if os.path.exists(CSL):
                cite_args += ['--csl', CSL]

        if output_format == 'docx':
            outfile = os.path.join(SCRIPT_DIR, 'tumbleweed_paper.docx')
            cmd = ['pandoc', tmp_path, '-o', outfile,
                   '--from', 'markdown', '--to', 'docx', '--standalone'] + cite_args
        elif output_format == 'pdf':
            outfile = os.path.join(SCRIPT_DIR, 'tumbleweed_paper.pdf')
            cmd = ['pandoc', tmp_path, '-o', outfile,
                   '--from', 'markdown', '--pdf-engine', engine,
                   '-V', 'geometry:margin=1in', '-V', 'fontsize=11pt'] + cite_args
        else:
            raise ValueError(f"Unknown format: {output_format}")

        # weasyprint (under conda) needs Homebrew's glib/pango at load time.
        env = dict(os.environ)
        if engine == 'weasyprint' and os.path.isdir('/opt/homebrew/lib'):
            prev = env.get('DYLD_FALLBACK_LIBRARY_PATH', '')
            env['DYLD_FALLBACK_LIBRARY_PATH'] = (
                '/opt/homebrew/lib' + (f':{prev}' if prev else ''))
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"Error:\n{result.stderr}")
            return None
        print(f"Saved: {outfile}")
        return outfile
    finally:
        os.unlink(tmp_path)


def main():
    p = argparse.ArgumentParser(description='Build the Tumbleweed paper')
    p.add_argument('--pdf', action='store_true', help='Generate PDF')
    p.add_argument('--both', action='store_true', help='Generate both DOCX and PDF')
    p.add_argument('--engine', default='weasyprint',
                   help='pandoc PDF engine (default weasyprint; use xelatex if installed)')
    args = p.parse_args()

    if args.both:
        build('docx', engine=args.engine)
        build('pdf', engine=args.engine)
    elif args.pdf:
        build('pdf', engine=args.engine)
    else:
        build('docx', engine=args.engine)


if __name__ == '__main__':
    main()
