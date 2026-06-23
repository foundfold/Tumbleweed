#!/usr/bin/env python3
"""Generate the no-contrastive (nc) headline + LOO configs from the v7_film_cnn `_mst2` set.

The headline Tumbleweed model (v7_film_cnn) trains with the supervised-InfoNCE contrastive
term ON (lam_contrast defaults to 1.0 in train_diffusion_hybrid.py). The component ablation
(Supplementary Table S3) showed that term is non-load-bearing: +0.002 on RecoveryBench (within
seed noise) and a slight drop in the conditioning delta. We therefore promote the simpler
contrastive-free model (lam_contrast=0.0) to the headline and keep contrastive as an ablation.

This script clones each of the 6 active `v7_film_cnn_*_mst2` configs to a `v8_film_cnn_nc_*_mst2`
config, doing a text-level transform so the original comments survive:
  - run_id:  v7_film_cnn  ->  v8_film_cnn_nc
  - inject `  lam_contrast: 0.0` into the training block (after lam_diff) if absent

Source configs were hand-written; see git history. No external data sources.
Inputs : training/configs/tumbleweed_60m_diffusion_v7_film_cnn{,_loo_<T>}_mst2.yaml
Outputs: training/configs/tumbleweed_60m_diffusion_v8_film_cnn_nc{,_loo_<T>}_mst2.yaml
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CFGD = os.path.join(HERE, "..", "training", "configs")

SRC_IDS = [
    "tumbleweed_60m_diffusion_v7_film_cnn_mst2",
    "tumbleweed_60m_diffusion_v7_film_cnn_loo_FGF9_mst2",
    "tumbleweed_60m_diffusion_v7_film_cnn_loo_IL1RL1_mst2",
    "tumbleweed_60m_diffusion_v7_film_cnn_loo_PARP1_mst2",
    "tumbleweed_60m_diffusion_v7_film_cnn_loo_MECP2_mst2",
    "tumbleweed_60m_diffusion_v7_film_cnn_loo_SNCA_mst2",
]


def transform(text):
    """Repoint run_id to the nc family and ensure lam_contrast: 0.0 is set."""
    lines = text.splitlines(keepends=True)
    out, have_lc = [], False
    for ln in lines:
        if ln.startswith("run_id:"):
            ln = ln.replace("v7_film_cnn", "v8_film_cnn_nc")
            # annotate the rename inline
            out.append(ln.rstrip("\n") + "  # nc = contrastive term OFF (lam_contrast=0)\n")
            continue
        if ln.strip().startswith("lam_contrast:"):
            have_lc = True
            ln = "  lam_contrast: 0.0           # contrastive term OFF (ablated as non-load-bearing, Table S3)\n"
        out.append(ln)
        # inject right after lam_diff if no explicit lam_contrast in the file
        if ln.strip().startswith("lam_diff:") and not have_lc:
            out.append("  lam_contrast: 0.0           # contrastive term OFF (ablated as non-load-bearing, Table S3)\n")
            have_lc = True
    return "".join(out)


def main():
    for sid in SRC_IDS:
        src = os.path.join(CFGD, f"{sid}.yaml")
        with open(src) as f:
            text = f.read()
        new = transform(text)
        dst_id = sid.replace("v7_film_cnn", "v8_film_cnn_nc")
        dst = os.path.join(CFGD, f"{dst_id}.yaml")
        with open(dst, "w") as f:
            f.write(new)
        has = "lam_contrast: 0.0" in new
        print(f"wrote {dst_id}.yaml  (lam_contrast=0 set: {has})")


if __name__ == "__main__":
    main()
