#!/usr/bin/env python3
"""Aggregate the matched-seed ablation ladder into a summary table.

Reads per-ckpt RecoveryBench CSVs (recovery_<run>.csv: per-target winner-vs-random/naive
AUROC) and null-conditioning CSVs (null_<run>.csv: own-zero conditioning AUROC), groups by
rung over the 3 seeds, and prints mean +/- std per rung. Pairs with run_ablation_ladder.sh.

Inputs : data_refs/ablation_results/{recovery,null}_abl_R*_s*.csv
Outputs: prints table; writes ablation_ladder_summary.csv
"""
import glob
import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "data_refs", "ablation_results")
RUNGS = ["abl_R0_uncond", "abl_R1_token", "abl_R2_film", "abl_R3_cnn", "abl_R4_contrast"]
LABEL = {
    "abl_R0_uncond": "R0 uncond (in-domain only)",
    "abl_R1_token": "R1 + target token",
    "abl_R2_film": "R2 + FiLM conditioning",
    "abl_R3_cnn": "R3 + CNN front-end",
    "abl_R4_contrast": "R4 + contrastive (full)",
}


def recovery_mean(run):
    """Mean AUROC(winner vs random) and (winner vs naive) across targets for one run."""
    f = os.path.join(RES, f"recovery_{run}.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    return (
        float(df["AUROC_winner_vs_random"].mean()),
        float(df["AUROC_winner_vs_naive"].mean()),
    )


def null_own_zero(run):
    """own-zero conditioning AUROC for one run (mean over whatever the null CSV reports)."""
    f = os.path.join(RES, f"null_{run}.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    for col in ("own_minus_zero", "own_zero", "delta_own_zero", "own_minus_zero_auroc"):
        if col in df.columns:
            return float(df[col].mean())
    # fall back: if own and zero columns exist separately
    if {"own", "zero"}.issubset(df.columns):
        return float((df["own"] - df["zero"]).mean())
    return np.nan


rows = []
for rung in RUNGS:
    rec_r, rec_n, nulls = [], [], []
    for s in (0, 1, 2):
        run = f"{rung}_s{s}"
        r = recovery_mean(run)
        if r is not None:
            rec_r.append(r[0])
            rec_n.append(r[1])
        n = null_own_zero(run)
        if n is not None:
            nulls.append(n)
    rows.append(
        {
            "rung": LABEL[rung],
            "n_seeds": len(rec_r),
            "recovery_vs_random_mean": np.mean(rec_r) if rec_r else np.nan,
            "recovery_vs_random_std": np.std(rec_r) if rec_r else np.nan,
            "recovery_vs_naive_mean": np.mean(rec_n) if rec_n else np.nan,
            "null_own_zero_mean": np.mean(nulls) if nulls else np.nan,
            "null_own_zero_std": np.std(nulls) if nulls else np.nan,
        }
    )

out = pd.DataFrame(rows)
pd.set_option("display.width", 160, "display.max_columns", 20)
print("\n=== Tumbleweed ablation ladder (matched corpus, 3 seeds) ===")
print("recovery_vs_random: mean AUROC(winner vs random) over 4 targets, EvoFlow baseline = 0.521\n")
for _, r in out.iterrows():
    print(
        f"  {r['rung']:<32} n={int(r['n_seeds'])}  "
        f"vs_random={r['recovery_vs_random_mean']:.3f} +/- {r['recovery_vs_random_std']:.3f}   "
        f"vs_naive={r['recovery_vs_naive_mean']:.3f}   "
        f"null(own-zero)={r['null_own_zero_mean']:.3f} +/- {r['null_own_zero_std']:.3f}"
    )

# deltas between consecutive rungs (what each component adds)
print("\n=== per-component delta on recovery_vs_random ===")
vals = out["recovery_vs_random_mean"].values
for i in range(1, len(out)):
    print(f"  {out['rung'].iloc[i]:<32} delta = {vals[i] - vals[i-1]:+.3f}")

dest = os.path.join(RES, "ablation_ladder_summary.csv")
out.to_csv(dest, index=False)
print(f"\nwrote {dest}")
