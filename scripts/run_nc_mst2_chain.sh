#!/bin/bash
# No-contrastive (nc) headline + LOO retrain + RecoveryBench chain (run ON the Thunder node).
#
# Promotes the simpler contrastive-free model (lam_contrast=0) to the headline. Identical to
# run_st2fix_mst2_chain.sh except the 6 run_ids are the v8_film_cnn_nc_*_mst2 set and every
# eval output is tagged `_nc` so it never collides with the v7 (contrastive-on) results.
#
# Trains all 6 nc models to step20000, then RecoveryBench-evaluates each:
#   - matched model : score all 5 targets on the EXACT matched seqs (apples-to-apples w/ EvoFlow)
#                     + null-conditioning ablation (Fig 5 panel A)
#   - each LOO model: score ONLY its held-out target (--only <uniprot>), zero-shot (Fig 5 panel B)
# Low-t regime (0.1,0.15,0.2) matches the published EvoFlow / RiNALMo comparison.
#
# Prereq (already baked into the snapshot from the mst2 run): centered ESM-2 bank carries P14719.
set -u
cd ~/Tumbleweed
CFGD=training/configs
RUNS=~/Desktop/autoRNA_data/tumbleweed/training_runs
DR=~/Tumbleweed/data_refs
EMB="$DR/target_protein_embeddings_centered.parquet"
TL=0.1,0.15,0.2
SEQS="$DR/recovery_seqs.parquet"
log(){ echo "[$(date -u '+%F %T')] $*"; }

MATCHED=tumbleweed_60m_diffusion_v8_film_cnn_nc_mst2

declare -A LOO=(
  [tumbleweed_60m_diffusion_v8_film_cnn_nc_loo_FGF9_mst2]="P31371"
  [tumbleweed_60m_diffusion_v8_film_cnn_nc_loo_IL1RL1_mst2]="P14719"
  [tumbleweed_60m_diffusion_v8_film_cnn_nc_loo_PARP1_mst2]="P09874"
  [tumbleweed_60m_diffusion_v8_film_cnn_nc_loo_MECP2_mst2]="P51608"
  [tumbleweed_60m_diffusion_v8_film_cnn_nc_loo_SNCA_mst2]="P37840"
)
ORDER=("$MATCHED" \
  tumbleweed_60m_diffusion_v8_film_cnn_nc_loo_FGF9_mst2 \
  tumbleweed_60m_diffusion_v8_film_cnn_nc_loo_IL1RL1_mst2 \
  tumbleweed_60m_diffusion_v8_film_cnn_nc_loo_PARP1_mst2 \
  tumbleweed_60m_diffusion_v8_film_cnn_nc_loo_MECP2_mst2 \
  tumbleweed_60m_diffusion_v8_film_cnn_nc_loo_SNCA_mst2)

STRIP='^(Some weights|You should|warnings\.warn|  warnings)'

# ---------- TRAIN all 6 ----------
for RID in "${ORDER[@]}"; do
  CK="$RUNS/$RID/ckpt_step20000.pt"
  if [ -f "$CK" ]; then
    log "TRAIN $RID -- ckpt exists, skip"
  else
    log "TRAIN $RID -- start"
    python3 training/train_diffusion_hybrid.py --config "$CFGD/$RID.yaml" 2>&1 | grep -vE "$STRIP"
    log "TRAIN $RID -- done"
  fi
done

# ---------- EVAL matched (all 5 targets, matched seqs) ----------
MCK="$RUNS/$MATCHED/ckpt_step20000.pt"
MOUT="$DR/recovery_likelihood_v8_film_cnn_nc_mst2_matchedseqs_lowt.csv"
log "EVAL matched RecoveryBench -> $MOUT"
python3 scripts/eval_recovery_likelihood.py --ckpt "$MCK" --n_eval 400 \
  --t_levels "$TL" --seqs "$SEQS" --target_embeddings "$EMB" \
  --out_csv "$MOUT" 2>&1 | grep -vE "$STRIP"

# null-conditioning ablation on the matched model (Fig 5 panel A)
NOUT="$DR/null_conditioning_${MATCHED}.csv"
log "EVAL matched null-conditioning -> $NOUT"
python3 scripts/eval_null_conditioning.py --ckpt "$MCK" --seqs "$SEQS" \
  --t_levels "$TL" --out_csv "$NOUT" 2>&1 | grep -vE "$STRIP"

# ---------- EVAL each LOO (held-out target only) ----------
for RID in "${!LOO[@]}"; do
  CK="$RUNS/$RID/ckpt_step20000.pt"
  TAG=${RID#tumbleweed_60m_diffusion_v8_film_cnn_nc_}      # loo_FGF9_mst2 ...
  OUT="$DR/recovery_nc_${TAG}_lowt.csv"
  TUID="${LOO[$RID]}"
  log "EVAL LOO $RID (only=$TUID) -> $OUT"
  python3 scripts/eval_recovery_likelihood.py --ckpt "$CK" --n_eval 400 \
    --t_levels "$TL" --only "$TUID" --target_embeddings "$EMB" \
    --out_csv "$OUT" 2>&1 | grep -vE "$STRIP"
done

log "=== NC MST2 CHAIN ALL DONE ==="
touch ~/NC_CHAIN_DONE
log "matched   -> $MOUT"
log "null-cond -> $NOUT"
for RID in "${!LOO[@]}"; do
  TAG=${RID#tumbleweed_60m_diffusion_v8_film_cnn_nc_}
  log "loo       -> $DR/recovery_nc_${TAG}_lowt.csv"
done
