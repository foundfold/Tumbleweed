#!/bin/bash
# Launch contrastive aptamer training on a Thunder A100 instance.
#
# Prerequisites on the Thunder instance:
#   - Python 3.10+ with torch (cu124), pyarrow, pandas, pyyaml, scipy
#   - Working dir contains the `training/` and `data_refs/` directories synced from Drive
#   - Per-source parquets synced to ~/data/aptamer/  (mirroring the Mac layout)
#
# Usage:
#   bash launch_thunder.sh                        # single-GPU
#   bash launch_thunder.sh --gpus 4               # 4-GPU DDP via torchrun
#   bash launch_thunder.sh --config configs/X.yaml
#   bash launch_thunder.sh --resume runs/<id>/ckpt_step100000.pt

set -euo pipefail
cd "$(dirname "$0")"

GPUS=1
CONFIG=configs/default.yaml
RESUME=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)   GPUS=$2;   shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --resume) RESUME=$2; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

EXTRA=""
[[ -n "$RESUME" ]] && EXTRA="--resume $RESUME"

if [[ "$GPUS" -gt 1 ]]; then
  echo "Launching DDP on $GPUS GPUs with $CONFIG"
  torchrun --nproc-per-node="$GPUS" --standalone \
    train_contrastive.py --config "$CONFIG" $EXTRA
else
  echo "Launching single-GPU with $CONFIG"
  python3 -u train_contrastive.py --config "$CONFIG" $EXTRA
fi
