#!/usr/bin/env bash
# 在已配置好的 AKS conda 环境中跑 1 条 LongVideoBench 样本（FOCUS 目录下已含 datasets/longvideobench_smoke）
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate AKS
cd "$(dirname "$0")"
exec python select_keyframe.py \
  --dataset_name longvideobench \
  --dataset_path ./datasets/longvideobench_smoke \
  --output_dir focus_aks_smoke \
  --num_keyframes 16 \
  --batch_size 8 \
  --blip_model large \
  --seed 42 \
  --coarse_every_sec 32.0 \
  --fine_every_sec 2.0 \
  "$@"
