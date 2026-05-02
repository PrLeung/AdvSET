#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1

SOURCE_MODELS=(ALBEF TCL CLIP_ViT)
EXPERIMENTS=(main wo_attn wo_topo)

for source_model in "${SOURCE_MODELS[@]}"; do
  for experiment in "${EXPERIMENTS[@]}"; do
    python eval_AET.py \
      --source_model "${source_model}" \
      --experiment "${experiment}" \
      --cuda_id 0 \
      --save_cluster_info
  done
done
