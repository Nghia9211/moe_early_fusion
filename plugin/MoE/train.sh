#!/bin/bash
# train_gating v2.0 — User-Context Gating
# Usage: bash train.sh [dataset] [epochs]
# Example: bash train.sh yelp 50

DATASET=${1:-goodreads}
EPOCHS=${2:-50}
SPLIT=${3:-val}       # dùng train split để train gating

echo "=============================================="
echo "  Training MoE Gating v2.0 (Context Mode)"
echo "  Dataset : $DATASET"
echo "  Epochs  : $EPOCHS"
echo "  Split   : $SPLIT"
echo "=============================================="

python3 -m train_gating \
    --data_dir    "./data/${DATASET}" \
    --model_path  "./saved_models/${DATASET}_best_model.pt" \
    --gcn_path    "./saved_models/${DATASET}_gcn_emb_remapped.pt" \
    --faiss_path  "./faiss_dbs/${DATASET}_rich" \
    --output_dir  "./saved_models/moe" \
    --dataset     "${DATASET}" \
    --epochs      "${EPOCHS}" \
    --split       "${SPLIT}" \
    --lr          1e-3 \
    --batch_size  256