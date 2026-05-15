#!/usr/bin/env bash
# =============================================================================
# SimpleX Training Script
# =============================================================================
# Run from: moe_early_fusion/plugin/simplex/
#
# Default hyperparameters (from the SimpleX paper, CIKM 2021):
#   emb_dim   = 64
#   neg_ratio = 256   (use 512 or 1024 if you have a large GPU)
#   margin    = 0.4
#   epochs    = 200
#   batch_size= 512
#   lr        = 1e-3
# =============================================================================

set -e
cd "$(dirname "$0")"   # Make sure we're in plugin/simplex/

# Use project venv
PYTHON="/home/research/nghialt/.venv/bin/python"

DATA_DIR="../../dataset/output_data_all"
EMB_DIM=64
NEG_RATIO=256
MARGIN=0.4
EPOCHS=200
BATCH_SIZE=512
LR=0.001

echo "============================================================"
echo "  SimpleX Training"
echo "============================================================"

# ---- Yelp ----
echo ""
echo ">>> Training on YELP ..."
$PYTHON train.py \
    --data_dir   "$DATA_DIR" \
    --dataset    yelp \
    --emb_dim    $EMB_DIM \
    --neg_ratio  $NEG_RATIO \
    --margin     $MARGIN \
    --epochs     $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr         $LR \
    --device     auto

# ---- Amazon ----
echo ""
echo ">>> Training on AMAZON ..."
$PYTHON train.py \
    --data_dir   "$DATA_DIR" \
    --dataset    amazon \
    --emb_dim    $EMB_DIM \
    --neg_ratio  $NEG_RATIO \
    --margin     $MARGIN \
    --epochs     $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr         $LR \
    --device     auto

# ---- Goodreads ----
echo ""
echo ">>> Training on GOODREADS ..."
$PYTHON train.py \
    --data_dir   "$DATA_DIR" \
    --dataset    goodreads \
    --emb_dim    $EMB_DIM \
    --neg_ratio  $NEG_RATIO \
    --margin     $MARGIN \
    --epochs     $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr         $LR \
    --device     auto

echo ""
echo "============================================================"
echo "  All training done. Embeddings saved to plugin/simplex/embeddings/"
echo "============================================================"
