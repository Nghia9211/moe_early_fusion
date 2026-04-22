#!/bin/bash

# Kiểm tra nếu người dùng không truyền tham số thì mặc định là 'amazon'
# Hoặc bạn có thể yêu cầu bắt buộc nhập
DATASET=${1:-amazon}
EPOCHS=${2:-50} # Mặc định 30 epochs nếu không truyền tham số thứ 2

echo "Đang huấn luyện với Dataset: $DATASET"
echo "Số Epochs: $EPOCHS"

python3 -m train_gating \
    --data_dir  "./data/${DATASET}" \
    --model_path "./saved_models/${DATASET}_best_model.pt" \
    --gcn_path   "./saved_models/${DATASET}_gcn_emb_remapped.pt" \
    --faiss_path "./faiss_dbs/${DATASET}_rich" \
    --output_dir "./saved_models/moe_seq" \
    --dataset    "${DATASET}" \
    --epochs     "${EPOCHS}"