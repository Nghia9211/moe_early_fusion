#!/bin/bash

# ==============================================================================
# SCRIPT TRAIN SASREC (BẢN ĐÃ FIX MASK & MULTI-BLOCKS)
# ==============================================================================

# 1. Khai báo đường dẫn
DATA_ROOT="../MoE/data"
OUTPUT_DIR="./saved_models"

# 2. Khai báo Hyperparameters (Tập trung ở đây để dễ tune)
BATCH_SIZE=256       # Có thể tăng lên 256 nếu GPU của bạn mạnh
LR=0.001
EPOCHS=50
HIDDEN_SIZE=32
NUM_HEADS=2     
DROPOUT=0.2
WEIGHT_DECAY=0.0001

# 3. Danh sách các dataset muốn train lần lượt
# DATASETS=( "yelp")
DATASETS=("yelp") 

echo "=========================================================="
echo "🚀 BẮT ĐẦU QUÁ TRÌNH TRAINING SASREC"
echo "=========================================================="

# Vòng lặp chạy qua từng dataset
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo ">>> ĐANG XỬ LÝ DATASET: ${DATASET^^} <<<"
    DATA_DIR="${DATA_ROOT}/${DATASET}"
    
    # Kiểm tra xem thư mục data có tồn tại không
    if [ ! -d "$DATA_DIR" ]; then
        echo "❌ Lỗi: Không tìm thấy thư mục dữ liệu ${DATA_DIR}."
        echo "👉 Hãy chắc chắn bạn đã chạy file process_data.py thành công."
        continue
    fi
    # Thực thi lệnh Python
    python train.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --epochs $EPOCHS \
        --hidden_size $HIDDEN_SIZE \
        --num_heads $NUM_HEADS \
        --dropout $DROPOUT

    echo "✅ ĐÃ TRAIN XONG DATASET: ${DATASET^^}"
done

echo ""
echo "=========================================================="
echo "🎉 TẤT CẢ TIẾN TRÌNH ĐÃ HOÀN TẤT!"
echo "📍 Model và Biểu đồ được lưu tại: ${OUTPUT_DIR}"
echo "=========================================================="