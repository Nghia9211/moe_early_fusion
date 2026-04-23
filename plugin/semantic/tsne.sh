#!/bin/bash

# Kiểm tra xem người dùng có truyền tên dataset vào không
if [ -z "$1" ]; then
    echo "Lỗi: Bạn chưa truyền tên dataset!"
    echo "Cách sử dụng: ./run_tsne_semantic.sh <tên_dataset> [số_lượng_item_muốn_vẽ]"
    echo "Ví dụ: ./run_tsne_semantic.sh amazon 300"
    exit 1
fi

DATASET=$1

# Lấy số lượng item từ tham số thứ 2, nếu không truyền thì mặc định là 300
NUM_ITEMS=${2:-300}

# Đường dẫn tới thư mục data (nơi chứa file id2name.txt)
# Bạn hãy điều chỉnh đường dẫn này nếu cấu trúc thư mục của bạn khác nhé
DATA_DIR="../MoE/data/${DATASET}"

echo "======================================================"
echo "🚀 Đang chạy t-SNE Semantic Visualization..."
echo "📦 Dataset      : $DATASET"
echo "📊 Số lượng Item: $NUM_ITEMS"
echo "📂 Thư mục data : $DATA_DIR"
echo "======================================================"

# Gọi file Python chạy t-SNE
python3 visualize_semantic.py \
    --data_dir "$DATA_DIR" \
    --dataset "$DATASET" \
    --num_items "$NUM_ITEMS"

echo "======================================================"
echo "🎉 Hoàn tất!"