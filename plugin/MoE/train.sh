#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  train_all.sh — MoE Gating v2.3 | Train tất cả dataset & loss
# ══════════════════════════════════════════════════════════════════
#
#  Usage:
#    bash train_all.sh [loss] [epochs] [split]
#
#  Args (tuỳ chọn, theo thứ tự):
#    loss   : ce | bpr | both  (default: both)
#    epochs : số epoch          (default: 50)
#    split  : train | val       (default: val)
#
#  Ví dụ:
#    bash train_all.sh              # train cả ce + bpr, 50 epochs
#    bash train_all.sh ce           # chỉ CE loss
#    bash train_all.sh bpr 100      # BPR loss, 100 epochs
#    bash train_all.sh both 80 train

LOSS=${1:-both}
EPOCHS=${2:-50}
SPLIT=${3:-val}

# ── Màu terminal ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ── Datasets ──────────────────────────────────────────────────────
DATASETS=("amazon" "yelp" "goodreads")

# ── CE hyperparams ────────────────────────────────────────────────
CE_LR=1e-3
CE_BATCH=256
CE_BALANCE_EPS=0.1

# ── BPR hyperparams ───────────────────────────────────────────────
BPR_LR=1e-3
BPR_BATCH=256
BPR_N_NEG=5
BPR_HARD_RATIO=0.5

# ── Kiểm tra loss arg hợp lệ ─────────────────────────────────────
if [[ "$LOSS" != "ce" && "$LOSS" != "bpr" && "$LOSS" != "both" ]]; then
    echo -e "${RED}[ERROR] loss phải là: ce | bpr | both${NC}"
    exit 1
fi

# ── Tracking kết quả ──────────────────────────────────────────────
declare -A RESULTS   # RESULTS["dataset_loss"] = "OK" | "FAIL"
TOTAL=0
FAILED=0
START_ALL=$SECONDS

# ══════════════════════════════════════════════════════════════════
print_header() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
}

run_training() {
    local dataset=$1
    local loss=$2

    local data_dir="./data/${dataset}"
    local model_path="./saved_models/${dataset}_best_model.pt"
    local gcn_path="./saved_models/${dataset}_gcn_emb_remapped.pt"
    local faiss_path="./faiss_dbs/${dataset}_rich"
    local output_dir="./saved_models/moe_v23/${dataset}/${loss}"

    # Kiểm tra file bắt buộc
    local missing=0
    for f in "$data_dir" "$model_path"; do
        if [[ ! -e "$f" ]]; then
            echo -e "  ${RED}[SKIP] Không tìm thấy: $f${NC}"
            missing=1
        fi
    done
    [[ $missing -eq 1 ]] && return 1

    # Cảnh báo nếu GCN / FAISS không có (không bắt buộc)
    [[ ! -f "$gcn_path"    ]] && echo -e "  ${YELLOW}[WARN] GCN không có: $gcn_path${NC}"
    [[ ! -d "$faiss_path"  ]] && echo -e "  ${YELLOW}[WARN] FAISS không có: $faiss_path${NC}"

    mkdir -p "$output_dir"
    local log_file="${output_dir}/train.log"

    echo -e "  ${GREEN}▶ Bắt đầu training...${NC}"
    echo -e "  Output dir : $output_dir"
    echo -e "  Log file   : $log_file"

    local start=$SECONDS

    if [[ "$loss" == "ce" ]]; then
        python3 -m train_gating \
            --data_dir    "$data_dir"    \
            --model_path  "$model_path"  \
            --gcn_path    "$gcn_path"    \
            --faiss_path  "$faiss_path"  \
            --output_dir  "$output_dir"  \
            --dataset     "$dataset"     \
            --epochs      "$EPOCHS"      \
            --split       "$SPLIT"       \
            --loss        ce             \
            --lr          $CE_LR         \
            --batch_size  $CE_BATCH      \
            --balance_eps $CE_BALANCE_EPS \
            2>&1 | tee "$log_file"

    else  # bpr
        python3 -m train_gating \
            --data_dir    "$data_dir"    \
            --model_path  "$model_path"  \
            --gcn_path    "$gcn_path"    \
            --faiss_path  "$faiss_path"  \
            --output_dir  "$output_dir"  \
            --dataset     "$dataset"     \
            --epochs      "$EPOCHS"      \
            --split       "$SPLIT"       \
            --loss        bpr            \
            --lr          $BPR_LR        \
            --batch_size  $BPR_BATCH     \
            --n_neg       $BPR_N_NEG     \
            --hard_ratio  $BPR_HARD_RATIO \
            2>&1 | tee "$log_file"
    fi

    local exit_code=${PIPESTATUS[0]}
    local elapsed=$(( SECONDS - start ))

    if [[ $exit_code -eq 0 ]]; then
        echo -e "  ${GREEN}✅ Xong! (${elapsed}s)${NC}"
        return 0
    else
        echo -e "  ${RED}❌ THẤT BẠI (exit=$exit_code, ${elapsed}s) — xem: $log_file${NC}"
        return 1
    fi
}

# ══════════════════════════════════════════════════════════════════
#  Xác định danh sách loss cần chạy
# ══════════════════════════════════════════════════════════════════
if [[ "$LOSS" == "both" ]]; then
    LOSSES=("ce" "bpr")
else
    LOSSES=("$LOSS")
fi

# ══════════════════════════════════════════════════════════════════
#  Banner
# ══════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     MoE Gating v2.3 — Full Training Pipeline        ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo -e "  Datasets : ${DATASETS[*]}"
echo -e "  Losses   : ${LOSSES[*]}"
echo -e "  Epochs   : $EPOCHS"
echo -e "  Split    : $SPLIT"
echo ""

# ══════════════════════════════════════════════════════════════════
#  Main loop: dataset × loss
# ══════════════════════════════════════════════════════════════════
for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        TOTAL=$(( TOTAL + 1 ))
        key="${dataset}_${loss}"

        print_header "Dataset: ${dataset^^}  |  Loss: ${loss^^}  (${TOTAL}/${#DATASETS[@]}×${#LOSSES[@]})"

        if run_training "$dataset" "$loss"; then
            RESULTS[$key]="✅ OK"
        else
            RESULTS[$key]="❌ FAIL"
            FAILED=$(( FAILED + 1 ))
        fi
    done
done

# ══════════════════════════════════════════════════════════════════
#  Bảng tổng kết
# ══════════════════════════════════════════════════════════════════
ELAPSED_ALL=$(( SECONDS - START_ALL ))
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                   KẾT QUẢ TỔNG KẾT                  ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
printf "  %-18s | %-8s | %s\n" "Dataset" "Loss" "Status"
echo "  ──────────────────────────────────────────"
for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        key="${dataset}_${loss}"
        printf "  %-18s | %-8s | %s\n" "$dataset" "$loss" "${RESULTS[$key]:-⚠️  SKIP}"
    done
done
echo "  ──────────────────────────────────────────"
echo -e "  Tổng: ${TOTAL} jobs | Thất bại: ${FAILED} | Thời gian: ${ELAPSED_ALL}s"

if [[ $FAILED -eq 0 ]]; then
    echo -e "\n  ${GREEN}${BOLD}🎉 Tất cả hoàn thành thành công!${NC}"
    exit 0
else
    echo -e "\n  ${RED}${BOLD}⚠️  Có ${FAILED} job thất bại. Kiểm tra log để debug.${NC}"
    exit 1
fi