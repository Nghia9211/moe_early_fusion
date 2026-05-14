#!/bin/bash
# ============================================================
#  run_moe_all.sh
#  Shell script chạy MoE pipeline qua toàn bộ 3 Datasets
# ============================================================

set -e  # Dừng ngay nếu có lỗi

work_dir="."
cd "$work_dir"

# ── Dataset List & Scenario ───────────────────────────────────────────────
# DATASETS=( "yelp" "amazon")
# DATASETS=( "goodreads" "yelp")
DATASETS=("amazon")
# DATASETS=("goodreads" "yelp" "amazon")

# Đổi SCENARIO thành mảng để loop
# SCENARIOS=("user_cold_start" "classic")
SCENARIOS=("classic")

# ── Global Configs (Dùng chung cho cả 3) ──────────────────────────────────
STAGE="test"
CANS_NUM=20
MAX_EPOCH=5
MAX_SAMPLES=-1    # -1 = toàn bộ dataset
MP=16
SEED=303
TEMPERATURE="0.0"
RERANKER_MODE="llm"                # llm | embed_only | hybrid
MODEL="${MODEL:-qwen-research}"
API_KEY="${API_KEY:-EMPTY}"
BASE_URL="http://localhost:11435/v1"

# ── Vòng lặp chính qua Dataset và Scenario ────────────────────────────────
for DS in "${DATASETS[@]}"; do
    for SCENARIO in "${SCENARIOS[@]}"; do
        echo ""
        echo "############################################################"
        echo "  RUNNING MOE PIPELINE FOR: $DS | SCENARIO: $SCENARIO"
        echo "############################################################"

        # --- Tự động cập nhật Paths dựa trên Dataset hiện tại ---
        DATA_DIR="./data/${DS}/"
        MODEL_PATH="./saved_models/${DS}_best_model.pt"
        CANDIDATE_DIR="../../dataset/tasks5/${SCENARIO}/${DS}/tasks"
        FAISS_DB_PATH="./faiss_dbs/${DS}_rich"
        GCN_PATH="./saved_models/${DS}_gcn_emb_remapped.pt"
        GATING_MODEL_PATH="./saved_models/moe/${DS}_gating_model.pt"
        ITEMFILE="../../dataset/output_data_all/item.json"
        INPUT_JSON_FILE="./data/ground_truth.json"

        # --- Định nghĩa Output riêng cho từng Dataset ---
        P_MODEL="SASRec_MoE"
        NAME="moe_rerank_fbl_v9_full"
        OUTPUT_FILE="./output/${DS}_${SCENARIO}_${NAME}/${P_MODEL}_${MODEL}_SEED${SEED}_ep${MAX_EPOCH}.jsonl"
        RESULT_FILE="./output/${DS}_${SCENARIO}_${NAME}/evaluation_results_${NAME}_${DS}.json"
        # Tạo thư mục output
        mkdir -p "$(dirname "$OUTPUT_FILE")"
        mkdir -p "$(dirname "$RESULT_FILE")"

        # --- Bắt đầu tính thời gian ---
        start_time=$(date +%s)
        echo ">>> Start running $DS - $SCENARIO at $(date)"

        python3 ./main_moe.py \
            --data_dir="$DATA_DIR" \
            --model_path="$MODEL_PATH" \
            --input_json_file="$INPUT_JSON_FILE" \
            --dataset="$DS" \
            --stage="$STAGE" \
            --cans_num=$CANS_NUM \
            --max_epoch=$MAX_EPOCH \
            --max_samples=$MAX_SAMPLES \
            --candidate_dir="$CANDIDATE_DIR" \
            --item_mapping_file="$ITEMFILE" \
            --faiss_db_path="$FAISS_DB_PATH" \
            --gcn_path="$GCN_PATH" \
            --embed_model_name="sentence-transformers/all-mpnet-base-v2" \
            --gating_model_path="$GATING_MODEL_PATH" \
            --reranker_mode="$RERANKER_MODE" \
            --reranker_top_llm=15 \
            --model="$MODEL" \
            --api_key="$API_KEY" \
            --base_url="$BASE_URL" \
            --seed=$SEED \
            --mp=$MP \
            --temperature=$TEMPERATURE \
            --output_file="$OUTPUT_FILE" \
            --result_file="$RESULT_FILE" \
            --save_info \
            --save_rec_dir="$SAVE_REC_DIR" 

        # --- Kết thúc và tính thời gian ---
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        # Định nghĩa file log thời gian (cùng thư mục với OUTPUT_FILE)
        TIME_LOG="$(dirname "$OUTPUT_FILE")/execution_time.txt"
        echo "============================================================" >> "$TIME_LOG"
        echo "Dataset: $DS" >> "$TIME_LOG"
        echo "Scenario: $SCENARIO" >> "$TIME_LOG"
        echo "Start: $(date -d @$start_time)" >> "$TIME_LOG"
        echo "End:   $(date -d @$end_time)" >> "$TIME_LOG"
        echo "Duration: ${duration}s ($((duration/60))m $((duration%60))s)" >> "$TIME_LOG"
        echo "============================================================" >> "$TIME_LOG"

        echo ">>> Finished $DS - $SCENARIO. Duration: ${duration}s. Results: $OUTPUT_FILE"
    done
done

echo ""
echo "============================================================"
echo "  ALL DATASETS AND SCENARIOS COMPLETED."
echo "============================================================"