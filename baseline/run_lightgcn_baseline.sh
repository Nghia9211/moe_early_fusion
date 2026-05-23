#!/bin/bash

# Khai báo môi trường Python theo yêu cầu trong SKILLS.md
PYTHON="/home/research/nghialt/.venv/bin/python"

echo "========================================================"
echo "       RUNNING LIGHTGCN BASELINE"
echo "========================================================"

# Cấu hình danh sách các tập dữ liệu và kịch bản cần chạy
DATASETS=("yelp" "amazon" "goodreads" "amazon_musical" "amazon_industrial")
SCENARIOS=("classic" "user_cold_start" )
NUM_TASKS="None"      # Đặt "None" để chạy toàn bộ file task, hoặc một số nguyên như 500 để test nhanh

for SCENARIO in "${SCENARIOS[@]}"; do
    for TASK_SET in "${DATASETS[@]}"; do
        echo "--------------------------------------------------------"
        echo "Starting evaluation for Dataset: $TASK_SET | Scenario: $SCENARIO"
        echo "--------------------------------------------------------"
        
        $PYTHON LightGCN_baseline.py \
            --task_set "$TASK_SET" \
            --scenario "$SCENARIO" \
            --num_tasks "$NUM_TASKS"

        if [ $? -ne 0 ]; then
            echo "[ERROR] Baseline execution failed for dataset: $TASK_SET | scenario: $SCENARIO"
            # Thoát nếu có lỗi xảy ra
            exit 1
        fi
        
        echo "[SUCCESS] Finished evaluating $TASK_SET on $SCENARIO."
        echo ""
    done
done

echo "========================================================"
echo "ALL DATASETS AND SCENARIOS COMPLETED SUCCESSFULLY!"
echo "Results are saved in the results/ directory."
