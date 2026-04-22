import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def debug_and_plot(log_dir="experiment_logs"):
    # 1. Kiểm tra file
    file_pattern = os.path.join(log_dir, "stats_*.csv")
    all_files = glob.glob(file_pattern)
    
    print(f"🔍 Bước 1: Tìm thấy {len(all_files)} files trong thư mục '{log_dir}'")
    if not all_files:
        return

    all_data_frames = []

    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # 2. Xử lý tên Model
        parts = filename.replace('.csv', '').split('_')
        if len(parts) >= 4:
            model_name = "_".join(parts[1:-2])
        else:
            model_name = filename.replace('stats_', '').replace('.csv', '')
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"⚠️ File rỗng: {filename}")
                continue
            
            # --- BƯỚC QUAN TRỌNG NHẤT: ÉP KIỂU HIT ---
            # Chuyển tất cả: True, "True", "  True" -> 1. Còn lại -> 0
            # Cách này không bao giờ lỗi kể cả khi dữ liệu là String hay Boolean
            df['Hit_Numeric'] = (df['Hit'].astype(str).str.strip().str.upper() == 'TRUE').astype(int)
            
            print(f"✅ Đã đọc {filename} -> Model: {model_name} | Số dòng: {len(df)} | Số lần Hit=True: {df['Hit_Numeric'].sum()}")
            
            df['Model_Group'] = model_name
            all_data_frames.append(df[['Stage', 'Hit_Numeric', 'Model_Group']])
            
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {filename}: {e}")

    if not all_data_frames:
        print("❌ Không có dữ liệu hợp lệ để xử lý tiếp.")
        return

    # 3. Gộp dữ liệu
    final_df = pd.concat(all_data_frames, ignore_index=True)

    # 4. Tính toán trung bình
    # Chúng ta gom nhóm theo Model_Group và Stage để lấy giá trị trung bình (Hit Rate)
    summary = final_df.groupby(['Model_Group', 'Stage'])['Hit_Numeric'].mean().reset_index()
    summary['Hit_Rate_Percent'] = summary['Hit_Numeric'] * 100

    print("\n📊 BẢNG DỮ LIỆU CUỐI CÙNG TRƯỚC KHI VẼ:")
    print(summary)
    
    if summary['Hit_Rate_Percent'].sum() == 0:
        print("\n‼️ CẢNH BÁO: Tất cả Hit Rate đều bằng 0%. Đồ thị sẽ không có cột!")

    # 5. Vẽ đồ thị
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Sử dụng barplot của Seaborn
    ax = sns.barplot(data=summary, x='Stage', y='Hit_Rate_Percent', hue='Model_Group')

    # Thêm nhãn % trên đầu cột
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f'{h:.1f}%', 
                    (p.get_x() + p.get_width() / 2., h), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=10, fontweight='bold')

    plt.title('Tổng hợp Hit Rate (%) từ các thí nghiệm', fontsize=15)
    plt.ylabel('Hit Rate (%)')
    plt.ylim(0, 110)
    plt.legend(title='Phiên bản', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.show()

# Gọi hàm
debug_and_plot("experiment_logs")