import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình đường dẫn
DATASETS = ['yelp', 'amazon', 'goodreads']
BASE_DIR = '/home/research/nghialt/moe_early_fusion/plugin/MoE/output'
# Đổi scenario name nếu bạn test ở folder khác (ví dụ: moe_rr_fbl_tesst)
SCENARIO_SUFFIX = 'classic_moe_rr_test' 

def main():
    all_data = []
    
    for ds in DATASETS:
        # Format: /path/to/output/yelp_classic_moe_rr_test/gating_weights_yelp.csv
        file_path = os.path.join(BASE_DIR, f"{ds}_{SCENARIO_SUFFIX}", f"gating_weights_{ds}.csv")
        
        if os.path.exists(file_path):
            print(f"Loading {file_path}...")
            df = pd.read_csv(file_path)
            # Thêm cột tên dataset để group lúc vẽ
            df['Dataset'] = ds.capitalize()
            all_data.append(df)
        else:
            print(f"Warning: File not found -> {file_path}")
            
    if not all_data:
        print("Không tìm thấy file data nào để vẽ!")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Cấu hình style
    sns.set_theme(style="whitegrid")
    
    # ==========================================
    # 1. Vẽ Boxplot: Phân phối của các weight
    # ==========================================
    melted_df = pd.melt(combined_df, id_vars=['Dataset'], 
                        value_vars=['seq_weight', 'gcn_weight', 'sem_weight'],
                        var_name='Weight_Type', value_name='Weight_Value')
    
    weight_labels = {
        'seq_weight': 'Sequential (SASRec)',
        'gcn_weight': 'Collaborative (GCN)',
        'sem_weight': 'Semantic (LLM)'
    }
    melted_df['Weight_Type'] = melted_df['Weight_Type'].map(weight_labels)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Dataset', y='Weight_Value', hue='Weight_Type', data=melted_df,
                     palette=['#4C72B0', '#DD8452', '#55A868'])
    
    plt.title('Phân phối Gating Weights của các Experts', fontsize=14, pad=15)
    plt.ylabel('Weight Value', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title='Expert', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('gating_weights_distribution.png', dpi=300)
    print("=> Đã lưu hình: gating_weights_distribution.png")

    # ==========================================
    # 2. Vẽ Stacked Bar Chart: Trung bình các weight
    # ==========================================
    avg_weights = combined_df.groupby('Dataset')[['seq_weight', 'gcn_weight', 'sem_weight']].mean()
    
    fig, ax2 = plt.subplots(figsize=(8, 6))
    avg_weights.plot(kind='bar', stacked=True, ax=ax2, 
                     color=['#4C72B0', '#DD8452', '#55A868'], edgecolor='white')
    
    plt.title('Trung bình Gating Weights trên mỗi Dataset', fontsize=14, pad=15)
    plt.ylabel('Average Weight', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.xticks(rotation=0)
    
    # Hiển thị giá trị cụ thể trên từng thanh bar
    for c in ax2.containers:
        # Lấy giá trị từng block, format 2 chữ số thập phân
        labels = [f'{v.get_height():.2f}' if v.get_height() > 0.05 else '' for v in c]
        ax2.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=10)

    # Đổi tên label trong legend
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, [weight_labels[l] for l in labels], title='Expert', 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('gating_weights_average.png', dpi=300)
    print("=> Đã lưu hình: gating_weights_average.png")

if __name__ == "__main__":
    main()
