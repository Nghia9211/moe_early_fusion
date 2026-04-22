import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='logs/gating_weights_amazon.csv')
    parser.add_argument('--dataset', type=str, default='Amazon')
    parser.add_argument('--max_len', type=int, default=50, help='Độ dài chuỗi tối đa để vẽ')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Đọc dữ liệu
    df = pd.read_csv(args.log_file)
    
    # 2. Gom nhóm (Group by) theo len_seq và tính trung bình trọng số
    # Cắt bỏ những user có len_seq > max_len để đồ thị không bị nhiễu ở phần đuôi
    df_filtered = df[df['len_seq'] <= args.max_len]
    grouped = df_filtered.groupby('len_seq').mean().reset_index()
    
    x = grouped['len_seq'].values
    y_gcn = grouped['gcn_weight'].values
    y_seq = grouped['seq_weight'].values
    y_sem = grouped['sem_weight'].values
    
    # Đảm bảo tổng 3 weights luôn = 1.0 (Tránh sai số float)
    total = y_gcn + y_seq + y_sem
    y_gcn = y_gcn / total
    y_seq = y_seq / total
    y_sem = y_sem / total

    # 3. Vẽ biểu đồ Stacked Area
    plt.figure(figsize=(10, 6))
    
    # Dùng màu sắc học thuật (Pastel hoặc màu tách bạch)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue (GCN), Orange (Seq), Green (Sem)
    
    plt.stackplot(x, y_gcn, y_seq, y_sem, 
                  labels=['GCN Expert (Collaborative)', 'SASRec Expert (Sequential)', 'Semantic Expert (Content)'],
                  colors=colors, alpha=0.85)

    # 4. Tinh chỉnh giao diện chuẩn Paper
    plt.title(f'Dynamic MoE Gating Weights Distribution - {args.dataset}', fontsize=14, pad=15)
    plt.xlabel('User Sequence Length (Interaction History)', fontsize=12)
    plt.ylabel('Average Gating Weight', fontsize=12)
    
    # Đặt giới hạn trục X, Y cho gọn
    plt.xlim(1, args.max_len)
    plt.ylim(0, 1.0)
    
    # Legend đặt ở ngoài hoặc góc hợp lý
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=11)
    
    plt.grid(True, axis='x', linestyle='--', alpha=0.5, color='black')
    
    # Đánh dấu vùng Cold-Start (Giả sử <= 5 là cold-start)
    plt.axvline(x=5, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.text(2.5, 0.95, 'Cold-Start', color='white', fontweight='bold', ha='center', va='center', 
             bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

    # 5. Xuất file
    os.makedirs('figures', exist_ok=True)
    out_file = f"figures/gating_distribution_{args.dataset.lower()}.pdf"
    plt.savefig(out_file, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"✅ Đã lưu biểu đồ tại: {out_file}")

if __name__ == '__main__':
    main()