import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import os

# Import langchain HuggingFace để lấy embedding
from langchain_huggingface import HuggingFaceEmbeddings

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Semantic Embeddings with Auto-Naming Topics")
    parser.add_argument('--data_dir', type=str, required=True, help='Đường dẫn tới data chứa id2name.txt')
    parser.add_argument('--dataset', type=str, default='amazon', help='Tên dataset')
    parser.add_argument('--num_items', type=int, default=300, help='Số lượng Item muốn plot')
    parser.add_argument('--num_clusters', type=int, default=5, help='Số cụm K-Means')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"--- Semantic t-SNE for {args.dataset.upper()} ---")

    # 1. Load Dictionary (Item ID -> Name)
    id2name = {}
    id2name_file = os.path.join(args.data_dir, 'id2name.txt')
    with open(id2name_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('::', 1)
            if len(parts) == 2:
                id2name[int(parts[0])] = parts[1].strip()

    all_names = list(id2name.values())
    if len(all_names) == 0:
        print("Lỗi: Không tìm thấy item names!")
        return

    sampled_names = random.sample(all_names, min(args.num_items, len(all_names)))

    # 2. Khởi tạo HuggingFace Embedding Model
    print("Loading Sentence-Transformer Model...")
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Trích xuất Vector
    print(f"Embedding {len(sampled_names)} items...")
    vectors = np.array(embed_model.embed_documents(sampled_names), dtype=np.float32)

    # 4. Chạy K-Means
    kmeans = KMeans(n_clusters=5, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(vectors)

    # 5. Chạy t-SNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)

    # 6. Vẽ Hình chuẩn Academic
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(10, 7)) # Tăng nhẹ chiều ngang để trừ hao chỗ cho Legend
    
    scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap='Set2', alpha=0.8, s=80, edgecolors='w')
    
    circled_nums = ['①', '②', '③', '④', '⑤']
    for c in range(5):
        idx_c = np.where(labels == c)[0]
        if len(idx_c) > 0:
            center_x = np.median(vectors_2d[idx_c, 0])
            center_y = np.median(vectors_2d[idx_c, 1])
            ax.text(center_x, center_y, circled_nums[c], fontsize=26, fontweight='bold', 
                     ha='center', va='center', color='black',
                     bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="black", lw=1.5, alpha=0.95))

    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # TINH CHỈNH LEGEND: Đẩy sang bên phải và không đè lên điểm
    handles, _ = scatter.legend_elements(prop="colors")
    legend_labels = [f"Cluster {circled_nums[i]}" for i in range(5)]
    
    # Cách 1: Nằm trong biểu đồ nhưng ép vào góc trên bên phải với khoảng cách an toàn
    ax.legend(handles, legend_labels, fontsize=16, loc='upper right', framealpha=0.9)

    # Cách 2: Nằm hẳn ra ngoài biểu đồ bên phải (Khuyên dùng nếu điểm dữ liệu quá rộng)
    # ax.legend(handles, legend_labels, fontsize=16, 
    #           loc='upper left', 
    #           bbox_to_anchor=(1.02, 1), # Đẩy sang phải tọa độ x=1.02
    #           borderaxespad=0, 
    #           framealpha=0.9)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Lưu ý: bbox_inches='tight' cực kỳ quan trọng khi đặt legend ở ngoài
    output_path = f'figures/tsne_semantic_{args.dataset}.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✅ Đã lưu: {output_path}")
    

if __name__ == "__main__":
    main()