import pandas as pd
import matplotlib.pyplot as plt

def add_value_labels(ax, is_percentage=False):
    for p in ax.patches:
        height = p.get_height()
        if is_percentage:
            label = f'{height:.2%}'
        else:
            label = f'{int(height):,}'
            
        ax.annotate(label, 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=9, fontweight='bold')

print("--- 1. Processing Data ---")

# Khởi tạo các Series rỗng
item_counts = pd.Series(dtype=int)
user_counts = pd.Series(dtype=int)
review_counts = pd.Series(dtype=int)

# Đọc Item
try:
    df_item = pd.read_json('item.json', lines=True)
    item_counts = df_item.groupby('source')['item_id'].nunique()
    total_items = df_item['item_id'].nunique()
    print(f"✔ Đã đọc Item. Tổng unique: {total_items:,}")
except Exception as e:
    print(f"❌ Lỗi đọc item.json: {e}")

# Đọc User
try:
    df_user = pd.read_json('user.json', lines=True)
    user_counts = df_user.groupby('source')['user_id'].nunique()
    total_users = df_user['user_id'].nunique()
    print(f"✔ Đã đọc User. Tổng unique: {total_users:,}")
except Exception as e:
    print(f"❌ Lỗi đọc user.json: {e}")

# Đọc Review (Interactions)
try:
    df_review = pd.read_json('review.json', lines=True)
    review_counts = df_review.groupby('source')['review_id'].nunique()
    total_reviews = df_review['review_id'].nunique()
    print(f"✔ Đã đọc Review. Tổng unique: {total_reviews:,}")
except Exception as e:
    print(f"❌ Lỗi đọc review.json: {e}")

print("\n--- 2. Calculating Sparsity ---")

# Kết hợp tất cả thống kê vào một DataFrame để tính toán theo từng Source
summary_df = pd.DataFrame({
    'items': item_counts,
    'users': user_counts,
    'reviews': review_counts
}).fillna(0)

# Công thức Sparsity = 1 - (Interactions / (Users * Items))
def calculate_sparsity(row):
    if row['users'] > 0 and row['items'] > 0:
        return 1 - (row['reviews'] / (row['users'] * row['items']))
    return 1.0

summary_df['sparsity'] = summary_df.apply(calculate_sparsity, axis=1)

# Tính Sparsity tổng thể (Toàn bộ dataset)
global_sparsity = 1 - (total_reviews / (total_users * total_items))

print(summary_df)
print(f"\n=> Global Sparsity: {global_sparsity:.4%}")

print("\n--- 3. Drawing Charts ---")

# Tạo layout 2x2
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Dataset Analysis: Items, Users, Interactions & Sparsity', fontsize=20, fontweight='bold', y=0.98)

# 1. Biểu đồ Items
summary_df['items'].plot(kind='bar', ax=axes[0, 0], color='#4c72b0', width=0.6)
axes[0, 0].set_title(f'Items by Source (Total: {total_items:,})', fontsize=14)
axes[0, 0].tick_params(axis='x', rotation=0)
add_value_labels(axes[0, 0])

# 2. Biểu đồ Users
summary_df['users'].plot(kind='bar', ax=axes[0, 1], color='#c44e52', width=0.6)
axes[0, 1].set_title(f'Users by Source (Total: {total_users:,})', fontsize=14)
axes[0, 1].tick_params(axis='x', rotation=0)
add_value_labels(axes[0, 1])

# 3. Biểu đồ Interactions
summary_df['reviews'].plot(kind='bar', ax=axes[1, 0], color='#55a868', width=0.6)
axes[1, 0].set_title(f'Interactions by Source (Total: {total_reviews:,})', fontsize=14)
axes[1, 0].tick_params(axis='x', rotation=0)
add_value_labels(axes[1, 0])

# 4. Biểu đồ Sparsity
# Chú ý: Sparsity càng gần 100% thì dữ liệu càng thưa
summary_df['sparsity'].plot(kind='bar', ax=axes[1, 1], color='#8172b3', width=0.6)
axes[1, 1].set_title(f'Sparsity Score (Higher = Sparser)\nGlobal: {global_sparsity:.4%}', fontsize=14)
axes[1, 1].set_ylim(0, 1.1) # Để dành chỗ cho label
axes[1, 1].tick_params(axis='x', rotation=0)
add_value_labels(axes[1, 1], is_percentage=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

output_file = 'Dataset_Sparsity_Analysis.png'
plt.savefig(output_file, dpi=300)
print(f"--> Đã lưu biểu đồ vào file: {output_file}")

plt.show()