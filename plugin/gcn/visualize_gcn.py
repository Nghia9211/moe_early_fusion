"""
visualize_gcn.py
─────────────────
Visualize GCN embeddings theo ranking quality thay vì global clustering.

Thay vì plot users vs items globally (vô nghĩa với hub-heavy datasets),
ta plot theo "user + liked items vs random items" — phản ánh đúng
thứ GCN thật sự học được: separation giữa relevant và irrelevant items.

Màu sắc:
  🔵 Xanh dương = Users
  🟠 Cam        = Liked items (user đã interact — positive)
  🔴 Đỏ         = Random items (không liên quan — negative)

Nếu GCN tốt: user gần liked items, xa random items.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file',  type=str, required=True)
    parser.add_argument('--emb_file',    type=str, required=True)
    parser.add_argument('--dataset',     type=str, default='amazon')
    parser.add_argument('--num_users',   type=int, default=150,
                        help='Số users sample. Mỗi user sinh 1 liked + 1 random item.')
    parser.add_argument('--min_degree',  type=int, default=5,
                        help='Chỉ sample users có ít nhất min_degree interactions.')
    parser.add_argument('--n_liked',     type=int, default=2,
                        help='Số liked items lấy per user.')
    parser.add_argument('--n_random',    type=int, default=2,
                        help='Số random items lấy per user.')
    parser.add_argument('--perplexity',  type=float, default=20.0)
    parser.add_argument('--pca_dim',     type=int, default=50,
                        help='PCA trước t-SNE. 0 = tắt PCA.')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--out_dir',     type=str, default='figures')
    return parser.parse_args()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"\n{'='*50}")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"{'='*50}")

    # ── Load data ──────────────────────────────────────────────────────────
    graph_data   = torch.load(args.graph_file)
    node_mapping = graph_data['node_item_mapping']   # node_idx → orig_id
    train_dict   = graph_data['train_dict']          # u_idx → [i_idx, ...]
    num_users    = graph_data['num_users']

    embs_dict = torch.load(args.emb_file, map_location='cpu')
    print(f"Embeddings loaded: {len(embs_dict)} nodes, dim={next(iter(embs_dict.values())).shape[0]}")

    # ── Build item pool (tất cả item nodes) ───────────────────────────────
    all_item_orig_ids = set()
    for u_idx, items in train_dict.items():
        for i_idx in items:
            orig_id = node_mapping[i_idx]
            if orig_id in embs_dict:
                all_item_orig_ids.add(orig_id)
    all_item_orig_ids = list(all_item_orig_ids)
    print(f"Total items in pool: {len(all_item_orig_ids)}")

    # ── Sample users có đủ degree ──────────────────────────────────────────
    valid_users = [
        u_idx for u_idx in train_dict
        if len(train_dict[u_idx]) >= args.min_degree
        and node_mapping[u_idx] in embs_dict
    ]
    print(f"Valid users (degree>={args.min_degree}): {len(valid_users)}")

    if len(valid_users) < args.num_users:
        print(f"WARNING: Chỉ có {len(valid_users)} valid users, "
              f"giảm num_users xuống {len(valid_users)}")
        args.num_users = len(valid_users)

    sampled_users = random.sample(valid_users, args.num_users)

    # ── Extract vectors theo 3 nhóm ───────────────────────────────────────
    user_vecs   = []
    liked_vecs  = []
    random_vecs = []

    # Để vẽ đường nối user → liked/random
    liked_connections  = []   # (user_idx_in_list, liked_idx_in_list)
    random_connections = []

    print("Extracting vectors...")
    for u_idx in sampled_users:
        u_orig = node_mapping[u_idx]
        u_vec  = embs_dict[u_orig].numpy()

        # Liked items: lấy từ train_dict, đảm bảo có embedding
        liked_pool = [
            node_mapping[i_idx]
            for i_idx in train_dict[u_idx]
            if node_mapping[i_idx] in embs_dict
        ]
        if len(liked_pool) < args.n_liked:
            continue

        chosen_liked = random.sample(liked_pool, args.n_liked)

        # Random items: không nằm trong liked_pool của user này
        liked_set   = set(liked_pool)
        random_pool = [x for x in all_item_orig_ids if x not in liked_set]
        if len(random_pool) < args.n_random:
            continue

        chosen_random = random.sample(random_pool, args.n_random)

        # Lưu vectors
        u_pos = len(user_vecs)
        user_vecs.append(u_vec)

        for orig_id in chosen_liked:
            liked_connections.append((u_pos, len(liked_vecs)))
            liked_vecs.append(embs_dict[orig_id].numpy())

        for orig_id in chosen_random:
            random_connections.append((u_pos, len(random_vecs)))
            random_vecs.append(embs_dict[orig_id].numpy())

    print(f"Users: {len(user_vecs)} | "
          f"Liked: {len(liked_vecs)} | "
          f"Random: {len(random_vecs)}")

    if len(user_vecs) == 0:
        print("ERROR: Không có vectors nào. Giảm min_degree.")
        return

    # ── Ranking quality report ─────────────────────────────────────────────
    # Đây là metric thật sự — GCN tốt hay không
    print("\n=== Ranking Quality (lý do dùng GCN) ===")
    sims_liked  = []
    sims_random = []

    for u_pos, l_pos in liked_connections:
        sims_liked.append(cosine_sim(user_vecs[u_pos], liked_vecs[l_pos]))
    for u_pos, r_pos in random_connections:
        sims_random.append(cosine_sim(user_vecs[u_pos], random_vecs[r_pos]))

    print(f"  user↔liked  cosine sim: mean={np.mean(sims_liked):.3f}, "
          f"std={np.std(sims_liked):.3f}")
    print(f"  user↔random cosine sim: mean={np.mean(sims_random):.3f}, "
          f"std={np.std(sims_random):.3f}")
    gap = np.mean(sims_liked) - np.mean(sims_random)
    print(f"  Gap (liked - random):   {gap:.3f}  "
          f"{'✅ GCN học tốt' if gap > 0.05 else '⚠️  Gap nhỏ'}")

    # ── t-SNE ──────────────────────────────────────────────────────────────
    X_all = np.vstack([user_vecs, liked_vecs, random_vecs])
    n_u   = len(user_vecs)
    n_l   = len(liked_vecs)
    n_r   = len(random_vecs)

    # PCA trước t-SNE nếu cần
    if args.pca_dim > 0 and X_all.shape[1] > args.pca_dim:
        pca   = PCA(n_components=args.pca_dim, random_state=args.seed)
        X_all = pca.fit_transform(X_all)
        explained = pca.explained_variance_ratio_.sum()
        print(f"\nPCA {X_all.shape[1]}→{args.pca_dim} dims, "
              f"explained variance: {explained:.1%}")

    print(f"\nRunning t-SNE on {X_all.shape[0]} points...")
    tsne  = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        max_iter=2000,
        random_state=args.seed,
        init='pca',       # ổn định hơn random init
        learning_rate='auto',
    )
    X_2d = tsne.fit_transform(X_all)

    X_users_2d  = X_2d[:n_u]
    X_liked_2d  = X_2d[n_u: n_u + n_l]
    X_random_2d = X_2d[n_u + n_l:]

    # ── Fix 1: Filter outlier spike (loại points > 3 std) ─────────────────
    # Hub nodes với degree cực cao tạo spike cluster tách biệt trong t-SNE
    # → loại bỏ để visualization sạch hơn, không ảnh hưởng đến GCN quality
    from scipy import stats as scipy_stats

    def filter_outliers(coords_2d, threshold=3.0):
        """Trả về boolean mask — True = giữ lại, False = outlier."""
        z = np.abs(scipy_stats.zscore(coords_2d, axis=0))
        return (z < threshold).all(axis=1)

    mask_u = filter_outliers(X_users_2d)
    mask_l = filter_outliers(X_liked_2d)
    mask_r = filter_outliers(X_random_2d)

    n_removed = (~mask_u).sum() + (~mask_l).sum() + (~mask_r).sum()
    if n_removed > 0:
        print(f"Filtered {n_removed} outlier points (hub nodes) from visualization.")

    X_users_2d  = X_users_2d[mask_u]
    X_liked_2d  = X_liked_2d[mask_l]
    X_random_2d = X_random_2d[mask_r]

    # Build index remapping sau khi filter để đường nối vẫn đúng
    u_remap = {old: new for new, old in enumerate(np.where(mask_u)[0])}
    l_remap = {old: new for new, old in enumerate(np.where(mask_l)[0])}
    r_remap = {old: new for new, old in enumerate(np.where(mask_r)[0])}

    liked_connections_f  = [
        (u_remap[u], l_remap[l])
        for u, l in liked_connections
        if u in u_remap and l in l_remap
    ]
    random_connections_f = [
        (u_remap[u], r_remap[r])
        for u, r in random_connections
        if u in u_remap and r in r_remap
    ]

    # ── Fix 2: Giới hạn số đường nối để ảnh không bị rối ─────────────────
    MAX_LINES = 80
    liked_to_draw  = random.sample(
        liked_connections_f,  min(MAX_LINES, len(liked_connections_f))
    )
    random_to_draw = random.sample(
        random_connections_f, min(MAX_LINES, len(random_connections_f))
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(9, 7))

    # Đường nối user → liked (xanh nhạt) — chỉ subset MAX_LINES
    for u_pos, l_pos in liked_to_draw:
        ax.plot(
            [X_users_2d[u_pos, 0], X_liked_2d[l_pos, 0]],
            [X_users_2d[u_pos, 1], X_liked_2d[l_pos, 1]],
            color='#4169E1', alpha=0.15, linewidth=0.8,
        )

    # Đường nối user → random (đỏ nhạt) — chỉ subset MAX_LINES
    for u_pos, r_pos in random_to_draw:
        ax.plot(
            [X_users_2d[u_pos, 0], X_random_2d[r_pos, 0]],
            [X_users_2d[u_pos, 1], X_random_2d[r_pos, 1]],
            color='#DC143C', alpha=0.10, linewidth=0.6,
        )

    # Scatter — vẽ random trước (layer dưới), liked và user lên trên
    ax.scatter(X_random_2d[:, 0], X_random_2d[:, 1],
               c='#DC143C', alpha=0.55, s=60,
               edgecolors='w', linewidths=0.4, zorder=2)
    ax.scatter(X_liked_2d[:, 0],  X_liked_2d[:, 1],
               c='#FF8C00', alpha=0.75, s=70,
               edgecolors='w', linewidths=0.4, zorder=3)
    ax.scatter(X_users_2d[:, 0],  X_users_2d[:, 1],
               c='#4169E1', alpha=0.85, s=90,
               edgecolors='w', linewidths=0.5, zorder=4)

    # Legend
    legend_handles = [
        mpatches.Patch(color='#4169E1', label='Users'),
        mpatches.Patch(color='#FF8C00', label='Interacted items'),
        mpatches.Patch(color='#DC143C', label='Random items'),
    ]
    ax.legend(handles=legend_handles, fontsize=17,
              loc='best', framealpha=0.9, markerscale=1.2)

    # Fix 3: Bỏ sim gap text annotation

    ax.tick_params(labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.4)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'tsne_{args.dataset}.pdf')
    plt.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"\n✅ Saved: {out_path}")
    print(f"   Gap={gap:.3f} | "
          f"liked_sim={np.mean(sims_liked):.3f} | "
          f"random_sim={np.mean(sims_random):.3f}")


if __name__ == '__main__':
    main()