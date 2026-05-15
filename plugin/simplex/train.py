"""
SimpleX Training Script
=======================
Trains a SimpleX (CCL) model on the shared review.json dataset and
exports user/item embeddings in the same format as the LightGCN plugin:

    embeddings/{dataset}/
        user_embs.pt       → dict { user_id_str: tensor(emb_dim,) }
        item_embs.pt       → dict { item_id_str: tensor(emb_dim,) }
        user_id2idx.json   → { user_id_str: int }
        item_id2idx.json   → { item_id_str: int }

Usage:
    cd plugin/simplex
    python train.py --dataset yelp --emb_dim 64 --neg_ratio 256 --epochs 200
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow running from plugin/simplex/ directly
sys.path.insert(0, os.path.dirname(__file__))
from model import SimpleXModel, CCLLoss
from utils import (
    load_ground_truth_pairs,
    load_interactions,
    build_id_maps,
    build_train_val_dicts,
    CCLDataset,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleX Collaborative Filtering Model")

    # Data
    parser.add_argument(
        "--data_dir", type=str,
        default="../../dataset/output_data_all",
        help="Path to directory containing review.json"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["amazon", "yelp", "goodreads"],
        help="Dataset to train on"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="embeddings",
        help="Root directory for saved embeddings"
    )

    parser.add_argument(
        "--gt_path", type=str,
        default="../../plugin/MoE/data/ground_truth.json",
        help="Path to consolidated ground_truth.json (used to exclude GT pairs from training)"
    )

    # Model hyperparameters
    parser.add_argument("--emb_dim",   type=int,   default=64,    help="Embedding dimension")
    parser.add_argument("--neg_ratio", type=int,   default=256,   help="Number of negatives per positive (K)")
    parser.add_argument("--margin",    type=float, default=0.4,   help="CCL margin (m)")

    # Training
    parser.add_argument("--epochs",     type=int,   default=200,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=512,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,  help="Learning rate")
    parser.add_argument("--num_workers",type=int,   default=4,     help="DataLoader workers")
    parser.add_argument("--eval_every", type=int,   default=10,    help="Evaluate Recall@20 every N epochs")
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[SimpleX] Device : {device}")
    print(f"[SimpleX] Dataset: {args.dataset}")
    print(f"[SimpleX] emb_dim={args.emb_dim}  neg_ratio={args.neg_ratio}  margin={args.margin}")
    print(f"[SimpleX] epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print("-" * 60)

    # -----------------------------------------------------------------------
    # 1. Load & prepare data
    # -----------------------------------------------------------------------
    gt_exclude = load_ground_truth_pairs(args.gt_path)
    interactions = load_interactions(args.data_dir, args.dataset, gt_exclude=gt_exclude)
    user2idx, item2idx, idx2user, idx2item = build_id_maps(interactions)
    train_dict, val_dict = build_train_val_dicts(interactions, user2idx, item2idx)

    num_users = len(user2idx)
    num_items = len(item2idx)

    dataset = CCLDataset(train_dict, num_items, neg_ratio=args.neg_ratio)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[SimpleX] Training pairs: {len(dataset):,}  |  Batches/epoch: {len(loader):,}")
    print("-" * 60)

    # -----------------------------------------------------------------------
    # 2. Model, loss, optimizer
    # -----------------------------------------------------------------------
    model     = SimpleXModel(num_users, num_items, emb_dim=args.emb_dim).to(device)
    criterion = CCLLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_recall = 0.0
    best_epoch  = 0

    # -----------------------------------------------------------------------
    # 3. Training loop
    # -----------------------------------------------------------------------
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for user_ids, pos_ids, neg_ids in loader:
            user_ids = user_ids.to(device)           # (B,)
            pos_ids  = pos_ids.to(device)            # (B,)
            neg_ids  = neg_ids.to(device)            # (B, K)

            # Positive scores
            u_emb   = F.normalize(model.user_emb(user_ids), dim=-1)     # (B, d)
            pos_emb = F.normalize(model.item_emb(pos_ids),  dim=-1)     # (B, d)
            pos_scores = (u_emb * pos_emb).sum(dim=-1)                  # (B,)

            # Negative scores  — flatten neg_ids, embed, reshape
            B, K = neg_ids.shape
            neg_flat  = neg_ids.view(-1)                                 # (B*K,)
            neg_emb   = F.normalize(model.item_emb(neg_flat), dim=-1)   # (B*K, d)
            neg_emb   = neg_emb.view(B, K, -1)                          # (B, K, d)
            u_expand  = u_emb.unsqueeze(1).expand_as(neg_emb)           # (B, K, d)
            neg_scores = (u_expand * neg_emb).sum(dim=-1)               # (B, K)

            loss = criterion(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % args.eval_every == 0 or epoch == 1:
            recall = recall_at_k(
                model, val_dict, train_dict, num_items,
                k=20, device=device, sample_users=1000
            )
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:03d}/{args.epochs}  "
                f"Loss={avg_loss:.4f}  "
                f"Recall@20={recall:.4f}  "
                f"({elapsed:.1f}s)"
            )
            if recall > best_recall:
                best_recall = recall
                best_epoch  = epoch
        else:
            if epoch % 5 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:03d}/{args.epochs}  Loss={avg_loss:.4f}  ({elapsed:.1f}s)")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"[SimpleX] Training done in {total_time:.1f}s")
    print(f"[SimpleX] Best Recall@20 = {best_recall:.4f} (epoch {best_epoch})")

    # -----------------------------------------------------------------------
    # 4. Export embeddings
    # -----------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        user_embs_matrix = model.get_user_embeddings().cpu()  # (num_users, d)
        item_embs_matrix = model.get_item_embeddings().cpu()  # (num_items, d)

    # Build dicts: original_id → tensor
    user_embs_dict = {idx2user[idx]: user_embs_matrix[idx] for idx in range(num_users)}
    item_embs_dict = {idx2item[idx]: item_embs_matrix[idx] for idx in range(num_items)}

    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    torch.save(user_embs_dict, os.path.join(out_dir, "user_embs.pt"))
    torch.save(item_embs_dict, os.path.join(out_dir, "item_embs.pt"))

    with open(os.path.join(out_dir, "user_id2idx.json"), "w") as f:
        json.dump(user2idx, f)
    with open(os.path.join(out_dir, "item_id2idx.json"), "w") as f:
        json.dump(item2idx, f)

    print(f"[SimpleX] Embeddings saved to: {out_dir}/")
    print(f"          user_embs.pt  — {len(user_embs_dict):,} users, dim={args.emb_dim}")
    print(f"          item_embs.pt  — {len(item_embs_dict):,} items, dim={args.emb_dim}")


if __name__ == "__main__":
    main()
