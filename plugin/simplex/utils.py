"""
Data utilities for SimpleX training.

Reads interaction data from the shared dataset/output_data_all/review.json
(JSONL format, one JSON object per line).

Key helpers:
  - load_ground_truth_pairs(gt_path)      → set of (user_id, item_id) to exclude
  - load_interactions(data_dir, dataset,
                      gt_exclude)         → list of (user_id, item_id) pairs
  - build_id_maps(interactions)           → user2idx, item2idx dicts
  - CCLDataset                            → PyTorch Dataset for CCL training
  - recall_at_k(model, val_dict, ...)     → quick Recall@K for monitoring
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 1. Loading & building ID maps
# ---------------------------------------------------------------------------

DATASET_SOURCES = {
    "amazon":    "amazon",
    "yelp":      "yelp",
    "goodreads": "goodreads",
}


def load_ground_truth_pairs(gt_path: str) -> set:
    """
    Load the consolidated ground truth file (plugin/MoE/data/ground_truth.json)
    and return a set of (user_id, item_id) pairs to exclude from training.

    Args:
        gt_path: path to ground_truth.json
                 (e.g. '../../plugin/MoE/data/ground_truth.json')
    Returns:
        set of (user_id_str, item_id_str)
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"ground_truth.json not found at '{gt_path}'.\n"
            "Pass gt_path=None to skip exclusion (NOT recommended)."
        )
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = {(r["user_id"], r["item_id"]) for r in data if "user_id" in r and "item_id" in r}
    print(f"  [GT Exclusion] Loaded {len(pairs):,} (user, item) ground-truth pairs to exclude")
    return pairs


def load_interactions(
    data_dir: str,
    dataset: str,
    gt_exclude: set = None,
) -> List[Tuple[str, str]]:
    """
    Read review.json (JSONL) and return (user_id, item_id) pairs for the
    target dataset, excluding any ground-truth pairs to prevent data leakage.

    Args:
        data_dir  : path to dataset/output_data_all/
        dataset   : 'amazon' | 'yelp' | 'goodreads'
        gt_exclude: set of (user_id, item_id) to exclude (from load_ground_truth_pairs).
                    Pass None to skip exclusion (not recommended).
    Returns:
        List of (user_id_str, item_id_str) with GT pairs removed
    """
    source_name = DATASET_SOURCES.get(dataset.lower())
    if source_name is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_SOURCES)}")

    review_path = os.path.join(data_dir, "review.json")
    if not os.path.exists(review_path):
        raise FileNotFoundError(f"review.json not found at {review_path}")

    if gt_exclude is None:
        print("  [WARNING] gt_exclude is None — ground truth pairs NOT excluded from training!")
        gt_exclude = set()

    interactions = []
    excluded_count = 0
    print(f"Loading interactions for '{dataset}' from {review_path} …")
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("source") == source_name:
                uid = record.get("user_id")
                iid = record.get("item_id")
                if uid and iid:
                    if (uid, iid) in gt_exclude:
                        excluded_count += 1
                    else:
                        interactions.append((uid, iid))

    print(f"  → {len(interactions):,} interactions loaded for '{dataset}'")
    if excluded_count:
        print(f"  → {excluded_count:,} ground-truth pairs excluded (data leakage prevention ✓)")
    return interactions


def build_id_maps(
    interactions: List[Tuple[str, str]],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    """
    Build bidirectional int-index maps for users and items.

    Returns:
        user2idx, item2idx, idx2user, idx2item
    """
    users = sorted({u for u, _ in interactions})
    items = sorted({i for _, i in interactions})

    user2idx = {u: idx for idx, u in enumerate(users)}
    item2idx = {i: idx for idx, i in enumerate(items)}
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}

    print(f"  Users: {len(user2idx):,}  |  Items: {len(item2idx):,}")
    return user2idx, item2idx, idx2user, idx2item


def build_train_val_dicts(
    interactions: List[Tuple[str, str]],
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Split interactions into train / validation dicts.

    train_dict[u_idx] = [i_idx, ...]
    val_dict  [u_idx] = [i_idx, ...]   (1 item per user for monitoring)

    Uses leave-one-out strategy: the last interaction per user goes to val.
    """
    # Group by user, keeping insertion order (file order ≈ time order)
    user_items: Dict[int, List[int]] = defaultdict(list)
    for uid, iid in interactions:
        u_idx = user2idx.get(uid)
        i_idx = item2idx.get(iid)
        if u_idx is not None and i_idx is not None:
            user_items[u_idx].append(i_idx)

    train_dict: Dict[int, List[int]] = {}
    val_dict:   Dict[int, List[int]] = {}

    for u_idx, items in user_items.items():
        if len(items) <= 1:
            train_dict[u_idx] = items
            # No val entry — skip users with only one interaction
        else:
            val_dict[u_idx]   = [items[-1]]
            train_dict[u_idx] = items[:-1]

    print(f"  Train users: {len(train_dict):,}  |  Val users: {len(val_dict):,}")
    return train_dict, val_dict


# ---------------------------------------------------------------------------
# 2. PyTorch Dataset for CCL training
# ---------------------------------------------------------------------------

class CCLDataset(Dataset):
    """
    Each sample = (user_idx, pos_item_idx, neg_item_indices[K]).

    Negative items are sampled uniformly from all items except the user's
    positive items (non-observed negatives). This is done online each epoch.
    """

    def __init__(
        self,
        train_dict: Dict[int, List[int]],
        num_items: int,
        neg_ratio: int = 256,
    ):
        self.train_dict  = train_dict
        self.num_items   = num_items
        self.neg_ratio   = neg_ratio

        # Flatten into (user, pos_item) pairs
        self.pairs: List[Tuple[int, int]] = []
        for u, items in train_dict.items():
            for i in items:
                self.pairs.append((u, i))

        # Convert pos sets for fast lookup during negative sampling
        self.pos_sets: Dict[int, set] = {
            u: set(items) for u, items in train_dict.items()
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        user, pos_item = self.pairs[idx]
        pos_set = self.pos_sets[user]

        # Sample K negatives (rejection sampling, fast for small pos_set)
        neg_items = []
        while len(neg_items) < self.neg_ratio:
            candidates = np.random.randint(0, self.num_items, size=self.neg_ratio * 2)
            for c in candidates:
                if c not in pos_set:
                    neg_items.append(c)
                if len(neg_items) == self.neg_ratio:
                    break

        return (
            torch.tensor(user,     dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_items[:self.neg_ratio], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# 3. Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def recall_at_k(
    model,
    val_dict: Dict[int, List[int]],
    train_dict: Dict[int, List[int]],
    num_items: int,
    k: int = 20,
    device: torch.device = torch.device("cpu"),
    sample_users: int = 500,
) -> float:
    """
    Compute Recall@K on val_dict.
    Exclude train items from ranked list (standard protocol).

    Args:
        model        : trained SimpleXModel (in eval mode)
        val_dict     : {u_idx: [val_item_idx]}
        train_dict   : {u_idx: [train_item_idxs]} — items to exclude from ranking
        num_items    : total number of items
        k            : top-K cutoff
        device       : torch device
        sample_users : evaluate on a random subset for speed
    Returns:
        Recall@K (float)
    """
    model.eval()
    all_item_embs = model.get_item_embeddings().to(device)  # (num_items, d)

    users = list(val_dict.keys())
    if sample_users and len(users) > sample_users:
        users = random.sample(users, sample_users)

    hits = 0
    total = 0
    for u in users:
        gt_items = set(val_dict[u])
        exclude  = set(train_dict.get(u, []))

        u_emb = F.normalize(model.user_emb.weight[u].unsqueeze(0), dim=-1).to(device)
        scores = (u_emb @ all_item_embs.T).squeeze(0)  # (num_items,)

        # Mask out training items
        scores[list(exclude)] = -1e9

        top_k = scores.topk(k).indices.tolist()
        hits  += int(bool(gt_items & set(top_k)))
        total += 1

    return hits / total if total > 0 else 0.0


# needed inside recall_at_k
import torch.nn.functional as F
