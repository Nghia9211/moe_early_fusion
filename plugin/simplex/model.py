"""
SimpleX: A Simple and Strong Baseline for Collaborative Filtering
Paper: https://arxiv.org/abs/2109.12613 (CIKM 2021)

Model: Simple Matrix Factorization (two-tower embedding)
Loss : Cosine Contrastive Loss (CCL)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleXModel(nn.Module):
    """
    SimpleX two-tower model.
    Both users and items are represented by a single embedding table.
    Inference score = cosine_similarity(user_emb, item_emb).
    """

    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        # Xavier uniform init (same as the paper)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return cosine similarity scores for (user, item) pairs."""
        u = F.normalize(self.user_emb(user_ids), dim=-1)
        i = F.normalize(self.item_emb(item_ids), dim=-1)
        return (u * i).sum(dim=-1)  # (B,)

    def get_user_embeddings(self) -> torch.Tensor:
        """Return L2-normalised user embedding matrix (num_users, emb_dim)."""
        return F.normalize(self.user_emb.weight.detach(), dim=-1)

    def get_item_embeddings(self) -> torch.Tensor:
        """Return L2-normalised item embedding matrix (num_items, emb_dim)."""
        return F.normalize(self.item_emb.weight.detach(), dim=-1)

    def score_user_items(self, user_id: int, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Score a single user against a set of items.
        Args:
            user_id:      scalar user index
            item_indices: 1-D tensor of item indices
        Returns:
            1-D tensor of cosine scores
        """
        u = F.normalize(self.user_emb.weight[user_id].unsqueeze(0), dim=-1)  # (1, d)
        i = F.normalize(self.item_emb.weight[item_indices], dim=-1)           # (K, d)
        return (u * i).sum(dim=-1)                                             # (K,)


class CCLLoss(nn.Module):
    """
    Cosine Contrastive Loss from the SimpleX paper.

    L = Σ_{(u,i+)} [ -s(u,i+) + (1/K) * Σ_{j ∈ N_u} max(0, s(u,j) - m) ]

    where:
        s(u,x) = cosine_similarity(user_emb_u, item_emb_x)
        K      = number of negative samples per positive
        m      = margin (default 0.4 in paper)
    """

    def __init__(self, margin: float = 0.4, neg_weight: float = 1.0):
        """
        Args:
            margin:     Hinge margin m. Paper default = 0.4.
            neg_weight: Weight for negative term (unused in basic CCL but
                        allows future weighting strategies).
        """
        super().__init__()
        self.margin = margin
        self.neg_weight = neg_weight

    def forward(
        self,
        pos_scores: torch.Tensor,   # (B,)        cosine sim for positive pairs
        neg_scores: torch.Tensor,   # (B, K)      cosine sim for K negatives per user
    ) -> torch.Tensor:
        """
        Args:
            pos_scores: shape (B,)   – scores for B positive (user, item+) pairs
            neg_scores: shape (B, K) – scores for K negatives per user
        Returns:
            scalar loss
        """
        # Positive term: -s(u, i+)
        pos_loss = -pos_scores  # (B,)

        # Negative term: mean of hinge per user
        hinge = torch.clamp(neg_scores - self.margin, min=0.0)  # (B, K)
        neg_loss = hinge.mean(dim=-1)                            # (B,)

        loss = (pos_loss + neg_loss).mean()
        return loss
