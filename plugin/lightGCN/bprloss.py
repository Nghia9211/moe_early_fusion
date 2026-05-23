import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss theo LightGCN paper (Equation 15):

        L_BPR = -sum[ log sigma(y_ui - y_uj) ] + lambda * ||E^(0)||^2

    Trong đó:
    - y_ui = e_u^T * e_i  (inner product của final embeddings)
    - Regularization chỉ tính trên E^(0) (initial embeddings), KHÔNG phải final embeddings.
    - Paper dùng lambda = 1e-4 (reg_weight).
    """

    def __init__(self, reg_weight: float = 1e-4):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self,
                users, pos_items, neg_items,
                user_e_final, pos_e_final, neg_e_final,
                user_e0, pos_e0, neg_e0):
        """
        Args:
            users, pos_items, neg_items: index tensors (batch,)
            user_e_final, pos_e_final, neg_e_final: final embeddings dùng để tính score
            user_e0, pos_e0, neg_e0: initial embeddings E^(0) dùng để tính regularization

        Returns:
            total_loss, bpr_loss, reg_loss
        """
        # BPR score = inner product của final embeddings (Equation 14)
        pos_scores = (user_e_final * pos_e_final).sum(dim=1)
        neg_scores = (user_e_final * neg_e_final).sum(dim=1)

        # BPR loss: -mean[ log sigma(pos - neg) ] (Equation 15)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 Regularization: chỉ trên E^(0), chia cho batch size (Equation 15)
        # Paper: (lambda/2) * (||e_u^(0)||^2 + ||e_i^(0)||^2 + ||e_j^(0)||^2)
        reg_loss = (0.5 / users.shape[0]) * (
            user_e0.norm(2).pow(2) +
            pos_e0.norm(2).pow(2) +
            neg_e0.norm(2).pow(2)
        )

        total_loss = bpr_loss + self.reg_weight * reg_loss

        return total_loss, bpr_loss, reg_loss