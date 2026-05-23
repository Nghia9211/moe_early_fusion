import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    He et al., SIGIR 2020. https://arxiv.org/abs/2002.02126

    Paper specs:
    - Chỉ dùng neighborhood aggregation, KHÔNG có feature transformation hay nonlinear activation.
    - Trainable parameters: chỉ có embedding E0 của users và items (tách biệt).
    - Layer combination: E_final = (1 / (K+1)) * sum(E0, E1, ..., EK) — trung bình đều các layer.
    - Adjacency matrix: D^{-1/2} A D^{-1/2}, KHÔNG có self-loop.
    - Số layer K: paper thực nghiệm với K=1,2,3,4; mặc định K=3.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        # Paper: "The only trainable model parameter is the 0-th layer embedding E^(0)"
        # Tách riêng user embedding và item embedding như trong paper
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Paper: initializer — dùng normal distribution, std=0.1
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, A: torch.Tensor):
        """
        Args:
            A: Sparse normalized adjacency matrix D^{-1/2} A D^{-1/2},
               shape (num_users + num_items, num_users + num_items).
               KHÔNG có self-loop — đây là đặc điểm của LightGCN.

        Returns:
            E_final: Final embeddings sau layer combination, shape (num_users+num_items, d).
            E0: Initial embeddings (dùng để tính L2 regularization trong BPR loss).
        """
        # Ghép user và item embedding thành ma trận E0 chung — đúng theo paper (Equation 5)
        E0 = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        # Light Graph Convolution qua K layers — không có W, không có activation (Equation 3)
        # E^(k+1) = D^{-1/2} A D^{-1/2} E^(k)
        all_embeddings = [E0]
        E_k = E0
        for _ in range(self.num_layers):
            E_k = torch.sparse.mm(A, E_k)
            all_embeddings.append(E_k)

        # Layer combination: E_final = (1 / (K+1)) * (E0 + E1 + ... + EK) — Equation 11
        # Paper: alpha_k = 1/(K+1) cho tất cả layers (equal weight)
        E_final = torch.stack(all_embeddings, dim=1).mean(dim=1)

        return E_final, E0