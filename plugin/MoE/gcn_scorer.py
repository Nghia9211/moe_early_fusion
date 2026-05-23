"""
moe_fusion/gcn_scorer.py
────────────────────────
GCN collaborative scorer: tính s_gcn(u, i) dựa trên graph embeddings.

GCN embeddings được load sẵn trong ARAGRecAgent._shared_gcn_embeddings.

s_gcn(u, i) = cosine_similarity(h_user, h_item)
  với h_user = user embedding trực tiếp từ LightGCN node embeddings.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


class GCNScorer:
    """
    Tính GCN-based collaborative score cho từng candidate item.

    Flow:
      1. Lấy trực tiếp user embedding từ node_embs (learned bởi LightGCN)
      2. s_gcn(i) = cosine_sim(h_user, gcn_emb[i])

    Interface:
        scorer = GCNScorer(gcn_embeddings, node_embs, id2name, name2id)
        scores = scorer.score(user_id, candidate_ids)
        # → Dict[item_name, float]
    """

    def __init__(
        self,
        gcn_embeddings: torch.Tensor,       # shape: [num_items, embed_dim]
        node_embs:      Dict[str, torch.Tensor],  # { original_id_str: tensor(dim,) }
        id2name:        Dict[int, str],
        name2id:        Dict[str, int],
        device:         torch.device = None,
    ):
        self.device = device or torch.device("cpu")

        # Normalize item embeddings 1 lần duy nhất khi khởi tạo
        gcn_emb = gcn_embeddings.to(self.device).float()
        
        # Đánh dấu các cold items (zero vectors) trước khi normalize
        original_norms = gcn_emb.norm(dim=-1)
        self.is_zero_vector = (original_norms < 1e-6)
        
        norms = gcn_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.gcn_norm = gcn_emb / norms             # (num_items, dim)

        # node_embs: dict chứa cả user lẫn item embeddings từ LightGCN
        self.node_embs = node_embs

        self.id2name   = id2name
        self.name2id   = name2id
        self.num_items = gcn_emb.shape[0]

    # ─────────────────────────────────────────────────────────────────────
    # User representation
    # ─────────────────────────────────────────────────────────────────────

    def _user_embedding(self, user_id: str) -> Optional[torch.Tensor]:
        """
        Lấy trực tiếp user embedding từ LightGCN node_embs.

        Returns:
            Normalized user embedding (dim,) hoặc None nếu user không có trong graph (cold-start).
        """
        u_emb = self.node_embs.get(user_id)
        if u_emb is None:
            return None
        return F.normalize(u_emb.to(self.device).float(), dim=0)  # (dim,)

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def score(
        self,
        user_id:       str,
        candidate_ids: List[int],
    ) -> Dict[str, float]:
        """
        Tính GCN score cho từng candidate item.

        Args:
            user_id:       Original user ID string (key trong node_embs)
            candidate_ids: List integer item IDs cần rank

        Returns:
            Dict[item_name → cosine_score ∈ [-1, 1]]
            Trả về score 0.0 cho tất cả nếu user cold-start.
        """
        if not candidate_ids:
            return {}

        h_user = self._user_embedding(user_id)

        # Cold-start: user không có trong graph → uniform score
        if h_user is None:
            return {
                self.id2name.get(cid, f"item_{cid}"): 0.0
                for cid in candidate_ids
                if 0 <= cid < self.num_items
            }

        # Lọc valid candidate ids
        valid_cans = [
            (cid, self.id2name.get(cid, f"item_{cid}"))
            for cid in candidate_ids
            if 0 <= cid < self.num_items
        ]
        if not valid_cans:
            return {}

        # Batch cosine similarity — nhanh hơn vòng for
        can_ids  = torch.tensor([c[0] for c in valid_cans],
                                dtype=torch.long, device=self.device)
        can_embs = self.gcn_norm[can_ids]              # (n_cans, dim)

        # Dot product = cosine sim vì cả 2 đã normalize
        sims = (can_embs @ h_user)
        
        # Gán -1.0 cho các cold items (zero vectors) để chúng luôn nằm chót bảng
        zero_mask = self.is_zero_vector[can_ids]
        sims = sims.masked_fill(zero_mask, -1.0).cpu().tolist()

        return {
            name: float(sim)
            for (_, name), sim in zip(valid_cans, sims)
        }

    def top_k_names(
        self,
        user_id:       str,
        candidate_ids: List[int],
        k:             int = 20,
    ) -> List[str]:
        """Top-k item names theo GCN score. Dùng để build C_gcn."""
        scores = self.score(user_id, candidate_ids)
        return sorted(scores, key=scores.get, reverse=True)[:k]

    # ─────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_shared(cls, shared: dict) -> "GCNScorer":
        """
        Tạo GCNScorer từ dict shared resources.

        shared phải có keys:
          'gcn_embeddings' : torch.Tensor shape (num_items, dim)
          'node_embs'      : Dict[str, Tensor] — toàn bộ node embeddings từ LightGCN
          'id2name'        : Dict[int, str]
          'name2id'        : Dict[str, int]
          'device'         : torch.device (optional)
        """
        gcn_emb = shared.get('gcn_embeddings')
        if gcn_emb is None:
            raise ValueError(
                "[GCNScorer] 'gcn_embeddings' not found in shared resources. "
                "Ensure args.gcn_path is set và file tồn tại."
            )

        node_embs = shared.get('node_embs')
        if node_embs is None:
            import warnings
            warnings.warn(
                "[GCNScorer] 'node_embs' not found in shared resources — "
                "user embeddings sẽ fallback về cold-start (score 0.0). "
                "Để có user embedding đầy đủ, hãy dùng file được tạo bởi remap_gcn_embedding.py.",
                UserWarning, stacklevel=2
            )
            node_embs = {}

        return cls(
            gcn_embeddings=gcn_emb,
            node_embs=node_embs,
            id2name=shared['id2name'],
            name2id=shared['name2id'],
            device=shared.get('device', torch.device('cpu')),
        )
