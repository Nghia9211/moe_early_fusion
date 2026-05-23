"""
moe_fusion/config.py
────────────────────
Hyperparameters của MoE pipeline.

Goodreads update:
  - top_sem=35, top_seq=5, top_gcn=5 (đảo ngược so với Amazon)
  - default_weights=[0.1, 0.1, 0.8] cho Goodreads (FAISS dẫn dắt)
  - alpha thấp hơn cho Goodreads (tin reranker hơn s0 khi s0 yếu)
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RetrievalConfig:
    """Số lượng candidates lấy từ mỗi nguồn trước khi union."""
    top_seq: int = 20
    top_gcn: int = 20
    top_sem: int = 20
    top_M:   int = 20
    top_K:   int = 5


@dataclass
class GatingConfig:
    """Config cho MLP gating network.
    
    gating_mode:
      'context'  — input là user-context features (v2.2, mặc định mới)
      'score'    — input là expert scores của item (v1 cũ, bị bias)
    
    Context features (khi mode='context', input_dim=7):
      0: norm_seq_len         — len_seq / max_seq_len
      1: agree_gcn            — Spearman rank corr seq vs gcn, mapped [0,1]
      2: agree_sem            — Spearman rank corr seq vs sem, mapped [0,1]
      3: agree_gcn_sem        — Spearman rank corr gcn vs sem, mapped [0,1]  ← v2.2
      4: seq_confidence       — max - mean của SASRec scores  ∈ [0,1]        ← v2.2
      5: gcn_confidence       — max - mean của GCN scores     ∈ [0,1]
      6: sem_confidence       — max - mean của Sem scores     ∈ [0,1]

    NOTE: semantic_scorer trả về raw cosine scores (không minmax nội bộ).
          moe_fusion._minmax() normalize tập trung cho cả 3 experts.
    """
    input_dim:    int   = 7         # 7 context features (v2.2)
    hidden_dims:  list  = field(default_factory=lambda: [32, 16])
    dropout:      float = 0.2
    lr:           float = 1e-3
    epochs:       int   = 50
    batch_size:   int   = 256
    weight_decay: float = 1e-4

    default_weights: list = field(
        default_factory=lambda: [1/3, 1/3, 1/3]
    )
    
    # Gating mode
    gating_mode:   str   = 'context'   # 'context' | 'score'
    # Entropy regularization — buộc gates phân tán (DEPRECATED, dùng concentration_weight thay thế)
    entropy_reg_weight: float = 0.05
    # Context feature thresholds
    max_seq_len:       int   = 50
    cold_threshold:    int   = 5
    
    # Legacy compat
    use_seq_len_in_gating: bool = False


@dataclass
class ScoreConfig:
    """
    s1(u,i) = alpha * s0(u,i) + (1 - alpha) * s_rerank(u,i)
    """
    alpha: float = 0.5

    dataset_alpha: Dict[str, float] = field(default_factory=lambda: {
        'yelp':      0.5,   
        'amazon':    0.5,
        'goodreads': 0.5,  
    })
    use_adaptive_alpha: bool = False


@dataclass
class MoEConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    gating:    GatingConfig    = field(default_factory=GatingConfig)
    scoring:   ScoreConfig     = field(default_factory=ScoreConfig)

    gating_model_path: str = None

    use_seq:        bool = True
    use_gcn:        bool = True
    use_semantic:   bool = True
    use_reranker:   bool = True
    use_user_agent: bool = True

DEFAULT_CONFIG = MoEConfig()


def get_config_for_dataset(dataset: str = None) -> MoEConfig:
    """
    Trả về MoEConfig được tinh chỉnh cho từng dataset.

    Amazon:    GCN tốt → balanced
    Yelp:      SASRec tốt → seq heavy, Semantic yếu → suppress
    Goodreads: cả GCN và SASRec fail vì sparse →
               FAISS dẫn dắt hoàn toàn
    """
    cfg = MoEConfig()

    cfg.retrieval.top_seq = 20
    cfg.retrieval.top_gcn = 20
    cfg.retrieval.top_sem = 20
    cfg.retrieval.top_M   = 20
    cfg.retrieval.top_K   = 5
    cfg.gating.default_weights = [1/3, 1/3, 1/3]

    return cfg