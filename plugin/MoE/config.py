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
      'context'  — input là user-context features (v2, mặc định mới)
      'score'    — input là expert scores của item (v1 cũ, bị bias)
    
    Context features (khi mode='context', input_dim=6):
      0: norm_seq_len         — len_seq / max_seq_len
      1: cold_start_flag      — 1 nếu len_seq <= cold_threshold
      2: gcn_coverage         — tỉ lệ items trong history có GCN embedding
      3: seq_score_entropy    — entropy phân phối điểm SASRec trên candidates
      4: expert_agreement_gcn — Spearman rank correlation seq vs gcn
      5: expert_agreement_sem — Spearman rank correlation seq vs sem
    """
    input_dim:    int   = 6         # 6 context features
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
    
    # Entropy regularization — buộc gates phân tán
    entropy_reg_weight: float = 0.15
    
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

    use_seq:      bool = False
    use_gcn:      bool = False
    use_semantic: bool = True
    use_reranker: bool = False


DEFAULT_CONFIG = MoEConfig()


def get_config_for_dataset() -> MoEConfig:
    """
    Trả về MoEConfig được tinh chỉnh cho từng dataset.

    Amazon:    GCN tốt → balanced
    Yelp:      SASRec tốt → seq heavy
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