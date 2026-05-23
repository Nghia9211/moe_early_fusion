"""
moe_fusion/score_combiner.py
─────────────────────────────
"""

import numpy as np
import math
from typing import Dict, List, Tuple

from config import ScoreConfig, DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sasrec_confidence(fused_scores: Dict[str, float]) -> float:
    """
    Tính độ tự tin của MoE dựa trên khoảng cách điểm giữa Top 1 và Top 2.
    Lưu ý: Tính trên điểm gốc (chưa đổi qua rank) để đánh giá độ chênh lệch.
    """
    if len(fused_scores) < 2:
        return 1.0
    sorted_vals = sorted(fused_scores.values(), reverse=True)
    top1, top2  = sorted_vals[0], sorted_vals[1]
    score_range = sorted_vals[0] - sorted_vals[-1]
    if score_range == 0:
        return 0.5
    margin = (top1 - top2) / score_range
    return float(np.clip(margin, 0.0, 1.0))

def _scores_to_rank_scores(scores_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Chuyển đổi điểm s0 (MoE thô) thành Rank Score.
    Công thức: 1.0 / log2(rank + 2) để cùng dải tỉ lệ với điểm của Reranker.
    """
    if not scores_dict:
        return {}
    
    # Sắp xếp các item theo điểm giảm dần
    sorted_items = sorted(scores_dict, key=scores_dict.get, reverse=True)
    
    # Gán rank score
    rank_scores = {}
    for rank, item in enumerate(sorted_items):
        rank_scores[item] = 1.0 / math.log2(rank + 2)
        
    return rank_scores


def moe_confidence_score(fused_scores: Dict[str, float]) -> float:
    """
    Đo độ tự tin của MoE dựa trên phân phối fused scores.
    Trả về giá trị trong [0, 1]. Cao hơn = MoE tự tin hơn.

    Dùng 2 signals:
      1. Margin: khoảng cách top-1 vs top-2 (normalized by range)
      2. Concentration: tỷ lệ score mass trong top-3 vs toàn bộ
    """
    if len(fused_scores) < 2:
        return 1.0

    sorted_vals = sorted(fused_scores.values(), reverse=True)
    score_range = sorted_vals[0] - sorted_vals[-1]

    if score_range == 0:
        return 0.0  # Tất cả bằng nhau → không confidence

    # Signal 1: Margin giữa top-1 và top-2
    margin = (sorted_vals[0] - sorted_vals[1]) / score_range

    # Signal 2: Concentration trong top-3 so với tổng
    top3_mass = sum(sorted_vals[:min(3, len(sorted_vals))])
    total_mass = sum(sorted_vals)
    concentration = top3_mass / total_mass if total_mass > 0 else 0.5

    # Combined confidence: weighted average
    confidence = 0.6 * margin + 0.4 * concentration
    return float(np.clip(confidence, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# ScoreCombiner
# ─────────────────────────────────────────────────────────────────────────────

class ScoreCombiner:
    # Điểm mấu chốt: Giảm Alpha ở các vòng feedback để trao quyền cho LLM
    EPOCH_ALPHA_BOOST: float = 0.0 

    def __init__(self, cfg: ScoreConfig = None):
        self.cfg = cfg or DEFAULT_CONFIG.scoring

    # ─────────────────────────────────────────────────────────────────────
    # Alpha selection
    # ─────────────────────────────────────────────────────────────────────
    def get_alpha(
        self,
        dataset:       str,
        len_seq:       int,
        fused_scores:  Dict[str, float] = None,
        epoch:         int = 1,
    ) -> float:
        if not self.cfg.use_adaptive_alpha:
            # Non-adaptive (Vẫn áp dụng epoch boost cho Feedback Loop)
            base = self.cfg.alpha
            epoch_boost = (epoch - 1) * self.EPOCH_ALPHA_BOOST
            return float(np.clip(base + epoch_boost, 0.1, 0.9))

        # ── Base alpha theo dataset ───────────────────────────────────────
        alpha = self.cfg.dataset_alpha.get(dataset, self.cfg.alpha)

        # ── Epoch boost (feedback loop fix) ──────────────────────────────
        epoch_boost = (epoch - 1) * self.EPOCH_ALPHA_BOOST
        alpha      += epoch_boost

        final_alpha = float(np.clip(alpha, 0.1, 0.9))
        if epoch > 1:
            print(f"[ScoreCombiner] epoch={epoch} alpha_boost={epoch_boost:.2f} "
                  f"final_alpha={final_alpha:.3f}")
        return final_alpha

    # ─────────────────────────────────────────────────────────────────────
    # Core combine
    # ─────────────────────────────────────────────────────────────────────

    def combine(
        self,
        fused_scores:   Dict[str, float],
        rerank_scores:  Dict[str, float],
        dataset:        str  = 'amazon',
        len_seq:        int  = 0,
        top_k:          int  = None,
        epoch:          int  = 1,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        top_k = top_k or DEFAULT_CONFIG.retrieval.top_K

        if not fused_scores:
            c_k = sorted(rerank_scores, key=rerank_scores.get, reverse=True)[:top_k]
            return c_k, rerank_scores, {'alpha': 0.0, 'fallback': True}

        # ── TÍNH ALPHA TRƯỚC KHI CHUYỂN QUA RANK ────────────────────────
        # Để hàm tự tin (_sasrec_confidence) dùng điểm s0 thô đánh giá độ cách biệt
        alpha = self.get_alpha(dataset, len_seq, fused_scores, epoch=epoch)
        beta  = 1.0 - alpha

        # ── CHUYỂN s0 THÀNH RANK SCORES (Bỏ Z-Score) ────────────────────
        rank_s0 = _scores_to_rank_scores(fused_scores)

        # ── s1 = alpha * rank_s0 + beta * s_rerank ───────────────────────
        all_items = set(rank_s0.keys()) | set(rerank_scores.keys())
        s1_scores: Dict[str, float] = {}

        for item in all_items:
            s0_val    = rank_s0.get(item, 0.0)
            s_rer_val = rerank_scores.get(item, 0.0)
            s1_scores[item] = alpha * s0_val + beta * s_rer_val

        # ── C_K = TopK(s1) ────────────────────────────────────────────────
        c_k = sorted(s1_scores, key=s1_scores.get, reverse=True)[:top_k]

        debug_info = {
            'alpha':   round(alpha, 3),
            'beta':    round(beta, 3),
            'dataset': dataset,
            'len_seq': len_seq,
            'epoch':   epoch,
            'top_k':   top_k,
            'n_items': len(all_items),
            'top_item': c_k[0] if c_k else None,
        }
        
        return c_k, s1_scores, debug_info

    def combine_from_pipeline(
        self,
        fused_scores:  Dict[str, float],
        rerank_scores: Dict[str, float],
        data:          dict,
        args,
        top_k:         int = None,
        epoch:         int = 1,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        # Ưu tiên data['dataset'] (inject bởi moe_rec_agent.py) để nhất quán
        # với reranker.py. Fallback về args.data_dir nếu chưa có.
        dataset = (data.get('dataset') or
                   next((d for d in ['yelp', 'amazon', 'goodreads']
                         if d in getattr(args, 'data_dir', '')),
                        'amazon'))
        len_seq = data.get('len_seq', 0)
        top_k   = top_k or DEFAULT_CONFIG.retrieval.top_K

        return self.combine(
            fused_scores  = fused_scores,
            rerank_scores = rerank_scores,
            dataset       = dataset,
            len_seq       = len_seq,
            top_k         = top_k,
            epoch         = epoch,
        )