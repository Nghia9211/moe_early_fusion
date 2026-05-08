"""
moe_fusion.py — v3.0 (Runtime Expert Agreement Correction)
────────────────────────────────────────────────────────────
v3.0 changes:
  - Added runtime expert agreement check after gating prediction.
  - If an expert's ranking DISAGREES with the consensus of the other experts
    (negative or very low Spearman correlation), its gate weight is dynamically
    suppressed for that specific user.
  - This fixes the fundamental limitation of the MLP gating: it produces
    USER-LEVEL weights based on context features, but can't adapt to each
    user's specific candidate set. The agreement check adds per-candidate-set
    intelligence.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr

from config          import MoEConfig, DEFAULT_CONFIG
from gating_network  import GatingNetwork, extract_context_features


def _minmax(score_dict: Dict[str, float]) -> Dict[str, float]:
    if not score_dict: return {}
    vals = np.array(list(score_dict.values()))
    lo, hi = np.min(vals), np.max(vals)
    if hi == lo: return {k: 0.5 for k in score_dict}
    return {k: float((v-lo)/(hi-lo)) for k, v in score_dict.items()}


def _extract_signal(signal_scores: Dict[str, Dict[str, float]], key: str) -> Dict[str, float]:
    return {name: s.get(key, 0.0) for name, s in signal_scores.items()}


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Safe Spearman rank correlation. Returns 0.0 if insufficient data."""
    if len(a) < 3:
        return 0.0
    if np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    corr, _ = spearmanr(a, b)
    return float(corr) if not np.isnan(corr) else 0.0


def _apply_agreement_correction(
    g_seq: float, g_gcn: float, g_sem: float,
    norm_seq: Dict[str, float],
    norm_gcn: Dict[str, float],
    norm_sem: Dict[str, float],
    min_agreement: float = 0.15,
) -> Tuple[float, float, float, dict]:
    """
    Runtime expert agreement check.

    For each expert, compute how well its ranking agrees with the consensus
    of the OTHER two experts. If agreement < min_agreement, suppress the gate.

    This catches cases where the MLP gating gives weight to an expert that
    is actively harmful for a specific user's candidate set.

    Args:
        min_agreement: minimum Spearman correlation with consensus to keep
                       full gate weight. Below this, gate is scaled down.
                       Set to 0.0 to disable.

    Returns:
        (g_seq, g_gcn, g_sem, correction_info)
    """
    if min_agreement <= 0:
        return g_seq, g_gcn, g_sem, {}

    # Get common items across all experts
    common = set(norm_seq.keys()) & set(norm_gcn.keys()) & set(norm_sem.keys())
    if len(common) < 5:
        return g_seq, g_gcn, g_sem, {'skipped': 'too_few_common_items'}

    items = sorted(common)
    s_arr = np.array([norm_seq[n] for n in items])
    g_arr = np.array([norm_gcn[n] for n in items])
    m_arr = np.array([norm_sem[n] for n in items])

    # Consensus of each pair (for evaluating the third expert)
    consensus_no_seq = (g_arr + m_arr) / 2.0
    consensus_no_gcn = (s_arr + m_arr) / 2.0
    consensus_no_sem = (s_arr + g_arr) / 2.0

    corr_seq = _rank_corr(s_arr, consensus_no_seq)
    corr_gcn = _rank_corr(g_arr, consensus_no_gcn)
    corr_sem = _rank_corr(m_arr, consensus_no_sem)

    # Apply suppression: scale gate by max(corr, 0) / min_agreement
    # If corr < 0, gate → 0 (expert actively disagrees)
    # If 0 < corr < min_agreement, gate is proportionally reduced
    # If corr >= min_agreement, gate stays unchanged
    def _scale(gate: float, corr: float) -> float:
        if corr < 0:
            return gate * 0.0  # Expert disagrees with consensus → kill
        elif corr < min_agreement:
            return gate * (corr / min_agreement)  # Proportional reduction
        return gate

    g_seq_new = _scale(g_seq, corr_seq)
    g_gcn_new = _scale(g_gcn, corr_gcn)
    g_sem_new = _scale(g_sem, corr_sem)

    info = {
        'corr_seq': round(corr_seq, 3),
        'corr_gcn': round(corr_gcn, 3),
        'corr_sem': round(corr_sem, 3),
        'gate_pre_correction':  {'seq': round(g_seq, 3), 'gcn': round(g_gcn, 3), 'sem': round(g_sem, 3)},
        'gate_post_correction': {'seq': round(g_seq_new, 3), 'gcn': round(g_gcn_new, 3), 'sem': round(g_sem_new, 3)},
    }
    return g_seq_new, g_gcn_new, g_sem_new, info


class MoEFusion:
    def __init__(self, gating: GatingNetwork, cfg: MoEConfig = None,
                 gcn_norm=None):
        self.gating   = gating
        self.cfg      = cfg or DEFAULT_CONFIG
        self.gcn_norm = gcn_norm   # GCN embedding matrix — cần cho coverage feature

    def fuse(
        self,
        signal_scores: Dict[str, Dict[str, float]],
        len_seq:  int         = 0,
        seq_ids:  List[int]   = None,   # padded sequence IDs
        top_m:    int         = None,
        debug:    bool        = False,
    ) -> Tuple[List[str], Dict[str, float], dict]:

        top_m = top_m or self.cfg.retrieval.top_M
        if not signal_scores: return [], {}, {}

        raw_seq = _extract_signal(signal_scores, 'seq')
        raw_gcn = _extract_signal(signal_scores, 'gcn')
        raw_sem = _extract_signal(signal_scores, 'sem')

        norm_seq = _minmax(raw_seq)
        norm_gcn = _minmax(raw_gcn)
        norm_sem = _minmax(raw_sem)

        # ── Gate: context-mode ───────────────────────────────────────────────
        gating_cfg = self.gating.cfg
        mode = getattr(gating_cfg, 'gating_mode', 'context')

        if mode == 'context' and self.gating.trained:
            ctx = extract_context_features(
                seq       = seq_ids or [],
                len_seq   = len_seq,
                seq_scores= norm_seq,
                gcn_scores= norm_gcn,
                sem_scores= norm_sem,
                gcn_norm  = self.gcn_norm,
                cfg       = gating_cfg,
            )
            g_seq, g_gcn, g_sem = self.gating.predict_from_context(ctx)
        else:
            # Fallback: legacy score-based hoặc chưa train
            gate_weights = self.gating.predict(signal_scores, len_seq=len_seq)
            first = next(iter(gate_weights.values()), (1/3, 1/3, 1/3))
            g_seq, g_gcn, g_sem = first

        # Disable unused experts
        if not self.cfg.use_seq:      g_seq = 0.0
        if not self.cfg.use_gcn:      g_gcn = 0.0
        if not self.cfg.use_semantic: g_sem = 0.0

        # ── Runtime Expert Agreement Correction (v3.0) ───────────────────────
        # Dynamically suppress experts that disagree with the consensus
        # of the other experts for THIS user's candidate set.
        agreement_info = {}
        min_agree = getattr(self.cfg.gating, 'min_agreement_corr', 0.15)
        if min_agree > 0:
            g_seq, g_gcn, g_sem, agreement_info = _apply_agreement_correction(
                g_seq, g_gcn, g_sem,
                norm_seq, norm_gcn, norm_sem,
                min_agreement=min_agree,
            )

        total_g = g_seq + g_gcn + g_sem
        if total_g > 0:
            g_seq, g_gcn, g_sem = g_seq/total_g, g_gcn/total_g, g_sem/total_g

        # ── Fuse scores ──────────────────────────────────────────────────────
        fused_scores: Dict[str, float] = {
            name: g_seq * norm_seq.get(name, 0.0)
                + g_gcn * norm_gcn.get(name, 0.0)
                + g_sem * norm_sem.get(name, 0.0)
            for name in signal_scores
        }

        c_m = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_m]

        debug_info = {}
        if debug:
            debug_info = {
                'avg_gates':      {'seq': round(g_seq,3), 'gcn': round(g_gcn,3), 'sem': round(g_sem,3)},
                'gating_mode':    mode,
                'n_candidates':   len(signal_scores),
                'top_m':          top_m,
                'top_item':       c_m[0] if c_m else None,
                'top_score':      round(fused_scores.get(c_m[0], 0), 4) if c_m else 0,
                'trained_gating': self.gating.trained,
                'len_seq':        len_seq,
                'agreement_correction': agreement_info,
            }

        return c_m, fused_scores, debug_info

    def fuse_from_data(
        self,
        data:          dict,
        signal_scores: Dict[str, Dict[str, float]],
        top_m:         int  = None,
        debug:         bool = True,
    ) -> Tuple[List[str], Dict[str, float], dict]:

        len_seq = data.get('len_seq', 0)
        if isinstance(len_seq, (list, tuple)): len_seq = len_seq[0]
        try: len_seq = int(len_seq)
        except: len_seq = 0

        # Lấy padded seq IDs để tính gcn_coverage
        seq_ids = None
        raw_seq = data.get('seq', [])
        if hasattr(raw_seq, 'tolist'): seq_ids = raw_seq.tolist()
        elif isinstance(raw_seq, list): seq_ids = raw_seq

        return self.fuse(
            signal_scores = signal_scores,
            len_seq       = len_seq,
            seq_ids       = seq_ids,
            top_m         = top_m or self.cfg.retrieval.top_M,
            debug         = debug,
        )