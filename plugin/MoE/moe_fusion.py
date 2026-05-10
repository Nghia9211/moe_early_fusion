"""
moe_fusion.py
────────────────────────────────────────────────────────────
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

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