"""
gating_network.py — v2.2 (User-Context Gating, 7-feature)
─────────────────────────────────────────────────────────────────────
GatingMLP v2.2: nhận USER-CONTEXT features (dim=7).

Thay đổi so với v2.1:
  - Thêm agree_gcn_sem  → Spearman rank corr gcn vs sem, mapped [0,1]
                          Đo sự đồng thuận trực tiếp giữa 2 non-seq experts.
                          Nếu cao: cả gcn và sem cùng hướng → tăng tin cậy cả hai.
                          Nếu thấp: 2 experts mâu thuẫn → seq làm trọng tài.
  - Thêm seq_confidence → max(seq_scores) - mean(seq_scores)  ∈ [0,1]
                          Đo mức độ tự tin của SASRec expert (đối xứng với gcn/sem).

Context features (dim=7):
  0: norm_seq_len      — len_seq / max_seq_len               ∈ [0,1]
  1: agree_gcn         — Spearman rank corr seq vs gcn        ∈ [0,1]
  2: agree_sem         — Spearman rank corr seq vs sem        ∈ [0,1]
  3: agree_gcn_sem     — Spearman rank corr gcn vs sem        ∈ [0,1]  ← NEW
  4: seq_confidence    — max(seq_scores) - mean(seq_scores)   ∈ [0,1]  ← NEW
  5: gcn_confidence    — max(gcn_scores) - mean(gcn_scores)   ∈ [0,1]
  6: sem_confidence    — max(sem_scores) - mean(sem_scores)   ∈ [0,1]

Lý do thiết kế:
  - norm_seq_len         → phân biệt cold vs warm user
  - agree_gcn/sem        → consensus của từng non-seq expert với seq
  - agree_gcn_sem        → consensus trực tiếp giữa gcn và sem (góc nhìn bổ sung)
  - seq/gcn/sem_confidence → mỗi expert tự đánh giá mức độ "chắc chắn" của mình
  - Cả 7 features độc lập, không có circular dependency

NOTE (Double Normalization Fix v2.2):
  semantic_scorer._embed_and_score() đã được sửa để trả về raw cosine scores
  thay vì scores đã minmax. moe_fusion._minmax() normalize tập trung cho cả
  3 experts (seq, gcn, sem) trước khi truyền vào extract_context_features().
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from config import GatingConfig, DEFAULT_CONFIG


class GatingMLP(nn.Module):
    """
    MLP gating network.
    Input : user-context features (dim=5 mặc định)
    Output: softmax gate weights [g_seq, g_gcn, g_sem]
    """

    def __init__(self, cfg: GatingConfig = None):
        super().__init__()
        cfg = cfg or DEFAULT_CONFIG.gating

        layers = []
        in_dim = cfg.input_dim          # phải là 5 trong v2.1
        for h_dim in cfg.hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),    # stabilise gradients
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 3))   # → [logit_seq, logit_gcn, logit_sem]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)      # (B, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation giữa 2 score vectors."""
    if len(a) < 3:
        return 0.0
    from scipy.stats import spearmanr
    corr, _ = spearmanr(a, b)
    return float(corr) if not np.isnan(corr) else 0.0


def _expert_confidence(scores: Dict[str, float]) -> float:
    """
    Đo mức độ tự tin của expert: max(scores) - mean(scores).
    - Cao  → expert tập trung vào ít item → tự tin
    - Thấp → expert cho điểm đều → không chắc nên tin ai

    Scores đã được minmax-normalize vào [0, 1] trước khi gọi hàm này,
    nên kết quả tự nhiên nằm trong [0, 1].
    """
    if not scores:
        return 0.0
    vals = np.array(list(scores.values()), dtype=np.float32)
    return float(vals.max() - vals.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Context feature extractor  (public API)
# ─────────────────────────────────────────────────────────────────────────────

def extract_context_features(
    seq:        List[int],
    len_seq:    int,
    seq_scores: Dict[str, float],       # minmax-normalized SASRec scores
    gcn_scores: Dict[str, float],       # minmax-normalized GCN scores
    sem_scores: Dict[str, float],       # minmax-normalized Semantic scores
    gcn_norm:   Optional[torch.Tensor], # GCN embedding matrix (unused, kept for compat)
    cfg:        GatingConfig,
) -> List[float]:
    """
    Trích xuất 7 user-context features cho gating network v2.2.

    Args:
        seq        : padded sequence (list of item IDs)
        len_seq    : actual sequence length
        seq_scores : {item_name: score} từ SASRec (đã minmax-normalize bởi moe_fusion)
        gcn_scores : {item_name: score} từ GCN    (đã minmax-normalize bởi moe_fusion)
        sem_scores : {item_name: score} từ Semantic (đã minmax-normalize bởi moe_fusion)
        gcn_norm   : GCN embedding matrix — không dùng trong v2.2, giữ để backward-compat
        cfg        : GatingConfig

    Returns:
        List[float] độ dài 7  (= cfg.input_dim)

    Feature layout:
        [0] norm_seq_len   — user warmth
        [1] agree_gcn      — Spearman(seq, gcn) → [0,1]
        [2] agree_sem      — Spearman(seq, sem) → [0,1]
        [3] agree_gcn_sem  — Spearman(gcn, sem) → [0,1]  ← v2.2 NEW
        [4] seq_confidence — max-mean of seq scores       ← v2.2 NEW
        [5] gcn_confidence — max-mean of gcn scores
        [6] sem_confidence — max-mean of sem scores
    """
    max_seq = getattr(cfg, 'max_seq_len', 50)

    # ── Feature 0: norm_seq_len ──────────────────────────────────────────────
    norm_seq_len = min(len_seq / max(max_seq, 1), 1.0)

    # ── Feature 1: agree_gcn  (Spearman: seq vs gcn) ─────────────────────────
    agree_gcn = 0.0
    if seq_scores and gcn_scores:
        common = set(seq_scores) & set(gcn_scores)
        if len(common) >= 3:
            s_seq = np.array([seq_scores[n] for n in common], dtype=np.float32)
            s_gcn = np.array([gcn_scores[n] for n in common], dtype=np.float32)
            agree_gcn = (_spearman_corr(s_seq, s_gcn) + 1.0) / 2.0  # [-1,1] → [0,1]

    # ── Feature 2: agree_sem  (Spearman: seq vs sem) ─────────────────────────
    agree_sem = 0.0
    if seq_scores and sem_scores:
        common = set(seq_scores) & set(sem_scores)
        if len(common) >= 3:
            s_seq = np.array([seq_scores[n] for n in common], dtype=np.float32)
            s_sem = np.array([sem_scores[n] for n in common], dtype=np.float32)
            agree_sem = (_spearman_corr(s_seq, s_sem) + 1.0) / 2.0

    # ── Feature 3: agree_gcn_sem  (Spearman: gcn vs sem) ─────────────────────
    # Đo sự đồng thuận trực tiếp giữa 2 non-seq experts.
    # Nếu cao → cả gcn và sem cùng hướng → gating có thể tăng cả hai.
    # Nếu thấp → 2 experts mâu thuẫn → seq làm trọng tài tự nhiên.
    agree_gcn_sem = 0.0
    if gcn_scores and sem_scores:
        common = set(gcn_scores) & set(sem_scores)
        if len(common) >= 3:
            s_gcn = np.array([gcn_scores[n] for n in common], dtype=np.float32)
            s_sem = np.array([sem_scores[n] for n in common], dtype=np.float32)
            agree_gcn_sem = (_spearman_corr(s_gcn, s_sem) + 1.0) / 2.0

    # ── Feature 4: seq_confidence ─────────────────────────────────────────────
    # Đo mức độ tự tin của SASRec: expert tập trung vào ít item → confidence cao.
    seq_confidence = _expert_confidence(seq_scores)

    # ── Feature 5: gcn_confidence ────────────────────────────────────────────
    gcn_confidence = _expert_confidence(gcn_scores)

    # ── Feature 6: sem_confidence ────────────────────────────────────────────
    sem_confidence = _expert_confidence(sem_scores)

    return [
        norm_seq_len,   # 0
        agree_gcn,      # 1
        agree_sem,      # 2
        agree_gcn_sem,  # 3 ← NEW
        seq_confidence, # 4 ← NEW
        gcn_confidence, # 5
        sem_confidence, # 6
    ]


# ─────────────────────────────────────────────────────────────────────────────
# GatingNetwork wrapper  (high-level API)
# ─────────────────────────────────────────────────────────────────────────────

class GatingNetwork:
    """
    High-level wrapper: load / predict / save GatingMLP.

    Inference (preferred):
        weights = gating.predict_from_context(context_features)
        # → (g_seq, g_gcn, g_sem)

    Legacy backward-compat:
        weights = gating.predict(signal_scores, len_seq)
    """

    def __init__(
        self,
        cfg:        GatingConfig  = None,
        model_path: str           = None,
        device:     torch.device  = None,
    ):
        self.cfg    = cfg or DEFAULT_CONFIG.gating
        self.device = device or torch.device('cpu')
        self.model  = GatingMLP(self.cfg).to(self.device)
        self.trained = False

        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std:  Optional[np.ndarray] = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            print(f'[GatingNetwork] No checkpoint — using default weights {self.cfg.default_weights}')

    # ── Prediction (context-mode) ─────────────────────────────────────────────

    def predict_from_context(
        self,
        context_features: List[float],
    ) -> Tuple[float, float, float]:
        """
        Predict gate weights từ user-context feature vector (5-dim).

        Returns:
            (g_seq, g_gcn, g_sem) — sum = 1
        """
        feat = np.array(context_features, dtype=np.float32)

        if not self.trained:
            dw = self.cfg.default_weights
            return float(dw[0]), float(dw[1]), float(dw[2])

        if self.feat_mean is not None and self.feat_std is not None:
            feat = (feat - self.feat_mean) / (self.feat_std + 1e-8)

        x = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            w = self.model(x).cpu().numpy()[0]

        return float(w[0]), float(w[1]), float(w[2])

    # ── Legacy backward-compat predict ───────────────────────────────────────

    def predict(
        self,
        signal_scores: Dict[str, Dict[str, float]],
        len_seq: int = 0,
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Backward-compatible interface dùng cho moe_fusion.py cũ.
        Context-mode: tất cả items nhận cùng gate (user-level).
        """
        if not signal_scores:
            return {}

        items = list(signal_scores.keys())

        if not self.trained:
            if len_seq <= self.cfg.cold_threshold:
                dw = [0.15, 0.35, 0.50]   # cold: trust semantic more
            else:
                dw = self.cfg.default_weights
            return {it: (float(dw[0]), float(dw[1]), float(dw[2])) for it in items}

        # Context-mode đúng: gọi predict_from_context với features đầy đủ
        # Nếu không có context → fallback default
        dw = self.cfg.default_weights
        return {it: (float(dw[0]), float(dw[1]), float(dw[2])) for it in items}

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'cfg':              self.cfg,
            'norm_mean':        self.feat_mean.tolist() if self.feat_mean is not None else None,
            'norm_std':         self.feat_std.tolist()  if self.feat_std  is not None else None,
            'gating_mode':      getattr(self.cfg, 'gating_mode', 'context'),
            'feature_version':  'v2.2',
            'feature_names':    [
                'norm_seq_len',
                'agree_gcn',
                'agree_sem',
                'agree_gcn_sem',   # v2.2 NEW
                'seq_confidence',  # v2.2 NEW
                'gcn_confidence',
                'sem_confidence',
            ],
        }, path)
        print(f'[GatingNetwork] Saved → {path}')

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if 'cfg' in ckpt:
            self.cfg   = ckpt['cfg']
            self.model = GatingMLP(self.cfg).to(self.device)
            mode    = getattr(self.cfg, 'gating_mode', 'context')
            fv      = ckpt.get('feature_version', 'v2.0')
            fnames  = ckpt.get('feature_names', [])
            print(f'[GatingNetwork] Loaded: mode={mode}, input_dim={self.cfg.input_dim}, '
                  f'feature_version={fv}')
            if fnames:
                print(f'[GatingNetwork] Features: {fnames}')

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.trained = True

        if ckpt.get('norm_mean') is not None:
            self.feat_mean = np.array(ckpt['norm_mean'], dtype=np.float32)
            self.feat_std  = np.array(ckpt['norm_std'],  dtype=np.float32)
            print(f'[GatingNetwork] Norm params: mean={self.feat_mean.round(3)}')
        elif 'norm_min' in ckpt:
            # Legacy checkpoint format
            self.feat_mean = np.array(ckpt['norm_min'], dtype=np.float32)
            self.feat_std  = np.array(ckpt['norm_max'], dtype=np.float32)
            print('[GatingNetwork] Legacy norm params loaded.')