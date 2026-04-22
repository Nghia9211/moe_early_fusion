"""
inspect_gating_model.py
────────────────────────
Chạy script này tại máy để chẩn đoán tại sao sem gate = 0.

Usage:
    python inspect_gating_model.py --ckpt ./saved_models/moellm/goodreads_gating_model.pt
"""

import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GatingMLP(nn.Module):
    def __init__(self, hidden_dims=(16, 8)):
        super().__init__()
        layers, in_dim = [], 4
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']

    # ── Detect architecture ──────────────────────────────────────────────────
    hidden_dims = []
    for k, v in state.items():
        if 'weight' in k and len(v.shape) == 2 and v.shape[1] == 4:
            hidden_dims.append(v.shape[0])
        elif 'weight' in k and len(v.shape) == 2 and v.shape[1] not in (4, 3):
            hidden_dims.append(v.shape[0])
    # fallback
    if not hidden_dims:
        hidden_dims = [16, 8]

    model = GatingMLP()
    try:
        model.load_state_dict(state)
    except Exception as e:
        print(f"Load state dict error: {e} — trying strict=False")
        model.load_state_dict(state, strict=False)
    model.eval()

    # ── Normalization params ─────────────────────────────────────────────────
    mean = np.array(
        ckpt.get('norm_mean', ckpt.get('norm_min', [4.04, 0.618, 0.574, 0.322])),
        dtype=np.float32)
    std = np.array(
        ckpt.get('norm_std', ckpt.get('norm_max', [6.55, 0.470, 0.294, 0.311])),
        dtype=np.float32)

    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"feat_mean: {mean.tolist()}")
    print(f"feat_std:  {std.tolist()}")

    # ── Layer weights ────────────────────────────────────────────────────────
    print("\n=== MLP Layer Stats ===")
    for k, v in state.items():
        print(f"  {k:40s} | shape={str(tuple(v.shape)):15s} | "
              f"min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")

    # ── Last layer bias — KEY diagnostic ────────────────────────────────────
    last_bias = None
    for k, v in state.items():
        if 'bias' in k and v.shape == (3,):
            last_bias = v.numpy()
    if last_bias is not None:
        print(f"\n=== Last Layer Bias (→ [seq_logit, gcn_logit, sem_logit]) ===")
        print(f"  seq={last_bias[0]:.4f}  gcn={last_bias[1]:.4f}  sem={last_bias[2]:.4f}")
        if last_bias[2] < -2.0:
            print(f"  ⚠ sem bias = {last_bias[2]:.4f} << 0")
            print(f"  → MLP đã học HARD-SUPPRESS sem qua bias.")
            print(f"  → Nguyên nhân: sem=0 trong phần lớn training samples")
            print(f"    (FAISS chưa được truyền vào lúc train, hoặc bug cross-user).")
            print(f"  → FIX: Retrain sau khi sửa compute_batch_scores_optimized.")

    # ── Forward pass sweep ──────────────────────────────────────────────────
    print("\n=== Sweep s_sem [0.0 → 1.0] (seq=mean, gcn=mean, len=mean) ===")
    print(f"{'s_sem':>6} {'normed_sem':>10} {'g_seq':>6} {'g_gcn':>6} {'g_sem':>8}")
    for s_sem in np.linspace(0, 1, 11):
        raw = np.array([mean[0], mean[1], s_sem, mean[3]], dtype=np.float32)
        normed = (raw - mean) / std
        with torch.no_grad():
            g = model(torch.tensor(normed).unsqueeze(0)).squeeze().numpy()
        print(f"{s_sem:>6.2f} {normed[2]:>10.4f} {g[0]:>6.4f} {g[1]:>6.4f} {g[2]:>8.6f}")

    print("\n=== Inference-like samples (từ log) ===")
    test_cases = [
        ("User 1 mean score",  [-1.69, 0.386, 0.545, 0.322]),
        ("User 2 mean score",  [-1.83, 0.446, 0.559, 0.322]),
        ("sem=1.0 (max)",      [mean[0], mean[1], 1.000, mean[3]]),
        ("sem=0.0 (min)",      [mean[0], mean[1], 0.000, mean[3]]),
        ("all at train mean",  mean.tolist()),
    ]
    for desc, raw in test_cases:
        raw_np = np.array(raw, dtype=np.float32)
        normed = (raw_np - mean) / std
        with torch.no_grad():
            g = model(torch.tensor(normed).unsqueeze(0)).squeeze().numpy()
        print(f"  {desc}")
        print(f"    raw_sem={raw[2]:.3f}  normed_sem={normed[2]:.4f} "
              f"→ g_seq={g[0]:.4f}  g_gcn={g[1]:.4f}  g_sem={g[2]:.6f}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    raw_max_sem = np.array([mean[0], mean[1], 1.0, mean[3]], dtype=np.float32)
    with torch.no_grad():
        g_max = model(torch.tensor((raw_max_sem - mean) / std).unsqueeze(0)).squeeze().numpy()
    if g_max[2] < 0.05:
        print("  ❌ CONFIRMED: MLP hoàn toàn suppress sem kể cả khi s_sem=1.0")
        print("  ❌ Cần RETRAIN gating sau khi fix train_gating.py")
    else:
        print(f"  ✓ MLP có thể activate sem (g_sem={g_max[2]:.4f} khi s_sem=1.0)")
        print("  → Vấn đề có thể ở normalization hoặc inference pipeline, không phải model weights")


if __name__ == '__main__':
    main()