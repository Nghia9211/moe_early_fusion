"""
train_gating.py — v2.1 (User-Context Gating, 5-feature)
────────────────────────────────────────────────────────────────────
Thay đổi so với v2.0:
  - input_dim = 7 
  - CE + entropy reg, balanced target
"""

import os, sys, argparse, math, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
root_dir    = os.path.dirname(parent_dir)
for p in [current_dir, parent_dir, root_dir, os.path.join(root_dir, 'baseline')]:
    if p not in sys.path: sys.path.insert(0, p)

from config          import MoEConfig, GatingConfig, DEFAULT_CONFIG
from gating_network  import GatingMLP, GatingNetwork, extract_context_features
from seq_scorer      import SeqScorer
from gcn_scorer      import GCNScorer
from semantic_scorer import SemanticScorer
from dataset.general_dataset import GeneralDataset


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Feature labels khớp với v2.2
FEATURE_LABELS = [
    'norm_seq_len',
    'agree_gcn',
    'agree_sem',
    'agree_gcn_sem',   # v2.2 NEW: Spearman(gcn, sem)
    'seq_confidence',  # v2.2 NEW: max-mean of seq scores
    'gcn_confidence',
    'sem_confidence',
]


def _minmax(d: Dict[str, float]) -> Dict[str, float]:
    if not d: return {}
    vals = np.array(list(d.values()))
    lo, hi = vals.min(), vals.max()
    if hi == lo: return {k: 0.5 for k in d}
    return {k: float((v - lo) / (hi - lo)) for k, v in d.items()}


def _ndcg_quality(gt_name: str, scores: Dict[str, float]) -> float:
    if not scores or gt_name not in scores: return 0.0
    rank = sorted(scores, key=scores.get, reverse=True).index(gt_name) + 1
    return 1.0 / math.log2(rank + 1)


def _softmax(arr: np.ndarray, temp: float) -> np.ndarray:
    a = arr / max(temp, 1e-8)
    a -= a.max()
    e = np.exp(a)
    return e / e.sum()


def _quality_target_v3(q_arr: np.ndarray, min_quality: float = 0.1) -> Optional[np.ndarray]:
    """
    Quality-proportional target with weak expert suppression.
    
    If an expert's NDCG quality < min_quality, its target weight → 0.
    If NO expert passes the threshold, return None (skip sample).
    """
    q_masked = q_arr.copy()
    q_masked[q_masked < min_quality] = 0.0
    
    total = q_masked.sum()
    if total < 1e-8:
        return None  # No expert is good → skip
    
    return (q_masked / total).astype(np.float32)


def build_context_feature(
    seq: List[int], len_seq: int,
    seq_sc: Dict, gcn_sc: Dict, sem_sc: Dict,
    gcn_norm, cfg: GatingConfig,
) -> List[float]:
    """Wrapper gọi extract_context_features từ gating_network.py (5-feature v2.1)."""
    return extract_context_features(
        seq=seq, len_seq=len_seq,
        seq_scores=seq_sc, gcn_scores=gcn_sc, sem_scores=sem_sc,
        gcn_norm=gcn_norm,   # không dùng trong v2.1, passed for API compat
        cfg=cfg,
    )


def print_feature_report(X: np.ndarray):
    print(f"\n{'Signal':<18} | {'Mean':>7} | {'Std':>7} | {'%NonZero':>9}")
    print('-' * 54)
    for i in range(X.shape[1]):
        col = X[:, i]
        nz  = (np.abs(col) > 1e-9).mean() * 100
        lbl = FEATURE_LABELS[i] if i < len(FEATURE_LABELS) else f'f{i}'
        print(f"{lbl:<18} | {col.mean():>7.4f} | {col.std():>7.4f} | {nz:>8.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Batch score computation  (unchanged from v2.0)
# ─────────────────────────────────────────────────────────────────────────────

def compute_batch_scores(batch_seqs, batch_lens, batch_pools,
                         seq_scorer, gcn_scorer, sem_scorer, id2name, device):
    B = len(batch_seqs)
    with torch.no_grad():
        all_logits = seq_scorer.model.forward_eval(batch_seqs, batch_lens.cpu().numpy())

    h_users = None
    if gcn_scorer:
        hl = []
        for i in range(B):
            h = gcn_scorer._user_embedding(batch_seqs[i].tolist(), int(batch_lens[i]))
            hl.append(h if h is not None
                      else torch.zeros(gcn_scorer.gcn_norm.shape[1], device=device))
        h_users = torch.stack(hl)

    sem_map = [{} for _ in range(B)]
    if sem_scorer and sem_scorer.embedding_function:
        all_unique = set()
        for pool in batch_pools:
            all_unique.update(id2name.get(cid, f'item_{cid}') for cid in pool)
        unique_list = list(all_unique)
        rich   = sem_scorer._get_candidate_texts(unique_list)
        q_strs = [
            'User interested in: ' + ' '.join(
                id2name.get(iid, '') for iid in batch_seqs[i].tolist() if iid in id2name
            )
            for i in range(B)
        ]
        q_vecs = np.array(sem_scorer.embedding_function.embed_documents(q_strs), dtype=np.float32)
        d_vecs = np.array(sem_scorer.embedding_function.embed_documents(rich),   dtype=np.float32)
        q_vecs /= (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-8)
        d_vecs /= (np.linalg.norm(d_vecs, axis=1, keepdims=True) + 1e-8)
        sim  = q_vecs @ d_vecs.T
        n2i  = {n: i for i, n in enumerate(unique_list)}
        for i, pool in enumerate(batch_pools):
            raw = {
                id2name[c]: sim[i, n2i[id2name[c]]]
                for c in pool
                if c in id2name and id2name[c] in n2i
            }
            sem_map[i] = _minmax(raw)

    results = []
    for i in range(B):
        pool  = batch_pools[i]
        s_seq = _minmax({id2name[c]: all_logits[i, c].item() for c in pool if c in id2name})
        s_gcn = {}
        if h_users is not None:
            can_ids = torch.tensor([c for c in pool if c < gcn_scorer.num_items], device=device)
            if len(can_ids):
                sims  = (gcn_scorer.gcn_norm[can_ids] @ h_users[i]).cpu().tolist()
                s_gcn = _minmax({
                    id2name[c.item()]: s
                    for c, s in zip(can_ids, sims)
                    if c.item() in id2name
                })
        results.append((s_seq, s_gcn, sem_map[i]))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Build training data — 7-feature context + CE target
# ─────────────────────────────────────────────────────────────────────────────

def build_training_data_context(
    loader, seq_scorer, gcn_scorer, sem_scorer,
    id2name, cfg: GatingConfig,
    balance_eps:  float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X : (N, 5) context features
        Y : (N, 3) CE target gate distribution
    """
    X_list, Y_list = [], []
    gcn_norm = gcn_scorer.gcn_norm if gcn_scorer else None
    device   = seq_scorer.device

    pbar = tqdm(loader, desc='Building Context Training Data', unit='batch')
    for batch in pbar:
        seqs     = batch['seq'].to(device)
        lens     = batch['len_seq'].to(device)
        next_ids = batch['next'].cpu().numpy()
        cans_t   = batch.get('cans')

        batch_pools = []
        for i in range(len(next_ids)):
            gt_id = int(next_ids[i])
            cans  = cans_t[i].tolist() if cans_t is not None else []
            batch_pools.append(list(set(cans) | {gt_id}))

        results = compute_batch_scores(
            seqs, lens, batch_pools,
            seq_scorer, gcn_scorer, sem_scorer, id2name, device,
        )

        for i in range(len(next_ids)):
            gt_id   = int(next_ids[i])
            gt_name = id2name.get(gt_id)
            if not gt_name:
                continue

            s_seq, s_gcn, s_sem = results[i]
            len_seq_i = int(lens[i])
            seq_list  = seqs[i].tolist()

            # ── 7-feature context vector ─────────────────────────────────────
            ctx = build_context_feature(
                seq_list, len_seq_i, s_seq, s_gcn, s_sem, gcn_norm, cfg
            )

            # ── Target: NDCG quality per expert ──────────────────────────
            q_seq = _ndcg_quality(gt_name, s_seq)
            q_gcn = _ndcg_quality(gt_name, s_gcn) if s_gcn else 0.0
            q_sem = _ndcg_quality(gt_name, s_sem) if s_sem else 0.0
            q_arr = np.array([q_seq, q_gcn, q_sem], dtype=np.float32)

            # v3: quality-aware target with weak expert suppression
            target_gate = _quality_target_v3(q_arr, min_quality=balance_eps)
            if target_gate is None:
                continue  # Skip samples where no expert is good

            X_list.append(ctx)
            Y_list.append(target_gate)

        pbar.set_postfix({'N': len(X_list)})
    pbar.close()

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_features(X: np.ndarray):
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0        # constant features → keep as-is (shouldn't happen in v2.1)
    return (X - mu) / std, mu, std


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_gating_context(
    X:        np.ndarray,
    Y_target: np.ndarray,
    cfg:      GatingConfig,
    device:   torch.device,
    val_ratio: float = 0.1,
) -> GatingMLP:
    """
    Train GatingMLP với Cross Entropy loss + Entropy regularization.
    Entropy bonus buộc gates phân tán, tránh collapse về 1 expert.
    """
    n     = len(X)
    n_val = max(1, int(n * val_ratio))
    idx   = np.random.permutation(n)
    v_idx, t_idx = idx[:n_val], idx[n_val:]

    ds     = TensorDataset(
        torch.tensor(X[t_idx],        dtype=torch.float32),
        torch.tensor(Y_target[t_idx], dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model     = GatingMLP(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    X_vl = torch.tensor(X[v_idx],        dtype=torch.float32).to(device)
    Y_vl = torch.tensor(Y_target[v_idx], dtype=torch.float32).to(device)

    best_val, best_state = float('inf'), None

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            gates = model(x_b)                              # (B, 3) — softmax

            # Cross-entropy loss: -Σ target_i * log(pred_i)
            ce = -(y_b * gates.log().clamp(min=-100)).sum(dim=-1).mean()
            # Concentration reg: penalize uniform gates (opposite of entropy bonus)
            neg_H = (gates * gates.log().clamp(min=-100)).sum(dim=-1).mean()
            loss = ce + conc_w * neg_H

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl_gates = model(X_vl)
            vl_ce    = -(Y_vl * vl_gates.log().clamp(min=-100)).sum(dim=-1).mean().item()
            vl_negH  = (vl_gates * vl_gates.log().clamp(min=-100)).sum(dim=-1).mean().item()
            val_loss = vl_ce + conc_w * vl_negH

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            avg_g = model(X_vl).mean(dim=0).detach().cpu().numpy()
            print(f'  Epoch {epoch+1:03d}/{cfg.epochs} | '
                  f'train={total_loss / max(len(loader), 1):.4f} | val={val_loss:.4f} | '
                  f'gates: seq={avg_g[0]:.3f} gcn={avg_g[1]:.3f} sem={avg_g[2]:.3f}')

    if best_state:
        model.load_state_dict(best_state)

    # Final distribution stats
    model.eval()
    with torch.no_grad():
        all_x = torch.tensor(X, dtype=torch.float32).to(device)
        avg_g = model(all_x).detach().cpu().numpy()

    print(f'\n📊 Final avg gates on ALL data:')
    print(f'   seq={avg_g[:,0].mean():.3f}±{avg_g[:,0].std():.3f} | '
          f'gcn={avg_g[:,1].mean():.3f}±{avg_g[:,1].std():.3f} | '
          f'sem={avg_g[:,2].mean():.3f}±{avg_g[:,2].std():.3f}')
    print(f'\n📊 Target gate distribution (mean):')
    print(f'   seq={Y_target[:,0].mean():.3f} | '
          f'gcn={Y_target[:,1].mean():.3f} | '
          f'sem={Y_target[:,2].mean():.3f}')
    print(f'\nBest val loss: {best_val:.4f}')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--model_path',  required=True)
    parser.add_argument('--gcn_path',    default=None)
    parser.add_argument('--faiss_path',  default=None)
    parser.add_argument('--output_dir',  default='./saved_models/moe')
    parser.add_argument('--dataset',     default='amazon', choices=['amazon', 'yelp', 'goodreads'])
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--balance_eps', type=float, default=0.1,
                        help='v3: expert quality threshold. Experts with NDCG < this → gate=0')
    parser.add_argument('--entropy_reg', type=float, default=0.0,
                        help='DEPRECATED. Use --conc_weight instead.')
    parser.add_argument('--conc_weight', type=float, default=0.02,
                        help='Concentration reg weight. Positive = penalize uniform gates')
    parser.add_argument('--split',       default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--hidden_size', type=int,   default=64)
    parser.add_argument('--embed_model', default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train_gating v2.2] Device={device} | dataset={args.dataset} | split={args.split}')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load SASRec ──────────────────────────────────────────────────────────
    from utils.model import SASRec
    data_statis = pd.read_pickle(os.path.join(args.data_dir, 'data_statis.df'))
    seq_size, item_num = data_statis['seq_size'][0], data_statis['item_num'][0]
    ckpt      = torch.load(args.model_path, map_location=device, weights_only=False)
    hidden_sz = ckpt.get('hidden_size', args.hidden_size) if isinstance(ckpt, dict) else args.hidden_size
    sasrec    = SASRec(hidden_sz, item_num, seq_size, 0.1, device).to(device)
    sasrec.load_state_dict(ckpt.get('model_state_dict', ckpt))
    sasrec.eval()

    id2name: Dict[int, str] = {}
    with open(os.path.join(args.data_dir, 'id2name.txt'), encoding='utf-8') as f:
        for line in f:
            ll = line.strip().split('::', 1)
            if len(ll) == 2:
                id2name[int(ll[0])] = ll[1].strip()

    shared = {
        'model': sasrec, 'id2name': id2name, 'name2id': {},
        'item_num': item_num, 'device': device,
    }
    seq_scorer = SeqScorer.from_shared(shared)

    # ── GCN ─────────────────────────────────────────────────────────────────
    gcn_scorer = None
    if args.gcn_path and os.path.exists(args.gcn_path):
        shared['gcn_embeddings'] = torch.load(args.gcn_path, map_location=device, weights_only=True)
        gcn_scorer = GCNScorer.from_shared(shared)
        print(f'[train_gating] GCN loaded: {gcn_scorer.num_items} items')

    # ── Semantic ─────────────────────────────────────────────────────────────
    sem_scorer = None
    if args.faiss_path and os.path.exists(args.faiss_path):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        embed_fn  = HuggingFaceEmbeddings(model_name=args.embed_model)
        vs        = FAISS.load_local(
            folder_path=args.faiss_path, embeddings=embed_fn,
            allow_dangerous_deserialization=True,
        )
        shared.update({'vector_store': vs, 'embedding_function': embed_fn})
        sem_scorer = SemanticScorer.from_shared(shared)
        print('[train_gating] Semantic FAISS loaded')

    # ── Dataset ──────────────────────────────────────────────────────────────
    class _Args:
        def __init__(self, d): self.data_dir = d

    dataset = GeneralDataset(
        _Args(args.data_dir),
        stage={'train': 'train', 'val': 'val', 'test': 'test'}[args.split],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── GatingConfig v2.2 ────────────────────────────────────────────────────
    cfg = GatingConfig(
        input_dim          = 7,
        hidden_dims        = [32, 16],
        dropout            = 0.2,
        epochs             = args.epochs,
        lr                 = args.lr,
        batch_size         = args.batch_size,
        gating_mode        = 'context',
        entropy_reg_weight = args.entropy_reg,
        expert_quality_threshold = args.balance_eps,
        concentration_weight     = args.conc_weight,
    )

    # ── Build training data ──────────────────────────────────────────────────
    print('\n📦 Building context training data (7-feature v2.2)...')
    X_raw, Y_target = build_training_data_context(
        loader, seq_scorer, gcn_scorer, sem_scorer, id2name,
        cfg=cfg, balance_eps=args.balance_eps,
    )
    print(f'Training samples: {len(X_raw):,}')
    print_feature_report(X_raw)

    X_norm, mu, std = normalize_features(X_raw)

    # ── Train ────────────────────────────────────────────────────────────────
    print('\n🚀 Training GatingMLP v2.2 (7-feature context-mode)...')
    trained_mlp = train_gating_context(X_norm, Y_target, cfg, device)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, f'{args.dataset}_gating_model.pt')
    torch.save({
        'model_state_dict': trained_mlp.state_dict(),
        'cfg':              cfg,
        'norm_mean':        mu.tolist(),
        'norm_std':         std.tolist(),
        'gating_mode':      'context',
        'feature_version':  'v2.2',
        'feature_names':    FEATURE_LABELS,
    }, out_path)
    print(f'\n✅ Saved → {out_path}')


if __name__ == '__main__':
    main()