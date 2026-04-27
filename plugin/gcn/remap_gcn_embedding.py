"""
remap_gcn_embeddings.py
────────────────────────
Re-export GCN embeddings từ {original_id: tensor} sang {int_item_id: tensor}
để GCNScorer có thể index đúng theo integer item ID.

Cách dùng:
    python remap_gcn_embeddings.py \
        --gcn_path     ./saved_models/amazon_gcn_emb.pt \
        --id2rawid     ./data/amazon/id2rawid.txt \
        --output_path  ./saved_models/amazon_gcn_emb_remapped.pt \
        --emb_dim      64
"""

import torch
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcn_path',    required=True)
    parser.add_argument('--id2rawid',    required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--emb_dim',     type=int, default=64)
    args = parser.parse_args()

    # 1. Load GCN dict: {original_id_string: tensor(64,)}
    print(f"Loading GCN embeddings from {args.gcn_path}...")
    gcn_dict = torch.load(args.gcn_path, map_location='cpu')
    print(f"  Total nodes in GCN: {len(gcn_dict):,}")

    # 2. Load id2rawid: int_id → ASIN
    print(f"Loading id2rawid from {args.id2rawid}...")
    int2asin = {}
    with open(args.id2rawid, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('::', 1)
            if len(parts) == 2:
                int2asin[int(parts[0])] = parts[1].strip()

    print(f"  Total item mappings: {len(int2asin):,}")
    print(f"  Sample: {list(int2asin.items())[:3]}")

    # 3. Build remapped tensor: index = int_item_id
    max_id   = max(int2asin.keys())
    emb_dim  = args.emb_dim
    remapped = torch.zeros(max_id + 1, emb_dim, dtype=torch.float32)

    matched   = 0
    unmatched = 0

    for int_id, asin in int2asin.items():
        if asin in gcn_dict:
            remapped[int_id] = gcn_dict[asin].float()
            matched += 1
        else:
            # Giữ zero vector — GCNScorer sẽ normalize thành uniform
            unmatched += 1

    print(f"\n  Matched:   {matched:,} / {len(int2asin):,}")
    print(f"  Unmatched: {unmatched:,} (zero vectors — cold items)")
    print(f"  Coverage:  {matched/len(int2asin)*100:.1f}%")

    if matched == 0:
        raise ValueError(
            "0 items matched! id2rawid values không khớp với GCN keys.\n"
            f"  id2rawid sample: {list(int2asin.values())[:3]}\n"
            f"  GCN key sample:  {list(gcn_dict.keys())[:3]}"
        )

    # 4. Save
    torch.save(remapped, args.output_path)
    print(f"\n✅ Remapped embeddings saved to {args.output_path}")
    print(f"   Shape: {remapped.shape}  (max_item_id+1, emb_dim)")

    # 5. Quick sanity check
    print("\n--- Sanity Check ---")
    sample_ids = list(int2asin.keys())[:3]
    for iid in sample_ids:
        asin = int2asin[iid]
        if asin in gcn_dict:
            orig = gcn_dict[asin]
            remap = remapped[iid]
            match = torch.allclose(orig.float(), remap)
            print(f"  int_id={iid} | ASIN={asin} | match={match}")


if __name__ == '__main__':
    main()