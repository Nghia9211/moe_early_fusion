"""
moe_fusion/reranker.py
───────────────────────
Feedback Loop 4.0: 
  - LLM Reranker được phép chọn lại item cũ nếu nó cho rằng User Simulator sai lầm.
"""

import json
import re
import traceback
from typing import Dict, List, Tuple
import numpy as np
import math

def rank_to_score(ranked_list: List[str]) -> Dict[str, float]:
    # Điểm: 1.0, 0.63, 0.5, 0.43, 0.38...
    return {item: 1.0 / math.log2(rank + 2) for rank, item in enumerate(ranked_list)}

def _build_user_query(data: dict, candidate_names: list = None, id2name: Dict[int, str] = None) -> str:
    """Build user query string, filtering out reviews of candidate items to avoid data leakage.
    Mirrors the filter in RecHackerAgent_baseline.py line 72:
        filtered = [r for r in all_reviews if r.get('item_id') not in candidate_ids]
    """
    parts = []
    reviews = data.get('reviews', [])
    if reviews:
        # ── FILTER: loại bỏ review của candidate items (tránh leak GT review) ──
        # reviews là list[dict] với key 'item_id' (raw id)
        # candidate_names là list[str] tên item → dùng thêm name2id + id2rawid để map
        if isinstance(reviews, list):
            # ── RecHacker-style filter ──────────────────────────────────────────
            # RecHacker: candidate_ids = set(self.task['candidate_list'])
            #            filtered = [r for r in all_reviews
            #                        if r.get('item_id') not in candidate_ids]
            # Ở đây data['cans'] là inner IDs → map sang raw IDs qua id2rawid
            id2rawid = data.get('id2rawid', {})
            candidate_ids = set()
            for inner_id in data.get('cans', []):
                raw_id = id2rawid.get(inner_id)
                if raw_id:
                    candidate_ids.add(str(raw_id))
            if candidate_ids:
                filtered_reviews = []
                # Build mapping from raw_id -> item_name
                rawid2name = {}
                if id2name:
                    rawid2name = {str(raw): id2name.get(inner) for inner, raw in id2rawid.items() if inner in id2name}
                
                for r in reviews:
                    item_id_str = str(r.get('item_id', ''))
                    if item_id_str not in candidate_ids:
                        r_copy = dict(r)
                        r_copy.pop('review_id', None)
                        r_copy.pop('user_id', None)
                        
                        # Thêm tên item vào review
                        item_name = rawid2name.get(item_id_str)
                        if item_name:
                            r_copy['item_name'] = item_name
                        
                        filtered_reviews.append(r_copy)
                reviews = filtered_reviews

        # Convert list → string nếu cần, rồi truncate
        if isinstance(reviews, list):
            history_review = str(reviews)
        else:
            history_review = reviews
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            encoded = enc.encode(history_review)
            if len(encoded) > 8000: history_review = enc.decode(encoded[:8000])
        except Exception:
            history_review = history_review[:15000]
        parts.append(f"User's Historical Reviews:\n{history_review}")
    else:
        seq_str = data.get('seq_str', '') or ''
        if seq_str and seq_str.strip() and seq_str != 'Empty History':
            words = seq_str.split()
            parts.append(f"User history items: {' '.join(words[-80:])}")
    return "\n\n".join(parts) if parts else ""

def _embed_similarity(query: str, candidate_names: List[str], candidate_texts: Dict[str, str], embedding_fn) -> Dict[str, float]:
    if not candidate_names or embedding_fn is None: return {n: 0.5 for n in candidate_names}
    try:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        query_vec  = embedding_fn.embed_query(query)
        texts      = [candidate_texts.get(n, n) for n in candidate_names]
        item_vecs  = embedding_fn.embed_documents(texts)
        sims       = cos_sim([query_vec], item_vecs)[0]
        min_s, max_s = float(sims.min()), float(sims.max())
        rng = max_s - min_s if max_s != min_s else 1.0
        return {n: float((sims[i] - min_s) / rng) for i, n in enumerate(candidate_names)}
    except Exception:
        return {n: 0.5 for n in candidate_names}

def _llm_rerank(
    llm, data: dict, name2id: Dict[str, int], candidate_names: List[str], 
    user_query: str, memory: List[str], max_candidates: int = 10,
    output_dir: str = None
) -> Tuple[List[str], str]:
    cans_to_rank = candidate_names[:max_candidates]
    dataset = next((d for d in ['yelp', 'amazon', 'goodreads'] if d in str(data)), 'amazon')
    task_type = {"goodreads": "Goodreads", "yelp": "Yelp", "amazon": "Amazon"}.get(dataset, "Platform")
    task_item = {"goodreads": "book", "yelp": "business", "amazon": "product"}.get(dataset, "item")

    interaction_tool = data.get('interaction_tool')
    id2rawid         = data.get('id2rawid', {})
    item_list_info   = []

    if interaction_tool and name2id and id2rawid:
        keys = ['item_id', 'name', 'stars', 'review_count', 'attributes',
        'title', 'average_rating', 'rating_number', 'description',
        'ratings_count', 'title_without_series']
        for name in cans_to_rank:
            inner_id = name2id.get(name)
            raw_id   = id2rawid.get(inner_id)
            info_dict = {'Target_Name': name}
            if raw_id:
                try:
                    fetched = interaction_tool.get_item(item_id=raw_id)
                    if fetched:
                        for k in keys:
                            if k in fetched: info_dict[k] = fetched[k]
                except Exception: pass
            item_list_info.append(info_dict)
    else:
        item_list_info = [{'Target_Name': n} for n in cans_to_rank]

    # ── FORMAT ITEMS AS NUMBERED ML RANKING ──────────────────────────────
    numbered_lines = []
    for idx, info in enumerate(item_list_info, 1):
        name = info.get('Target_Name', 'Unknown')
        details = []
        for k in ['average_rating', 'stars', 'rating_number', 'ratings_count',
                   'review_count', 'description', 'attributes']:
            v = info.get(k)
            if v:
                if k == 'description' and isinstance(v, str) and len(v) > 150:
                    v = v[:150] + '...'
                if k == 'attributes' and isinstance(v, str) and len(v) > 100:
                    v = v[:100] + '...'
                details.append(f"{k}: {v}")
        detail_str = ", ".join(details) if details else "No additional info"
        numbered_lines.append(f'  #{idx}: "{name}" — {detail_str}')
    ranked_display = "\n".join(numbered_lines)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        encoded_items = enc.encode(ranked_display)
        if len(encoded_items) > 6000: ranked_display = enc.decode(encoded_items[:6000])
    except Exception:
        ranked_display = ranked_display[:25000]

    # ── LLM PROMPT 5.0: MOE-ANCHORED REFINEMENT ──────────────────────────
    if len(memory) > 0:
        dialogue_hist = "\n".join(memory[-2:])
        prompt = f"""You are a recommendation refinement system for {task_item}s on {task_type}.
A specialized ML model ranked candidates for this user. The user rejected the previous recommendation.
Your job is to REFINE the ranking based on the user's feedback.

User's Profile & History:
{user_query}

Previous Dialogue & User's Critique:
{dialogue_hist}

ML Model's Current Ranking (most recommended first):
{ranked_display}

INSTRUCTIONS:
1. Listen to the user's critique carefully. Keep POSITIVE MATCHES near the TOP.
2. Only push DOWN items the user explicitly rejected.
3. PRESERVE the ML model's order for items not mentioned in the critique.
4. Output ONLY a JSON: {{"ranked_items": [list of Target_Names], "explanation": "brief reason"}}
5. Include ALL {len(cans_to_rank)} candidates. Do not add text outside the JSON."""

    else:
        prompt = f"""You are a recommendation refinement system for {task_item}s on {task_type}.
A specialized ML recommendation model has already ranked candidate {task_item}s for this user using multiple signals (sequential behavior patterns, collaborative filtering, and content similarity). Your job is to REFINE this ranking — not rebuild it from scratch.

User's Profile & Review History:
{user_query}

ML Model's Ranking (most recommended → least recommended):
{ranked_display}

REFINEMENT INSTRUCTIONS:
1. The ML model's ranking is based on strong statistical signals and is generally RELIABLE. Do NOT rearrange dramatically.
2. Focus on the TOP 5 positions — getting these right matters most.
3. Only swap two items if you see CLEAR, SPECIFIC evidence in the user's review history that a lower-ranked item better matches their preferences.
4. Key signals: category/genre alignment, rating patterns, specific features the user mentions in reviews.
5. When in DOUBT, PRESERVE the ML model's original order.
6. Output ONLY a JSON: {{"ranked_items": [list of ALL {len(cans_to_rank)} Target_Names in refined order], "explanation": "reason for changes, or 'ML ranking preserved' if no changes needed"}}
7. Use Target_Names exactly as shown. Do not introduce new items."""

    # ── DIALOGUE LOGGING TO FILE (for paper) ──
    import os
    _log_dir  = output_dir or data.get('output_dir', os.path.join(os.path.dirname(__file__), 'output'))
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, 'reranker_dialogue_log.txt')

    round_label = "FEEDBACK CORRECTION ROUND" if len(memory) > 0 else "INITIAL RANKING ROUND"
    SEP  = "=" * 80
    SEP2 = "-" * 80

    def _log(text: str):
        with open(_log_path, 'a', encoding='utf-8') as f:
            f.write(text + "\n")

    _log(f"\n{SEP}")
    _log(f"[RERANKER DIALOGUE] {round_label} | Platform: {task_type} | Item type: {task_item}")
    _log(SEP)
    _log("[PROMPT → RERANKER]")
    _log(SEP2)
    _log(prompt)
    _log(SEP2)

    try:
        response      = llm.invoke(prompt)
        content       = response.content if hasattr(response, 'content') else str(response)

        _log("[RERANKER RESPONSE]")
        _log(SEP2)
        _log(content)
        _log(SEP2)

        content_clean = content.replace("`" * 3 + "json", "").replace("`" * 3, "").strip()
        parsed        = json.loads(content_clean)
        ranked        = parsed.get('ranked_items', [])
        explanation   = parsed.get('explanation', '')

        _log(f"[PARSED] Explanation: {explanation}")
        _log(f"[PARSED] Ranked list ({len(ranked)} items): {ranked}")
        _log(SEP + "\n")

        valid_set    = set(cans_to_rank)
        ranked_valid = [r for r in ranked if r in valid_set]
        ranked_set   = set(ranked_valid)
        for n in cans_to_rank:
            if n not in ranked_set: ranked_valid.append(n)
        if len(candidate_names) > max_candidates: ranked_valid += candidate_names[max_candidates:]

        return ranked_valid, explanation
    except Exception as e:
        _log(f"[RERANKER ERROR] {e}")
        _log(SEP + "\n")
        return candidate_names, "Failed to parse LLM output."

class Reranker:
    def __init__(self, embedding_fn=None, llm=None, mode: str='embed_only', enabled: bool=True, top_llm: int=20, output_dir: str=None):
        self.embedding_fn, self.llm, self.mode, self.enabled, self.top_llm = embedding_fn, llm, mode, enabled, top_llm
        self.output_dir = output_dir

    def rerank(self, data: dict, c_m: List[str], id2name: Dict[int, str]=None, name2id: Dict[str, int]=None, memory: List[str]=None) -> Tuple[List[str], Dict[str, float], str]:
        if not self.enabled or not c_m: return c_m, rank_to_score(c_m), "Reranker disabled."
        memory = memory or []
        try:
            if self.mode == 'embed_only': return self._embed_rerank(data, c_m, memory, id2name)
            elif self.mode == 'llm': return self._llm_only_rerank(data, name2id, c_m, memory, id2name)
            elif self.mode == 'hybrid': return self._hybrid_rerank(data, name2id, c_m, memory, id2name)
            else: return self._embed_rerank(data, c_m, memory, id2name)
        except Exception as e: return c_m, rank_to_score(c_m), f"Reranker error: {e}"

    def _embed_rerank(self, data: dict, c_m: List[str], memory: List[str], id2name: Dict[int, str]=None) -> Tuple[List[str], Dict[str, float], str]:
        query = _build_user_query(data, id2name=id2name)
        if not query: return c_m, rank_to_score(c_m), "No user query available."
        sim_scores = _embed_similarity(query, c_m, {n: n for n in c_m}, self.embedding_fn)
        ranked = sorted(c_m, key=lambda n: sim_scores.get(n, 0.0), reverse=True)
        return ranked, rank_to_score(ranked), "Reranked by embedding similarity."

    def _llm_only_rerank(self, data: dict, name2id: Dict[str, int], c_m: List[str], memory: List[str], id2name: Dict[int, str]=None) -> Tuple[List[str], Dict[str, float], str]:
        if self.llm is None: return self._embed_rerank(data, c_m, memory, id2name)
        query = _build_user_query(data, candidate_names=c_m, id2name=id2name)
        ranked, explanation = _llm_rerank(self.llm, data, name2id, c_m, query, memory, max_candidates=self.top_llm, output_dir=self.output_dir)
        return ranked, rank_to_score(ranked), explanation

    def _hybrid_rerank(self, data: dict, name2id: Dict[str, int], c_m: List[str], memory: List[str], id2name: Dict[int, str]=None) -> Tuple[List[str], Dict[str, float], str]:
        query = _build_user_query(data, candidate_names=c_m, id2name=id2name)
        sim_scores = _embed_similarity(query, c_m, {n: n for n in c_m}, self.embedding_fn)
        embed_ranked = sorted(c_m, key=lambda n: sim_scores.get(n, 0.0), reverse=True)
        if self.llm is None: return embed_ranked, rank_to_score(embed_ranked), "Fallback to embed_only"
        
        top_pool, tail_pool = embed_ranked[:self.top_llm], embed_ranked[self.top_llm:]
        llm_ranked, explanation = _llm_rerank(self.llm, data, name2id, top_pool, query, memory, max_candidates=self.top_llm, output_dir=self.output_dir)
        final_ranked = llm_ranked + tail_pool
        return final_ranked, rank_to_score(final_ranked), explanation

    @classmethod
    def from_shared(cls, shared: dict, llm=None, mode: str='embed_only', enabled: bool=True, top_llm: int=15, output_dir: str=None) -> "Reranker":
        return cls(embedding_fn=shared.get('embedding_function'), llm=llm, mode=mode, enabled=enabled, top_llm=top_llm, output_dir=output_dir)