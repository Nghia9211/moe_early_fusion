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

def _build_user_query(data: dict, candidate_names: list = None) -> str:
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
                reviews = [r for r in reviews
                           if str(r.get('item_id', '')) not in candidate_ids]

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
    user_query: str, memory: List[str], max_candidates: int = 20,
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

    items_desc_str = str(item_list_info)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        encoded_items = enc.encode(items_desc_str)
        if len(encoded_items) > 6000: items_desc_str = enc.decode(encoded_items[:6000])
    except Exception:
        items_desc_str = items_desc_str[:25000]

    # ── LLM PROMPT 4.1: SOFT CORRECTION + KEEP GUARD ──
    if len(memory) > 0:
        dialogue_hist = "\n".join(memory[-2:])
        prompt = f"""You are a recommendation system for {task_item}s on {task_type}.
In the previous round, you recommended items but the user rejected them. 

User's Profile & History:
{user_query}

Previous Dialogue & User's Critique:
{dialogue_hist}

You must evaluate the ENTIRE candidate list again: {str(cans_to_rank)}
Candidate Details:
{items_desc_str}

CRITICAL INSTRUCTIONS:
1. Listen to the user's critique carefully. Distinguish between POSITIVE MATCHES and NEGATIVE NOISE in their feedback.
2. If the user's critique mentions POSITIVE MATCHES, you MUST keep those items near the TOP of your ranking. They are confirmed good fits.
3. Only push DOWN items that the user explicitly marked as NEGATIVE NOISE or truly irrelevant.
4. If you firmly believe an item from the previous list is still the best fit despite the critique, YOU MAY KEEP IT in your new ranking.
5. Output ONLY a JSON object with keys "ranked_items" (list of Target_Names) and "explanation" (explain your adjustment).
6. Do not add any text outside the JSON. Example format: {{"ranked_items": ["A", "B", "C"], "explanation": "..."}}"""

    else:
        prompt = f"""You are a real human user on {task_type}, a platform for crowd-sourced {task_item} reviews.
Here is your {task_type} profile and review history: {user_query}.
Your historical {task_item} reviews show your preference as follows: ['user_id', 'review_count', 'friends', 'stars'...].
Now you need to rank the following {len(cans_to_rank)} {task_item}: {str(cans_to_rank)} according to their match degree to your preference.
The information of the above {len(cans_to_rank)} candidate {task_item} is as follows: {items_desc_str}.

Your final output should be ONLY a JSON object with keys "ranked_items" (list of Target_Names in sorted order) and "explanation" (brief reason).
DO NOT introduce any other {task_item} ids!
Please rank the more interested {task_item} more front in your rank list.
You should think step by step before your final answer.
IMPORTANT: DO NOT output your analysis process!
Remember to output {task_item} Target_Names instead of {task_item} names.
Include ALL {len(cans_to_rank)} candidates. Do not add any text outside the JSON.
Example format: {{"ranked_items": ["A", "B", "C"], "explanation": "..."}}"""

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
    def __init__(self, embedding_fn=None, llm=None, mode: str='embed_only', enabled: bool=True, top_llm: int=15, output_dir: str=None):
        self.embedding_fn, self.llm, self.mode, self.enabled, self.top_llm = embedding_fn, llm, mode, enabled, top_llm
        self.output_dir = output_dir

    def rerank(self, data: dict, c_m: List[str], id2name: Dict[int, str]=None, name2id: Dict[str, int]=None, memory: List[str]=None) -> Tuple[List[str], Dict[str, float], str]:
        if not self.enabled or not c_m: return c_m, rank_to_score(c_m), "Reranker disabled."
        memory = memory or []
        try:
            if self.mode == 'embed_only': return self._embed_rerank(data, c_m, memory)
            elif self.mode == 'llm': return self._llm_only_rerank(data, name2id, c_m, memory)
            elif self.mode == 'hybrid': return self._hybrid_rerank(data, name2id, c_m, memory)
            else: return self._embed_rerank(data, c_m, memory)
        except Exception as e: return c_m, rank_to_score(c_m), f"Reranker error: {e}"

    def _embed_rerank(self, data: dict, c_m: List[str], memory: List[str]) -> Tuple[List[str], Dict[str, float], str]:
        query = _build_user_query(data)
        if not query: return c_m, rank_to_score(c_m), "No user query available."
        sim_scores = _embed_similarity(query, c_m, {n: n for n in c_m}, self.embedding_fn)
        ranked = sorted(c_m, key=lambda n: sim_scores.get(n, 0.0), reverse=True)
        return ranked, rank_to_score(ranked), "Reranked by embedding similarity."

    def _llm_only_rerank(self, data: dict, name2id: Dict[str, int], c_m: List[str], memory: List[str]) -> Tuple[List[str], Dict[str, float], str]:
        if self.llm is None: return self._embed_rerank(data, c_m, memory)
        query = _build_user_query(data, candidate_names=c_m)
        ranked, explanation = _llm_rerank(self.llm, data, name2id, c_m, query, memory, max_candidates=len(c_m), output_dir=self.output_dir)
        return ranked, rank_to_score(ranked), explanation

    def _hybrid_rerank(self, data: dict, name2id: Dict[str, int], c_m: List[str], memory: List[str]) -> Tuple[List[str], Dict[str, float], str]:
        query = _build_user_query(data, candidate_names=c_m)
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