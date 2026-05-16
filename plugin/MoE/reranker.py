"""
moe_fusion/reranker.py
───────────────────────
Feedback Loop 4.0: 
  - LLM Reranker được phép chọn lại item cũ nếu nó cho rằng User Simulator sai lầm.
"""

import json
import re
import traceback
import threading
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
import numpy as np
import math

# ── Module-level tiktoken cache (tránh load BPE vocab lại mỗi call) ──────────
try:
    import tiktoken as _tiktoken
    _TIKTOKEN_ENC = _tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None

# ── Module-level item metadata cache (thread-safe) ───────────────────────────
_item_cache: Dict[str, dict] = {}
_item_cache_lock = threading.Lock()

def _cached_get_item(tool, raw_id: str):
    """Fetch item metadata với cache để tránh gọi lại cùng raw_id nhiều lần."""
    with _item_cache_lock:
        if raw_id in _item_cache:
            return _item_cache[raw_id]
    result = tool.get_item(item_id=raw_id)
    with _item_cache_lock:
        _item_cache[raw_id] = result
    return result

# ── Structured output schema for LLM reranker ────────────────────────────
class _RankerOutput(BaseModel):
    ranked_items: List[str]
    explanation: str


def rank_to_score(ranked_list: List[str]) -> Dict[str, float]:
    # Điểm: 1.0, 0.63, 0.5, 0.43, 0.38...
    return {item: 1.0 / math.log2(rank + 2) for rank, item in enumerate(ranked_list)}

def _build_user_query(data: dict, candidate_names: list = None, id2name: Dict[int, str] = None) -> str:
    """Build user query string, filtering out reviews of candidate items to avoid data leakage.
    Mirrors the filter in RecHackerAgent_baseline.py line 72:
        filtered = [r for r in all_reviews if r.get('item_id') not in candidate_ids]
    """
    # Fields to strip from review history (noise / irrelevant metadata)
    _STRIP_FIELDS = {'date_added', 'date_updated', 'source', 'type',
                     'timestamp', 'image', 'sub_item_id', 'date',
                     'review_id', 'user_id'}

    parts = []
    reviews = data.get('reviews', [])
    if reviews:
        # ── FILTER: loại bỏ review của candidate items (tránh leak GT review) ──
        if isinstance(reviews, list):
            id2rawid        = data.get('id2rawid', {})
            interaction_tool = data.get('interaction_tool')
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
                    rawid2name = {str(raw): id2name.get(inner)
                                  for inner, raw in id2rawid.items() if inner in id2name}

                for r in reviews:
                    item_id_str = str(r.get('item_id', ''))
                    if item_id_str not in candidate_ids:
                        # 1. Bỏ các trường noise
                        r_copy = {k: v for k, v in r.items() if k not in _STRIP_FIELDS}

                        # 2. Thêm item_name (nếu chưa có)
                        item_name = rawid2name.get(item_id_str)
                        if item_name:
                            r_copy['item_name'] = item_name

                        # 3. Inject categories từ interaction_tool (nếu có)
                        if interaction_tool:
                            try:
                                fetched = interaction_tool.get_item(item_id=item_id_str)
                                if fetched:
                                    cats = fetched.get('categories')
                                    if cats:
                                        if isinstance(cats, list):
                                            cats = ', '.join(str(c) for c in cats)
                                        r_copy['categories'] = cats
                            except Exception:
                                pass

                        # 4. Đưa item_id và item_name lên đầu dict
                        ordered = {}
                        if 'item_id' in r_copy:   ordered['item_id']   = r_copy.pop('item_id')
                        if 'item_name' in r_copy:  ordered['item_name'] = r_copy.pop('item_name')
                        if 'categories' in r_copy: ordered['categories'] = r_copy.pop('categories')
                        ordered.update(r_copy)  # còn lại: stars, text, ...

                        filtered_reviews.append(ordered)
                reviews = filtered_reviews

        # Convert list → string nếu cần, rồi truncate
        if isinstance(reviews, list):
            history_review = str(reviews)
        else:
            history_review = reviews
        try:
            if _TIKTOKEN_ENC is not None:
                encoded = _TIKTOKEN_ENC.encode(history_review)
                if len(encoded) > 8000: history_review = _TIKTOKEN_ENC.decode(encoded[:8000])
            else:
                history_review = history_review[:15000]
        except Exception:
            history_review = history_review[:15000]
        parts.append(f"\n{history_review}")
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
        keys = ['item_id', 'name', 'stars', 'review_count', 'categories',
        'title', 'average_rating', 'rating_number', 'description',
        'ratings_count', 'title_without_series']
        for name in cans_to_rank:
            inner_id = name2id.get(name)
            raw_id   = id2rawid.get(inner_id)
            info_dict = {'Target_Name': name}
            if raw_id:
                try:
                    fetched = _cached_get_item(interaction_tool, raw_id)
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
                   'review_count', 'description', 'categories']:
            v = info.get(k)
            if v:
                if k == 'description' and isinstance(v, str) and len(v) > 150:
                    v = v[:150] + '...'
                if k == 'categories' and isinstance(v, str) and len(v) > 100:
                    v = v[:100] + '...'
                details.append(f"{k}: {v}")
        detail_str = ", ".join(details) if details else "No additional info"
        numbered_lines.append(f'  #{idx}: "{name}" — {detail_str}')
    ranked_display = "\n".join(numbered_lines)

    try:
        if _TIKTOKEN_ENC is not None:
            encoded_items = _TIKTOKEN_ENC.encode(ranked_display)
            if len(encoded_items) > 6000: ranked_display = _TIKTOKEN_ENC.decode(encoded_items[:6000])
        else:
            ranked_display = ranked_display[:25000]
    except Exception:
        ranked_display = ranked_display[:25000]

    # ── LLM PROMPT 5.2: UNIFIED FEEDBACK REFINEMENT ─────────────────────────
    if len(memory) > 0:
        dialogue_hist = "\n".join(memory[-2:])

        # Extract explicitly mentioned positive matches from the critique
        positive_items_text = ""
        pos_match = re.search(
            r'POSITIVE\s*MATCHES?\s*:(.*?)(?=\n\s*\d+\.\s*NEGATIVE|\Z)',
            dialogue_hist,
            re.IGNORECASE | re.DOTALL
        )
        if pos_match:
            pos_text = pos_match.group(1).strip()
            if pos_text and not re.match(r'^none\b', pos_text, re.IGNORECASE):
                positive_items_text = f"\nCRITICAL MUST-DO: The user explicitly identified the following as POSITIVE MATCHES: {pos_text}. You MUST place these items at the very TOP of your refined ranking (Rank 1 and/or Rank 2). Do NOT drop them, even if the user rejected the overall list."

        prompt = f"""You are a recommendation refinement system for {task_item}s on {task_type}.
A specialized ML model ranked candidates for this user. The user rejected the previous recommendation.
Your job is to REFINE the ranking based on the user's feedback.

User's Profile & History:
{user_query}

Previous Dialogue & User's Critique:
{dialogue_hist}

ML Model's Current Ranking (most recommended first):
{ranked_display}

INSTRUCTIONS:{positive_items_text}
1. CRITICAL: Keep POSITIVE MATCHES (items the user explicitly praised) near the TOP. Any item identified as a POSITIVE MATCH in the critique MUST be promoted to Rank 1 or Rank 2. Do NOT move it down.
2. Push DOWN items the user explicitly rejected as negative noise or a clear mismatch.
3. For items not mentioned in the critique: use the user's review history and the item categories/descriptions above to determine the best order.
4. Output ONLY a JSON: {{"ranked_items": [list of ALL {len(cans_to_rank)} Target_Names in refined order], "explanation": "brief reason"}}
5. MANDATORY COUNT CHECK: Before outputting, verify that your ranked_items list contains EXACTLY {len(cans_to_rank)} items. If it has fewer, you have made an error — add the missing items at the end. Outputting fewer than {len(cans_to_rank)} items is considered a FAILED response.
6. Do NOT add text outside the JSON."""

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
7. MANDATORY COUNT CHECK: Before outputting, verify that your ranked_items list contains EXACTLY {len(cans_to_rank)} items. If it has fewer, you have made an error — add the missing Target_Names at the end in their original order. Outputting fewer than {len(cans_to_rank)} items is considered a FAILED response.
8. Use Target_Names exactly as shown. Do not introduce new items."""

    # ── DIALOGUE LOGGING TO FILE (for paper) ──
    import os
    import threading
    _log_dir  = output_dir or data.get('output_dir', os.path.join(os.path.dirname(__file__), 'output'))
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, 'reranker_dialogue_log.txt')

    # Module-level lock: ensures each block (prompt or response) is written
    # atomically even when the pipeline processes multiple users in parallel.
    if not hasattr(_llm_rerank, '_file_lock'):
        _llm_rerank._file_lock = threading.Lock()

    round_label = ("FEEDBACK REFINEMENT ROUND" if len(memory) > 0 else "INITIAL RANKING ROUND")
    SEP  = "=" * 80
    SEP2 = "-" * 80

    def _log_block(lines):
        """Write a list of lines as one atomic file write, protected by a lock."""
        block = "\n".join(lines) + "\n"
        with _llm_rerank._file_lock:
            with open(_log_path, 'a', encoding='utf-8') as f:
                f.write(block)

    user_id_str = str(data.get('id', data.get('user_id', 'Unknown')))
    round_num   = len(memory) + 1   # Round 1 = initial, Round 2+ = feedback
    # Write the entire prompt block atomically in one lock-protected write
    _log_block([
        f"\n{SEP}",
        f"[RERANKER] User ID: {user_id_str} | ROUND {round_num} | {round_label} | Platform: {task_type} | Item type: {task_item}",
        SEP,
        f"[PROMPT → RERANKER] (Round {round_num})",
        SEP2,
        prompt,
        SEP2,
    ])

    content = "(no response yet)"   # sentinel for error logging
    try:
        # ── PRIMARY: with_structured_output (schema-enforced JSON) ────────────
        # Guarantees valid JSON with correct fields — no parsing needed.
        # Eliminates Extra data / Invalid \escape / Expecting ',' errors.
        structured_llm = llm.with_structured_output(_RankerOutput)
        result         = structured_llm.invoke(prompt)
        ranked         = result.ranked_items
        explanation    = result.explanation
        # Reconstruct content string for logging consistency
        content = json.dumps(
            {"ranked_items": ranked, "explanation": explanation},
            ensure_ascii=False, indent=2,
        )

    except Exception as e_structured:
        # ── FALLBACK: plain invoke + 3-layer parser ───────────────────────────
        # Activated if with_structured_output is unavailable or raises.
        _log_block([f"[RERANKER WARNING] structured_output failed ({type(e_structured).__name__}: {e_structured}), falling back to 3-layer parser"])
        try:
            response = llm.invoke(prompt)
            content  = response.content if hasattr(response, 'content') else str(response)

            content_clean = content.replace("`" * 3 + "json", "").replace("`" * 3, "").strip()

            # Layer 1: balanced brace extraction (fixes Extra data)
            def _first_json_object(text):
                start = text.find('{')
                if start == -1: return None
                depth, in_str, esc = 0, False, False
                for i, ch in enumerate(text[start:], start):
                    if esc: esc = False; continue
                    if ch == '\\' and in_str: esc = True; continue
                    if ch == '"': in_str = not in_str
                    if not in_str:
                        if ch == '{': depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0: return text[start:i + 1]
                return None

            json_str = _first_json_object(content_clean)
            if not json_str:
                raise ValueError(f"No JSON object found: {content_clean[:200]}")

            # Layer 2: progressive JSON repairs (fixes Invalid \escape, curly quotes)
            parsed, last_err = None, None
            for repair in [
                lambda s: json.loads(s),
                lambda s: json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)),
                lambda s: json.loads(s.replace('\u201c', '"').replace('\u201d', '"')),
                lambda s: json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\',
                                            s.replace('\u201c', '"').replace('\u201d', '"'))),
            ]:
                try: parsed = repair(json_str); break
                except Exception as e: last_err = e

            # Layer 3: regex ranked_items extraction (fixes Expecting ',' delimiter)
            if parsed is None:
                m = re.search(r'"ranked_items"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
                if m:
                    extracted = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
                    if extracted:
                        expl_m = re.search(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str)
                        parsed = {'ranked_items': extracted,
                                  'explanation': expl_m.group(1) if expl_m else '(explanation parse failed)'}
                if parsed is None:
                    raise last_err

            ranked      = parsed.get('ranked_items', [])
            explanation = parsed.get('explanation', '')

        except Exception as e:
            _log_block([f"[RERANKER ERROR] USER ID {user_id_str} {e}", SEP + "\n"])
            _log_block([f"[RERANKER ERROR] {content}", SEP + "\n"])
            return candidate_names, "Failed to parse LLM output."

    # ── Warn if LLM returned fewer items than requested ──────────────────────
    n_returned = len([r for r in ranked if r in set(cans_to_rank)])
    incomplete_warning = ""
    if n_returned < len(cans_to_rank):
        incomplete_warning = (
            f"[WARNING] LLM returned only {n_returned}/{len(cans_to_rank)} valid items "
            f"(ignored MANDATORY COUNT CHECK). Missing items will be appended in ML order."
        )

    # Write the entire response block atomically in one lock-protected write
    log_lines = [
        f"[RERANKER RESPONSE] User ID: {user_id_str} ",
        SEP2,
        content,
        SEP2,
        f"[PARSED] Explanation: {explanation}",
        f"[PARSED] Ranked list ({len(ranked)} items): {ranked}",
    ]
    if incomplete_warning:
        log_lines.append(incomplete_warning)
    log_lines.append(SEP + "\n")
    _log_block(log_lines)

    valid_set    = set(cans_to_rank)
    ranked_valid = [r for r in ranked if r in valid_set]
    ranked_set   = set(ranked_valid)
    for n in cans_to_rank:
        if n not in ranked_set: ranked_valid.append(n)
    if len(candidate_names) > max_candidates: ranked_valid += candidate_names[max_candidates:]

    return ranked_valid, explanation

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

    @classmethod
    def from_shared(cls, shared: dict, llm=None, mode: str='embed_only', enabled: bool=True, top_llm: int=20, output_dir: str=None) -> "Reranker":
        return cls(embedding_fn=shared.get('embedding_function'), llm=llm, mode=mode, enabled=enabled, top_llm=top_llm, output_dir=output_dir)