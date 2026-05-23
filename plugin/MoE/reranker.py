"""
moe_fusion/reranker.py
───────────────────────
Feedback Loop 4.0:
  - LLM Reranker được phép chọn lại item cũ nếu nó cho rằng User Simulator sai lầm.
"""

import json
import re
import threading
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
import numpy as np
import math

from utils.text_processing import (
    build_review_history,
    extract_item_text,
    ITEM_FETCH_KEYS,
)

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
# LLM trả về INDEX (1-based) thay vì tên → tránh hoàn toàn name-mismatch
class _RankerOutput(BaseModel):
    ranked_indices: List[int]   # 1-based indices tương ứng với thứ tự trong ML ranking
    explanation: str


def rank_to_score(ranked_list: List[str]) -> Dict[str, float]:
    # Điểm: 1.0, 0.63, 0.5, 0.43, 0.38...
    return {item: 1.0 / math.log2(rank + 2) for rank, item in enumerate(ranked_list)}

def _build_user_query(
    data: dict,
    candidate_names: list = None,
    id2name: Dict[int, str] = None,
) -> str:
    """Wrapper: build user review history string.
    Logic chi tiết nằm trong utils/text_processing.py::build_review_history.
    """
    return build_review_history(data, id2name=id2name)


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
    output_dir: str = None,
    # ── Token budgets (tổng prompt ≤ ~8k tokens để tránh LengthFinish / Timeout) ──
    _QUERY_TOKEN_CAP: int = 3000,   # user_query
    _ITEMS_TOKEN_CAP: int = 4000,   # ranked_display
) -> Tuple[List[str], str]:
    cans_to_rank = candidate_names[:max_candidates]
    # Ưu tiên data['dataset'] (inject bởi moe_rec_agent.py), fallback về output_dir path
    dataset = data.get('dataset') or next(
        (d for d in ['yelp', 'amazon', 'goodreads']
         if d in str(data.get('output_dir', ''))),
        'amazon'
    )
    task_type = {"goodreads": "Goodreads", "yelp": "Yelp", "amazon": "Amazon"}.get(dataset, "Platform")
    task_item = {"goodreads": "book", "yelp": "business", "amazon": "product"}.get(dataset, "item")

    interaction_tool = data.get('interaction_tool')
    id2rawid         = data.get('id2rawid', {})
    item_list_info   = []

    if interaction_tool and name2id and id2rawid:
        for name in cans_to_rank:
            inner_id  = name2id.get(name)
            raw_id    = id2rawid.get(inner_id)
            info_dict = {'Target_Name': name}
            if raw_id:
                try:
                    fetched = _cached_get_item(interaction_tool, raw_id)
                    if fetched:
                        for k in ITEM_FETCH_KEYS:
                            if k in fetched:
                                info_dict[k] = fetched[k]
                except Exception:
                    pass
            item_list_info.append(info_dict)
    else:
        item_list_info = [{'Target_Name': n} for n in cans_to_rank]

    # ── FORMAT ITEMS AS NUMBERED ML RANKING ──────────────────────────────
    numbered_lines = [
        f'  #{idx}: "{info.get("Target_Name", "Unknown")}" — {extract_item_text(info, dataset)}'
        for idx, info in enumerate(item_list_info, 1)
    ]
    ranked_display = "\n".join(numbered_lines)

    # ── Cap ranked_display để tránh prompt quá dài (LengthFinish / Timeout) ──
    try:
        if _TIKTOKEN_ENC is not None:
            encoded_items = _TIKTOKEN_ENC.encode(ranked_display)
            if len(encoded_items) > _ITEMS_TOKEN_CAP:
                ranked_display = _TIKTOKEN_ENC.decode(encoded_items[:_ITEMS_TOKEN_CAP])
        else:
            ranked_display = ranked_display[:(_ITEMS_TOKEN_CAP * 4)]
    except Exception:
        ranked_display = ranked_display[:(_ITEMS_TOKEN_CAP * 4)]

    # ── Cap user_query (đã được build bởi build_review_history, nhưng cap lại ở đây
    #    để đảm bảo tổng prompt ≤ ~8k tokens ngay cả khi caller truyền query dài) ──
    try:
        if _TIKTOKEN_ENC is not None:
            encoded_q = _TIKTOKEN_ENC.encode(user_query)
            if len(encoded_q) > _QUERY_TOKEN_CAP:
                user_query = _TIKTOKEN_ENC.decode(encoded_q[:_QUERY_TOKEN_CAP]) + "\n[... history truncated ...]"
        else:
            user_query = user_query[:(_QUERY_TOKEN_CAP * 4)]
    except Exception:
        user_query = user_query[:(_QUERY_TOKEN_CAP * 4)]

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

REFINEMENT INSTRUCTIONS:{positive_items_text}
1. CRITICAL: Keep POSITIVE MATCHES (items the user explicitly praised) near the TOP. Any item identified as a POSITIVE MATCH in the critique MUST be promoted to Rank 1 or Rank 2.
2. Push DOWN items the user explicitly rejected as negative noise or a clear mismatch.
3. For items not mentioned: use the user's review history and item descriptions to determine the best order.
4. Output ranked_indices as the item #N numbers from the ML ranking above, ordered from best to worst.
   Example: if #3 is best, then #1, then #2 → ranked_indices: [3, 1, 2, ...]. Include all {len(cans_to_rank)} indices.
5. Provide a brief explanation of your changes."""

    else:
        prompt = f"""You are a recommendation refinement system for {task_item}s on {task_type}.
A specialized ML recommendation model has already ranked candidate {task_item}s for this user using multiple signals (sequential behavior patterns, collaborative filtering, and content similarity). Your job is to REFINE this ranking — not rebuild it from scratch.
The ML ranking is statistically reliable. Make MINIMAL adjustments only when clearly justified.

User's Profile & Review History:
{user_query}

ML Model's Ranking (most recommended → least recommended):
{ranked_display}

REFINEMENT INSTRUCTIONS:
1. Only swap items if you see CLEAR, SPECIFIC evidence in the user's review history that a lower-ranked item better matches their preferences.
2. Key signals: category/genre alignment, rating patterns, specific features the user mentions in reviews.
3. Output ranked_indices as the item #N numbers from the ML ranking above, ordered from best to worst.
   Example: if #3 is best, then #1, then #2 → ranked_indices: [3, 1, 2, ...]. Include all {len(cans_to_rank)} indices.
4. Provide a brief explanation of your changes, or 'ML ranking preserved' if no changes were needed."""

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

    content     = "(no response yet)"   # sentinel for error logging
    ranked      = list(cans_to_rank)     # fallback: giữ nguyên ML order
    explanation = "(no explanation)"
    try:
        # ── PRIMARY: with_structured_output (schema-enforced JSON) ────────────
        # LLM trả về ranked_indices (List[int], 1-based) → map về tên gốc.
        # Loại bỏ hoàn toàn name-mismatch: index không bao giờ sai.
        #
        # CRITICAL: bind max_tokens TRƯỚC with_structured_output.
        # ChatOpenAI.max_tokens KHÔNG được forward tự động qua vLLM's structured-
        # output endpoint → model sinh vô hạn token → LengthFinishReasonError.
        # 512 tokens = 10 indices (~30 tok) + explanation (~100 tok) + JSON (~50 tok)
        # với buffer dư. Đủ dùng, không bao giờ trigger unbounded generation.
        _MAX_OUTPUT = 512
        structured_llm  = llm.bind(max_tokens=_MAX_OUTPUT).with_structured_output(_RankerOutput)
        result          = structured_llm.invoke(prompt)
        raw_indices     = result.ranked_indices   # e.g. [3, 1, 2, 5, 4, ...]
        explanation     = result.explanation

        # ── Map index → tên (1-based, clamp ngoài khoảng) ─────────────────
        n = len(cans_to_rank)
        seen   = set()
        ranked = []
        for idx in raw_indices:
            if 1 <= idx <= n and idx not in seen:
                ranked.append(cans_to_rank[idx - 1])
                seen.add(idx)
        # Append missing (theo ML order) nếu LLM bỏ sót
        for i, name in enumerate(cans_to_rank, 1):
            if i not in seen:
                ranked.append(name)

        # Reconstruct content string for logging consistency
        content = json.dumps(
            {"ranked_indices": raw_indices, "ranked_items": ranked, "explanation": explanation},
            ensure_ascii=False, indent=2,
        )

    except Exception as e_structured:
        # ── Phân loại lỗi để log rõ hơn ─────────────────────────────────────
        err_type = type(e_structured).__name__
        err_msg  = str(e_structured)

        # LengthFinishReasonError: LLM bị cắt vì max_tokens quá nhỏ so với prompt
        # → nguyên nhân gốc: prompt quá dài; đã được xử lý bằng token cap ở trên.
        # APITimeoutError: vLLM bị ngộp prompt dài; cũng được giảm thiểu bởi cap.
        if 'LengthFinishReason' in err_type or 'LengthFinish' in err_msg:
            warn = (
                f"[RERANKER WARNING] structured_output failed ({err_type}: {err_msg}), "
                "keeping ML order  ← Prompt still too long after cap; consider reducing _ITEMS_TOKEN_CAP further."
            )
        elif 'Timeout' in err_type or 'timeout' in err_msg.lower():
            warn = (
                f"[RERANKER WARNING] structured_output failed ({err_type}: {err_msg}), "
                "keeping ML order  ← vLLM timeout; prompt may still be too long."
            )
        else:
            warn = f"[RERANKER WARNING] structured_output failed ({err_type}: {err_msg}), keeping ML order"

        _log_block([warn])

    # ── Warn nếu LLM trả về index ngoài khoảng hoặc thiếu ───────────────────
    n_valid = len([r for r in ranked if r in set(cans_to_rank)])
    incomplete_warning = ""
    if n_valid < len(cans_to_rank):
        incomplete_warning = (
            f"[WARNING] Index mapping yielded only {n_valid}/{len(cans_to_rank)} valid items. "
            f"Missing items appended in ML order."
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

    # ranked đã đầy đủ (append missing đã xử lý trong try block)
    if len(candidate_names) > max_candidates:
        ranked += candidate_names[max_candidates:]

    return ranked, explanation

class Reranker:
    def __init__(self, embedding_fn=None, llm=None, mode: str='embed_only', enabled: bool=True, top_llm: int=10, output_dir: str=None):
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
    def from_shared(cls, shared: dict, llm=None, mode: str='embed_only', enabled: bool=True, top_llm: int=10, output_dir: str=None) -> "Reranker":
        return cls(embedding_fn=shared.get('embedding_function'), llm=llm, mode=mode, enabled=enabled, top_llm=top_llm, output_dir=output_dir)