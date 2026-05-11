"""
feedback_adjuster.py — v3.4 (Genre-aware penalties)
────────────────────────────────────────────────────────────
v3.4 thay đổi so với v3.3:
  - Thêm _genre_mismatch_counts riêng cho GENRE_MISMATCH (Amazon v2)
  - genre_mismatch_penalty mặc định 0.85 (yếu hơn TRUE_NOISE 0.80)
  - adjust() tích thêm factor genre_mismatch vào score
  - summary() và reset() cập nhật theo

Prompt templates cho LLM parser được tách ra file riêng:
  - amazon_feedback_prompt.py
  - goodreads_feedback_prompt.py
  - yelp_feedback_prompt.py

Mỗi file export:
  FEEDBACK_PARSER_PROMPT      : str   — template với {item_list}, {user_reason}
  BLANKET_REJECTION_PHRASES   : list  — phrases kích hoạt blanket rejection
  SOFT_CATEGORIES             : set   — categories → soft penalty (×0.95)
  IGNORE_CATEGORIES           : set   — categories → no penalty at all
  HARD_CATEGORIES             : set   — (optional) categories → hard penalty riêng
                                        nếu không có thì chỉ TRUE_NOISE mới hard

Phân loại feedback:
  TRUE_NOISE            → phạt mạnh        (×0.80) — non-gaming item / hoàn toàn vô nghĩa
  GENRE_MISMATCH        → phạt vừa         (×0.85) — game nhưng genre cách xa (Amazon v2)
  PERSONAL_PREFERENCE   → soft             (×0.95)
  PLATFORM_MISMATCH     → ignored          (Amazon only)
  EDITION_MISMATCH      → soft             (Goodreads)
  LOCATION_MISMATCH     → soft             (Yelp)
  LIFESTYLE_MISMATCH    → hard/soft        (Yelp v2, tùy config)
  Nếu không có LLM      → fallback về regex.
"""

from typing import Dict, List, Set, Optional
import importlib
import importlib.util
import json
import os


# ─────────────────────────────────────────────────────────────────────────────
# Dataset detection + prompt module loader
# ─────────────────────────────────────────────────────────────────────────────

_DATASET_MODULE_MAP = {
    "amazon":    "amazon_feedback_prompt",
    "goodreads": "goodreads_feedback_prompt",
    "yelp":      "yelp_feedback_prompt",
}


def _detect_dataset(data_dir: str = None, dataset: str = None) -> str:
    """
    Phát hiện dataset từ tên dataset hoặc data_dir path.
    Returns: 'amazon' | 'goodreads' | 'yelp'
    """
    combined = " ".join([dataset or "", data_dir or ""]).lower()
    if "goodreads" in combined:
        return "goodreads"
    if "yelp" in combined:
        return "yelp"
    return "amazon"


def _load_prompt_module(dataset: str):
    """
    Import động prompt module tương ứng với dataset.

    Thứ tự tìm kiếm:
      1. <this_dir>/constant/<module>.py   ← ưu tiên cao nhất (production layout)
      2. <this_dir>/<module>.py            ← cùng thư mục (legacy / flat layout)
      3. sys.path thông thường             ← fallback (installed / test)

    Returns module object với attributes:
      FEEDBACK_PARSER_PROMPT, BLANKET_REJECTION_PHRASES,
      SOFT_CATEGORIES, IGNORE_CATEGORIES
    Optional attributes (v2):
      HARD_CATEGORIES   — set of categories treated as hard penalty
                          nếu không có, mặc định chỉ TRUE_NOISE là hard
    """
    module_name = _DATASET_MODULE_MAP.get(dataset, _DATASET_MODULE_MAP["amazon"])
    this_dir    = os.path.dirname(os.path.abspath(__file__))

    search_paths = [
        os.path.join(this_dir, "constant", f"{module_name}.py"),
        os.path.join(this_dir, f"{module_name}.py"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            spec   = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"[FeedbackAdjuster] Loaded prompt module: {path}")
            return module

    print(f"[FeedbackAdjuster] Loading prompt module via sys.path: {module_name}")
    return importlib.import_module(module_name)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackScoreAdjuster:

    def __init__(
        self,
        negative_penalty:       float = 0.80,
        positive_boost:         float = 1.20,
        soft_penalty:           float = 0.90,
        genre_mismatch_penalty: float = 0.85,   # v3.4: riêng cho GENRE_MISMATCH
        max_penalty_rounds:     int   = 3,
        safe_rank_cutoff:       int   = 2,       # Rank 1-2 không bị penalize
        llm_client              = None,
        output_dir:             str   = None,
        dataset:                str   = None,
        data_dir:               str   = None,
    ):
        self.negative_penalty       = negative_penalty
        self.positive_boost         = positive_boost
        self.soft_penalty           = soft_penalty
        self.genre_mismatch_penalty = genre_mismatch_penalty  # v3.4
        self.max_penalty_rounds     = max_penalty_rounds
        self.safe_rank_cutoff       = safe_rank_cutoff
        self.llm_client             = llm_client
        self.output_dir             = output_dir
        self.dataset                = _detect_dataset(data_dir=data_dir, dataset=dataset)

        # ── Load prompt config từ file riêng ────────────────────────────
        _mod = _load_prompt_module(self.dataset)
        self._prompt_template   : str  = _mod.FEEDBACK_PARSER_PROMPT
        self._blanket_phrases   : list = _mod.BLANKET_REJECTION_PHRASES
        self._soft_categories   : set  = _mod.SOFT_CATEGORIES
        self._ignore_categories : set  = _mod.IGNORE_CATEGORIES
        # v3.4: HARD_CATEGORIES optional — nếu module không định nghĩa, default rỗng
        self._hard_categories   : set  = getattr(_mod, 'HARD_CATEGORIES', set())

        self._negative_counts:       Dict[str, int] = {}  # TRUE_NOISE → ×0.80
        self._positive_counts:       Dict[str, int] = {}  # POSITIVE   → ×1.20
        self._soft_counts:           Dict[str, int] = {}  # PERSONAL_PREFERENCE etc. → ×0.90
        self._genre_mismatch_counts: Dict[str, int] = {}  # GENRE_MISMATCH → ×0.85 (v3.4)

        print(f"[FeedbackAdjuster] Ready | dataset='{self.dataset}' | "
              f"module='{_DATASET_MODULE_MAP[self.dataset]}' | "
              f"genre_mismatch_penalty={self.genre_mismatch_penalty}")

    # ─────────────────────────────────────────────────────────────────────
    # Update từ 1 round feedback
    # ─────────────────────────────────────────────────────────────────────

    def update_from_memory(self, user_reason: str, rec_item_list: List[str]):
        if not rec_item_list:
            return

        reason_lower      = (user_reason or "").lower()
        safe_items        = set(rec_item_list[:self.safe_rank_cutoff])
        penalizable_items = rec_item_list[self.safe_rank_cutoff:]

        # ── 1. LLM parsing ───────────────────────────────────────────────
        llm_positive        = set()
        llm_negative        = set()   # TRUE_NOISE → hard penalty ×0.80
        llm_genre_mismatch  = set()   # GENRE_MISMATCH → penalty ×0.85 (v3.4)
        llm_soft            = set()   # soft-penalty categories → ×0.90
        llm_ignored         = set()   # no-penalty categories

        if self.llm_client is not None:
            try:
                parsed = self._llm_parse_feedback(user_reason, rec_item_list)
                llm_positive.update(
                    item.lower().strip() for item in parsed.get("positive", [])
                )
                for entry in parsed.get("negative", []):
                    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                        continue
                    item_name, category = entry[0], entry[1]
                    item_lower = item_name.lower().strip()

                    if category == "TRUE_NOISE":
                        llm_negative.add(item_lower)
                    elif category == "GENRE_MISMATCH":
                        # v3.4: bucket riêng, không merge vào llm_negative
                        llm_genre_mismatch.add(item_lower)
                    elif category in self._ignore_categories:
                        llm_ignored.add(item_lower)
                    elif category in self._soft_categories:
                        llm_soft.add(item_lower)
                    elif category in self._hard_categories:
                        # HARD_CATEGORIES từ module (ngoài TRUE_NOISE/GENRE_MISMATCH)
                        # → fallback về hard penalty giống TRUE_NOISE
                        llm_negative.add(item_lower)
                    else:
                        llm_soft.add(item_lower)   # unknown → safe fallback

                print(
                    f"[FeedbackAdjuster][LLM][{self.dataset}] "
                    f"pos={llm_positive} | true_noise={llm_negative} | "
                    f"genre_mismatch={llm_genre_mismatch} | "
                    f"soft={llm_soft} | ignored={llm_ignored}"
                )
            except Exception as e:
                print(f"[FeedbackAdjuster][LLM] error: {e}, fallback to regex")

        # ── 2. Blanket rejection ─────────────────────────────────────────
        is_blanket = (
            len(llm_negative) == 0
            and len(llm_genre_mismatch) == 0
            and len(llm_soft) == 0
            and any(p in reason_lower for p in self._blanket_phrases)
        )
        if is_blanket:
            for item in rec_item_list[:1]:
                self._soft_counts[item] = self._soft_counts.get(item, 0) + 1
            print(f"[FeedbackAdjuster] Blanket rejection → soft rank-1: {rec_item_list[:1]}")

        # ── 3. Áp dụng LLM hoặc fallback regex ──────────────────────────
        applied_any = False
        if self.llm_client is not None and (
            llm_positive or llm_negative or llm_genre_mismatch or llm_soft
        ):
            # Boost positive (bất kể rank)
            for item in rec_item_list:
                if item.lower() in llm_positive:
                    self._positive_counts[item] = self._positive_counts.get(item, 0) + 1
                    applied_any = True

            # Hard penalize TRUE_NOISE (ngoài safe zone)
            for item in penalizable_items:
                if item.lower() in llm_negative and item not in safe_items:
                    self._negative_counts[item] = self._negative_counts.get(item, 0) + 1
                    applied_any = True

            # v3.4: Genre mismatch penalty (ngoài safe zone)
            for item in penalizable_items:
                if item.lower() in llm_genre_mismatch and item not in safe_items:
                    self._genre_mismatch_counts[item] = (
                        self._genre_mismatch_counts.get(item, 0) + 1
                    )
                    applied_any = True

            # Soft penalize (ngoài safe zone)
            for item in penalizable_items:
                if item.lower() in llm_soft and item not in safe_items:
                    self._soft_counts[item] = self._soft_counts.get(item, 0) + 1
                    applied_any = True

            # llm_ignored → intentionally no counts updated

            parts = []
            if llm_positive:       parts.append(f"✅ Boost: {llm_positive}")
            if llm_negative:       parts.append(f"❌ TRUE_NOISE (×{self.negative_penalty}): {llm_negative}")
            if llm_genre_mismatch: parts.append(f"🎮 GENRE_MISMATCH (×{self.genre_mismatch_penalty}): {llm_genre_mismatch}")
            if llm_soft:           parts.append(f"⚠️  Soft (×{self.soft_penalty}): {llm_soft}")
            if llm_ignored:        parts.append(f"🚫 Ignored: {llm_ignored}")
            if safe_items:         parts.append(f"🛡️  Protected: {safe_items}")
            print(f"[FeedbackAdjuster] [{self.dataset}] | {' | '.join(parts)}")

        if not applied_any:
            # Fallback regex (section-header based, dataset-agnostic)
            positive_items = self._extract_section(
                reason_lower, rec_item_list, "positive matches"
            )
            negative_items = self._extract_section(
                reason_lower, penalizable_items, "negative noise"
            )
            for item in positive_items:
                self._positive_counts[item] = self._positive_counts.get(item, 0) + 1
            for item in negative_items:
                if item not in safe_items:
                    self._negative_counts[item] = self._negative_counts.get(item, 0) + 1
            self._log(positive_items, negative_items, is_blanket, safe_items, rec_item_list)

    # ─────────────────────────────────────────────────────────────────────
    # LLM call
    # ─────────────────────────────────────────────────────────────────────

    def _llm_parse_feedback(self, user_reason: str, rec_item_list: List[str]) -> dict:
        user_reason_trunc = user_reason
        if len(user_reason_trunc) > 4000:
            user_reason_trunc = user_reason_trunc[:4000] + "..."

        item_list_str = "\n".join(f"- {item}" for item in rec_item_list)
        prompt = self._prompt_template.format(
            item_list=item_list_str,
            user_reason=user_reason_trunc,
        )

        log_dir  = self.output_dir or os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'feedback_llm_parser_log.txt')

        def _log(text: str):
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(text + "\n")

        estimated_tokens = len(prompt) // 3.5

        _log("\n" + "=" * 80)
        _log(f"[FEEDBACK LLM PARSER] dataset={self.dataset}")
        _log(f"[STATS] Prompt Length: {len(prompt)} chars | Est. Tokens: ~{int(estimated_tokens)}")
        _log("=" * 80)
        _log("[PROMPT]")
        _log("-" * 80)
        _log(prompt)
        _log("-" * 80)

        try:
            response = self.llm_client.invoke(prompt)
            content  = response.content if hasattr(response, 'content') else str(response)

            _log("[LLM RESPONSE]")
            _log("-" * 80)
            _log(content)
            _log("-" * 80)

            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            parsed_json = json.loads(content)

            _log("[PARSED JSON]")
            _log("-" * 80)
            _log(json.dumps(parsed_json, indent=2))
            _log("=" * 80 + "\n")

            return parsed_json

        except Exception as e:
            _log(f"[ERROR] {e}")
            _log(f"[ERROR DETAILS] Prompt Chars: {len(prompt)} | "
                 f"Est. Tokens: ~{int(len(prompt)//3.5)}")
            _log("=" * 80 + "\n")
            raise

    # ─────────────────────────────────────────────────────────────────────
    # Score adjustment — v3.4: thêm genre_mismatch factor
    # ─────────────────────────────────────────────────────────────────────

    def adjust(self, fused_scores: Dict[str, float]) -> Dict[str, float]:
        if not (
            self._negative_counts
            or self._positive_counts
            or self._soft_counts
            or self._genre_mismatch_counts   # v3.4
        ):
            return fused_scores

        adjusted = {}
        for item, score in fused_scores.items():
            neg   = min(self._negative_counts.get(item, 0),       self.max_penalty_rounds)
            pos   = self._positive_counts.get(item, 0)
            soft  = min(self._soft_counts.get(item, 0),           self.max_penalty_rounds)
            genre = min(self._genre_mismatch_counts.get(item, 0), self.max_penalty_rounds)  # v3.4

            factor = (
                (self.negative_penalty       ** neg)
                * (self.positive_boost       ** pos)
                * (self.soft_penalty         ** soft)
                * (self.genre_mismatch_penalty ** genre)   # v3.4
            )
            adjusted[item] = score * factor

        return adjusted

    # ─────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────

    def _extract_section(
        self,
        reason_lower:    str,
        candidate_items: List[str],
        section_header:  str,
    ) -> List[str]:
        idx = reason_lower.find(section_header)
        if idx == -1:
            return []
        text_after = reason_lower[idx + len(section_header):]
        cut_at = ["positive matches", "negative noise", "decision:"]
        end = len(text_after)
        for marker in cut_at:
            if marker == section_header:
                continue
            pos = text_after.find(marker)
            if 0 < pos < end:
                end = pos
        return [item for item in candidate_items if item.lower() in text_after[:end]]

    def _log(self, positive, negative, blanket, safe_items, rec_list):
        parts = []
        if positive:   parts.append(f"✅ Boost: {positive}")
        if negative:   parts.append(f"❌ Penalize (rank 3-5): {negative}")
        if blanket:    parts.append(f"⚠️  Blanket → soft rank-1 only: {rec_list[:1]}")
        if safe_items: parts.append(f"🛡️  Protected (rank 1-2): {list(safe_items)}")
        if not any([positive, negative, blanket]):
            parts.append("ℹ️  No named items → no adjustment")
        print(f"[FeedbackAdjuster] {' | '.join(parts)}")

    # ─────────────────────────────────────────────────────────────────────
    # Public utils — v3.4: thêm genre_mismatch vào summary/reset
    # ─────────────────────────────────────────────────────────────────────

    def get_rejected_items(self) -> Set[str]:
        return {k for k, v in self._negative_counts.items() if v >= 1}

    def get_genre_mismatched_items(self) -> Set[str]:
        """v3.4: items bị GENRE_MISMATCH penalty."""
        return {k for k, v in self._genre_mismatch_counts.items() if v >= 1}

    def get_praised_items(self) -> Set[str]:
        return {k for k, v in self._positive_counts.items() if v >= 1}

    def summary(self) -> dict:
        return {
            "dataset":               self.dataset,
            "negative_counts":       dict(self._negative_counts),
            "genre_mismatch_counts": dict(self._genre_mismatch_counts),  # v3.4
            "positive_counts":       dict(self._positive_counts),
            "soft_counts":           dict(self._soft_counts),
        }

    def reset(self):
        self._negative_counts.clear()
        self._positive_counts.clear()
        self._soft_counts.clear()
        self._genre_mismatch_counts.clear()   # v3.4