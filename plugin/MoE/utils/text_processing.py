"""
moe_early_fusion/plugin/MoE/utils/text_processing.py
─────────────────────────────────────────────────────
Utility functions để xử lý và chuẩn hoá text từ review history và item metadata
trước khi đưa vào LLM Reranker prompt.

Public API:
    clean_review_text(text)             → str
    build_review_history(data, id2name) → str   (đã filter candidate leak, truncate)
    extract_item_text(info, dataset)    → str   (dataset-specific: amazon/goodreads/yelp)
"""

import re
import html
from typing import Dict, Optional

# ── Optional tiktoken (shared với reranker.py qua import) ────────────────────
try:
    import tiktoken as _tiktoken
    _TIKTOKEN_ENC = _tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None

# ─────────────────────────────────────────────────────────────────────────────
# REVIEW TEXT CLEANER
# ─────────────────────────────────────────────────────────────────────────────

def clean_review_text(text: str) -> str:
    """Strip HTML tags, normalize unicode/escape lỗi, và collapse whitespace.

    Xử lý theo thứ tự:
      1. Unescape HTML entities  (&amp; &lt; &gt; &#x27; &quot; ...)
      2. Xóa toàn bộ HTML tags  (<br> <br/> <p> <b> ...)
      3. Xóa replacement char \\ufffd và null byte \\x00
      4. Sửa stray backslash-escape không hợp lệ  (\\' → ', \\_ → _)
      5. Collapse khoảng trắng / dòng trống thừa
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ''

    # 1. HTML entities
    text = html.unescape(text)
    # 2. HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 3. Bad unicode
    text = text.replace('\ufffd', '').replace('\x00', '')
    # 4. Stray backslash-escapes (giữ lại \n, \t, các escape JSON hợp lệ)
    text = re.sub(r'\\([^"\\/bfnrtu\n\t])', r'\1', text)
    # 5. Whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# REVIEW HISTORY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

# Fields hoàn toàn bị loại bỏ (noise / metadata dư thừa)
_STRIP_FIELDS: frozenset = frozenset({
    'date_added', 'date_updated', 'source', 'type',
    'timestamp', 'image', 'images', 'sub_item_id', 'date',
    'review_id', 'user_id',
    # amazon
    'helpful_vote', 'verified_purchase', 'title',
    # goodreads
    'n_votes', 'n_comments', 'read_at', 'started_at',
    # yelp
    'useful', 'funny', 'cool',
})


def build_review_history(
    data: dict,
    id2name: Optional[Dict[int, str]] = None,
    token_limit: int = 8000,
) -> str:
    """Xây dựng chuỗi lịch sử review của user để đưa vào prompt LLM.

    - Lọc review của candidate items (tránh GT leakage).
    - Chỉ giữ lại: item_name [item_id], stars, categories, text.
    - Clean text (strip HTML, fix unicode).
    - Truncate theo token_limit (tiktoken cl100k_base hoặc char fallback).

    Args:
        data:        dict chứa keys 'reviews', 'cans', 'id2rawid',
                     'interaction_tool', 'seq_str'.
        id2name:     mapping inner_id → display_name (từ graph).
        token_limit: số token tối đa cho phần review history.

    Returns:
        Chuỗi text đã xử lý, sẵn sàng nhúng vào prompt.
    """
    reviews      = data.get('reviews', [])
    id2rawid     = data.get('id2rawid', {})
    interaction_tool = data.get('interaction_tool')

    # ── Fallback: seq_str nếu không có reviews ──────────────────────────────
    if not reviews:
        seq_str = data.get('seq_str', '') or ''
        if seq_str and seq_str.strip() and seq_str != 'Empty History':
            words = seq_str.split()
            return f"User history items: {' '.join(words[-80:])}"
        return ''

    # ── Tập candidate raw_ids cần loại bỏ (GT leakage guard) ────────────────
    candidate_ids: set = set()
    for inner_id in data.get('cans', []):
        raw_id = id2rawid.get(inner_id)
        if raw_id:
            candidate_ids.add(str(raw_id))

    # ── Mapping raw_id → display name ────────────────────────────────────────
    rawid2name: dict = {}
    if id2name:
        rawid2name = {
            str(raw): id2name.get(inner)
            for inner, raw in id2rawid.items()
            if inner in id2name
        }

    # ── Filter + clean từng review ───────────────────────────────────────────
    filtered: list = []
    for r in reviews:
        item_id_str = str(r.get('item_id', ''))

        # Bỏ qua nếu là candidate item
        if candidate_ids and item_id_str in candidate_ids:
            continue

        # Resolve item_name
        item_name = rawid2name.get(item_id_str) or r.get('item_name')

        # Lấy / inject categories
        categories = r.get('categories')
        if interaction_tool and not categories:
            try:
                fetched = interaction_tool.get_item(item_id=item_id_str)
                if fetched:
                    cats = fetched.get('categories')
                    if cats:
                        if isinstance(cats, list):
                            cats = ', '.join(str(c) for c in cats)
                        categories = cats
            except Exception:
                pass
        elif isinstance(categories, list):
            categories = ', '.join(str(c) for c in categories)

        # Tạo dict gọn: item_name [item_id], stars, categories, text
        display_name = f"{item_name} [{item_id_str}]" if item_name and item_id_str else item_name
        ordered: dict = {}
        if display_name:
            ordered['item_name'] = display_name
        stars = r.get('stars')
        if stars is not None:
            ordered['stars'] = stars
        if categories:
            ordered['categories'] = categories
        for text_key in ('text', 'review_text'):
            raw_text = r.get(text_key)
            if raw_text:
                ordered['text'] = clean_review_text(raw_text)
                break

        filtered.append(ordered)

    # Mới nhất lên đầu (danh sách hiện tại là chronological)
    history_str = str(list(reversed(filtered)))

    # ── Truncate ─────────────────────────────────────────────────────────────
    try:
        if _TIKTOKEN_ENC is not None:
            encoded = _TIKTOKEN_ENC.encode(history_str)
            if len(encoded) > token_limit:
                history_str = _TIKTOKEN_ENC.decode(encoded[:token_limit])
        else:
            history_str = history_str[:token_limit * 4]   # ~4 chars/token rough estimate
    except Exception:
        history_str = history_str[:token_limit * 4]

    return f"\n{history_str}"


# ─────────────────────────────────────────────────────────────────────────────
# ITEM TEXT EXTRACTOR  (dataset-specific)
# ─────────────────────────────────────────────────────────────────────────────

# Yelp: các attribute key boolean chỉ báo tính năng tích cực khi True
_YELP_BOOL_FEATURE_KEYS: frozenset = frozenset({
    'OutdoorSeating', 'HasTV', 'WiFi', 'BikeParking',
    'WheelchairAccessible', 'HappyHour', 'Caters',
    'RestaurantsDelivery', 'RestaurantsTakeOut',
    'RestaurantsReservations', 'GoodForKids', 'DogsAllowed',
    'BusinessAcceptsCreditCards',
})

# Keys cần fetch từ interaction_tool cho mỗi item candidate
ITEM_FETCH_KEYS: list = [
    'item_id', 'name', 'stars', 'review_count', 'categories',
    'title', 'average_rating', 'rating_number', 'description',
    'features', 'ratings_count', 'title_without_series',
    'popular_shelves', 'format', 'attributes', 'is_open',
]


def _normalize_categories(cats) -> str:
    if isinstance(cats, list):
        return ', '.join(str(c) for c in cats)
    return str(cats) if cats else ''


def _extract_sentences(text: str, n: int = 2, max_chars: int = 300) -> str:
    """Tách tối thiểu n câu đầu từ đoạn text, cắt nếu quá max_chars."""
    if not isinstance(text, str) or not text.strip():
        return ''
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s for s in sents if s.strip()]
    result = ' '.join(sents[:n])
    if len(result) > max_chars:
        result = result[:max_chars] + '...'
    return result


def extract_item_text(info: dict, dataset: str) -> str:
    """Trả về chuỗi mô tả ngắn gọn, có thông tin cho một item candidate.

    Args:
        info:    dict item đã fetch (có key 'Target_Name' + metadata).
        dataset: 'amazon' | 'goodreads' | 'yelp' | khác.

    Returns:
        Chuỗi dạng "key: value, key: value, ..."
    """
    details: list = []

    # ── Amazon ───────────────────────────────────────────────────────────────
    if dataset == 'amazon':
        rating = info.get('average_rating') or info.get('stars')
        if rating:
            details.append(f"rating: {rating}")

        rating_cnt = info.get('rating_number') or info.get('review_count')
        if rating_cnt:
            details.append(f"reviews: {rating_cnt}")

        cats = _normalize_categories(info.get('categories', ''))
        if cats:
            details.append(f"categories: {cats[:120]}")

        # Features (ưu tiên) → Description (fallback), lấy ≥ 2 câu
        content_str = ''
        features = info.get('features')
        if features and isinstance(features, list):
            bullets = [str(f).strip() for f in features if str(f).strip()][:3]
            content_str = '; '.join(bullets)

        if not content_str:
            desc = info.get('description')
            if desc and isinstance(desc, list):
                desc = ' '.join(str(d) for d in desc if d)
            content_str = _extract_sentences(desc, n=2, max_chars=300)

        if content_str:
            if len(content_str) > 300:
                content_str = content_str[:300] + '...'
            details.append(f"info: {content_str}")

    # ── Goodreads ─────────────────────────────────────────────────────────────
    elif dataset == 'goodreads':
        rating = info.get('average_rating')
        if rating:
            details.append(f"rating: {rating}")

        cnt = info.get('ratings_count') or info.get('review_count')
        if cnt:
            details.append(f"ratings: {cnt}")

        fmt = info.get('format')
        if fmt:
            details.append(f"format: {fmt}")

        shelves = info.get('popular_shelves')
        if shelves and isinstance(shelves, list):
            top3 = [s.get('name', '') for s in shelves[:3] if isinstance(s, dict)]
            top3 = [s for s in top3 if s]
            if top3:
                details.append(f"shelves: {', '.join(top3)}")

        desc_str = _extract_sentences(info.get('description', ''), n=2, max_chars=300)
        if desc_str:
            details.append(f"description: {desc_str}")

    # ── Yelp ─────────────────────────────────────────────────────────────────
    elif dataset == 'yelp':
        stars = info.get('stars')
        if stars:
            details.append(f"stars: {stars}")

        rc = info.get('review_count')
        if rc:
            details.append(f"reviews: {rc}")

        cats = _normalize_categories(info.get('categories', ''))
        if cats:
            details.append(f"categories: {cats[:120]}")

        is_open = info.get('is_open')
        if is_open is not None:
            details.append(f"open: {'yes' if is_open else 'no'}")

        # Boolean attributes → positive feature tags
        attrs = info.get('attributes')
        if attrs and isinstance(attrs, dict):
            pos_tags = []
            for k, v in attrs.items():
                if k not in _YELP_BOOL_FEATURE_KEYS:
                    continue
                flag = v if isinstance(v, bool) else str(v).strip().lower() == 'true'
                if flag:
                    # camelCase → "Camel Case"
                    readable = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', k)
                    pos_tags.append(readable)
            if pos_tags:
                details.append(f"features: {', '.join(pos_tags)}")

    # ── Generic fallback ──────────────────────────────────────────────────────
    else:
        for k in ['average_rating', 'stars', 'rating_number', 'ratings_count', 'review_count', 'categories']:
            v = info.get(k)
            if v:
                details.append(f"{k}: {v}")
        desc = info.get('description')
        if isinstance(desc, str) and desc.strip():
            desc = desc[:200] + '...' if len(desc) > 200 else desc
            details.append(f"description: {desc}")

    return ', '.join(details) if details else 'No additional info'
