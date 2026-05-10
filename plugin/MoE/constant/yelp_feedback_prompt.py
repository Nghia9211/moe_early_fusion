"""
yelp_feedback_prompt.py
────────────────────────
LLM parser prompt cho FeedbackScoreAdjuster — dataset: Yelp (local businesses).

Categories:
  POSITIVE            → boost
  TRUE_NOISE          → hard penalty  (×0.80)
  PERSONAL_PREFERENCE → soft penalty  (×0.95)
  LOCATION_MISMATCH   → soft penalty  (×0.95)  — venue ở xa / đóng cửa / không accessible
"""

# ── Blanket-rejection phrases ────────────────────────────────────────────────
BLANKET_REJECTION_PHRASES = [
    "all of the venues",
    "all of the businesses",
    "all 5 venues",
    "none of the",
    "all recommended",
    "all items",
    "does not include any",
    "does not contain any",
]

# ── Categories that carry NO penalty ────────────────────────────────────────
IGNORE_CATEGORIES = set()   # Yelp không có category nào bị ignore hoàn toàn

# ── Categories that carry a SOFT penalty ────────────────────────────────────
SOFT_CATEGORIES = {"PERSONAL_PREFERENCE", "LOCATION_MISMATCH"}

# ── LLM prompt template ─────────────────────────────────────────────────────
# Placeholders: {item_list}, {user_reason}
FEEDBACK_PARSER_PROMPT = """Analyze the following user feedback for a local business / venue recommendation system.

Recommended venues (in order):
{item_list}

User feedback:
{user_reason}

For each venue mentioned, classify it into ONE of these categories:
- "POSITIVE": user finds the venue appealing.
- "PERSONAL_PREFERENCE": venue is a legitimate business but does not match the user's taste, price range, or service type.
- "LOCATION_MISMATCH": venue is in a different city / too far / permanently closed.

Crucial rules:
- All recommended venues are real, legitimate businesses. There is NO "absurd" or "data error" venue in the list. DO NOT invent a "TRUE_NOISE" category.
- Use EXACT venue names from the recommended list.
- Classify EVERY venue mentioned by the user, even if the feedback seems negative.
- Output ONLY valid JSON (no extra text) in this exact format:
{{
  "positive": ["venue name"],
  "negative": [["venue name", "PERSONAL_PREFERENCE"], ...]
}}"""