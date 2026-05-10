"""
amazon_feedback_prompt.py
─────────────────────────
LLM parser prompt cho FeedbackScoreAdjuster — dataset: Amazon (video games).

Categories:
  POSITIVE            → boost
  TRUE_NOISE          → hard penalty  (×0.80)
  PERSONAL_PREFERENCE → soft penalty  (×0.95)
  PLATFORM_MISMATCH   → ignored       (no penalty — user owns all consoles)
"""

# ── Blanket-rejection phrases ────────────────────────────────────────────────
BLANKET_REJECTION_PHRASES = [
    "all of the products",
    "all 5 items",
    "none of the",
    "all recommended",
    "all items",
    "does not include any",
    "does not contain any",
]

# ── Categories that carry NO penalty ────────────────────────────────────────
IGNORE_CATEGORIES = {"PLATFORM_MISMATCH"}

# ── Categories that carry a SOFT penalty ────────────────────────────────────
SOFT_CATEGORIES = {"PERSONAL_PREFERENCE", "PLATFORM_MISMATCH"}

# ── LLM prompt template ─────────────────────────────────────────────────────
# Placeholders: {item_list}, {user_reason}
FEEDBACK_PARSER_PROMPT = """Analyze the following user feedback for a video game / gaming product recommendation system.

Recommended items (in order):
{item_list}

User feedback:
{user_reason}

For each item mentioned in the feedback, classify it into one of these categories:
- "POSITIVE": user says it matches their interests or gaming taste.
- "TRUE_NOISE": item is completely unrelated to gaming (e.g., kitchen appliances, baby products, office supplies). A video game, gaming accessory, or gaming peripheral can NEVER be TRUE_NOISE.
- "PERSONAL_PREFERENCE": item is gaming-related but does not match the user's taste, genre preference, or they already own something similar.
- "PLATFORM_MISMATCH": user complains about platform/console incompatibility. This category should be IGNORED — the user owns all platforms (PS2/3/4/5, Xbox 360/One/Series X, Nintendo, PC).

Important rules:
- Only classify items that are explicitly discussed in the user's feedback.
- Use EXACT item names from the recommended list above.
- ESCAPE any internal double quotes in item names (use \\" instead of ").
- Output ONLY valid JSON (no extra text, no markdown) in this exact format:
{{
  "positive": ["item name"],
  "negative": [["item name", "CATEGORY"], ...]
}}"""