"""
amazon_feedback_prompt.py — v2
──────────────────────────────
LLM parser prompt cho FeedbackScoreAdjuster — dataset: Amazon (video games).

v2 thêm GENRE_MISMATCH để tạo hard penalty khi genre hoàn toàn khác cluster,
thay vì chỉ phân biệt TRUE_NOISE (non-gaming) vs PERSONAL_PREFERENCE (mọi game).

Categories:
  POSITIVE            → boost          (+)
  TRUE_NOISE          → hard penalty   (×0.80)  — item không liên quan gaming
  GENRE_MISMATCH      → hard penalty   (×0.85)  — game nhưng genre cách xa hoàn toàn
  PERSONAL_PREFERENCE → soft penalty   (×0.95)  — game lân cận nhưng không hợp taste
  PLATFORM_MISMATCH   → ignored        (no penalty — user owns all consoles)
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
SOFT_CATEGORIES = {"PERSONAL_PREFERENCE"}

# ── Categories that carry a HARD penalty (stronger than default TRUE_NOISE) ─
# TRUE_NOISE   → ×0.80 (penalty mạnh nhất — non-gaming item)
# GENRE_MISMATCH → ×0.85 (penalty vừa — game nhưng genre cách xa)
# Cả hai đều được xử lý trong FeedbackScoreAdjuster qua _negative_counts,
# nhưng GENRE_MISMATCH dùng riêng _genre_mismatch_counts với factor nhẹ hơn.
HARD_CATEGORIES = {"TRUE_NOISE", "GENRE_MISMATCH"}

# ── LLM prompt template ─────────────────────────────────────────────────────
# Placeholders: {item_list}, {user_reason}
FEEDBACK_PARSER_PROMPT = """Analyze the following user feedback for a video game / gaming product recommendation system.

Recommended items (in order):
{item_list}

User feedback:
{user_reason}

For each item mentioned in the feedback, classify it into ONE of these categories:
- "POSITIVE": user says the item matches their genre taste, franchise interest, or gaming preferences.
- "TRUE_NOISE": item is completely unrelated to gaming — e.g., kitchen appliances, baby products, office supplies. A video game, gaming accessory, or gaming peripheral can NEVER be TRUE_NOISE.
- "GENRE_MISMATCH": item IS a video game or gaming product, but belongs to a genre cluster that is completely absent from the user's purchase history AND the user explicitly notes the mismatch. Examples: user only buys FPS/Action games → a farming simulator, a visual novel, or a rhythm game = GENRE_MISMATCH. Use this ONLY when the genre distance is large and clear. A neighboring genre (e.g., Action-RPG → hack-and-slash) is NEVER a GENRE_MISMATCH.
- "PERSONAL_PREFERENCE": item is a video game in a neighboring or related genre, but does not match the user's specific taste, difficulty preference, or art style. Or the user already owns something very similar.
- "PLATFORM_MISMATCH": user complains about platform/console incompatibility. This category must be IGNORED — the user owns all platforms (PS2/3/4/5, Xbox 360/One/Series X, Nintendo Wii/Switch, PC). Do not penalize for platform reasons.

Important rules:
- Only classify items that are explicitly discussed in the user's feedback.
- Use EXACT item names from the recommended list above.
- ESCAPE any internal double quotes in item names (use \\" instead of ").
- When in doubt between GENRE_MISMATCH and PERSONAL_PREFERENCE, prefer PERSONAL_PREFERENCE (softer penalty).
- Output ONLY valid JSON (no extra text, no markdown) in this exact format:
{{
  "positive": ["item name"],
  "negative": [["item name", "CATEGORY"], ...]
}}"""