"""
goodreads_feedback_prompt.py
─────────────────────────────
LLM parser prompt cho FeedbackScoreAdjuster — dataset: Goodreads (books).

Categories:
  POSITIVE            → boost
  TRUE_NOISE          → hard penalty  (×0.80)
  PERSONAL_PREFERENCE → soft penalty  (×0.95)
  EDITION_MISMATCH    → soft penalty  (×0.95)  — đã đọc / bản dịch khác / series xong
"""

# ── Blanket-rejection phrases ────────────────────────────────────────────────
BLANKET_REJECTION_PHRASES = [
    "all of the books",
    "all 5 books",
    "none of the books",
    "none of these",
    "all recommended",
    "does not include any",
    "does not contain any",
]

# ── Categories that carry NO penalty ────────────────────────────────────────
IGNORE_CATEGORIES = set()   # Goodreads không có category nào bị ignore hoàn toàn

# ── Categories that carry a SOFT penalty ────────────────────────────────────
SOFT_CATEGORIES = {"PERSONAL_PREFERENCE", "EDITION_MISMATCH"}

# ── LLM prompt template ─────────────────────────────────────────────────────
# Placeholders: {item_list}, {user_reason}
FEEDBACK_PARSER_PROMPT = """Analyze the following user feedback for a book recommendation system.

Recommended books (in order):
{item_list}

User feedback:
{user_reason}

For each book mentioned in the feedback, classify it into one of these categories:
- "POSITIVE": user says the book genuinely fits their reading taste, genre preference, or author loyalty.
- "TRUE_NOISE": book is completely irrelevant to any genre, theme, or author type in the user's reading history. Use this ONLY when the mismatch is severe and obvious (e.g., a technical programming manual recommended to a romance reader).
- "PERSONAL_PREFERENCE": book is within a plausible genre or theme range but does not match the user's specific taste — e.g., wrong sub-genre, writing style too dense/light, or topic not interesting to them.
- "EDITION_MISMATCH": user has already read this book (possibly a different edition or translation), or the book is part of a series they have already completed. Apply soft penalty.

Important rules:
- Only classify books that are explicitly discussed in the user's feedback.
- Genre proximity matters: a mystery reader may enjoy thrillers — do NOT classify these as TRUE_NOISE.
- Use EXACT book titles from the recommended list above.
- ESCAPE any internal double quotes in book titles (use \\" instead of ").
- Output ONLY valid JSON (no extra text, no markdown) in this exact format:
{{
  "positive": ["book title"],
  "negative": [["book title", "CATEGORY"], ...]
}}"""