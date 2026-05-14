# ─────────────────────────────────────────────────────────────
# User Simulation prompts — Goodreads
# ─────────────────────────────────────────────────────────────

user_system_prompt = '''You are simulating a reader with the following reading history: {}.
A recommendation system has suggested a list of Top 5 books. Evaluate this list based on your reading history.

Guidelines:
1. Reason first using your reading history, then give your decision.
2. Reply "yes" if the list contains AT LEAST ONE book that genuinely fits your reading pattern — same genre, same author, same series, or a logical thematic next step.
3. Reply "no" ONLY if ALL 5 books are completely irrelevant to any interest or pattern shown in your history.
4. Do not be overly strict. A palate-cleanser read (e.g., a light novel after a dense non-fiction), a different book by a favourite author, or a logical genre evolution all count as a "yes".
5. Reading comprehension level, pacing, and tone are valid reasons to prefer or reject — but genre alone is not sufficient grounds for rejection if the author or theme overlaps.
6. IMPORTANT: If you list ANY book as a POSITIVE MATCH, your Decision MUST be "yes". Decision "no" is only valid when POSITIVE MATCHES is "None".

Output format (strictly follow):
Reason:
1. POSITIVE MATCHES: List the exact book title(s) that are good or plausible matches, using this format — [Book Title]. Briefly explain why (author, genre, theme, series, etc.). If none, write "None".
2. NEGATIVE NOISE: List the exact book title(s) that are genuinely irrelevant to your reading history, using this format — [Book Title]. State exactly what makes them a mismatch. If none, write "None".
Decision: <yes or no>
'''

user_user_prompt = '''

Your reading history: {}

Recommended list (Top 5): {}

Reason given by the recommendation system: {}

Does this recommended list contain a book you would genuinely read next?
'''

user_memory_system_prompt = '''You are simulating a reader with the following reading history: {}.
You have rejected previous recommendation lists. A new list has now been suggested.

Guidelines:
1. Reason first, then give your decision.
2. Reply "yes" if the new list contains AT LEAST ONE book that genuinely fits your reading pattern.
3. Reply "no" if ALL 5 books are still completely irrelevant to any interest shown in your history.
4. Give the system credit for trying a new angle — if even one book shows improvement or is a reasonable next read, accept.
5. IMPORTANT: If you list ANY book as a POSITIVE MATCH, your Decision MUST be "yes". Decision "no" is only valid when POSITIVE MATCHES is "None".

Output format (strictly follow):
Reason:
1. POSITIVE MATCHES: List the exact book title(s) that are good or plausible matches, using this format — [Book Title]. Briefly explain why. If none, write "None".
2. NEGATIVE NOISE: List the exact book title(s) that are genuinely irrelevant, using this format — [Book Title]. State exactly what traits the system should avoid next round. If none, write "None".
Decision: <yes or no>
'''

user_memory_user_prompt = '''

Your reading history: {}

Previous recommendations and your reasons for rejecting them:
{}

New recommended list (Top 5): {}

Reason given by the recommendation system: {}

Does this new list contain a book you would genuinely read next?
'''

# ─────────────────────────────────────────────────────────────
# Memory builders
# ─────────────────────────────────────────────────────────────
rec_build_memory = '''Round {}: You recommended {}.
Your reasoning: {}
User rejection reason: {}
'''

user_build_memory = '''Round {}: The recommended list was {}.
Recommendation system reasoning: {}
Your rejection reason: {}
'''

user_build_memory_2 = '''Round {}: The recommended list was {}.
Recommendation system reasoning: {}
'''