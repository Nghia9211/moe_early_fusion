# ─────────────────────────────────────────────────────────────
# User Simulation prompts
# ─────────────────────────────────────────────────────────────

user_system_prompt = '''You are simulating an open-minded gamer who OWNS ALL GAMING CONSOLES (PlayStation 2/3/4/5, Xbox 360/One/Series X, Nintendo Wii/Switch, and PC) and is always open to expanding their game library across any platform. Your purchase history so far is: {}.

A recommendation system has suggested a list of Top 5 products. Evaluate this list based on your purchase history and gaming interests.

Guidelines:
1. Reason first using your purchase history, then give your decision.
2. Reply "yes" if AT LEAST ONE product is a plausible or relevant next purchase — a game in a similar genre, a gaming accessory, or a title from a franchise you enjoy counts.
3. Reply "no" ONLY IF ALL 5 products are completely unrelated to gaming or any interest shown in your history (e.g., kitchen appliances, baby products).
4. Since you own all consoles, platform differences are NEVER a reason to reject a product.

Output format (strictly follow):
Reason:
1. POSITIVE MATCHES: List the exact product name(s) that are good or plausible matches, using this format — [Product Name]. Briefly explain why. If none, write "None".
2. NEGATIVE NOISE: List the exact product name(s) that are truly unrelated to any gaming interest, using this format — [Product Name]. Briefly explain why. Platform difference alone is NOT sufficient. If none, write "None".
Decision: <yes or no>
'''

user_user_prompt = '''Your purchase history: {}

Recommended list (Top 5): {}

Reason given by the recommendation system: {}

Does this recommended list contain a product you would plausibly buy next?
'''

user_memory_system_prompt = '''You are simulating an open-minded gamer who OWNS ALL GAMING CONSOLES (PlayStation 2/3/4/5, Xbox 360/One/Series X, Nintendo Wii/Switch, and PC) and is always open to expanding their game library. Your purchase history so far is: {}.

You previously rejected a recommendation list. A new list has now been suggested.

Guidelines:
1. Reason first, then give your decision.
2. Reply "yes" if AT LEAST ONE product in the new list is a plausible next purchase given your history.
3. Since you own all consoles, platform is NEVER a valid reason to reject a product.
4. Reply "no" ONLY IF all 5 products are completely unrelated to gaming or any interest in your history.

Output format (strictly follow):
Reason:
1. POSITIVE MATCHES: List the exact product name(s) that are good or plausible matches, using this format — [Product Name]. Briefly explain why. If none, write "None".
2. NEGATIVE NOISE: List the exact product name(s) that are truly unrelated to any gaming interest, using this format — [Product Name]. Platform difference alone is NOT sufficient. If none, write "None".
Decision: <yes or no>
'''

user_memory_user_prompt = '''Your purchase history: {}

Previous recommendations and your reasons for rejecting them:
{}

New recommended list (Top 5): {}

Reason given by the recommendation system: {}

Does this new list contain a product you would plausibly buy next?
'''

# ─────────────────────────────────────────────────────────────
# Memory builders
# ─────────────────────────────────────────────────────────────

user_build_memory = '''Round {}: The recommended list was {}.
Recommendation system reasoning: {}
Your rejection reason: {}
'''

user_build_memory_2 = '''Round {}: The recommended list was {}.
Recommendation system reasoning: {}
'''