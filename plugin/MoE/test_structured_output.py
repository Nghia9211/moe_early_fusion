"""
test_structured_output.py
─────────────────────────
Test xem local vLLM server có support structured output không.

Chạy: /home/research/nghialt/.venv/bin/python3 test_structured_output.py

Kết quả:
  [OK]   → có thể dùng trong reranker.py
  [FAIL] → giữ 3-layer parser hiện tại
"""

import json
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List

# ── Config (giống run_moe.sh) ──────────────────────────────────────────────
MODEL    = "qwen-research"
BASE_URL = "http://localhost:11435/v1"
API_KEY  = "EMPTY"
TEMP     = 0.0

# ── Pydantic schema ────────────────────────────────────────────────────────
class RankerOutput(BaseModel):
    ranked_items: List[str]
    explanation: str

SIMPLE_PROMPT = (
    "You are a book reranker. "
    "Return a JSON with exactly these fields:\n"
    '  "ranked_items": ["Book A", "Book B", "Book C"],\n'
    '  "explanation": "reason here"\n'
    "Output ONLY the JSON, nothing else."
)

def make_llm(**extra):
    return ChatOpenAI(
        model=MODEL,
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=TEMP,
        max_retries=1,
        **extra,
    )

# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: with_structured_output (Pydantic / tool-calling)")
print("=" * 60)
try:
    llm = make_llm()
    structured_llm = llm.with_structured_output(RankerOutput)
    result = structured_llm.invoke(SIMPLE_PROMPT)
    print("[OK] with_structured_output works!")
    print(f"     ranked_items : {result.ranked_items}")
    print(f"     explanation  : {result.explanation[:80]}")
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {str(e)[:300]}")

print()

# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: response_format=json_object (JSON mode, no schema)")
print("=" * 60)
try:
    llm2 = make_llm(model_kwargs={"response_format": {"type": "json_object"}})
    r2 = llm2.invoke(SIMPLE_PROMPT)
    parsed = json.loads(r2.content)
    print("[OK] json_object mode works!")
    print(f"     ranked_items : {parsed.get('ranked_items')}")
    print(f"     explanation  : {str(parsed.get('explanation',''))[:80]}")
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {str(e)[:300]}")

print()

# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: Plain invoke + 3-layer parser (current approach)")
print("=" * 60)
try:
    import re
    llm3 = make_llm()
    r3 = llm3.invoke(SIMPLE_PROMPT)
    content = r3.content.replace("```json", "").replace("```", "").strip()

    def first_json(text):
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
                    if depth == 0: return text[start:i+1]
        return None

    js = first_json(content)
    parsed3 = json.loads(js)
    print("[OK] 3-layer parser works!")
    print(f"     ranked_items : {parsed3.get('ranked_items')}")
    print(f"     explanation  : {str(parsed3.get('explanation',''))[:80]}")
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {str(e)[:300]}")

print()
print("=" * 60)
print("SUMMARY — use whichever tests passed:")
print("  TEST 1 OK → use with_structured_output (best)")
print("  TEST 2 OK → use response_format=json_object (good, simpler)")
print("  TEST 3 OK → keep 3-layer parser (current, no model requirement)")
print("=" * 60)
