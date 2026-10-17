"""Gemini fallback for questions the rule solver can't answer."""
import json
import os
import re
import urllib.request
from pathlib import Path

MODEL = "gemini-3.8-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def _load_dotenv():
    """Read KEY=value lines from .env (local runs); real env vars win."""
    env = Path(__file__).parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if sep and key and not key.startswith("#") and value:
            os.environ.setdefault(key, value)


_load_dotenv()

PROMPT = """Answer this multiple-choice business arithmetic question. Work it out carefully, \
then end your reply with a final line of the form "ANSWER: X" where X is A, B, C or D.

{question}

A) {a}
B) {b}
C) {c}
D) {d}"""


def solve_llm(question, options):
    """Return a letter, or None if no API key is set or the call fails."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    content = PROMPT.format(question=question, a=options[0], b=options[1], c=options[2], d=options[3])
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": content}]}]}).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
    except Exception as e:  # never let the fallback crash a run; the round still gets written
        print(f"  LLM call failed: {e}")
        return None
    candidates = data.get("candidates") or []
    if not candidates:
        print(f"  LLM returned no answer: {data.get('promptFeedback')}")
        return None
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    m = re.findall(r"ANSWER:\s*([ABCD])", text)
    return m[-1] if m else None
