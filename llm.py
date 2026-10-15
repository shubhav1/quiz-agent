"""Claude fallback for questions the rule solver can't answer."""
import os
import re
from pathlib import Path

MODEL = "claude-opus-5-5"


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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    client = anthropic.Anthropic()
    content = PROMPT.format(question=question, a=options[0], b=options[1], c=options[2], d=options[3])
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:  # never let the fallback crash a run; the round still gets written
        print(f"  LLM call failed: {e}")
        return None
    if response.stop_reason == "refusal":
        print("  LLM refused")
        return None
    text = "".join(b.text for b in response.content if b.type == "text")
    m = re.findall(r"ANSWER:\s*([ABCD])", text)
    return m[-1] if m else None
