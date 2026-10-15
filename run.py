"""Fetch questions.csv and write answers/round-NNNN.json for every unanswered round.

Usage: python run.py [path-or-url-to-questions.csv]
"""
import csv
import io
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from llm import solve_llm
from solver import solve_rule

QUESTIONS_URL = "https://raw.githubusercontent.com/FaisalXL/dso429-hourly-quiz/main/questions.csv"
ANSWERS = Path(__file__).parent / "answers"


def load_questions(src):
    if src.startswith("http"):
        req = urllib.request.Request(src, headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8")
    else:
        text = Path(src).read_text(encoding="utf-8")
    rounds = defaultdict(dict)
    for row in csv.DictReader(io.StringIO(text)):
        rounds[int(row["round_id"])][int(row["q_index"])] = row
    return rounds


def answer(row):
    options = [row[f"option_{c}"] for c in "abcd"]
    letter = solve_rule(row["question_text"], options)
    if letter:
        return letter, "rule"
    letter = solve_llm(row["question_text"], options)
    if letter:
        return letter, "llm"
    return "A", "guess"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else QUESTIONS_URL
    rounds = load_questions(src)
    ANSWERS.mkdir(exist_ok=True)
    written = 0
    for round_id in sorted(rounds):
        path = ANSWERS / f"round-{round_id:04d}.json"
        if path.exists():
            continue
        answers, sources = {}, defaultdict(int)
        for q_index in sorted(rounds[round_id]):
            letter, how = answer(rounds[round_id][q_index])
            answers[f"q{q_index}"] = letter
            sources[how] += 1
            if how != "rule":
                print(f"  round {round_id} q{q_index}: {how} -> {letter} | {rounds[round_id][q_index]['question_text']}")
        path.write_text(json.dumps({"round_id": round_id, "answers": answers}, indent=2) + "\n")
        print(f"round {round_id}: wrote {path.name} ({dict(sources)})")
        written += 1
    print(f"done: {written} new round file(s), {len(rounds)} rounds published")


if __name__ == "__main__":
    main()
