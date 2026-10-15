"""Rule-based solvers for the recognized question templates, with an LLM fallback.

Each rule computes the expected answer, then the closest option is picked.
If no rule matches, or the closest option is not close enough, the question
goes to Claude (llm.py).
"""
import math
import re
from datetime import date, timedelta

NUM = r"\$?(-?[\d,]+(?:\.\d+)?)"
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


def num(s):
    return float(s.replace(",", "").replace("$", ""))


# ---------- option parsing / matching ----------

def parse_option(opt):
    """Return ('date', date) | ('num', float) | None."""
    o = opt.strip()
    m = re.fullmatch(r"(\d{1,2}) ([A-Za-z]+) (\d{4})", o)
    if m and m.group(2).lower() in MONTHS:
        return "date", date(int(m.group(3)), MONTHS.index(m.group(2).lower()) + 1, int(m.group(1)))
    m = re.search(r"-?\$?-?[\d,]*\.?\d+", o)
    if m:
        s = m.group(0).replace("$", "").replace(",", "")
        return "num", float(s)
    return None


def pick(value, options, rel_tol=0.005):
    """Pick the letter whose option is closest to value; None if nothing is close."""
    best, best_err = None, None
    for letter, opt in zip("ABCD", options):
        p = parse_option(opt)
        if p is None:
            continue
        kind, v = p
        if isinstance(value, date):
            if kind != "date":
                continue
            err = abs((v - value).days)
            ok = err == 0
        else:
            if kind != "num":
                continue
            err = abs(v - value)
            ok = err <= max(rel_tol * abs(value), 0.051)
        if ok and (best_err is None or err < best_err):
            best, best_err = letter, err
    return best


# ---------- rules: (regex, function returning the expected value) ----------

def _compound(m):
    p, r, n = num(m[1]), num(m[2]) / 100, num(m[3])
    return p * (1 + r) ** n


def _discount(m):
    return num(m[1]) * (1 - num(m[2]) / 100)


def _revenue(m):
    return num(m[1]) * num(m[2])


def _fx(m):
    return num(m[1]) * num(m[2])


def _weighted_avg(m):
    n1, a1, n2, a2 = map(num, (m[1], m[2], m[3], m[4]))
    return (n1 * a1 + n2 * a2) / (n1 + n2)


def _gross_margin(m):
    rev, cogs = num(m[1]), num(m[2])
    return (rev - cogs) / rev * 100


def _break_even(m):
    fixed, price, var = num(m[1]), num(m[2]), num(m[3])
    return round(fixed / (price - var))


def _probability(m):
    counts = [num(x) for x in re.findall(r"(\d+) [a-z]+ tickets", m.string)]
    target = num(re.search(r"(\d+) " + re.escape(m[1]) + r" tickets", m.string)[1])
    return target / sum(counts) * 100


def _pct_change(m):
    a, b = num(m[1]), num(m[2])
    return (b - a) / a * 100


def _ratio(m):
    total, x, y = num(m[1]), num(m[2]), num(m[3])
    share = x if m[4] == "first" else y
    return total * share / (x + y)


def _sequence(m):
    xs = [num(x) for x in re.findall(r"-?[\d.]+", m[1].replace(",", " "))]
    d = [b - a for a, b in zip(xs, xs[1:])]
    if all(abs(x - d[0]) < 1e-9 for x in d):
        return xs[-1] + d[0]
    if all(x != 0 for x in xs):
        r = [b / a for a, b in zip(xs, xs[1:])]
        if all(abs(x - r[0]) < 1e-9 for x in r):
            return xs[-1] * r[0]
    dd = [b - a for a, b in zip(d, d[1:])]
    if dd and all(abs(x - dd[0]) < 1e-9 for x in dd):
        return xs[-1] + d[-1] + dd[0]
    return None


def _settle(m):
    d = date(int(m[3]), MONTHS.index(m[2].lower()) + 1, int(m[1]))
    return d + timedelta(days=int(m[4]))


RULES = [
    (rf"investment of {NUM} grows at {NUM}% compounded annually.*?after {NUM} years", _compound),
    (rf"listed at {NUM} is discounted by {NUM}%", _discount),
    (rf"sells {NUM} units at {NUM} each.*total revenue", _revenue),
    (rf"invoice for {NUM} is converted to \w+ at {NUM} \w+ per", _fx),
    (rf"had {NUM} orders averaging {NUM}; region \w+ had {NUM} orders averaging {NUM}", _weighted_avg),
    (rf"generates {NUM} in revenue against {NUM} in cost of goods sold.*gross margin", _gross_margin),
    (rf"Fixed costs are {NUM}\. Each unit sells for {NUM} with a variable cost of {NUM}", _break_even),
    (r"probability it is an? (\w+) ticket", _probability),
    (rf"moved from {NUM} to {NUM}\. What is the percentage change", _pct_change),
    (r"A \$?([\d,]+(?:\.\d+)?) .*? split between two \w+ in the ratio (\d+):(\d+)\. What does the (first|second)", _ratio),
    (r"series runs ([-\d., ]+?)\. If the pattern continues", _sequence),
    (r"signed on (\d{1,2}) ([A-Za-z]+) (\d{4}) has a (\d+)-day settlement", _settle),
]


def solve_rule(question, options):
    """Return a letter if a rule matches and an option fits, else None."""
    for pattern, fn in RULES:
        m = re.search(pattern, question, re.IGNORECASE)
        if not m:
            continue
        try:
            value = fn(m)
        except (ValueError, ZeroDivisionError, TypeError, IndexError):
            return None
        if value is None:
            return None
        return pick(value, options)
    return None
