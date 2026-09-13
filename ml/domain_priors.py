"""Inference-time domain priors — outside the char n-gram ONNX model.

Weird / random-looking gambling hosts and clear lexical gambling terms are
handled here so the n-gram model is not forced to absorb those patterns
(which otherwise pulls probability away from readable keyword domains).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.config import DOMAIN_LEXICON_FLOOR, DOMAIN_WEIRD_FLOOR

# Align with labeling junk TLDs; keep local so priors stay independent of page gates.
_JUNK_TLDS = (
    ".sbs",
    ".vip",
    ".top",
    ".xyz",
    ".cc",
    ".icu",
    ".fun",
    ".site",
    ".club",
    ".online",
    ".live",
    ".bet",
    ".win",
    ".pro",
    ".cfd",
    ".buzz",
    ".shop",
    ".store",
    ".lol",
    ".click",
)

# Alone enough to flag a domain name (substring only if len >= 5).
_STRONG = (
    "judol",
    "togel",
    "gacor",
    "maxwin",
    "sbobet",
    "maxbet",
    "bandarqq",
    "baccarat",
    "blackjack",
    "roulette",
    "sportsbook",
    "bookie",
    "qiuqiu",
    "livedraw",
    "taruhan",
    "judionline",
    "situsjudi",
    "agenjudi",
    "bandarjudi",
    "tembakikan",
)

# Need a second gambling signal (another token, or junk TLD + digits).
_WEAK = (
    "judi",
    "slot",
    "casino",
    "poker",
    "bandar",
    "betting",
    "parlay",
    "jackpot",
)

_PART_RE = re.compile(r"[a-z]+|\d+")
_DIGIT_RUN_RE = re.compile(r"\d{4,}")


def _label(domain: str) -> str:
    return domain.split(".", 1)[0].lower()


def _parts(label: str) -> list[str]:
    return _PART_RE.findall(label)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def _has_strong(label: str, parts: list[str]) -> bool:
    part_set = {p for p in parts if p.isalpha()}
    for term in _STRONG:
        if len(term) >= 5:
            if term in label:
                return True
        elif term in part_set:
            return True
    return False


def _weak_hits(label: str, parts: list[str]) -> int:
    """Count distinct weak terms. Short 'judi' is token-only (avoid 'judicial')."""
    part_set = {p for p in parts if p.isalpha()}
    hits = 0
    for term in _WEAK:
        if term == "judi":
            if term in part_set:
                hits += 1
        elif term in label:
            hits += 1
    return hits


def has_gambling_lexicon(domain: str) -> bool:
    """True when the domain name itself carries clear gambling vocabulary."""
    if not domain:
        return False
    label = _label(domain)
    parts = _parts(label)
    if _has_strong(label, parts):
        return True
    weak = _weak_hits(label, parts)
    if weak >= 2:
        return True
    junk = any(domain.endswith(t) for t in _JUNK_TLDS)
    digits = sum(1 for c in label if c.isdigit())
    if weak >= 1 and junk and (digits >= 2 or bool(_DIGIT_RUN_RE.search(label))):
        return True
    return False


def is_weird_domain(domain: str) -> bool:
    """True for random-looking hosts typical of disposable gambling domains.

    High-precision gate: junk TLD plus digit-run / numeric prefix / randomish label.
    Does not retrain or alter the char n-gram model.
    """
    if not domain:
        return False
    if not any(domain.endswith(t) for t in _JUNK_TLDS):
        return False
    label = _label(domain)
    alnum = re.sub(r"[^a-z0-9]", "", label)
    digits = sum(c.isdigit() for c in label)
    if _DIGIT_RUN_RE.search(label) or label[:1].isdigit():
        return True
    if (
        re.fullmatch(r"[a-z0-9]{10,}", label)
        and _entropy(alnum) > 3.5
        and digits >= 2
    ):
        return True
    return False


def apply_domain_priors(domain: str, p_model: float) -> tuple[float, str | None]:
    """Raise probability via priors; never lower the n-gram score.

    Returns (adjusted_p, prior_name|None).
    """
    if has_gambling_lexicon(domain):
        return max(p_model, DOMAIN_LEXICON_FLOOR), "lexicon"
    if is_weird_domain(domain):
        return max(p_model, DOMAIN_WEIRD_FLOOR), "weird"
    return p_model, None
