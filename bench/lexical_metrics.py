"""Token-level F1 and BLEU-1, the lexical metrics the paper reports next to the judge (§4.1, Tab. 4).

Both compare normalized word tokens: lowercase, punctuation and the articles
"a", "an", "the" removed (the usual normalization of QA benchmarks). The paper
does not give its implementation; these are the standard definitions, with no
stemming and no smoothing.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalized_tokens(text: str) -> list[str]:
    lowered = text.lower().translate(_PUNCTUATION)
    return _ARTICLES.sub(" ", lowered).split()


def token_f1(prediction: str, gold: str) -> float:
    """Harmonic mean of the precision and recall of shared tokens (each token counted as often as it appears)."""
    predicted = normalized_tokens(prediction)
    expected = normalized_tokens(gold)
    if not predicted or not expected:
        return float(predicted == expected)
    shared = sum((Counter(predicted) & Counter(expected)).values())
    if shared == 0:
        return 0.0
    precision = shared / len(predicted)
    recall = shared / len(expected)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, gold: str) -> float:
    """Clipped unigram precision times the brevity penalty (Papineni et al., 2002, with n = 1)."""
    predicted = normalized_tokens(prediction)
    expected = normalized_tokens(gold)
    if not predicted or not expected:
        return float(predicted == expected)
    clipped = sum((Counter(predicted) & Counter(expected)).values())
    precision = clipped / len(predicted)
    brevity_penalty = 1.0 if len(predicted) > len(expected) else math.exp(1 - len(expected) / len(predicted))
    return brevity_penalty * precision
