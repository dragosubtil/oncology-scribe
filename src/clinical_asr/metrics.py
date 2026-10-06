"""Lexical metrics and explicitly heuristic clinical review flags."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import jiwer

NORMALIZATION_VERSION = "unicode-decimal-preserving-v3"
NUMBER_TOKEN = r"[+-]?(?:\d+(?:[.,]\d+)*|[.,]\d+)%?"
TOKEN_PATTERN = re.compile(NUMBER_TOKEN + r"|[^\W\d_]+(?:['-][^\W\d_]+)*", re.UNICODE)
NEGATION_CUES = {
    "en": {"no", "not", "without", "denies", "denied", "negative"},
    "de": {"kein", "keine", "keinen", "nicht", "ohne", "verneint", "negativ"},
    "gsw": {"kei", "keis", "keini", "nid", "nöd", "ohni", "kein", "nicht", "keine"},
    "fr": {"pas", "sans", "aucun", "aucune", "nie", "négatif", "négative"},
    "it": {"non", "nessun", "nessuna", "senza", "nega", "negativo"},
    "pt": {"não", "sem", "nega", "ausência", "negativo"},
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("\u2019", "'").replace("\u2212", "-")
    return " ".join(TOKEN_PATTERN.findall(text))


def _alignment_metrics(reference: str, hypothesis: str) -> dict:
    words = jiwer.process_words(reference, hypothesis)
    chars = jiwer.process_characters(reference, hypothesis)
    count = words.hits + words.substitutions + words.deletions
    char_count = chars.hits + chars.substitutions + chars.deletions
    return {
        "wer": words.wer if count else None,
        "cer": chars.cer if char_count else None,
        "reference_words": count,
        "reference_characters": char_count,
        "hits": words.hits,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "word_errors": words.substitutions + words.deletions + words.insertions,
    }


def _term_counts(text: str, terms: list[str]) -> Counter:
    tokens = normalize_text(text).split()
    choices = sorted({normalize_text(term) for term in terms}, key=lambda x: -len(x.split()))
    patterns = [(term, term.split()) for term in choices if term]
    result: Counter = Counter()
    cursor = 0
    while cursor < len(tokens):
        for term, pattern in patterns:
            if tokens[cursor:cursor + len(pattern)] == pattern:
                result[term] += 1
                cursor += len(pattern)
                break
        else:
            cursor += 1
    return result


def term_metrics(reference: str, hypothesis: str, terms: list[str]) -> dict:
    ref = _term_counts(reference, terms)
    hyp = _term_counts(hypothesis, terms)
    matched = sum((ref & hyp).values())
    nref, nhyp = sum(ref.values()), sum(hyp.values())
    precision = matched / nhyp if nhyp else (0.0 if nref else None)
    recall = matched / nref if nref else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else (0.0 if nref else None)
    )
    return {"reference_occurrences": nref, "hypothesis_occurrences": nhyp,
            "matched_occurrences": matched, "precision": precision, "recall": recall,
            "f1": f1, "per_term_reference": dict(ref), "per_term_hypothesis": dict(hyp)}


def score_text(reference: str, hypothesis: str, terms: list[str], language: str) -> dict:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    flags = []
    cues = NEGATION_CUES.get(language, set())
    if Counter(t for t in ref.split() if t in cues) != Counter(t for t in hyp.split() if t in cues):
        flags.append("negation_change")
    numbers = re.compile(r"(?<!\w)" + NUMBER_TOKEN + r"(?:\s*(?:mg|mcg|g|ml|mmol|gy|%))?(?!\w)")
    if Counter(numbers.findall(ref)) != Counter(numbers.findall(hyp)):
        flags.append("numeric_or_unit_change")
    return {
        "raw": _alignment_metrics(reference, hypothesis),
        "normalized": _alignment_metrics(ref, hyp),
        "normalization_version": NORMALIZATION_VERSION,
        "terms": term_metrics(reference, hypothesis, terms),
        "empty_reference": not bool(ref),
        "review_flags": flags,
        "review_flags_are_heuristic": True,
    }
