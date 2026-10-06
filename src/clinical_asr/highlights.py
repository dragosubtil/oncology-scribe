"""Highlight review cues without treating decoder scores as clinical certainty."""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

from .metrics import NEGATION_CUES, normalize_text
from .word_audio import valid_times


def review_spans(text: str, words: list[dict], language: str, threshold: float = 0.6) -> list[dict]:
    if not 0 <= threshold <= 1:
        raise ValueError("Review threshold must be between zero and one")
    spans = []
    previous_end = 0
    for word in words:
        start, end = word["char_start"], word["char_end"]
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            raise ValueError("Invalid word offset")
        if start < previous_end or text[start:end] != word["text"]:
            raise ValueError("Word offset does not match the raw transcript")
        previous_end = end
        score = word.get("model_score")
        if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Invalid word model score")
        reasons = []
        if score is not None and score < threshold:
            reasons.append("low_model_score")
        if re.search(r"\d", word["text"]):
            reasons.append("numeric_review")
        if set(normalize_text(word["text"]).split()) & NEGATION_CUES.get(language, set()):
            reasons.append("negation_review")
        if reasons:
            spans.append({"start": start, "end": end, "text": word["text"],
                          "model_score": score, "reasons": reasons,
                          "audio_start": word.get("start"), "audio_end": word.get("end")})
    return spans


def render_highlight_html(text: str, spans: list[dict], source_sha256: str | None = None,
                          revision: str | None = None, words: list[dict] | None = None) -> str:
    cues = {}
    previous = 0
    for span in spans:
        start, end = span['start'], span['end']
        if not previous <= start < end <= len(text) or text[start:end] != span['text']:
            raise ValueError('Invalid highlight offset')
        cues[(start, end)] = span
        previous = end
    if words is None:
        words = [dict(text=s['text'], char_start=s['start'], char_end=s['end'],
                      start=s.get('audio_start'), end=s.get('audio_end')) for s in spans]
    parts = []
    cursor = 0
    for word in words:
        start, end = word['char_start'], word['char_end']
        if (any(not isinstance(i, int) or isinstance(i, bool) for i in (start, end))
                or not cursor <= start < end <= len(text) or text[start:end] != word['text']):
            raise ValueError('Invalid aligned word offset')
        gap = text[cursor:start]
        parts.append(f'<span class="untimed-audio-word">{html.escape(gap)}</span>'
                     if gap.strip() else html.escape(gap))
        span = cues.get((start, end))
        color = ('#ffdddd' if 'low_model_score' in span['reasons'] else '#fff0bd') if span else 'transparent'
        label = html.escape(', '.join(span['reasons']), quote=True) if span else ''
        token = html.escape(text[start:end])
        if source_sha256 and revision and valid_times(word.get('start'), word.get('end')):
            parts.append(f'<button type="button" class="word-audio-cue" '
                         f'data-word-start="{start}" data-word-end="{end}" '
                         f'data-source-sha256="{html.escape(source_sha256, quote=True)}" '
                         f'data-review-revision="{html.escape(revision, quote=True)}" '
                         f'aria-pressed="false" '
                         f'aria-label="Listen to {html.escape(text[start:end], quote=True)}" '
                         f'title="Play original audio; estimated alignment{": " + label if label else ""}" '
                         f'style="background:{color};color:#171717">{token}</button>')
        elif span:
            parts.append(f'<mark class="untimed-audio-word" style="background:{color};color:#171717" title="{label}">{token}</mark>')
        else:
            parts.append(f'<span class="untimed-audio-word" title="Audio alignment unavailable">{token}</span>')
        cursor = end
    tail = text[cursor:]
    parts.append(f'<span class="untimed-audio-word">{html.escape(tail)}</span>'
                 if tail.strip() else html.escape(tail))
    toolbar = ''
    if source_sha256 and revision and words:
        icon = (Path(__file__).parent / 'assets/icons/audio-lines.svg').read_text(encoding='utf-8')
        toolbar = ('<div class="audio-linked-toolbar"><span>Original audio</span>'
                   '<button type="button" class="audio-selection-play" disabled '
                   'title="Listen to selected words" aria-label="Play selection">'
                   f'{icon}<span>Play selection</span></button></div>')
    return toolbar + '<div class="audio-transcript-text">' + ''.join(parts) + '</div>'
