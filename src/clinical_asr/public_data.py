"""Pre-inference selection for a bounded public mock-consultation probe."""

from __future__ import annotations

import math
import random
from pathlib import Path


def select_clean_intervals(entries, limit: int, seed: int):
    if not 1 <= limit <= 100:
        raise ValueError("Sample limit must be between 1 and 100")
    candidates = []
    for start, end, text in entries:
        if (math.isfinite(start) and math.isfinite(end) and start >= 0
                and 3 <= end - start <= 18 and len(text.split()) >= 4
                and "<" not in text and ">" not in text):
            candidates.append((start, end, text))
    if len(candidates) < limit:
        raise ValueError("Insufficient clean intervals for the frozen sample count")
    random.Random(seed).shuffle(candidates)
    return sorted(candidates[:limit])
def require_new_probe(root: Path):
    if any((root / name).exists() for name in ("manifest.jsonl", "dev.jsonl", "test.jsonl")):
        raise ValueError("Use a new output folder rather than overwriting a frozen probe")


