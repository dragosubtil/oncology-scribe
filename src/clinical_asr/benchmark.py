"""Policy-separated corpus reports and paired cluster-bootstrap comparisons."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .metrics import NORMALIZATION_VERSION, score_text
from .schema import Sample


def _check_predictions(samples: list[Sample], predictions: dict[str, str]):
    expected = {sample.sample_id for sample in samples}
    if len(expected) != len(samples):
        raise ValueError("Duplicate sample IDs")
    missing, extra = expected - predictions.keys(), predictions.keys() - expected
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} samples")
    if extra:
        raise ValueError(f"Unknown predictions for {len(extra)} samples")
    if any(not isinstance(text, str) for text in predictions.values()):
        raise ValueError("Predictions must be strings")
    if not samples:
        raise ValueError("Cannot evaluate an empty corpus")


def _cluster_labels(samples: list[Sample]) -> list[int]:
    parent = list(range(len(samples)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for i, sample in enumerate(samples):
        for key in (("speaker", sample.speaker_id), ("encounter", sample.encounter_id)):
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    return [find(i) for i in range(len(samples))]


def _micro(scores: list[dict]) -> float | None:
    denominator = sum(score["reference_words"] for score in scores)
    return sum(score["word_errors"] for score in scores) / denominator if denominator else None


def _bootstrap_ci(scores: list[dict], labels: list[int], iterations: int, seed: int,
                  other_scores: list[dict] | None = None) -> list[float] | None:
    clusters = sorted(set(labels))
    if iterations <= 0 or len(clusters) < 2:
        return None
    grouped = {cluster: [i for i, label in enumerate(labels) if label == cluster]
               for cluster in clusters}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        indexes = [i for cluster in rng.choice(clusters, size=len(clusters), replace=True)
                   for i in grouped[int(cluster)]]
        value = _micro([scores[i] for i in indexes])
        if value is not None:
            if other_scores is not None:
                other = _micro([other_scores[i] for i in indexes])
                if other is None:
                    continue
                value -= other
            values.append(value)
    return np.quantile(values, [0.025, 0.975]).tolist() if values else None


def _partitions(samples: list[Sample]) -> dict:
    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[(sample.language, sample.target_policy, sample.split)].append(index)
    return dict(sorted(groups.items()))


def evaluate_predictions(samples: list[Sample], predictions: dict[str, str],
                         bootstrap: int = 1000, seed: int = 42) -> dict:
    _check_predictions(samples, predictions)
    if bootstrap < 0:
        raise ValueError("Bootstrap iterations cannot be negative")
    rows = []
    for sample in samples:
        rows.append({"sample_id": sample.sample_id, "language": sample.language,
                     "target_policy": sample.target_policy, "split": sample.split,
                     "speaker_id": sample.speaker_id, "encounter_id": sample.encounter_id,
                     "data_use": sample.data_use, "reference_reviewed": sample.reference_reviewed,
                     "scores": score_text(sample.reference, predictions[sample.sample_id],
                                          sample.clinical_terms, sample.language)})
    groups = []
    for (language, policy, split), indexes in _partitions(samples).items():
        normalized = [rows[i]["scores"]["normalized"] for i in indexes]
        raw = [rows[i]["scores"]["raw"] for i in indexes]
        labels = _cluster_labels([samples[i] for i in indexes])
        nonempty = [score["wer"] for score in normalized if score["wer"] is not None]
        independent = len(set(labels))
        groups.append({
            "language": language, "target_policy": policy, "split": split,
            "clips": len(indexes), "independent_groups": independent,
            "reference_words": sum(score["reference_words"] for score in normalized),
            "micro_wer": _micro(normalized), "raw_micro_wer": _micro(raw),
            "macro_wer": float(np.mean(nonempty)) if nonempty else None,
            "micro_wer_ci95": _bootstrap_ci(normalized, labels, bootstrap, seed),
            "empty_reference_clips": sum(not score["reference_words"] for score in normalized),
            "empty_reference_insertions": sum(score["insertions"] for score in normalized
                                              if not score["reference_words"]),
            "all_references_reviewed": all(samples[i].reference_reviewed for i in indexes),
            "data_use_categories": sorted({samples[i].data_use for i in indexes}),
            "small_sample_warning": independent < 20,
        })
    return {"schema_version": 1, "normalization_version": NORMALIZATION_VERSION,
            "bootstrap": {"iterations": bootstrap, "seed": seed,
                          "unit": "connected speaker/encounter cluster", "method": "percentile"},
            "clinical_validation": "not_established_by_this_report",
            "groups": groups, "rows": rows}


def paired_comparison(samples: list[Sample], baseline: dict[str, str],
                      candidate: dict[str, str], bootstrap: int = 1000, seed: int = 42) -> dict:
    if bootstrap < 0:
        raise ValueError("Bootstrap iterations cannot be negative")
    if baseline.keys() != candidate.keys():
        raise ValueError("Paired comparisons require the same sample IDs")
    _check_predictions(samples, baseline)
    _check_predictions(samples, candidate)
    groups = []
    for (language, policy, split), indexes in _partitions(samples).items():
        subset = [samples[i] for i in indexes]
        before = [score_text(s.reference, baseline[s.sample_id], [], language)["normalized"]
                  for s in subset]
        after = [score_text(s.reference, candidate[s.sample_id], [], language)["normalized"]
                 for s in subset]
        labels = _cluster_labels(subset)
        a, b = _micro(before), _micro(after)
        groups.append({"language": language, "target_policy": policy, "split": split,
                       "independent_groups": len(set(labels)),
                       "baseline_micro_wer": a, "candidate_micro_wer": b,
                       "delta_micro_wer": b - a if a is not None and b is not None else None,
                       "delta_ci95": _bootstrap_ci(after, labels, bootstrap, seed, before)})
    return {"direction": "candidate_minus_baseline; negative favors candidate",
            "bootstrap_iterations": bootstrap, "seed": seed, "groups": groups}
