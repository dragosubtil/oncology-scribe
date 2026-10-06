"""Offline note replay; explicit engineering controls, not clinical ground truth."""

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import sys
import urllib.error
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from clinical_asr import notes

LABELS = frozenset(("subjective", "objective", "assessment", "plan", "unassigned"))
MODELS = ("qwen3.5:4b", "medgemma:4b", "llama3.1:latest")


class AuditIntegrityError(RuntimeError):
    pass


def strict_json(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise AuditIntegrityError("Duplicate capture JSON field")
            result[key] = item
        return result

    def nonfinite(value):
        raise AuditIntegrityError("Nonfinite capture JSON")

    try:
        return json.loads(value, object_pairs_hook=unique, parse_constant=nonfinite)
    except (ValueError, TypeError) as exc:
        raise AuditIntegrityError("Invalid capture JSON") from exc


def require(condition, message):
    if not condition:
        raise AuditIntegrityError(message)


def check_call(request, raw, model):
    require(request.get("model") == model, "Captured request model mismatch")
    try:
        envelope = strict_json(raw)
    except AuditIntegrityError:
        return None  # The adapter must reconstruct and report this failure.
    if isinstance(envelope, dict) and ("model" in envelope or envelope.get("done") is True):
        require(envelope.get("model") == model, "Captured response model mismatch")
    return envelope


def reconstruct(adapter, case, model, captures, policy=None):
    class Replay:
        def open(self, request, **options):
            route = request.full_url.rsplit("/", 1)[-1]
            require(route in captures, "Missing captured route: " + route)
            captured = captures[route]
            body = strict_json(request.data)
            if route == "chat" and policy is not None:
                body = copy.deepcopy(body)
                body["messages"][0]["content"] = policy
            require(
                body == captured["request"],
                "Request identity/configuration differs from frozen capture",
            )
            if "transport_error" in captured:
                raise urllib.error.URLError(captured["transport_error"])
            raw = captured["response"]
            if route == "chat":
                check_call(body, raw, model)
            return io.BytesIO(raw)

    try:
        with patch("urllib.request.build_opener", lambda *args, **kwargs: Replay()):
            return adapter(case["source"], model=model, timeout=60), None
    except (ValueError, urllib.error.URLError, OSError) as exc:
        return None, str(exc)


def score_note(source, expected, report):
    total = len(expected)
    result = dict(
        source_preserved=False,
        category_correct=0,
        category_total=total,
        question_as_assertion=0,
        mismatches=[],
        actual_labels=[None] * total,
    )
    if not isinstance(report, dict) or not isinstance(report.get("sections"), dict):
        return result
    expected_keys = {(s["start"], s["end"], s["quote"]) for s in expected}
    counts, assigned, covered = Counter(), {}, set()
    valid = (
        report.get("raw_transcript") == source
        and report.get("source_transcript_sha256") == hashlib.sha256(source.encode()).hexdigest()
        and report.get("requires_human_review") is True
        and report.get("clinical_validation") == "not_established"
    )
    for label, spans in report["sections"].items():
        if label not in LABELS or not isinstance(spans, list):
            valid = False
            continue
        for span in spans:
            if not isinstance(span, dict):
                valid = False
                continue
            start, end, quote = (span.get(k) for k in ("start", "end", "quote"))
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(source)
                or source[start:end] != quote
            ):
                valid = False
                continue
            key = (start, end, quote)
            counts[key] += 1
            assigned[key] = label
            covered.update(range(start, end))
    valid = (
        valid
        and set(counts) == expected_keys
        and all(n == 1 for n in counts.values())
        and all(i in covered for i, char in enumerate(source) if not char.isspace())
    )
    actual = [
        assigned.get((s["start"], s["end"], s["quote"]))
        if counts[(s["start"], s["end"], s["quote"])] == 1
        else None
        for s in expected
    ]
    result.update(
        source_preserved=valid,
        actual_labels=actual,
        category_correct=sum(
            label == span["label"] for label, span in zip(actual, expected, strict=True)
        )
        if valid
        else 0,
        question_as_assertion=sum(
            s["quote"].rstrip().endswith("?") and label not in (None, "unassigned")
            for s, label in zip(expected, actual, strict=True)
        ),
        mismatches=[
            {"quote": s["quote"], "expected": s["label"], "actual": label}
            for s, label in zip(expected, actual, strict=True)
            if label != s["label"]
        ],
    )
    return result


def check_matrix(rows, cases, models):
    planned = [
        (model, variant, case)
        for model in models
        for variant in ("baseline", "enhanced")
        for case in cases
    ]
    require(len(rows) == len(planned), "Incomplete or extra note matrix rows")
    for row, (model, variant, case) in zip(rows, planned, strict=True):
        require(
            row.get("model") == model
            and row.get("model_digest") == models[model]
            and row.get("variant") == variant
            and all(row.get(k) == case[k] for k in ("id", "language", "group")),
            "Note matrix identity/digest/order mismatch",
        )


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_inside(root, name):
    path = (root / name).resolve()
    require(
        path.is_relative_to(root.resolve()) and path.is_file(),
        "Capture missing or outside approved root",
    )
    return path.read_bytes()


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("_frozen_note_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ollama_positional_note


def run_audit(history, workspace, output):
    protocol_raw = read_inside(history, "protocol.json")
    protocol = strict_json(protocol_raw)
    cases_raw = read_inside(history, "cases.json")
    require(sha(cases_raw) == protocol["cases_sha256"], "Cases hash differs from frozen protocol")
    require(
        tuple(protocol["models"]) == MODELS and protocol["calls"] == 48,
        "Unexpected note experiment matrix",
    )
    require(
        protocol["independently_clinically_verified"] is False,
        "This replay does not validate clinical gold",
    )
    adapter_path = None
    for name, digest in protocol["files"].items():
        raw = read_inside(workspace, name)
        require(sha(raw) == digest, "Frozen source hash mismatch: " + name)
        if name.replace("\\", "/").endswith("/src/clinical_asr/notes.py"):
            adapter_path = (workspace / name).resolve()
    require(adapter_path is not None, "Frozen adapter not identified")
    progress = strict_json(read_inside(history, "run/progress.json"))
    require(
        progress["status"] == "complete"
        and progress["attempted"] == progress["planned"] == 48
        and progress["models_after"] == protocol["models"]
        and progress["identities_unchanged"] is True,
        "Incomplete execution or model identity drift",
    )
    cases = strict_json(cases_raw)
    rows_raw = read_inside(history, "run/results.jsonl")
    rows = [strict_json(line) for line in rows_raw.splitlines()]
    check_matrix(rows, cases, protocol["models"])
    frozen = load_adapter(adapter_path)
    by_id = {c["id"]: c for c in cases}
    require(len(by_id) == len(cases) == 8, "Duplicate or unexpected cases")
    reconstructed = []
    hashes = {
        "protocol.json": sha(protocol_raw),
        "cases.json": sha(cases_raw),
        "run/results.jsonl": sha(rows_raw),
    }
    for i, row in enumerate(rows):
        case = by_id[row["id"]]
        captures = {}
        for route in ("show", "chat"):
            prefix = f"run/calls/{i:03d}-{route}"
            request_raw = read_inside(history, prefix + "-request.json")
            response_raw = read_inside(history, prefix + "-response.json")
            captures[route] = {"request": strict_json(request_raw), "response": response_raw}
            hashes[prefix + "-request.json"] = sha(request_raw)
            hashes[prefix + "-response.json"] = sha(response_raw)
        require(
            row["actual_generation_called"] is True, "Generation flag differs from captured call"
        )
        policy = protocol["policy"] if row["variant"] == "enhanced" else None
        old_note, old_error = reconstruct(frozen, case, row["model"], captures, policy)
        require(
            (row["status"] == "ok" and old_error is None and row["report"] == old_note)
            or (row["status"] == "failed" and row.get("error") == old_error and old_note is None),
            "Historical report/failure differs from frozen raw replay",
        )
        original_score = score_note(case["source"], case["expected_spans"], old_note)
        require(
            row["metrics"]["category_correct"] == original_score["category_correct"]
            and row["metrics"]["category_total"] == original_score["category_total"],
            "Historical metric mismatch",
        )
        new_note, new_error = reconstruct(
            notes.ollama_positional_note, case, row["model"], captures, policy
        )
        reconstructed.append(
            {
                "index": i,
                "model": row["model"],
                "variant": row["variant"],
                "case": case["id"],
                "language": case["language"],
                "historical": original_score,
                "candidate": score_note(case["source"], case["expected_spans"], new_note),
                "candidate_report": new_note,
                "candidate_error": new_error,
            }
        )
    summary = {}
    for model in MODELS:
        for variant in ("baseline", "enhanced"):
            selected = [r for r in reconstructed if r["model"] == model and r["variant"] == variant]
            summary[f"{model}/{variant}"] = {
                version: {
                    "notes": len(selected),
                    "correct": sum(r[version]["category_correct"] for r in selected),
                    "total": sum(r[version]["category_total"] for r in selected),
                    "source_preserved_notes": sum(r[version]["source_preserved"] for r in selected),
                    "question_as_assertion": sum(
                        r[version]["question_as_assertion"] for r in selected
                    ),
                }
                for version in ("historical", "candidate")
            }
    output.mkdir(parents=True, exist_ok=False)
    (output / "rows.json").write_text(
        json.dumps(reconstructed, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    result = {
        "scope": "48 historical authored/development notes only; engineering labels, not clinical gold",
        "inference_calls": 0,
        "clinical_validation": "not_established",
        "summary": summary,
        "input_hashes": hashes,
        "candidate_notes_sha256": sha(Path(notes.__file__).read_bytes()),
        "evaluator_sha256": sha(Path(__file__).read_bytes()),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.history.resolve(), args.workspace.resolve(), args.output.resolve())
    print(json.dumps(result["summary"], indent=2))
