"""Bounded local model probes on authored fixtures; never clinical validation."""

import argparse
import gc
import hashlib
import io
import json
import os
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

from clinical_asr.corrections import (
    _SPELLING_CONTEXT,
    apply_reviewed_corrections,
    ollama_corrections,
)
from clinical_asr.demo_examples import load_example
from clinical_asr.metrics import score_text
from clinical_asr.notes import validate_local_model
from clinical_asr.schema import file_sha256
from clinical_asr.transcribe import ASRConfig, Transcriber

MODELS = ("qwen3.5:4b", "medgemma:4b", "llama3.1:latest")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def check_identity(envelope, model):
    if not isinstance(envelope, dict) or envelope.get("model") != model:
        raise ValueError("Response model identity does not match the requested model")


def score_correction(case, report):
    expected = {tuple(item) for item in case["expected"]}
    if report is None:
        return {
            "status": "failed",
            "passed": False,
            "successful_abstention": False,
            "expected_edits_found": 0,
            "expected_edits": len(expected),
        }
    if report["raw_transcript"] != case["text"]:
        raise ValueError("Correction report source mismatch")
    eligible = [item for item in report["proposals"] if not item["protected"]]
    simulated = apply_reviewed_corrections(report, [item["id"] for item in eligible], True)
    actual = {(item["before"], item["after"]) for item in eligible}
    rejected = report.get("rejected_proposals", [])
    unexpected_raw = sorted(
        {
            (item["before"], item["after"])
            for item in report["proposals"] + rejected
            if item["before"].casefold() != item["after"].casefold()
        }
        - expected
    )
    target = case["text"]
    for before, after in case["expected"]:
        target = target.replace(before, after)
    return {
        "status": "scored",
        "passed": actual == expected
        and not unexpected_raw
        and not rejected
        and simulated["reviewed_transcript"] == target,
        "output_matches_expected": simulated["reviewed_transcript"] == target,
        "successful_abstention": not expected and not report["proposals"] and not rejected,
        "expected_edits_found": len(actual & expected),
        "expected_edits": len(expected),
        "unexpected_eligible_edits": sorted(actual - expected),
        "protected_proposals": sum(item["protected"] for item in report["proposals"]),
        "rejected_proposals": len(rejected),
        "unexpected_raw_suggestions": unexpected_raw,
        "simulated_reviewed_text": simulated["reviewed_transcript"],
        "audio_confirmation_is_simulated": True,
    }


def reference_terms(reference, candidates):
    return [term for term in candidates if term.casefold() in reference.casefold()]


def tags():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:11434/api/tags", timeout=10) as response:
        result = json.loads(response.read(1024 * 1024 + 1))
    found = {item["name"]: item["digest"] for item in result["models"]}
    for model in MODELS:
        if model not in found or len(found[model]) != 64:
            raise ValueError(f"Required installed model unavailable: {model}")
    return {model: found[model] for model in MODELS}


class Capture:
    """Capture exact adapter traffic, including failed requests, without changing it."""

    def __init__(self):
        self.calls = []
        self.factory = urllib.request.build_opener

    def opener(self, *handlers):
        delegate = self.factory(*handlers)
        owner = self

        class Opener:
            def open(self, request, timeout):
                entry = {
                    "url": request.full_url,
                    "request": json.loads(request.data),
                    "request_sha256": hashlib.sha256(request.data).hexdigest(),
                }
                owner.calls.append(entry)
                try:
                    with delegate.open(request, timeout=timeout) as response:
                        raw = response.read(1024 * 1024 + 1)
                    entry["response_utf8"] = raw.decode("utf-8")
                    entry["response_sha256"] = hashlib.sha256(raw).hexdigest()
                    if request.full_url.endswith("/api/chat"):
                        check_identity(json.loads(raw), entry["request"]["model"])
                    return io.BytesIO(raw)
                except Exception as exc:
                    entry["transport_or_identity_error"] = str(exc)
                    raise

        return Opener()


def run_llm(output, deadline, cases_path):
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = document["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    digest = file_sha256(cases_path)
    pinned = tags()
    save(
        output / "protocol.json",
        {
            "cases": document,
            "cases_sha256": digest,
            "model_digests": pinned,
            "adapter_sha256": file_sha256(
                Path(__file__).parents[1] / "src/clinical_asr/corrections.py"
            ),
            "runner_sha256": file_sha256(Path(__file__)),
            "planned_calls": len(cases) * len(MODELS),
            "timeout_seconds": 45,
            "zero_temperature": True,
            "new_download_bytes": 0,
            "clinical_validation": "not_established",
            "max_proposals": {"medgemma": 1, "others": 16},
            "expected_edits_per_case_at_most": 1,
            "weight_training": False,
        },
    )
    rows = []
    for model in MODELS:
        if tags() != pinned:
            raise ValueError("Model tags changed before inference")
        for case in cases:
            if time.monotonic() >= deadline or (output / "STOP").exists():
                break
            capture, report, error = Capture(), None, None
            started = time.perf_counter()
            try:
                with patch("urllib.request.build_opener", capture.opener):
                    validate_local_model(model, "http://127.0.0.1:11434", 10)
                    report = ollama_corrections(
                        case["text"], case["language"], model=model, timeout=45
                    )
            except (ValueError, OSError) as exc:
                error = str(exc)
            row = {
                "case_id": case["id"],
                "language": case["language"],
                "kind": case["kind"],
                "model": model,
                "digest": pinned[model],
                "report": report,
                "error": error,
                "seconds": time.perf_counter() - started,
                "calls": capture.calls,
                "score": score_correction(case, report),
            }
            rows.append(row)
            save(output / f"{len(rows) - 1:03d}.json", row)
            save(
                output / "progress.json",
                {"completed": len(rows), "planned": len(cases) * len(MODELS)},
            )
            print(
                f"{len(rows)}/{len(cases) * len(MODELS)} {model} {case['id']} passed={row['score']['passed']} error={error}",
                flush=True,
            )
        if tags() != pinned:
            raise ValueError("Model tags changed after inference")
    if file_sha256(cases_path) != digest:
        raise ValueError("Frozen cases changed during inference")
    save(
        output / "summary.json",
        {
            "completed": len(rows),
            "planned": len(cases) * len(MODELS),
            "models": {
                model: {
                    "completed": sum(row["model"] == model for row in rows),
                    "planned": len(cases),
                    "passed": sum(row["model"] == model and row["score"]["passed"] for row in rows),
                    "output_matches_expected": sum(
                        row["model"] == model and row["score"].get("output_matches_expected", False)
                        for row in rows
                    ),
                    "successful_control_abstentions": sum(
                        row["model"] == model and row["score"]["successful_abstention"]
                        for row in rows
                    ),
                    "planned_controls": sum(not case["expected"] for case in cases),
                    "failed_calls": sum(
                        row["model"] == model and row["error"] is not None for row in rows
                    ),
                    "unsafe_eligible_suggestions": sum(
                        len(row["score"].get("unexpected_eligible_edits", []))
                        for row in rows
                        if row["model"] == model
                    ),
                }
                for model in MODELS
            },
            "clinical_validation": "not_established",
            "models_unchanged": tags() == pinned,
        },
    )


def run_asr(output, deadline, protocol_path):
    from launch_candidate import verify_models

    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    installed = verify_models(protocol)
    save(
        output / "protocol.json",
        {
            "installed_models": protocol["models"],
            "runner_sha256": file_sha256(Path(__file__)),
            "reference_kind": "generation_script_not_independent_gold",
            "new_download_bytes": 0,
            "clinical_validation": "not_established",
        },
    )
    rows = []
    for model, path in installed.items():
        engine = Transcriber(
            ASRConfig(model=path, device="cpu", compute_type="int8", local_files_only=True)
        )
        for identifier in ("en-01", "de-01", "fr-01", "it-01"):
            if time.monotonic() >= deadline or (output / "STOP").exists():
                break
            audio, language, _, reference = load_example(identifier)
            audio_hash = file_sha256(Path(audio))
            started, result, metrics, error = time.perf_counter(), None, None, None
            try:
                result = engine.transcribe_file(audio, language=language)
            except (ValueError, OSError, RuntimeError) as exc:
                error = str(exc)
            if result is not None:
                terms = reference_terms(
                    reference,
                    (*_SPELLING_CONTEXT, "cisplatine", "cisplatino", "HER2", "PD-L1", "EGFR"),
                )
                metrics = score_text(reference, result["text"], terms, language)
            if file_sha256(Path(audio)) != audio_hash:
                raise ValueError("Audio changed during inference")
            row = {
                "example_id": identifier,
                "model": model,
                "language": language,
                "audio_path": audio,
                "audio_sha256": audio_hash,
                "generation_reference": reference,
                "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                "reference_kind": "generation_script_not_independent_gold",
                "result": result,
                "metrics": metrics,
                "error": error,
                "seconds": time.perf_counter() - started,
            }
            rows.append(row)
            save(output / f"{len(rows) - 1:03d}.json", row)
            print(f"{model} {identifier} error={error} metrics={metrics}", flush=True)
        del engine
        gc.collect()
    verify_models(protocol)
    save(
        output / "summary.json",
        {
            "completed": len(rows),
            "planned": 8,
            "models_verified_after": True,
            "clinical_validation": "not_established",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("llm", "asr"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("examples/critical-cases.json"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.mode == "asr" and args.protocol is None:
        parser.error("ASR requires the installed-artifact protocol")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parents[1]
    source_files = [
        Path(__file__),
        *[
            root / "src/clinical_asr" / name
            for name in ("corrections.py", "transcribe.py", "audio.py", "metrics.py")
        ],
    ]
    source_hashes = {str(path): file_sha256(path) for path in source_files}
    frozen = args.output / "frozen-source"
    frozen.mkdir()
    for path in source_files:
        (frozen / path.name).write_bytes(path.read_bytes())
    save(args.output / "source-hashes.json", source_hashes)
    deadline = time.monotonic() + args.seconds
    if args.mode == "llm":
        run_llm(args.output, deadline, args.cases)
    else:
        run_asr(args.output, deadline, args.protocol)
    if any(file_sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise ValueError("Source code changed during the model probe")
    save(args.output / "source-confirmation.json", {"unchanged": True})


if __name__ == "__main__":
    main()
