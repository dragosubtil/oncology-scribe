"""Summarize completed public probe runs without selecting models on test data."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import jiwer

from clinical_asr.cli import write_json
from clinical_asr.metrics import normalize_text
from clinical_asr.schema import file_sha256, load_manifest

ROOT = Path(__file__).resolve().parents[1]


def error_spans(reference, hypothesis):
    aligned = jiwer.process_words(normalize_text(reference), normalize_text(hypothesis))
    return [{"type": item.type,
             "reference": aligned.references[0][item.ref_start_idx:item.ref_end_idx],
             "hypothesis": aligned.hypotheses[0][item.hyp_start_idx:item.hyp_end_idx]}
            for item in aligned.alignments[0] if item.type != "equal"]


def main():
    manifests = {
        "primock-dev": ROOT / "data/primock-probe/dev.jsonl",
        "primock-test": ROOT / "data/primock-probe/test.jsonl",
        "fleurs-test": ROOT / "data/public-evaluation/manifest.jsonl",
    }
    samples = {key: load_manifest(path, require_audio=True) for key, path in manifests.items()}
    hashes = {key: {file_sha256(Path(item.audio_path)) for item in rows}
              for key, rows in samples.items()}
    if hashes["primock-dev"] & hashes["primock-test"]:
        raise ValueError("Development and held-out recordings overlap by file hash")
    results = []
    for corpus in manifests:
        models = ["small", "turbo", "large-v3"]
        if corpus == "primock-dev":
            models.append("turbo-no-vad")
        for model in models:
            folder = ROOT / "runs" / f"{corpus}-{model}"
            run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            if run["status"] != "complete" or run["manifest_sha256"] != file_sha256(manifests[corpus]):
                raise ValueError(f"Incomplete run or changed manifest: {folder.name}")
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            predictions = {row["sample_id"]: row for row in map(
                json.loads, (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
            if len(predictions) != len(samples[corpus]):
                raise ValueError("Prediction count does not match manifest")
            details = []
            flags = Counter()
            for sample, row in zip(samples[corpus], report["rows"], strict=True):
                if row["sample_id"] != sample.sample_id:
                    raise ValueError("Unexpected report ordering")
                prediction = predictions[sample.sample_id]
                if prediction["source_audio_sha256"] != file_sha256(Path(sample.audio_path)):
                    raise ValueError("Prediction recording identity changed")
                flags.update(row["scores"]["review_flags"])
                details.append({"sample_id": sample.sample_id, "language": sample.language,
                                "reference": sample.reference, "hypothesis": prediction["text"],
                                "scores": row["scores"],
                                "errors": error_spans(sample.reference, prediction["text"])})
            write_json(folder / "error_analysis.json", {
                "license": "CC-BY-4.0", "source": str(manifests[corpus].relative_to(ROOT)),
                "clinical_adjudication": "not_performed", "rows": details})
            results.append({"run": folder.name, "groups": report["groups"],
                            "heuristic_review_flags": dict(flags),
                            "manifest_sha256": run["manifest_sha256"], "config": run["config"],
                            "report_sha256": file_sha256(folder / "report.json"),
                            "predictions_sha256": file_sha256(folder / "predictions.jsonl")})
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "clinical_validation": "not_established", "free_local_inference": True,
        "candidate_selected_on_development": "turbo, VAD on; see PUBLIC_PROBE_PROTOCOL.md",
        "test_data_used_for_tuning": False, "source_audio_hash_overlap_dev_test": False,
        "models_fine_tuned": False, "medical_llm_edits_in_wer": False,
        "latency_experiment": "not_controlled; other local verification ran concurrently",
        "no_swiss_german_or_oncology_data": True, "results": results,
    }
    write_json(ROOT / "verification/public_probe_summary.json", summary)
    print(json.dumps([{ "run": row["run"], "scores": [
        {"language": g["language"], "clips": g["clips"], "words": g["reference_words"],
         "normalized_wer_percent": round(100 * g["micro_wer"], 2),
         "raw_wer_percent": round(100 * g["raw_micro_wer"], 2)}
        for g in row["groups"]]} for row in results], indent=2))


if __name__ == "__main__":
    main()
