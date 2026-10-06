import importlib.util
import sys
import time
import types
from pathlib import Path

import pytest

from clinical_asr.corrections import ollama_corrections

SPEC = importlib.util.spec_from_file_location(
    "critical_probe", Path(__file__).parents[1] / "tools" / "critical_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_failed_call_is_not_successful_abstention():
    result = probe.score_correction({"text": "cisplatin", "expected": []}, None)
    assert not result["passed"]
    assert not result["successful_abstention"]
    assert result["status"] == "failed"


def test_empty_valid_report_is_successful_control():
    report = ollama_corrections("", "en")
    result = probe.score_correction({"text": "", "expected": []}, report)
    assert result["passed"]
    assert result["successful_abstention"]


def test_source_mismatch_is_integrity_failure():
    report = ollama_corrections("", "en")
    with pytest.raises(ValueError, match="source"):
        probe.score_correction({"text": "cisplatin", "expected": []}, report)


def test_missing_expected_edit_does_not_pass():
    result = probe.score_correction(
        {"text": "", "expected": [["cis plat in", "cisplatin"]]},
        ollama_corrections("", "en"),
    )
    assert not result["passed"]
    assert result["expected_edits_found"] == 0


def test_envelope_identity_is_required():
    with pytest.raises(ValueError, match="identity"):
        probe.check_identity({"model": "other", "done": True}, "qwen3.5:4b")


def test_reference_terms_are_derived_only_from_script():
    assert probe.reference_terms("cisplatin and Cisplatin", ["cisplatin", "tamoxifen"]) == [
        "cisplatin"
    ]


def asr_fixture(tmp_path, monkeypatch):
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture-only-no-decoding")
    output = tmp_path / "run"
    output.mkdir()
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"models": {}}')
    monkeypatch.setitem(
        sys.modules,
        "launch_candidate",
        types.SimpleNamespace(
            verify_models=lambda _: {"turbo": str(tmp_path), "large-v3": str(tmp_path)}
        ),
    )
    monkeypatch.setattr(probe, "load_example", lambda _: (str(audio), "en", "script", "cisplatin"))

    class Engine:
        def __init__(self, config):
            pass

        def transcribe_file(self, path, language):
            return {"text": "cisplatin"}

    monkeypatch.setattr(probe, "Transcriber", Engine)
    return output, protocol


def test_asr_string_asset_path_is_hashed_and_bound(tmp_path, monkeypatch):
    output, protocol = asr_fixture(tmp_path, monkeypatch)
    probe.run_asr(output, time.monotonic() + 10, protocol)
    assert probe.json.loads((output / "summary.json").read_text())["completed"] == 8


def test_asr_evaluator_failure_is_not_an_inference_error(tmp_path, monkeypatch):
    output, protocol = asr_fixture(tmp_path, monkeypatch)

    def broken(*args, **kwargs):
        raise ValueError("Evaluator defect")

    monkeypatch.setattr(probe, "score_text", broken)
    with pytest.raises(ValueError, match="Evaluator defect"):
        probe.run_asr(output, time.monotonic() + 10, protocol)


def test_rejected_noop_preserves_output_but_is_not_model_abstention():
    text = "cisplatin"
    report = ollama_corrections("", "en")
    report.update(
        raw_transcript=text,
        source_transcript_sha256=probe.hashlib.sha256(text.encode()).hexdigest(),
        rejected_proposals=[
            {
                "span_id": "s0",
                "before": text,
                "after": text,
                "reason": "Already correct",
                "rejection": "unchanged_text",
            }
        ],
    )
    result = probe.score_correction({"text": text, "expected": []}, report)
    assert not result["passed"]
    assert result["output_matches_expected"]
    assert not result["successful_abstention"]
