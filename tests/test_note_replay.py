import hashlib
import json
import urllib.request

import pytest

from tools import replay_note_audit as audit


def test_json_rejects_duplicate_fields_and_nonfinite_values():
    for raw in ('{"done":true,"done":false}', '{"x":NaN}', '{"x":Infinity}'):
        with pytest.raises(audit.AuditIntegrityError):
            audit.strict_json(raw)


@pytest.mark.parametrize(
    "body,response",
    [
        ({"model": "wrong"}, {"model": "qwen3.5:4b", "done": True}),
        ({"model": "qwen3.5:4b"}, {"model": "wrong", "done": True}),
    ],
)
def test_call_model_identity_cannot_be_relabelled(body, response):
    with pytest.raises(audit.AuditIntegrityError, match="model"):
        audit.check_call(body, json.dumps(response).encode(), "qwen3.5:4b")


def test_malformed_raw_response_is_left_for_adapter_failure_not_success_credit():
    assert audit.check_call({"model": "qwen3.5:4b"}, b"broken JSON", "qwen3.5:4b") is None


def test_internal_evaluator_error_is_never_counted_as_a_model_failure():
    def broken(*args, **kwargs):
        raise audit.AuditIntegrityError("request identity mismatch")

    with pytest.raises(audit.AuditIntegrityError):
        audit.reconstruct(broken, {"source": "No fever."}, "qwen3.5:4b", {})


@pytest.mark.parametrize("failure", ["transport", "malformed", "incomplete"])
def test_raw_failed_calls_replay_as_failure_and_receive_no_category_credit(failure):
    body = {"model": "qwen3.5:4b"}
    capture = {"request": body}
    if failure == "transport":
        capture["transport_error"] = "connection closed"
    else:
        capture["response"] = (
            b"broken JSON" if failure == "malformed" else b'{"model":"qwen3.5:4b","done":false}'
        )

    def adapter(source, model, timeout):
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode()
        )
        with urllib.request.build_opener().open(request, timeout=timeout) as response:
            envelope = json.loads(response.read())
        if not envelope.get("done"):
            raise ValueError("Incomplete response")
        raise AssertionError("Failed call must not reach a successful report")

    report, error = audit.reconstruct(
        adapter, {"source": "No fever."}, "qwen3.5:4b", {"chat": capture}
    )
    assert report is None and error
    assert (
        audit.score_note(
            "No fever.",
            [{"start": 0, "end": 9, "quote": "No fever.", "label": "subjective"}],
            report,
        )["category_correct"]
        == 0
    )


def test_internal_assertion_is_not_swallowed_under_replay():
    def broken(*args, **kwargs):
        raise AssertionError("Evaluator defect")

    with pytest.raises(AssertionError):
        audit.reconstruct(broken, {"source": "No fever."}, "qwen3.5:4b", {})


def fixture_note():
    source = "Has the tumour progressed?\nNo fever."
    spans = [
        {"start": 0, "end": 26, "quote": "Has the tumour progressed?", "label": "unassigned"},
        {"start": 27, "end": 36, "quote": "No fever.", "label": "subjective"},
    ]
    report = {
        "raw_transcript": source,
        "source_transcript_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "requires_human_review": True,
        "clinical_validation": "not_established",
        "sections": {
            "assessment": [dict(quote=spans[0]["quote"], start=0, end=26)],
            "subjective": [dict(quote="No fever.", start=27, end=36)],
        },
    }
    return source, spans, report


def test_score_counts_question_as_assertion_even_with_exact_source_preserved():
    source, spans, report = fixture_note()
    result = audit.score_note(source, spans, report)
    assert result["source_preserved"] is True
    assert result["category_correct"] == 1
    assert result["category_total"] == 2
    assert result["question_as_assertion"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "tampered_quote", "wrong_hash"])
def test_integrity_failure_cannot_receive_category_credit(mutation):
    source, spans, report = fixture_note()
    if mutation == "duplicate":
        report["sections"]["subjective"] *= 2
    elif mutation == "missing":
        report["sections"]["subjective"] = []
    elif mutation == "tampered_quote":
        report["sections"]["subjective"][0]["quote"] = "Fever."
    else:
        report["source_transcript_sha256"] = "wrong"
    result = audit.score_note(source, spans, report)
    assert result["source_preserved"] is False
    assert result["category_correct"] == 0


def test_matrix_requires_every_declared_model_variant_case_once():
    cases = [{"id": "one", "group": "authored", "language": "en", "source": "No fever."}]
    models = {"qwen3.5:4b": "frozen-digest"}
    rows = [
        {
            "id": "one",
            "group": "authored",
            "language": "en",
            "model": "qwen3.5:4b",
            "model_digest": "frozen-digest",
            "variant": variant,
        }
        for variant in ("baseline", "enhanced")
    ]
    audit.check_matrix(rows, cases, models)
    for mutated in (
        rows[:1],
        rows + [rows[0]],
        rows[::-1],
        [rows[0] | {"model_digest": "wrong"}, rows[1]],
    ):
        with pytest.raises(audit.AuditIntegrityError):
            audit.check_matrix(mutated, cases, models)
