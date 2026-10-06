import hashlib
import io
import json
import urllib.error
import urllib.request

import pytest

from clinical_asr import notes


def api():
    function = getattr(notes, "ollama_positional_note", None)
    assert callable(function), "The production positional note adapter is missing"
    return function


@pytest.fixture
def wire(monkeypatch):
    state = {"content": '{"labels":["subjective","plan"]}', "requests": []}

    class Opener:
        def open(self, request, timeout):
            if request.full_url.endswith("/api/show"):
                metadata = {"details": {"format": "gguf"},
                            "model_info": {"general.architecture": "llama"}, "template": ""}
                return io.BytesIO(json.dumps(metadata | state.get("metadata", {})).encode())
            state["requests"].append((json.loads(request.data), timeout))
            if state.get("error"):
                raise state["error"]
            envelope = {"done": True, "done_reason": "stop", "model": "llama3.1:latest",
                        "message": {"role": "assistant", "content": state["content"]}}
            return io.BytesIO(json.dumps(envelope | state.get("envelope", {})).encode())

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    return state


def test_positional_note_retains_all_source_facts_and_offsets(wire):
    text = "No fever. Continue cisplatin 2.5 mg."
    note = api()(text)
    assert note["sections"]["subjective"] == [{"quote": "No fever.", "start": 0, "end": 9}]
    assert note["sections"]["plan"] == [
        {"quote": "Continue cisplatin 2.5 mg.", "start": 10, "end": 36}]
    assert note["raw_transcript"] == text
    assert note["source_transcript_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert note["uncovered_source"] == []
    assert note["requires_human_review"] is True
    assert note["clinical_validation"] == "not_established"
    assert note["not_abstractive_summary"] is True
    body, timeout = wire["requests"][0]
    assert body["model"] == "llama3.1:latest"
    assert "think" not in body
    assert timeout == 45
    assert body["format"]["properties"]["labels"]["minItems"] == 2
    assert body["format"]["properties"]["labels"]["maxItems"] == 2
    assert json.loads(body["messages"][1]["content"])["evidence_spans"] == [
        "No fever.", "Continue cisplatin 2.5 mg."]


def test_prompt_distinguishes_patient_denials_from_measured_findings(wire):
    api()("She denies fever. Temperature measured at 37 C.")
    system = wire["requests"][0][0]["messages"][0]["content"]
    assert "including denied or absent symptoms" in system
    assert "Only measured examination findings" in system
    assert "not an objective finding" in system


def test_line_aware_spans_do_not_merge_speakers_or_change_questions(wire):
    wire["content"] = '{"labels":["unassigned","subjective","objective"]}'
    text = "Doctor: Pain?\nPatient: No pain\nHER2 negative."
    note = api()(text)
    assert note["sections"]["unassigned"][0]["quote"] == "Doctor: Pain?"
    assert note["sections"]["subjective"][0]["quote"] == "Patient: No pain"
    assert note["sections"]["objective"][0]["quote"] == "HER2 negative."
    assert note["uncovered_source"] == []


@pytest.mark.parametrize("content", [
    '{"labels":["subjective"]}',
    '{"labels":["subjective","plan","objective"]}',
    '{"labels":["subjective","prescribe"]}',
    '{"labels":["subjective",1]}',
    '{"labels":["subjective","plan"],"advice":"change dose"}',
    '{"labels":["subjective","plan"],"labels":["objective","plan"]}',
    '{"labels":["subjective",NaN]}',
    '{"subjective":["s0"],"plan":["s1"]}',
])
def test_incomplete_extra_or_invented_category_outputs_fail_closed(wire, content):
    wire["content"] = content
    with pytest.raises(ValueError):
        api()("No fever. Continue 2.5 mg.")


@pytest.mark.parametrize("envelope", [
    {"done": False}, {"done_reason": "length"}, {"error": "failed"},
    {"message": {"content": None}},
])
def test_transport_success_is_not_credit_for_incomplete_generation(wire, envelope):
    wire["envelope"] = envelope
    with pytest.raises(ValueError):
        api()("No fever. Continue 2.5 mg.")


def test_qwen_note_request_disables_thinking_without_changing_facts(wire):
    note = api()("No fever. Continue cisplatin 2.5 mg.", model="qwen3.5:4b")
    assert wire["requests"][0][0]["think"] is False
    assert note["sections"]["plan"][0]["quote"] == "Continue cisplatin 2.5 mg."


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected_before_any_inference(wire, timeout):
    with pytest.raises(ValueError, match="timeout"):
        api()("No fever.", timeout=timeout)
    assert wire["requests"] == []


def test_external_endpoint_is_rejected_before_inference(wire):
    with pytest.raises(ValueError, match="local"):
        api()("No fever.", endpoint="https://paid.example.com")
    assert wire["requests"] == []


def test_empty_source_does_not_infer_or_invent(wire):
    note = api()(" \n")
    assert all(not section for section in note["sections"].values())
    assert wire["requests"] == []


@pytest.mark.parametrize("text", ["x" * 12001, "Pain. " * 257], ids=["characters", "spans"])
def test_oversized_inputs_are_not_silently_truncated(wire, text):
    with pytest.raises(ValueError, match="limit"):
        api()(text)
    assert wire["requests"] == []


def test_offline_service_returns_an_actionable_error_not_an_empty_note(wire):
    wire["error"] = urllib.error.URLError("offline")
    with pytest.raises(ValueError, match="Ollama"):
        api()("No fever.")


def test_frontend_uses_positional_backend_and_retains_dose(wire):
    from clinical_asr.demo import note_preview

    markup, note = note_preview("No fever. Continue cisplatin 2.5 mg.",
                                "positional", "llama3.1:latest")
    assert note["method"] == "ollama_positional_categories"
    assert "Continue cisplatin 2.5 mg." in markup
    assert note["uncovered_source"] == []


def test_frontend_defaults_to_evaluated_english_candidate():
    from clinical_asr.demo import build_app

    components = build_app().config["components"]
    by_label = {c["props"].get("label"): c["props"] for c in components}
    assert by_label["Medical correction model"]["value"] == "qwen3.5:4b"
    assert by_label["Note method"]["value"] == "positional"
    assert by_label["Local note model"]["value"] == "llama3.1:latest"


def test_unknown_note_method_cannot_silently_call_a_model(wire):
    from clinical_asr.demo import note_preview

    with pytest.raises(ValueError, match="method"):
        note_preview("No fever.", "typo", "llama3.1:latest")
    assert wire["requests"] == []


def test_cli_positional_note_writes_source_bound_output(wire, tmp_path):
    from clinical_asr.cli import main

    source = tmp_path / "fictional.txt"
    source.write_text("No fever. Continue cisplatin 2.5 mg.", encoding="utf-8")
    output = tmp_path / "note.json"
    assert main(["notes", str(source), "--method", "positional", "--output", str(output)]) == 0
    note = json.loads(output.read_text(encoding="utf-8"))
    assert note["method"] == "ollama_positional_categories"
    assert note["sections"]["plan"][0]["quote"] == "Continue cisplatin 2.5 mg."


@pytest.mark.parametrize("model", ["gemma4:cloud", "remote/model", "unknown:latest"])
def test_unverified_or_cloud_model_names_never_receive_source(wire, model):
    with pytest.raises(ValueError, match="model"):
        api()("No fever.", model=model)
    assert wire["requests"] == []


@pytest.mark.parametrize("metadata", [
    {"remote_host": "https://ollama.com", "remote_model": "some-model"},
    {"details": {"format": "remote"}}, {"model_info": {}},
])
def test_remote_or_unverifiable_installed_alias_never_receives_source(wire, metadata):
    wire["metadata"] = metadata
    with pytest.raises(ValueError, match="local model"):
        api()("No fever.")
    assert wire["requests"] == []


def test_token_heavy_source_is_rejected_before_chat_not_truncated(wire):
    with pytest.raises(ValueError, match="context"):
        api()("abc_xyz/123!" * 600)
    assert wire["requests"] == []


def test_legacy_cli_also_rejects_cloud_models_without_sending_source(wire, tmp_path):
    from clinical_asr.cli import main

    source = tmp_path / "fictional.txt"
    source.write_text("No fever.", encoding="utf-8")
    output = tmp_path / "note.json"
    assert main(["notes", str(source), "--method", "ollama", "--model", "gemma4:cloud",
                 "--output", str(output)]) == 2
    assert wire["requests"] == []
    assert not output.exists()
