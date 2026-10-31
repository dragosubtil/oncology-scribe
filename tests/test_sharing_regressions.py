"""Regression cases for source publication and quoted clinical questions."""

import io
import json
import urllib.request

import gradio as gr
import pytest

from clinical_asr import demo, notes
from clinical_asr.timing import TimingHistory


@pytest.mark.parametrize("ending", [
    '?"', "?'", '?\u201d', '?\u2019', '?\u00bb', '?")', '?" ]  ',
    '?\u201c', '?\u2018', '?\u203a', '?\u00a0\u00bb', '?\u202f\u00bb',
])
def test_quoted_question_is_not_an_assessment(monkeypatch, ending):
    class Opener:
        def open(self, request, **options):
            if request.full_url.endswith("/api/show"):
                return io.BytesIO(json.dumps({"details": {"format": "gguf"},
                    "model_info": {"general.architecture": "llama"}, "template": ""}).encode())
            spans = json.loads(json.loads(request.data)["messages"][1]["content"])["evidence_spans"]
            return io.BytesIO(json.dumps({"done": True, "done_reason": "stop", "message": {
                "content": json.dumps({"labels": ["assessment"] * len(spans)})}}).encode())

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    source = 'Clinician: "Has the tumour progressed' + ending
    result = notes.ollama_positional_note(source)
    assert not result["sections"]["assessment"], result["sections"]
    assert result["sections"]["unassigned"][0]["quote"] == source
    assert result["category_overrides"][0]["reason"] == "explicit_question"


@pytest.mark.parametrize(("question", "answer"), [
    ('Clinician: "Has the tumour progressed?"', 'Patient: No.'),
    ('Arzt: \u201eIst der Tumor fortgeschritten?\u201c', 'Patient: Nein.'),
    ('Medecin: \u00abLa tumeur a-t-elle progresse?\u202f\u00bb', 'Patient: Non.'),
    ('Medico: "Il tumore e progredito?"', 'Paziente: No.'),
])
def test_quoted_question_before_another_speaker_is_protected(monkeypatch, question, answer):
    class Opener:
        def open(self, request, **options):
            if request.full_url.endswith("/api/show"):
                return io.BytesIO(json.dumps({"details": {"format": "gguf"},
                    "model_info": {"general.architecture": "llama"}, "template": ""}).encode())
            spans = json.loads(json.loads(request.data)["messages"][1]["content"])["evidence_spans"]
            return io.BytesIO(json.dumps({"done": True, "done_reason": "stop", "message": {
                "content": json.dumps({"labels": ["assessment"] * len(spans)})}}).encode())

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    source = question + " " + answer
    result = notes.ollama_positional_note(source)
    assert result["sections"]["unassigned"] == [{"quote": question, "start": 0, "end": len(question)}]
    assert result["sections"]["assessment"] == [
        {"quote": answer, "start": len(question) + 1, "end": len(source)}]
    assert result["uncovered_source"] == []


@pytest.mark.parametrize("publisher", ["publish_note", "publish_transcription", "publish_proposals"])
def test_source_invalidated_after_validation_cannot_publish(monkeypatch, tmp_path, publisher):
    app = demo.build_app()
    functions = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
    history = TimingHistory(tmp_path / "timing.json")
    tracker = {"generation": 0}
    if publisher == "publish_note":
        args = ["fictional source", "extractive", "unused"]
        value = ("old preview", {"raw_transcript": args[0]})
        kind = "note"
    elif publisher == "publish_proposals":
        args = ["fictional source", "en", "qwen3.5:4b"]
        report = {"raw_transcript": args[0], "proposals": []}
        value = ("old proposals", report, report, [])
        kind = "corrections"
    else:
        args = [None, "en", "verbatim", "turbo", "cpu", .6, False]
        value = ("old text", "old markup", {"text": "old text"}, "old text", {})
        kind = "asr"
    state = demo.start_timed_request(kind, args, history, tracker)
    job = list(demo.run_timed_request(lambda *_: value, args, state, history, tracker))[-1][0]
    original = demo.current_request_result

    def invalidated_after_validation(*inputs, **options):
        result = original(*inputs, **options)
        demo.invalidate_task(tracker)
        return result

    monkeypatch.setattr(demo, "current_request_result", invalidated_after_validation)
    if publisher == "publish_transcription":
        response = functions[publisher](job, tracker, *args)
    else:
        response = functions[publisher](job, tracker, None, *args)
    assert response[0] == gr.skip(), "A superseded result was returned for display"
    assert not tracker.get("published")
    assert not tracker.get("issued")
