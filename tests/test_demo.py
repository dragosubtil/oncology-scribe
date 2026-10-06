import pytest

from clinical_asr.demo import (
    _export,
    build_app,
    compare_texts,
    note_preview,
    transcribe_ui,
    update_highlights,
)


def test_ui_empty_upload_has_explicit_error():
    try:
        transcribe_ui(None, "en", "verbatim", "small", "cpu", 0.6)
    except ValueError as exc:
        assert "audio" in str(exc).lower()
    else:
        raise AssertionError("An empty upload cannot produce a success state")


def test_ui_initial_asr_is_the_frozen_development_candidate():
    app = build_app()
    model = next(c for c in app.config["components"] if c["props"].get("label") == "ASR model")
    assert model["props"]["value"] == "turbo"


def test_note_preview_escapes_html_in_source():
    output, result = note_preview("<script>alert(1)</script>.", "extractive", "llama3.1:latest")
    assert "<script>" not in output
    assert "&lt;script&gt;" in output
    assert result["requires_human_review"]


def test_ui_comparison_preserves_dose_difference():
    result = compare_texts("Dose 2.5 mg", "Dose 25 mg", "en", "verbatim")
    assert result["scores"]["normalized"]["wer"] == 1 / 3
    assert "numeric_or_unit_change" in result["scores"]["review_flags"]


def test_threshold_update_changes_exported_state_without_mutating_input():
    original = {
        "text": "pain",
        "spoken_language": "en",
        "words": [{"text": "pain", "char_start": 0, "char_end": 4, "model_score": 0.7}],
    }
    marked, details, state, old_export = update_highlights(original, 0.8)
    assert "low_model_score" in marked
    assert state["review_threshold"] == 0.8
    assert details == state
    assert state["review_spans"][0]["reasons"] == ["low_model_score"]
    assert "review_spans" not in original
    assert old_export is None


def test_failed_results_cannot_be_exported_as_drafts():
    with pytest.raises(ValueError, match="successful"):
        _export({"status": "failed", "error": "No audio"})


def test_review_threshold_event_supports_keyboard_and_numeric_input():
    app = build_app()
    event = next(
        item for item in app.config["dependencies"] if item["api_name"] == "update_highlights"
    )
    assert event["targets"][0][1] == "change"


def test_audio_change_invalidates_both_export_links():
    app = build_app()
    components = app.config["components"]
    audio_id = next(c["id"] for c in components if c["props"].get("label") == "Consultation audio")
    export_ids = {
        c["id"]
        for c in components
        if c["props"].get("label") in {"Transcript export", "Note export"}
    }
    change = next(
        d
        for d in app.config["dependencies"]
        if any(t[0] == audio_id and t[1] == "change" for t in d["targets"])
    )
    assert len(export_ids) == 2
    assert export_ids <= set(change["outputs"])


def test_frontpage_is_a_consultation_workbench_with_medical_review():
    app = build_app()
    components = app.config["components"]
    tabs = [c["props"].get("label") for c in components if c["type"] == "tabitem"]
    assert tabs[0] == "Consultation"
    assert "Medical Review" in tabs
    medical_model = next(
        c for c in components if c["props"].get("label") == "Medical correction model"
    )
    assert medical_model["props"]["value"] == "qwen3.5:4b"
    assert any(
        c["props"].get("label") == "I checked the selected changes against the audio"
        for c in components
    )


def test_medical_proposal_markup_escapes_untrusted_text():
    from clinical_asr.demo import correction_preview

    result = correction_preview(
        {
            "proposals": [
                {
                    "id": "e0",
                    "before": "<script>",
                    "after": "<img onerror=alert(1)>",
                    "reason": "<b>test</b>",
                    "protected": True,
                }
            ]
        }
    )
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    assert "&lt;img" in result
    assert "Manual review" in result


def test_review_rejects_stale_transcript_before_calling_backend():
    from clinical_asr.demo import review_corrections_ui

    with pytest.raises(ValueError, match="changed"):
        review_corrections_ui("new text", {"raw_transcript": "old text"}, ["e0"], True)


def test_source_bound_export_rejects_stale_note_or_corrections():
    from clinical_asr.demo import export_matching_source

    with pytest.raises(ValueError, match="changed"):
        export_matching_source({"raw_transcript": "old source"}, "new source")


def test_withdrawing_audio_confirmation_invalidates_reviewed_export():
    app = build_app()
    components = app.config["components"]
    confirmation = next(
        c["id"]
        for c in components
        if c["props"].get("label") == "I checked the selected changes against the audio"
    )
    reviewed_export = next(
        c["id"] for c in components if c["props"].get("label") == "Reviewed draft export"
    )
    event = next(d for d in app.config["dependencies"]
                 if any(t[0] == confirmation and t[1] == "change" for t in d["targets"]))
    assert reviewed_export in event["outputs"]


def test_confirmation_alone_cannot_create_a_corrected_draft():
    from clinical_asr.demo import review_corrections_ui

    with pytest.raises(ValueError, match="No correction selected"):
        review_corrections_ui("cis plat in", {"raw_transcript": "cis plat in"}, [], True)


@pytest.mark.parametrize("label", [
    "Changes to include", "Transcript for medical review", "Review language",
    "Medical correction model",
])
def test_medical_input_changes_invalidate_transferred_note_source(label):
    app = build_app()
    components = app.config["components"]
    changed = next(c["id"] for c in components if c["props"].get("label") == label)
    note_source = next(c["id"] for c in components
                       if c["props"].get("label") == "Transcript for note")
    event = next(d for d in app.config["dependencies"]
                 if any(t[0] == changed and t[1] == "change" for t in d["targets"]))
    assert note_source in event["outputs"]


@pytest.mark.parametrize("current", [
    ["new source", "en", "medgemma:4b"],
    ["old source", "de", "medgemma:4b"],
    ["old source", "en", "llama3.1:latest"],
])
def test_delayed_completion_is_rejected_after_inputs_change(current):
    from clinical_asr.demo import current_request_result

    completed = {"inputs": ["old source", "en", "medgemma:4b"], "result": "old result"}
    with pytest.raises(ValueError, match="changed"):
        current_request_result(completed, current)


def test_current_completion_returns_only_its_result():
    from clinical_asr.demo import current_request_result

    inputs = ["source", "en", "medgemma:4b"]
    assert current_request_result({"inputs": inputs, "result": "result"}, inputs) == "result"


def test_delayed_failure_does_not_replace_newer_result():
    from clinical_asr.demo import current_request_result

    with pytest.raises(ValueError, match="changed"):
        current_request_result({"inputs": ["old"], "error": "old model failed"}, ["new"])


def test_review_export_rejects_changed_selection_even_before_ui_reset():
    from clinical_asr.demo import export_reviewed_source

    with pytest.raises(ValueError, match="selection"):
        export_reviewed_source(
            {"raw_transcript": "source", "audio_confirmed": True, "accepted_ids": ["e0"]},
            "source", True, [],
        )


def test_note_transfer_rejects_stale_selected_edits_and_empty_source():
    from clinical_asr.demo import note_source_for_review

    with pytest.raises(ValueError, match="changed"):
        note_source_for_review("new source", {"raw_transcript": "old source"}, ["e0"], True)
    assert note_source_for_review("raw draft", None, [], False) == "raw draft"
    with pytest.raises(ValueError, match="required"):
        note_source_for_review("", None, [], False)


def test_note_preview_shows_uncovered_source_instead_of_hiding_dose(monkeypatch):
    from clinical_asr.notes import validate_note

    source = "No fever. Recorded dose 2.5 mg."
    partial = validate_note(source, {"subjective": [{"quote": "No fever.", "start": 0, "end": 9}]})
    monkeypatch.setattr("clinical_asr.demo.ollama_note", lambda *_: partial)
    preview, _ = note_preview(source, "ollama", "medgemma:4b")
    assert "Recorded dose 2.5 mg." in preview
    assert "Uncovered source" in preview


@pytest.mark.parametrize("label", ["Consultation audio", "ASR model", "Device", "Spoken language"])
def test_asr_input_changes_clear_old_result_and_export(label):
    app = build_app()
    components = app.config["components"]
    changed = next(c["id"] for c in components if c["props"].get("label") == label)
    required = {c["id"] for c in components
                if c["props"].get("label") in {"Raw transcript", "Transcript export"}}
    event = next(d for d in app.config["dependencies"]
                 if any(t[0] == changed and t[1] == "change" for t in d["targets"])
                 and required <= set(d["outputs"]))
    assert event["queue"] is False


def test_selected_corrections_cannot_bypass_creating_reviewed_draft():
    from clinical_asr.demo import note_source_for_review

    with pytest.raises(ValueError, match="reviewed draft"):
        note_source_for_review("source", {"raw_transcript": "source"}, ["e0"], True)


def test_highlights_and_export_reject_audio_cleared_before_result_is_applied(tmp_path):
    from clinical_asr.demo import export_matching_audio, update_matching_highlights
    from clinical_asr.schema import file_sha256

    audio = tmp_path / "fictional.bin"
    audio.write_bytes(b"fictional test audio identity")
    old = {"text": "pain", "spoken_language": "en", "words": [],
           "source_audio_sha256": file_sha256(audio)}
    with pytest.raises(ValueError, match="audio"):
        update_matching_highlights(old, 0.8, None)
    with pytest.raises(ValueError, match="audio"):
        export_matching_audio(old, None)
    audio.write_bytes(b"different audio identity")
    with pytest.raises(ValueError, match="audio"):
        update_matching_highlights(old, 0.8, str(audio))


@pytest.mark.parametrize("name", ["Transcribe audio", "Propose terminology edits", "Generate note draft"])
def test_slow_pipeline_disables_source_controls_until_completion(name):
    app = build_app()
    components = app.config["components"]
    button_id = next(c["id"] for c in components
                     if c["type"] == "button" and c["props"].get("value") == name)
    editable_ids = {c["id"] for c in components if c["props"].get("label") in
                    {"Transcript for medical review", "Transcript for note", "Consultation audio",
                     "I checked the selected changes against the audio"}}
    start = next(d for d in app.config["dependencies"]
                 if any(t[0] == button_id and t[1] == "click" for t in d["targets"]))
    assert editable_ids <= set(start["outputs"])
    updates = app.fns[start["id"]].fn()
    assert all(update["interactive"] is False for update in updates)
    assert any(d["api_name"].startswith("unlock_inputs") for d in app.config["dependencies"])
