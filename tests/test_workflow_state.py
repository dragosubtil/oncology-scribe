"""Request identity and server-owned proposal tests; no real model calls."""

import copy
import hashlib
import inspect

import pytest

from clinical_asr import demo
from clinical_asr.timing import TimingHistory


@pytest.fixture
def history(tmp_path):
    return TimingHistory(tmp_path / "times.json")


def completed(history, tracker, args, task=lambda *_: "draft", kind="note", audio=None):
    state = demo.start_timed_request(kind, args, history, tracker, audio=audio)
    return list(demo.run_timed_request(task, args, state, history, tracker))[-1][0]


def test_queued_request_has_distinct_identity_even_for_identical_inputs(history):
    tracker = {"generation": 0}
    args = ["source", "extractive", "unused"]
    first = demo.start_timed_request("note", args, history, tracker)
    second = demo.start_timed_request("note", args, history, tracker)
    assert first.get("request_id"), "A generation is not a request identity"
    assert first["request_id"] != second["request_id"]
    assert first["session_id"] == second["session_id"]


def test_invalidation_after_worker_completion_blocks_publication(history):
    tracker = {"generation": 0}
    args = ["source", "extractive", "unused"]
    job = completed(history, tracker, args)
    demo.invalidate_task(tracker)
    with pytest.raises(ValueError, match="changed|superseded"):
        demo.current_request_result(job, args, tracker)


def test_identical_input_retry_cannot_publish_previous_completion(history):
    tracker = {"generation": 0}
    args = ["source", "extractive", "unused"]
    old = completed(history, tracker, args)
    new = completed(history, tracker, args, lambda *_: "new draft")
    with pytest.raises(ValueError, match="changed|superseded"):
        demo.current_request_result(old, args, tracker)
    assert demo.current_request_result(new, args, tracker) == "new draft"


def test_other_session_cannot_publish_equal_input_completion(history):
    args = ["source", "extractive", "unused"]
    first, other = {"generation": 0}, {"generation": 0}
    job = completed(history, first, args)
    completed(history, other, args)
    with pytest.raises(ValueError, match="changed|superseded"):
        demo.current_request_result(job, args, other)


def test_publication_returns_independent_snapshot(history):
    tracker = {"generation": 0}
    args = ["source", "extractive", "unused"]
    value = {"nested": ["original"]}
    job = completed(history, tracker, args, lambda *_: value)
    value["nested"].append("late worker mutation")
    result = demo.current_request_result(job, args, tracker)
    assert result == {"nested": ["original"]}
    result["nested"].append("client mutation")
    assert demo.current_request_result(job, args, tracker) == {"nested": ["original"]}


def test_audio_replaced_at_same_path_after_enqueue_never_executes(history, tmp_path):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"original fictional audio")
    tracker = {"generation": 0}
    args = [str(audio), "en", "verbatim", "turbo", "cpu", .6, False]
    state = demo.start_timed_request("asr", args, history, tracker)
    audio.write_bytes(b"different fictional audio")
    calls = []
    job = list(demo.run_timed_request(lambda *_: calls.append(True), args, state, history, tracker))[-1][0]
    assert calls == []
    assert "error" in job


def test_audio_replaced_between_worker_and_publication_is_rejected(history, tmp_path):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"original fictional audio")
    tracker = {"generation": 0}
    args = [str(audio), "en", "verbatim", "turbo", "cpu", .6, False]
    job = completed(history, tracker, args, kind="asr")
    audio.write_bytes(b"different fictional audio")
    with pytest.raises(ValueError, match="changed|superseded"):
        demo.current_request_result(job, args, tracker)


def proposal_report():
    source = "Trastuzumap and Rituximap 5 mg."
    return {"status": "proposed", "raw_transcript": source,
            "source_transcript_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "language": "en", "model": "qwen3.5:4b",
            "proposals": [{"id": "e0", "span_id": "s0", "before": "Trastuzumap",
                           "after": "Trastuzumab", "start": 0, "end": 11,
                           "reason": "Spelling; listen to the audio.", "protected": False,
                           "requires_audio_confirmation": True}],
            "rejected_proposals": [], "clinical_validation": "not_established",
            "multilingual_validation": "not_established", "requires_human_review": True,
            "requires_clinician_final_review": True, "clinician_final_review": "pending"}


def issued(history, tracker, audio=None):
    report = proposal_report()
    args = [report["raw_transcript"], report["language"], report["model"]]
    job = completed(history, tracker, args, lambda *_: ("preview", report, report, []),
                    kind="corrections", audio=audio)
    demo.issue_proposals(job, args, tracker)
    return report, args, job


def test_self_consistent_retargeted_proposal_is_rejected_at_issuing_boundary(history):
    tracker = {"generation": 0}
    report, args, _ = issued(history, tracker)
    forged = copy.deepcopy(report)
    forged["proposals"][0].update(before="Rituximap", after="Rituximab", start=16, end=25)
    with pytest.raises(ValueError, match="issued|changed"):
        demo.review_corrections_ui(args[0], forged, ["e0"], True, tracker=tracker)


def test_recomputed_hash_cannot_authorize_source_rewrite(history):
    tracker = {"generation": 0}
    report, _, _ = issued(history, tracker)
    forged = copy.deepcopy(report)
    source = forged["raw_transcript"].replace("5 mg", "50 mg")
    forged.update(raw_transcript=source, source_transcript_sha256=hashlib.sha256(source.encode()).hexdigest())
    with pytest.raises(ValueError, match="issued|changed"):
        demo.review_corrections_ui(source, forged, ["e0"], True, tracker=tracker)


def test_other_session_and_superseded_request_cannot_apply_issued_proposals(history):
    tracker, other = {"generation": 0}, {"generation": 0}
    report, args, _ = issued(history, tracker)
    with pytest.raises(ValueError, match="issued|changed"):
        demo.review_corrections_ui(args[0], report, ["e0"], True, tracker=other)
    demo.start_timed_request("corrections", args, history, tracker)
    with pytest.raises(ValueError, match="issued|changed"):
        demo.review_corrections_ui(args[0], report, ["e0"], True, tracker=tracker)


def test_server_issued_report_is_frozen_and_valid_selection_still_applies(history):
    tracker = {"generation": 0}
    report, args, _ = issued(history, tracker)
    reviewed = demo.review_corrections_ui(args[0], report, ["e0"], True, tracker=tracker)
    assert reviewed[0] == "Trastuzumab and Rituximap 5 mg."
    assert reviewed[1]["clinician_final_review"] == "pending"
    report["proposals"][0]["after"] = "Rituximab"
    with pytest.raises(ValueError, match="issued|changed"):
        demo.review_corrections_ui(args[0], report, ["e0"], True, tracker=tracker)


def test_proposals_bound_to_audio_cannot_be_applied_after_replacement(history, tmp_path):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"original fictional audio")
    tracker = {"generation": 0}
    report, args, _ = issued(history, tracker, str(audio))
    audio.write_bytes(b"different fictional audio")
    with pytest.raises(ValueError, match="changed|issued"):
        demo.review_corrections_ui(args[0], report, ["e0"], True, tracker=tracker,
                                   audio=str(audio))


def test_ui_engine_never_permits_implicit_model_downloads(monkeypatch):
    configs = []
    demo._engine.cache_clear()
    monkeypatch.setattr(demo, "Transcriber", lambda config: configs.append(config))
    demo._engine("turbo", "cpu", False)
    assert configs[0].local_files_only is True
    demo._engine.cache_clear()


@pytest.mark.parametrize("publisher", ["publish_transcription", "publish_proposals", "publish_note"])
def test_every_ui_publisher_consumes_its_server_request_tracker(publisher):
    app = demo.build_app()
    event = next(d for d in app.config["dependencies"] if d["api_name"] == publisher)
    fn = app.fns[event["id"]].fn
    inputs = ["source", "extractive", "unused"]
    old = {"inputs": inputs, "result": ("old preview", {"raw_transcript": "source"}),
           "request": {"generation": 1, "request_id": "old", "session_id": "s"}}
    tracker = {"generation": 2, "active_request_id": "new", "session_id": "s"}
    assert "tracker" in inspect.signature(fn).parameters, "Publication still trusts input equality only"
    if publisher == "publish_note":
        result = fn(old, tracker, None, *inputs)
        assert all(item == {"__type__": "update"} for item in result)


def test_ui_applies_only_server_owned_issued_proposals():
    app = demo.build_app()
    event = next(d for d in app.config["dependencies"] if d["api_name"] == "safe_review")
    fn = app.fns[event["id"]].fn
    report = proposal_report()
    result = fn(report["raw_transcript"], report, ["e0"], True, {"generation": 0}, None)
    assert result[0] == ""
    assert result[1]["status"] == "failed"


def test_note_export_rejects_changed_model_before_change_handler_runs(history, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = demo.build_app()
    event = next(d for d in app.config["dependencies"] if d["api_name"] == "export_note_result")
    callback = app.fns[event["id"]].fn
    tracker = {"generation": 0}
    args = ["No fever.", "extractive", "llama3.1:latest"]
    job = completed(history, tracker, args, demo.note_preview)
    value = job["result"][1]
    demo.remember_published(value, job, tracker)
    with pytest.raises(ValueError, match="changed"):
        callback(value, tracker, "No fever.", "positional", "medgemma:4b", None)
    assert not (tmp_path / ".build/demo-exports").exists()


def test_export_rejects_superseded_generation_even_when_source_is_identical(history, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = {"generation": 0}
    args = ["No fever.", "extractive", "unused"]
    job = completed(history, tracker, args, demo.note_preview)
    value = job["result"][1]
    demo.remember_published(value, job, tracker)
    demo.invalidate_task(tracker)
    with pytest.raises(ValueError, match="changed"):
        demo.require_published_result(value, tracker)


def test_review_rejects_changed_language_before_change_handler_runs(history):
    tracker = {"generation": 0}
    report, args, _ = issued(history, tracker)
    with pytest.raises(ValueError, match="changed|issued"):
        demo.require_issued_proposals(args[0], report, tracker, configuration=["de", args[2]])


def test_invalidation_during_audio_hash_blocks_publication(history, tmp_path, monkeypatch):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"fictional audio")
    tracker = {"generation": 0}
    args = [str(audio), "en", "verbatim", "turbo", "cpu", .6, False]
    job = completed(history, tracker, args, kind="asr")
    genuine = demo.file_sha256

    def hash_then_invalidate(path):
        digest = genuine(path)
        demo.invalidate_task(tracker)
        return digest

    monkeypatch.setattr(demo, "file_sha256", hash_then_invalidate)
    with pytest.raises(ValueError, match="changed|superseded"):
        demo.current_request_result(job, args, tracker)


def test_replaced_audio_publishes_failed_clock_without_publishing_draft(history, tmp_path):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"fictional audio")
    app = demo.build_app()
    event = next(d for d in app.config["dependencies"] if d["api_name"] == "publish_transcription")
    callback = app.fns[event["id"]].fn
    tracker = {"generation": 0}
    args = [str(audio), "en", "verbatim", "turbo", "cpu", .6, False]
    state = demo.start_timed_request("asr", args, history, tracker)
    audio.write_bytes(b"changed fictional audio")
    job = list(demo.run_timed_request(lambda *_: "must not run", args, state, history, tracker))[-1][0]
    result = callback(job, tracker, *args)
    assert result[:6] == ({"__type__": "update"},) * 6
    assert 'data-status="failed"' in result[-1]


def test_changing_cutoff_before_transcription_remains_an_empty_state():
    app = demo.build_app()
    event = next(d for d in app.config["dependencies"] if d["api_name"] == "update_highlights")
    callback = app.fns[event["id"]].fn
    assert callback(None, .8, None, {"generation": 0}) == ("", None, None, None)


def test_same_request_ticket_cannot_execute_twice(history):
    tracker = {"generation": 0}
    args = ["source", "extractive", "unused"]
    state = demo.start_timed_request("note", args, history, tracker)
    calls = []
    def task(*_):
        return calls.append("executed")
    list(demo.run_timed_request(task, args, state, history, tracker))
    job = list(demo.run_timed_request(task, args, state, history, tracker))[-1][0]
    assert calls == ["executed"]
    assert "error" in job


def test_note_publication_rejects_new_audio_before_change_handler(history, tmp_path):
    first, second = tmp_path / "first.bin", tmp_path / "second.bin"
    first.write_bytes(b"first fictional audio")
    second.write_bytes(b"second fictional audio")
    tracker = {"generation": 0}
    args = ["No fever.", "extractive", "unused"]
    job = completed(history, tracker, args, demo.note_preview, audio=str(first))
    app = demo.build_app()
    event = next(d for d in app.config["dependencies"] if d["api_name"] == "publish_note")
    callback = app.fns[event["id"]].fn
    result = callback(job, tracker, str(second), *args)
    assert result == ({"__type__": "update"},) * 3
    assert tracker.get("published") is None


def test_late_unchanged_source_event_does_not_cancel_new_request(history):
    app = demo.build_app()
    control = next(c["id"] for c in app.config["components"]
                   if c["props"].get("label") == "Transcript for note")
    event = next(d for d in app.config["dependencies"]
                 if any(t[0] == control and t[1] == "change" for t in d["targets"])
                 and d["api_name"].startswith("invalidate"))
    tracker = {"generation": 0}
    args = ["No fever.", "extractive", "unused"]
    state = demo.start_timed_request("note", args, history, tracker)
    app.fns[event["id"]].fn(tracker, 2, None, *args)
    assert tracker["generation"] == state["generation"]
    assert tracker["active_request_id"] == state["request_id"]


def test_highlight_revision_does_not_mutate_original_nested_evidence():
    original = {"text": "pain", "spoken_language": "en", "words": [],
                "workflow": {"generation": 1}}
    _, updated, _, _ = demo.update_highlights(original, .8)
    updated["workflow"]["generation"] = 2
    assert original["workflow"] == {"generation": 1}


def test_obsolete_different_source_event_cannot_cancel_latest_request(history):
    app = demo.build_app()
    event = next(d for d in app.config["dependencies"]
                 if d["api_name"].startswith("invalidate")
                 and any(app.blocks[i].label == "Transcript for note" for i, _ in d["targets"] if i))
    tracker = {"generation": 0}
    old = ["old source", "extractive", "unused"]
    new = ["new source", "extractive", "unused"]
    demo.start_timed_request("note", old, history, tracker, event_order=2)
    latest = demo.start_timed_request("note", new, history, tracker, event_order=4)
    app.fns[event["id"]].fn(tracker, 1, None, *old)
    assert tracker["active_request_id"] == latest["request_id"]
    assert tracker["generation"] == latest["generation"]
    assert tracker["busy"] is True


def test_cancellation_cleanup_does_not_unlock_a_newer_busy_task(history):
    app = demo.build_app()
    event = next(d for d in app.config["dependencies"] if d["api_name"] == "unlock_review")
    fn = app.fns[event["id"]].fn
    idle = {"generation": 0}
    busy = {"generation": 0}
    demo.start_timed_request("note", ["source", "extractive", "unused"], history, busy)
    assert all(item == {"__type__": "update"} for item in fn(idle, busy))
    demo.invalidate_task(busy)
    assert all(item["interactive"] is True for item in fn(idle, busy))


def test_older_starter_cannot_commit_after_newer_starter_during_hash(history, tmp_path, monkeypatch):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"fictional audio")
    tracker = {"generation": 0}
    args = ["source", "extractive", "unused"]
    genuine = demo.file_sha256
    newer = []

    def interleave(path):
        monkeypatch.setattr(demo, "file_sha256", genuine)
        newer.append(demo.start_timed_request("note", args, history, tracker,
                                              audio=str(audio), event_order=5))
        return genuine(path)

    monkeypatch.setattr(demo, "file_sha256", interleave)
    with pytest.raises(ValueError, match="Obsolete"):
        demo.start_timed_request("note", args, history, tracker, audio=str(audio), event_order=4)
    assert tracker["last_event_order"] == 5
    assert tracker["active_request_id"] == newer[0]["request_id"]


def test_invalidator_hash_cannot_cancel_a_newer_starter(history, tmp_path, monkeypatch):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"fictional audio")
    tracker = {"generation": 0}
    args = ["source", "extractive", "unused"]
    demo.start_timed_request("note", args, history, tracker, audio=str(audio), event_order=2)
    app = demo.build_app()
    event = next(d for d in app.config["dependencies"]
                 if d["api_name"].startswith("invalidate")
                 and any(app.blocks[i].label == "Transcript for note" for i, _ in d["targets"] if i))
    genuine = demo.file_sha256
    newer = []

    def interleave(path):
        monkeypatch.setattr(demo, "file_sha256", genuine)
        newer.append(demo.start_timed_request("note", args, history, tracker,
                                              audio=str(audio), event_order=4))
        return genuine(path)

    monkeypatch.setattr(demo, "file_sha256", interleave)
    app.fns[event["id"]].fn(tracker, 3, str(audio), *args)
    assert tracker["active_request_id"] == newer[0]["request_id"]
    assert tracker["busy"] is True


def test_rejected_starter_has_unconditional_busy_aware_cleanup():
    app = demo.build_app()
    dependencies = app.config["dependencies"]
    starters = [d for d in dependencies if d["api_name"].startswith("start")]
    assert len(starters) == 3
    for starter in starters:
        cleanup = [d for d in dependencies if d["trigger_after"] == starter["id"]
                   and d["api_name"].startswith("unlock_review")
                   and not d["trigger_only_on_success"]]
        assert len(cleanup) == 1
        result = app.fns[cleanup[0]["id"]].fn(*({"busy": False} for _ in range(3)))
        assert all(item["interactive"] for item in result)


def test_cleanup_cannot_unlock_while_a_newer_starter_hashes(history, tmp_path, monkeypatch):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"fictional audio")
    tracker = {"generation": 0}
    args = ["source", "extractive", "unused"]
    app = demo.build_app()
    cleanup = next(d for d in app.config["dependencies"] if d["api_name"] == "unlock_review")
    genuine = demo.file_sha256

    def interleave(path):
        with pytest.raises(ValueError, match="Obsolete"):
            demo.start_timed_request("note", args, history, tracker, event_order=4)
        assert all(item == {"__type__": "update"}
                   for item in app.fns[cleanup["id"]].fn(tracker))
        return genuine(path)

    monkeypatch.setattr(demo, "file_sha256", interleave)
    demo.start_timed_request("note", args, history, tracker, audio=str(audio), event_order=5)
    assert tracker["busy"] is True
    assert not tracker.get("pending_start")


def test_failed_hash_releases_only_its_pending_starter(history, tmp_path, monkeypatch):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"fictional audio")
    tracker = {"generation": 0}

    def failed_hash(path):
        assert tracker.get("pending_start")
        raise OSError("test-only hash failure")

    monkeypatch.setattr(demo, "file_sha256", failed_hash)
    with pytest.raises(OSError, match="test-only"):
        demo.start_timed_request("note", ["source", "extractive", "unused"], history,
                                 tracker, audio=str(audio), event_order=1)
    assert not tracker.get("pending_start")
    assert not tracker.get("busy")


def test_unchanged_upload_event_during_starter_hash_does_not_cancel_it(history, tmp_path, monkeypatch):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"fictional audio")
    args = [str(audio), "en", "verbatim", "turbo", "cpu", .6, False]
    tracker = {"generation": 0}
    app = demo.build_app()
    event = next(d for d in app.config["dependencies"] if d["api_name"].startswith("invalidate")
                 and any(app.blocks[i].label == "Consultation audio" for i, _ in d["targets"] if i))
    genuine = demo.file_sha256

    def interleave(path):
        app.fns[event["id"]].fn(tracker, 4, *args)
        return genuine(path)

    monkeypatch.setattr(demo, "file_sha256", interleave)
    value = demo.start_timed_request("asr", args, history, tracker, event_order=3)
    assert tracker["active_request_id"] == value["request_id"]
    assert tracker["busy"]
