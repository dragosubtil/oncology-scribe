import importlib
import json

import pytest

from clinical_asr.timing import TimingHistory


def api():
    module = importlib.import_module("clinical_asr.demo")
    assert hasattr(module, "start_timed_request"), "Queued task clock is missing"
    assert hasattr(module, "run_timed_request"), "Worker task clock is missing"
    return module


def test_worker_publishes_processing_before_call_and_completion_after(tmp_path):
    module = api()
    history = TimingHistory(tmp_path / "times.json")
    state = module.start_timed_request("note", ["secret text", "extractive", "unused"], history)
    called = []

    def task(*args):
        called.append(args)
        return "answer"

    stream = module.run_timed_request(task, ["secret text", "extractive", "unused"], state, history)
    pending = next(stream)
    assert not called
    assert pending[0] is None
    assert pending[1]["status"] == "processing"
    assert pending[3] is True
    done = next(stream)
    assert called
    assert done[0]["result"] == "answer"
    assert done[0]["inputs"] == ["secret text", "extractive", "unused"]
    assert done[1]["status"] == "completed"
    assert done[3] is False
    assert done[0]["timing"]["processing_seconds"] >= 0
    assert "secret" not in json.dumps(history.records)
    assert "secret" not in json.dumps(done[1])
    with pytest.raises(StopIteration):
        next(stream)


def test_failure_stops_clock_and_does_not_train_eta(tmp_path):
    module = api()
    history = TimingHistory(tmp_path / "times.json")
    args = ["abc", "en", "qwen3.5:4b"]
    state = module.start_timed_request("corrections", args, history)

    def fail(*args):
        raise ValueError("Unavailable model")

    done = list(module.run_timed_request(fail, args, state, history))[-1]
    assert done[1]["status"] == "failed"
    assert done[3] is False
    assert done[0]["error"] == "Unavailable model"
    assert history.records == {}


def test_eta_uses_comparable_successes_without_source_text(tmp_path):
    module = api()
    history = TimingHistory(tmp_path / "times.json")
    args = ["a" * 100, "en", "qwen3.5:4b"]
    first = module.start_timed_request("corrections", args, history)
    assert first["expected_seconds"] is None
    for seconds in [20, 30, 25]:
        history.record(first["history_key"], first["units"], seconds)
    second = module.start_timed_request("corrections", ["b" * 100, *args[1:]], history)
    assert second["expected_seconds"] == 25
    other_model = module.start_timed_request("corrections", [args[0], "en", "medgemma:4b"], history)
    other_language = module.start_timed_request("corrections", [args[0], "fr", args[2]], history)
    assert other_model["expected_seconds"] is None
    assert other_language["expected_seconds"] is None


def test_missing_audio_produces_unknown_eta_instead_of_crashing(tmp_path):
    module = api()
    history = TimingHistory(tmp_path / "times.json")
    state = module.start_timed_request("asr", [None, "en", "verbatim", "turbo", "cpu", .6, False], history)
    assert state["status"] == "queued"
    assert state["expected_seconds"] is None
    assert state["units"] is None


def test_each_slow_task_has_a_live_session_clock():
    module = api()
    app = module.build_app()
    components = app.config["components"]
    for label in ["Transcription timing", "Terminology review timing", "Note draft timing"]:
        assert any(c["props"].get("label") == label for c in components)
    assert not any(c["type"] == "timer" for c in components)
    for label in ["Transcription timing", "Terminology review timing", "Note draft timing"]:
        clock = next(c for c in components if c["props"].get("label") == label)
        assert "watch('value'" in clock["props"]["js_on_load"]
        assert "setInterval" in clock["props"]["js_on_load"]


@pytest.mark.parametrize("label,timing_label", [
    ("Consultation audio", "Transcription timing"),
    ("Spoken language", "Transcription timing"),
    ("Transcript for medical review", "Terminology review timing"),
    ("Transcript for note", "Note draft timing"),
])
def test_changing_source_clears_the_old_task_duration(label, timing_label):
    app = api().build_app()
    components = app.config["components"]
    source = next(c["id"] for c in components if c["props"].get("label") == label)
    status = next(c["id"] for c in components if c["props"].get("label") == timing_label)
    events = [d for d in app.config["dependencies"]
              if any(t[0] == source and t[1] == "change" for t in d["targets"])
              and status in d["outputs"]]
    assert events, "An old completion clock must not describe a new source"
    event = events[0]
    values = [app.blocks[key].value for key in event["inputs"]]
    values[:2] = [{"generation": 0}, 1]
    update = app.fns[event["id"]].fn(*values)
    markup = update[event["outputs"].index(status)] if isinstance(update, tuple) else update
    assert "Ready" in markup


def revision_api():
    module = api()
    assert hasattr(module, "invalidate_task"), "Task revision guard is missing"
    return module


def test_source_reset_does_not_destroy_queued_request_and_prevents_execution(tmp_path):
    module = revision_api()
    history = TimingHistory(tmp_path / "times.json")
    tracker = {"generation": 0}
    args = ["source", "extractive", "unused"]
    queued = module.start_timed_request("note", args, history, tracker)
    assert "Ready" in module.invalidate_task(tracker)
    called = []
    done = list(module.run_timed_request(lambda *_: called.append(True), args, queued, history, tracker))[-1]
    assert not called
    assert "changed" in done[0]["error"]
    assert done[2] is None
    assert history.records == {}


def test_old_completion_after_source_reset_cannot_publish_a_clock(tmp_path):
    module = revision_api()
    history = TimingHistory(tmp_path / "times.json")
    tracker = {"generation": 0}
    args = ["source", "extractive", "unused"]
    queued = module.start_timed_request("note", args, history, tracker)
    stream = module.run_timed_request(lambda *_: "old result", args, queued, history, tracker)
    assert next(stream)[1]["status"] == "processing"
    module.invalidate_task(tracker)
    done = next(stream)
    assert "result" not in done[0]
    assert "changed" in done[0]["error"]
    assert done[2] is None
    assert history.records == {}


def test_two_sessions_have_independent_generations(tmp_path):
    module = revision_api()
    history = TimingHistory(tmp_path / "times.json")
    first, second = {"generation": 0}, {"generation": 0}
    args = ["source", "extractive", "unused"]
    queued = module.start_timed_request("note", args, history, second)
    module.invalidate_task(first)
    done = list(module.run_timed_request(lambda *_: "result", args, queued, history, second))[-1]
    assert done[0]["result"] == "result"
    assert done[1]["status"] == "completed"
