import gradio as gr
import pytest

from clinical_asr import demo
from clinical_asr.timing import TimingHistory


@pytest.fixture
def fixture(tmp_path):
    audio = tmp_path / "fiction.wav"
    audio.write_bytes(b"fictional fixture")
    args = [str(audio), "en", "verbatim", "turbo", "cpu", 0.6, False]
    tracker = {"generation": 0}
    history = TimingHistory(tmp_path / "times.json")
    request = demo.start_timed_request("asr", args, history, tracker, event_order=5)
    job = list(
        demo.run_timed_request(lambda *_: {"text": "source"}, args, request, history, tracker)
    )[-1][0]
    demo.remember_published(job["result"], job, tracker)
    app = demo.build_app()
    audio_id = next(
        c["id"] for c in app.config["components"] if c["props"].get("label") == "Consultation audio"
    )
    raw_id = next(
        c["id"] for c in app.config["components"] if c["props"].get("label") == "Raw transcript"
    )
    event = next(
        d
        for d in app.config["dependencies"]
        if raw_id in d["outputs"] and [audio_id, "change"] in [list(t) for t in d["targets"]]
    )
    return app.fns[event["id"]].fn, tracker, args, audio


def test_unchanged_audio_callback_preserves_completed_transcript_and_exports(fixture):
    callback, tracker, args, _ = fixture
    generation = tracker["generation"]
    result = callback(tracker, 6, *args)
    assert len(result) == 11
    assert all(item == gr.skip() for item in result)
    assert tracker["generation"] == generation
    assert tracker["published"]["value"]["text"] == "source"


@pytest.mark.parametrize("mutation", ["new_path", "same_path_bytes", "language"])
def test_real_asr_input_changes_clear_old_source_and_invalidate_receipt(
    fixture, tmp_path, mutation
):
    callback, tracker, args, audio = fixture
    changed = list(args)
    if mutation == "new_path":
        other = tmp_path / "new.wav"
        other.write_bytes(b"other fiction")
        changed[0] = str(other)
    elif mutation == "same_path_bytes":
        audio.write_bytes(b"modified fiction")
    else:
        changed[1] = "de"
    generation = tracker["generation"]
    result = callback(tracker, 6, *changed)
    assert result[:10] == ("", "", None, "", None, "", None, None, None, "")
    assert 'data-status="idle"' in result[10]
    assert tracker["generation"] == generation + 1
    assert tracker["published"] is None


def test_late_old_input_event_cannot_clear_newer_display(fixture):
    callback, tracker, args, _ = fixture
    result = callback(tracker, 4, *[args[0], "de", *args[2:]])
    assert all(item == gr.skip() for item in result)
    assert tracker["published"]["value"]["text"] == "source"
