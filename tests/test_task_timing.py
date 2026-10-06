import importlib
import math

import pytest


def timing():
    try:
        return importlib.import_module("clinical_asr.timing")
    except ModuleNotFoundError:
        pytest.fail("Task timing and empirical ETA are not implemented")


def test_eta_needs_three_comparable_successful_runs(tmp_path):
    history = timing().TimingHistory(tmp_path / "history.json")
    history.record("asr-turbo-en-cpu", 30, 60)
    history.record("asr-turbo-en-cpu", 30, 90)
    assert history.estimate("asr-turbo-en-cpu", 30) is None
    history.record("asr-turbo-en-cpu", 30, 75)
    assert history.estimate("asr-turbo-en-cpu", 30) == 75
    assert history.estimate("asr-small-en-cpu", 30) is None
    assert history.estimate("asr-turbo-de-cpu", 30) is None
    assert history.estimate("asr-turbo-en-cpu", 300) is None


def test_empirical_history_survives_restart_without_source_text(tmp_path):
    path = tmp_path / "history.json"
    module = timing()
    history = module.TimingHistory(path)
    for seconds in (3, 4, 5):
        history.record("notes-llama-positional", 100, seconds)
    assert module.TimingHistory(path).estimate("notes-llama-positional", 100) == 4
    assert "raw_transcript" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("units,seconds", [(0, 3), (1, -1), (math.nan, 1), (1, math.inf)])
def test_invalid_measurements_do_not_poison_eta(tmp_path, units, seconds):
    history = timing().TimingHistory(tmp_path / "history.json")
    with pytest.raises(ValueError):
        history.record("notes", units, seconds)
    assert history.estimate("notes", 100) is None


def test_corrupt_history_does_not_break_task_execution(tmp_path):
    path = tmp_path / "history.json"
    path.write_text('{"broken":NaN}', encoding="utf-8")
    assert timing().TimingHistory(path).estimate("notes", 100) is None


def test_timer_counts_queue_and_processing_separately():
    module = timing()
    state = module.begin("Transcription", expected_seconds=30, now=100)
    assert "Queued" in module.render(state, now=103)
    assert "00:03" in module.render(state, now=103)
    state = module.processing(state, now=105)
    markup = module.render(state, now=110)
    assert "Processing" in markup
    assert "00:10" in markup
    assert "00:25" in markup
    state = module.finish(state, succeeded=True, now=120)
    markup = module.render(state, now=999)
    assert "Completed" in markup
    assert "00:20" in markup
    assert "00:15" in markup
    assert state["processing_seconds"] == 15


def test_overrunning_eta_does_not_pretend_task_is_complete():
    module = timing()
    state = module.processing(module.begin("Note generation", 5, now=0), now=1)
    markup = module.render(state, now=20)
    assert "longer than estimated" in markup
    assert "Completed" not in markup


def test_failure_is_terminal_and_unknown_eta_is_honest():
    module = timing()
    state = module.begin("Corrections", None, now=0)
    assert "Not enough timing data" in module.render(state, now=1)
    state = module.finish(module.processing(state, now=2), succeeded=False, now=7)
    assert "Failed" in module.render(state, now=99)
    assert "00:07" in module.render(state, now=99)


def test_timing_markup_escapes_labels():
    module = timing()
    assert "<script>" not in module.render(module.begin("<script>", None, now=0), now=1)
