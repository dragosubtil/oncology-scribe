import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from clinical_asr.cli import main, read_predictions


def test_duplicate_prediction_ids_fail(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"sample_id":"s1","text":"a"}\n{"sample_id":"s1","text":"b"}', encoding="utf-8")
    try:
        read_predictions(path)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Duplicate predictions must not overwrite earlier output")


def test_note_cli_writes_reviewable_points(tmp_path):
    source, output = tmp_path / "source.txt", tmp_path / "notes.json"
    source.write_text("No fever. Continue 2.5 mg daily.", encoding="utf-8")
    assert main(["notes", str(source), "--output", str(output)]) == 0
    note = json.loads(output.read_text(encoding="utf-8"))
    assert note["requires_human_review"] is True
    assert len(note["sections"]["unassigned"]) == 2


def test_malformed_prediction_fails_without_echoing_clinical_text(tmp_path, capsys):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("not a manifest with sensitive content", encoding="utf-8")
    assert main(["validate", str(manifest)]) == 2
    assert "sensitive content" not in capsys.readouterr().err


def test_module_entrypoint_help_works_from_new_process():
    result = subprocess.run([sys.executable, "-m", "clinical_asr", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "benchmark" in result.stdout
    assert "notes" in result.stdout


def test_cancelled_benchmark_records_interruption_and_partial_count(tmp_path, monkeypatch):
    sf.write(tmp_path / "audio.wav", np.ones(16000) * 0.1, 16000)
    row = dict(sample_id="s1", speaker_id="speaker", encounter_id="visit",
               audio_path="audio.wav", language="en", split="test", reference="No pain.",
               data_use="simulated", source="fixture", license="project-generated")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row), encoding="utf-8")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("clinical_asr.cli.Transcriber.transcribe_file", interrupted)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        main(["benchmark", str(manifest), "--output-dir", str(output)])
    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "interrupted"
    assert metadata["completed_samples"] == 0
    assert metadata["finished_utc"]
    assert (output / "predictions.partial.jsonl").exists()
