import json

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from clinical_asr.audio import load_audio, prepare_audio
from clinical_asr.schema import Sample, load_manifest


def record(**changes):
    value = {
        "sample_id": "sample-1", "speaker_id": "speaker-1", "encounter_id": "visit-1",
        "audio_path": "audio.wav", "language": "en", "target_policy": "verbatim",
        "split": "test", "reference": "No fever.", "data_use": "simulated",
        "source": "test fixture", "license": "project-generated",
    }
    return value | changes


def manifest(tmp_path, rows):
    path = tmp_path / "samples.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_swiss_reference_policy_is_explicit():
    with pytest.raises(ValidationError):
        Sample(**record(language="gsw"))
    assert Sample(**record(language="gsw", target_policy="dialect")).language == "gsw"


def test_standard_german_target_is_not_accepted_for_french():
    with pytest.raises(ValidationError):
        Sample(**record(language="fr", target_policy="standard_de"))


def test_duplicate_ids_fail_instead_of_overwriting(tmp_path):
    with pytest.raises(ValueError, match="Duplicate sample"):
        load_manifest(manifest(tmp_path, [record(), record()]))


def test_same_speaker_cannot_leak_between_splits(tmp_path):
    with pytest.raises(ValueError, match="speaker"):
        load_manifest(manifest(tmp_path, [record(), record(sample_id="s2", split="train")]))


def test_same_encounter_cannot_leak_between_splits(tmp_path):
    with pytest.raises(ValueError, match="encounter"):
        load_manifest(manifest(tmp_path, [record(), record(sample_id="s2", speaker_id="s2", split="dev")]))


def test_identical_audio_detects_leakage_even_with_renamed_file(tmp_path):
    audio = np.sin(np.arange(8000) * 0.03).astype(np.float32) * 0.1
    sf.write(tmp_path / "a.wav", audio, 8000)
    (tmp_path / "b.wav").write_bytes((tmp_path / "a.wav").read_bytes())
    rows = [record(audio_path="a.wav"), record(sample_id="s2", speaker_id="s2", encounter_id="v2", audio_path="b.wav", split="train")]
    with pytest.raises(ValueError, match="audio"):
        load_manifest(manifest(tmp_path, rows), require_audio=True)


def test_empty_manifest_is_not_a_successful_benchmark(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        load_manifest(manifest(tmp_path, []))


def test_missing_audio_is_checked_only_when_required(tmp_path):
    path = manifest(tmp_path, [record()])
    assert len(load_manifest(path)) == 1
    with pytest.raises(ValueError, match="Missing audio"):
        load_manifest(path, require_audio=True)


def test_stereo_resampling_preserves_duration_and_amplitude():
    t = np.arange(48000) / 48000
    tone = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = prepare_audio(np.column_stack((tone, tone)), 48000)
    assert result.sample_rate == 16000
    assert len(result.samples) == 16000
    assert result.duration_seconds == pytest.approx(1.0)
    assert np.max(np.abs(result.samples)) == pytest.approx(0.05, abs=0.002)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), 16000.5])
def test_invalid_sample_rates_fail(rate):
    with pytest.raises(ValueError, match="sample rate"):
        prepare_audio(np.ones(10), rate)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_audio_is_rejected_instead_of_hidden(bad):
    with pytest.raises(ValueError, match="finite"):
        prepare_audio(np.array([0.1, bad]), 16000)


def test_empty_audio_has_zero_duration():
    result = prepare_audio(np.array([], dtype=np.float32), 16000)
    assert result.is_silent
    assert result.duration_seconds == 0


def test_low_volume_speech_is_not_peak_normalized_or_silenced():
    result = prepare_audio(np.ones(16000, dtype=np.float32) * 0.0001, 16000)
    assert not result.is_silent
    assert result.samples.max() == pytest.approx(0.0001)


def test_pcm_integer_audio_scales_to_float_range():
    result = prepare_audio(np.array([0, 16384, -16384], dtype=np.int16), 16000)
    assert result.samples.tolist() == [0.0, 0.5, -0.5]


def test_load_audio_resolves_real_wav_sampling_rate(tmp_path):
    path = tmp_path / "sound.wav"
    sf.write(path, np.ones(8000) * 0.05, 8000)
    result = load_audio(path)
    assert len(result.samples) == 16000
    assert result.duration_seconds == pytest.approx(1.0)


def test_invalid_channel_layout_rejected():
    with pytest.raises(ValueError, match="channel"):
        prepare_audio(np.ones((3, 2, 1)), 16000)


def test_missing_reference_is_not_mistaken_for_annotated_silence():
    with pytest.raises(ValidationError):
        Sample(**record(reference=" "))
    assert Sample(**record(reference="", reference_kind="silence")).reference == ""
    with pytest.raises(ValidationError):
        Sample(**record(reference_kind="silence"))


def test_antiphase_stereo_is_not_silently_discarded():
    tone = (0.1 * np.sin(np.arange(16000) * 0.03)).astype(np.float32)
    with pytest.raises(ValueError, match="cancellation"):
        prepare_audio(np.column_stack((tone, -tone)), 16000)


def test_identical_lossless_audio_detects_leakage_across_container_formats(tmp_path):
    audio = (np.sin(np.arange(16000) * 0.03) * 1000).astype(np.int16)
    sf.write(tmp_path / "a.wav", audio, 16000, subtype="PCM_16")
    sf.write(tmp_path / "b.flac", audio, 16000, subtype="PCM_16")
    rows = [record(audio_path="a.wav"), record(sample_id="s2", speaker_id="s2",
            encounter_id="v2", audio_path="b.flac", split="train")]
    with pytest.raises(ValueError, match="audio"):
        load_manifest(manifest(tmp_path, rows), require_audio=True)


@pytest.mark.parametrize("require_audio", [False, True])
def test_identity_validation_does_not_apply_the_asr_duration_cap(tmp_path, require_audio):
    sf.write(tmp_path / "long.wav", np.zeros(1801 * 160, dtype=np.float32), 160)
    rows = load_manifest(manifest(tmp_path, [record(audio_path="long.wav")]), require_audio=require_audio)
    assert len(rows) == 1
