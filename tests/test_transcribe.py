import hashlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from clinical_asr.highlights import render_highlight_html, review_spans
from clinical_asr.transcribe import ASRConfig, Transcriber


class FakeBackend:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **options):
        self.calls.append((audio, options))

        def segments():
            for i in range(2):
                yield SimpleNamespace(start=i, end=i + 1, text=" No pain.",
                                      avg_logprob=-0.2, no_speech_prob=0.01,
                                      words=[SimpleNamespace(start=i, end=i + 0.4, word=" No", probability=0.9),
                                             SimpleNamespace(start=i + 0.4, end=i + 1, word=" pain.", probability=0.3)])
        return segments(), SimpleNamespace(language="en", language_probability=0.9,
                                            duration_after_vad=2.0)


def test_silence_does_not_load_a_model():
    def forbidden(*args, **kwargs):
        raise AssertionError("silent audio must not load a model")
    result = Transcriber(ASRConfig(), model_factory=forbidden).transcribe_array(np.zeros(100), 16000, "en")
    assert result["text"] == ""
    assert result["status"] == "no_speech"
    assert result["segments"] == []


def test_repeated_phrases_are_preserved_with_word_offsets():
    backend = FakeBackend()
    result = Transcriber(ASRConfig(), model_factory=lambda **_: backend).transcribe_array(np.ones(48000) * 0.1, 24000, "en")
    assert result["text"] == "No pain. No pain."
    assert len(backend.calls[0][0]) == 32000
    assert len(result["words"]) == 4
    for word in result["words"]:
        assert result["text"][word["char_start"]:word["char_end"]] == word["text"]
    options = backend.calls[0][1]
    assert options["vad_filter"] is True
    assert options["word_timestamps"] is True
    assert options["condition_on_previous_text"] is False


def test_swiss_standard_target_records_language_mapping():
    backend = FakeBackend()
    result = Transcriber(ASRConfig(), model_factory=lambda **_: backend).transcribe_array(np.ones(16000) * 0.1, 16000, "gsw", "standard_de")
    assert backend.calls[0][1]["language"] == "de"
    assert result["spoken_language"] == "gsw"
    assert result["experimental_language_support"]


def test_dialect_inference_requires_explicit_experimental_opt_in():
    with pytest.raises(ValueError, match="experimental"):
        Transcriber(ASRConfig()).transcribe_array(np.ones(16000) * 0.1, 16000, "gsw", "dialect")


def test_invalid_language_or_policy_never_reaches_a_model():
    with pytest.raises(ValueError):
        Transcriber(ASRConfig()).transcribe_array(np.ones(100), 16000, "klingon")
    with pytest.raises(ValueError):
        Transcriber(ASRConfig()).transcribe_array(np.ones(100), 16000, "fr", "standard_de")


def test_cpu_precision_is_validated_before_download():
    with pytest.raises(ValidationError):
        ASRConfig(device="cpu", compute_type="float16")


def test_low_scores_and_critical_tokens_have_distinct_reasons():
    words = [{"text": "No", "char_start": 0, "char_end": 2, "model_score": 0.9},
             {"text": "pain", "char_start": 3, "char_end": 7, "model_score": 0.3},
             {"text": "5", "char_start": 8, "char_end": 9, "model_score": 0.9}]
    spans = review_spans("No pain 5 mg", words, "en", 0.6)
    assert spans[0]["reasons"] == ["negation_review"]
    assert spans[1]["reasons"] == ["low_model_score"]
    assert spans[2]["reasons"] == ["numeric_review"]


def test_markup_in_audio_transcript_is_escaped():
    text = '<script>alert("x")</script>'
    html = render_highlight_html(text, [])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_wrong_offsets_fail_instead_of_highlighting_other_words():
    with pytest.raises(ValueError, match="offset"):
        review_spans("No pain", [{"text": "pain", "char_start": 0, "char_end": 4, "model_score": 0.2}], "en", 0.6)


def test_artifact_preparation_is_separate_from_model_load_and_weights_are_hashed(tmp_path, monkeypatch):
    clock = [10.0]
    (tmp_path / "model.bin").write_bytes(b"test weights")
    module = ModuleType("faster_whisper")
    utility = ModuleType("faster_whisper.utils")

    def download(*args, **kwargs):
        clock[0] += 100
        return str(tmp_path)

    def load(*args, **kwargs):
        clock[0] += 3
        return FakeBackend()

    module.WhisperModel = load
    utility.download_model = download
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", utility)
    monkeypatch.setattr("clinical_asr.transcribe.time.perf_counter", lambda: clock[0])
    result = Transcriber(ASRConfig()).transcribe_array(np.ones(16000) * 0.1, 16000)
    assert result["timing"]["artifact_preparation_seconds"] == 100
    assert result["timing"]["model_load_seconds"] == 3
    assert result["provenance"]["model_files_sha256"]["model.bin"] == hashlib.sha256(b"test weights").hexdigest()
