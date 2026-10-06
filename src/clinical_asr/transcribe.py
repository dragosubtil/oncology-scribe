"""Lazy faster-whisper adapter preserving raw text, repeats and word evidence."""

from __future__ import annotations

import importlib.metadata
import platform
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .audio import PreparedAudio, load_audio, prepare_audio
from .schema import file_sha256


class ASRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(default="small", min_length=1)
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = Field(default=4, ge=1, le=32)
    beam_size: int = Field(default=5, ge=1, le=10)
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    initial_prompt: str | None = None
    revision: str | None = None
    download_root: str = "models"
    local_files_only: bool = False
    allow_experimental_dialect: bool = False
    max_duration_seconds: float = Field(default=1800, gt=0, le=14400)

    @model_validator(mode="after")
    def validate_device(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Select cpu or cuda explicitly for reproducible experiments")
        types = {"int8", "float32"} if self.device == "cpu" else {"int8_float16", "float16", "float32"}
        if self.compute_type not in types:
            raise ValueError("Compute type is incompatible with the selected device")
        return self


def _validate_policy(language: str, target_policy: str, experimental: bool):
    if language not in {"en", "de", "gsw", "fr", "it", "pt"}:
        raise ValueError("Unsupported spoken language")
    if language == "gsw":
        if target_policy not in {"standard_de", "dialect"}:
            raise ValueError("Swiss German requires an explicit target policy")
        if target_policy == "dialect" and not experimental:
            raise ValueError("Dialect inference needs experimental opt-in and appropriate references")
    elif target_policy != "verbatim":
        raise ValueError("Non-Swiss languages require verbatim transcription")


class Transcriber:
    def __init__(self, config: ASRConfig, model_factory=None):
        self.config = config
        self._factory = model_factory
        self._model = None
        self._resolved_revision = None
        self._load_seconds = 0.0
        self._artifact_seconds = 0.0
        self._model_hashes = {}

    def _load(self):
        if self._model is not None:
            return
        if self._factory is not None:
            started = time.perf_counter()
            self._model = self._factory(model_size_or_path=self.config.model)
            self._resolved_revision = "injected-backend"
        else:
            from faster_whisper import WhisperModel
            from faster_whisper.utils import download_model

            prepared = time.perf_counter()
            location = Path(self.config.model)
            if location.is_dir():
                model_path = str(location.resolve())
                self._resolved_revision = "local-directory"
            else:
                model_path = download_model(
                    self.config.model, cache_dir=self.config.download_root,
                    revision=self.config.revision, local_files_only=self.config.local_files_only,
                )
                self._resolved_revision = Path(model_path).name
            for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json",
                         "vocabulary.txt", "preprocessor_config.json"):
                artifact = Path(model_path) / name
                if artifact.is_file():
                    self._model_hashes[name] = file_sha256(artifact)
            self._artifact_seconds = time.perf_counter() - prepared
            started = time.perf_counter()
            self._model = WhisperModel(model_path, device=self.config.device,
                                       compute_type=self.config.compute_type,
                                       cpu_threads=self.config.cpu_threads)
        self._load_seconds = time.perf_counter() - started

    def transcribe_array(self, samples, sample_rate: int, language: str = "en",
                         target_policy: str = "verbatim") -> dict:
        _validate_policy(language, target_policy, self.config.allow_experimental_dialect)
        audio = prepare_audio(samples, sample_rate, self.config.max_duration_seconds)
        return self._transcribe(audio, language, target_policy)

    def transcribe_file(self, path: Path | str, language: str = "en",
                        target_policy: str = "verbatim") -> dict:
        _validate_policy(language, target_policy, self.config.allow_experimental_dialect)
        return self._transcribe(load_audio(path, self.config.max_duration_seconds), language, target_policy)

    def _transcribe(self, audio: PreparedAudio, language: str, target_policy: str) -> dict:
        warnings = list(audio.warnings)
        if language == "gsw":
            warnings.append("gsw_maps_to_de_token; target_output_requires_human_validation")
            if target_policy == "dialect":
                warnings.append("dialect_output_not_guaranteed_by_whisper")
        result = {
            "schema_version": 1, "text": "", "segments": [], "words": [],
            "spoken_language": language, "target_policy": target_policy,
            "backend_language": "de" if language == "gsw" else language,
            "experimental_language_support": language == "gsw",
            "status": "no_speech" if audio.is_silent else "transcribed",
            "audio": {"duration_seconds": audio.duration_seconds, "sample_rate": 16000,
                      "rms": audio.rms, "prepared_pcm_sha256": audio.sha256},
            "warnings": warnings, "config": self.config.model_dump(),
            "provenance": {"backend": "faster-whisper", "python": platform.python_version(),
                           "platform": platform.platform(), "model_revision": None,
                           "model_files_sha256": {}},
            "timing": {"artifact_preparation_seconds": 0.0, "model_load_seconds": 0.0,
                       "inference_seconds": 0.0, "rtf": None},
            "requires_human_review": True,
        }
        if audio.is_silent:
            return result
        first_load = self._model is None
        self._load()
        result["timing"]["model_load_seconds"] = self._load_seconds if first_load else 0.0
        result["timing"]["artifact_preparation_seconds"] = self._artifact_seconds if first_load else 0.0
        result["provenance"]["model_revision"] = self._resolved_revision
        result["provenance"]["model_files_sha256"] = dict(self._model_hashes)
        try:
            result["provenance"]["backend_version"] = importlib.metadata.version("faster-whisper")
        except importlib.metadata.PackageNotFoundError:
            result["provenance"]["backend_version"] = "not-installed"
        started = time.perf_counter()
        segments, info = self._model.transcribe(
            audio.samples, language=result["backend_language"], task="transcribe",
            beam_size=self.config.beam_size, temperature=0.0,
            vad_filter=self.config.vad_filter, word_timestamps=True,
            condition_on_previous_text=self.config.condition_on_previous_text,
            initial_prompt=self.config.initial_prompt,
        )
        chunks = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            char_base = sum(len(chunk) for chunk in chunks) + len(chunks)
            chunks.append(text)
            result["segments"].append({"start": segment.start, "end": segment.end,
                                       "text": text, "avg_logprob": segment.avg_logprob,
                                       "no_speech_score": segment.no_speech_prob})
            cursor = 0
            for word in segment.words or []:
                token = word.word.strip()
                if not token:
                    continue
                offset = text.find(token, cursor)
                if offset < 0:
                    warnings.append("unaligned_word_score")
                    continue
                cursor = offset + len(token)
                result["words"].append({"text": token, "start": word.start, "end": word.end,
                                        "char_start": char_base + offset,
                                        "char_end": char_base + cursor,
                                        "model_score": word.probability})
        elapsed = time.perf_counter() - started
        result["text"] = " ".join(chunks)
        result["timing"]["inference_seconds"] = elapsed
        result["timing"]["rtf"] = elapsed / audio.duration_seconds
        result["detected_language"] = info.language
        result["detected_language_score"] = info.language_probability
        result["audio"]["duration_after_vad_seconds"] = info.duration_after_vad
        if not result["text"]:
            result["status"] = "no_speech_detected"
        return result
