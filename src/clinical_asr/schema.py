"""Typed corpus records and checks against evaluation leakage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Language = Literal["en", "de", "gsw", "fr", "it", "pt"]
TargetPolicy = Literal["verbatim", "standard_de", "dialect"]


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.-]+$")
    speaker_id: str = Field(min_length=1, max_length=120)
    encounter_id: str = Field(min_length=1, max_length=120)
    audio_path: str = Field(min_length=1)
    language: Language
    target_policy: TargetPolicy = "verbatim"
    split: Literal["train", "dev", "test"]
    reference: str
    reference_kind: Literal["speech", "silence"] = "speech"
    clinical_terms: list[str] = Field(default_factory=list)
    data_use: Literal["simulated", "public", "consented"]
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    dialect: str | None = None
    reference_reviewed: bool = False

    @model_validator(mode="after")
    def validate_target(self):
        if self.language == "gsw" and self.target_policy == "verbatim":
            raise ValueError("Swiss German needs an explicit standard_de or dialect target")
        if self.language != "gsw" and self.target_policy != "verbatim":
            raise ValueError("Only Swiss German accepts standard_de or dialect targets")
        if any(not term.strip() for term in self.clinical_terms):
            raise ValueError("Clinical terms cannot be blank")
        if bool(self.reference.strip()) != (self.reference_kind == "speech"):
            raise ValueError("An empty reference requires explicitly annotated silence")
        return self


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path | str, require_audio: bool = False) -> list[Sample]:
    path = Path(path).resolve()
    samples = []
    seen_ids: set[str] = set()
    owners: dict[tuple[str, str], str] = {}
    audio_identities: dict[Path, tuple[str, str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = Sample.model_validate(json.loads(line))
        except (ValueError, TypeError) as exc:
            # Do not echo clinical content from a malformed record into logs.
            raise ValueError(f"Invalid manifest record on line {line_number}") from exc
        if sample.sample_id in seen_ids:
            raise ValueError(f"Duplicate sample ID: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        audio = Path(sample.audio_path)
        if not audio.is_absolute():
            audio = path.parent / audio
        audio = audio.resolve()
        sample = sample.model_copy(update={"audio_path": str(audio)})
        if require_audio and not audio.is_file():
            raise ValueError(f"Missing audio for sample {sample.sample_id}")
        keys = [("speaker", sample.speaker_id), ("encounter", sample.encounter_id)]
        if audio.is_file():
            if audio not in audio_identities:
                from .audio import decoded_pcm_sha256

                audio_identities[audio] = (file_sha256(audio), decoded_pcm_sha256(audio))
            container_hash, pcm_hash = audio_identities[audio]
            keys.extend([("audio", container_hash), ("decoded_audio", pcm_hash)])
        for kind, key in keys:
            if (kind, key) in owners and owners[(kind, key)] != sample.split:
                raise ValueError(f"Cross-split {kind} leakage for sample {sample.sample_id}")
            owners[(kind, key)] = sample.split
        samples.append(sample)
    if not samples:
        raise ValueError("Manifest is empty")
    return samples
