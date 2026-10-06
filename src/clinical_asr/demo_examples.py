"""Verified offline synthetic fiction for demonstration, not clinical evaluation."""

import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path, PureWindowsPath

ASSETS = Path(str(files("clinical_asr").joinpath("assets", "dialogues")))
_ID = re.compile(r"[a-z]{2}-[0-9]{2}\Z")


def _asset_path(filename: str) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError("Invalid dialogue asset path")
    relative = Path(filename)
    if (
        relative.is_absolute()
        or PureWindowsPath(filename).drive
        or "\\" in filename
        or any(part in {".", ".."} for part in filename.split("/"))
    ):
        raise ValueError("Unsafe dialogue asset path")
    root = ASSETS.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("Dialogue asset path escapes catalogue")
    return resolved


def _entries() -> list[dict]:
    manifest = json.loads(_asset_path("index.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("examples"), list):
        raise ValueError("Invalid dialogue example catalogue")
    entries = manifest["examples"]
    identifiers = []
    for entry in entries:
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
            raise ValueError("Invalid dialogue example id")
        if entry.get("language") not in {"en", "de", "fr", "it"}:
            raise ValueError("Invalid dialogue example language")
        if not isinstance(entry.get("label"), str) or not entry["label"].strip():
            raise ValueError("Missing dialogue example label")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate dialogue example id")
    return entries


def choices() -> list[tuple[str, str]]:
    """Return ordered (readable label, opaque example id) pairs for a UI selector."""
    return [(entry["label"], entry["id"]) for entry in _entries()]


def _verified_asset(filename: str, expected_hash: str) -> tuple[Path, bytes]:
    path = _asset_path(filename)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"Dialogue asset SHA-256 mismatch: {path.name}")
    return path, content


def load_example(identifier: str) -> tuple[str, str, str, str]:
    """Return absolute WAV, language, labeled script, and unlabeled generation reference.

    Missing assets raise FileNotFoundError. Unknown IDs, unsafe paths and corrupt
    assets raise ValueError. References are synthetic generation inputs, not gold.
    """
    if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
        raise ValueError("Unknown dialogue example id")
    entry = next((item for item in _entries() if item["id"] == identifier), None)
    if entry is None:
        raise ValueError(f"Unknown dialogue example id: {identifier}")
    audio_path, _ = _verified_asset(entry["audio_file"], entry["audio_sha256"])
    _, source_bytes = _verified_asset(entry["source_file"], entry["source_sha256"])
    source = json.loads(source_bytes.decode("utf-8"))
    if source["id"] != identifier or source["language"] != entry["language"]:
        raise ValueError("Dialogue example source does not match catalogue")
    turns = source["turns"]
    labels = source["role_labels"]
    if {turn["role"] for turn in turns} != {"doctor", "patient"}:
        raise ValueError("Dialogue example must have two roles")
    script = "\n".join(f"{labels[turn['role']]}: {turn['text']}" for turn in turns)
    reference = "\n".join(turn["text"] for turn in turns)
    return str(audio_path), entry["language"], script, reference
