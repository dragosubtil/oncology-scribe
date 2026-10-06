"""Asset and loader contracts; no ASR or language model is used."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import wave
from array import array
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def catalogue():
    # Missing implementation is an assertion failure during the first red run.
    assert importlib.util.find_spec("clinical_asr.demo_examples") is not None, (
        "The fictional dialogue catalogue loader has not been implemented"
    )
    from clinical_asr import demo_examples

    return demo_examples


def index(module):
    return json.loads((module.ASSETS / "index.json").read_text(encoding="utf-8"))


def test_catalogue_has_eight_distinct_fictional_two_voice_examples():
    module = catalogue()
    entries = index(module)["examples"]
    options = module.choices()
    assert len(options) == len(entries) == 8
    assert len({identifier for _, identifier in options}) == 8
    assert len({label for label, _ in options}) == 8
    assert Counter(entry["language"] for entry in entries) == {"en": 2, "de": 2, "fr": 2, "it": 2}
    assert {identifier for _, identifier in options} == {entry["id"] for entry in entries}
    assert len({entry["audio_sha256"] for entry in entries}) == 8
    for entry in entries:
        assert entry["synthetic"] is True and entry["fictional"] is True
        assert entry["clinically_reviewed"] is False
        assert entry["native_language_reviewed"] is False
        assert entry["reference_kind"] == "generation_script_not_independent_gold"
        assert entry["voices"] == {
            "doctor": entry["language"] + "+m3",
            "patient": entry["language"] + "+f3",
        }
        assert entry["engine"]["name"] == "eSpeak NG"
        assert entry["engine"]["version"]
        assert entry["engine"]["license"] == "GPL-3.0-or-later"
        assert entry["engine"]["rate_wpm"] == 160


def test_every_example_loads_absolute_verified_audio_and_separate_references():
    module = catalogue()
    for entry in index(module)["examples"]:
        audio, language, script, plain = module.load_example(entry["id"])
        assert Path(audio).is_absolute() and Path(audio).is_relative_to(module.ASSETS.resolve())
        assert language == entry["language"]
        source_path = module.ASSETS / entry["source_file"]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == entry["source_sha256"]
        assert hashlib.sha256(Path(audio).read_bytes()).hexdigest() == entry["audio_sha256"]
        assert script == "\n".join(
            f"{source['role_labels'][turn['role']]}: {turn['text']}" for turn in source["turns"]
        )
        assert plain == "\n".join(turn["text"] for turn in source["turns"])
        assert "doctor:" not in plain and "patient:" not in plain
        assert {turn["role"] for turn in source["turns"]} == {"doctor", "patient"}
        assert source["role_labels"] == {
            "en": {"doctor": "Doctor", "patient": "Patient"},
            "de": {"doctor": "Arzt", "patient": "Patientin"},
            "fr": {"doctor": "Médecin", "patient": "Patiente"},
            "it": {"doctor": "Medico", "patient": "Paziente"},
        }[language]


def test_actual_waves_have_speech_and_sample_accurate_nonoverlapping_turns():
    module = catalogue()
    total_bytes = sum(path.stat().st_size for path in module.ASSETS.rglob("*") if path.is_file())
    assert total_bytes < 35 * 1024 * 1024
    for entry in index(module)["examples"]:
        with wave.open(str(module.ASSETS / entry["audio_file"]), "rb") as audio:
            assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
            assert audio.getframerate() == entry["sample_rate"] == 22050
            assert audio.getcomptype() == "NONE"
            frames = audio.getnframes()
            samples = array("h", audio.readframes(frames))
        assert 35 <= frames / entry["sample_rate"] <= 80
        assert entry["duration_seconds"] == pytest.approx(frames / entry["sample_rate"])
        assert entry["frames"] == frames
        source = json.loads((module.ASSETS / entry["source_file"]).read_text(encoding="utf-8"))
        assert len(entry["turns"]) == len(source["turns"])
        previous_end = 0
        for turn, source_turn in zip(entry["turns"], source["turns"], strict=True):
            start, end = turn["start_frame"], turn["end_frame"]
            assert previous_end <= start < end <= frames
            assert turn["role"] == source_turn["role"]
            assert turn["voice"] == entry["voices"][turn["role"]]
            assert turn["start_seconds"] == pytest.approx(start / entry["sample_rate"])
            assert turn["end_seconds"] == pytest.approx(end / entry["sample_rate"])
            assert max(abs(value) for value in samples[start:end]) > 100
            assert sum(value != 0 for value in samples[start:end]) > (end - start) * 0.1
            assert all(value == 0 for value in samples[previous_end:start])
            previous_end = end


@pytest.mark.parametrize("identifier", ["unknown", "../en-01", "..\\en-01", "/en-01", "C:\\en-01", "", None])
def test_unknown_and_traversal_ids_are_rejected(identifier):
    with pytest.raises(ValueError, match="example"):
        catalogue().load_example(identifier)


@pytest.fixture
def copied_assets(tmp_path, monkeypatch):
    module = catalogue()
    destination = tmp_path / "dialogues"
    shutil.copytree(module.ASSETS, destination)
    monkeypatch.setattr(module, "ASSETS", destination)
    return module, index(module)["examples"][0]


@pytest.mark.parametrize("field", ["audio_file", "source_file"])
def test_missing_assets_fail_explicitly(copied_assets, field):
    module, entry = copied_assets
    (module.ASSETS / entry[field]).unlink()
    with pytest.raises(FileNotFoundError):
        module.load_example(entry["id"])


@pytest.mark.parametrize("field", ["audio_file", "source_file"])
def test_tampered_assets_fail_hash_validation(copied_assets, field):
    module, entry = copied_assets
    with (module.ASSETS / entry[field]).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        module.load_example(entry["id"])


@pytest.mark.parametrize("field", ["audio_file", "source_file"])
@pytest.mark.parametrize("unsafe", ["../outside.wav", "..\\outside.wav", "C:/outside.wav", "/outside.wav"])
def test_manifest_paths_cannot_escape_assets(copied_assets, field, unsafe):
    module, entry = copied_assets
    manifest = index(module)
    manifest["examples"][0][field] = unsafe
    (module.ASSETS / "index.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        module.load_example(entry["id"])


def test_generator_repeats_actual_offline_audio_and_manifest(tmp_path):
    engine_dir = ROOT / ".build" / "tts-deps" / "espeakng_loader"
    if not (engine_dir / "espeak-ng-data").is_dir() or not any(
        (engine_dir / name).is_file()
        for name in ("espeak-ng.dll", "libespeak-ng.so", "libespeak-ng.dylib")
    ):
        pytest.skip("Optional local eSpeak NG engine/data absent; no download attempted")
    module = catalogue()
    generator = ROOT / "tools" / "generate_dialogues.py"
    assert generator.is_file(), "Repeatable offline generator is missing"
    output = tmp_path / "regenerated"
    result = subprocess.run(
        [sys.executable, str(generator), "--engine-dir",
         str(engine_dir),
         "--output-dir", str(output)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    regenerated = json.loads((output / "index.json").read_text(encoding="utf-8"))
    original = index(module)
    assert regenerated == original
    for entry in original["examples"]:
        assert hashlib.sha256((output / entry["audio_file"]).read_bytes()).hexdigest() == entry["audio_sha256"]


def generator_module():
    spec = importlib.util.spec_from_file_location("dialogue_generator", ROOT / "tools" / "generate_dialogues.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalogue_records_complete_canonical_engine_data_provenance():
    for entry in index(catalogue())["examples"]:
        engine = entry["engine"]
        assert "data_files" in engine, "Voice/dictionary/phoneme file hashes are missing"
        file_hashes = engine["data_files"]
        paths = [item["path"] for item in file_hashes]
        assert paths == sorted(paths) and len(set(paths)) == len(paths)
        assert {"en_dict", "de_dict", "fr_dict", "it_dict", "phondata", "phonindex", "phontab"} <= set(paths)
        assert {"voices/!v/m3", "voices/!v/f3"} <= set(paths)
        assert all(len(item["sha256"]) == 64 and "\\" not in item["path"] for item in file_hashes)
        canonical = json.dumps(file_hashes, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == engine["data_sha256"]
        engine_dir = ROOT / ".build" / "tts-deps" / "espeakng_loader" / "espeak-ng-data"
        if engine_dir.is_dir():
            expected = [
                {"path": path.relative_to(engine_dir).as_posix(),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(engine_dir.rglob("*"), key=lambda path: path.relative_to(engine_dir).as_posix())
                if path.is_file()
            ]
            assert file_hashes == expected


def test_engine_data_change_changes_recorded_fingerprint(tmp_path):
    data = tmp_path / "espeak-ng-data"
    (data / "voices" / "!v").mkdir(parents=True)
    (data / "phondata").write_bytes(b"phonemes")
    (data / "voices" / "!v" / "m3").write_bytes(b"voice")
    generator = generator_module()
    assert hasattr(generator, "hash_engine_data"), "Engine data fingerprinting is missing"
    before_files, before_hash = generator.hash_engine_data(data)
    (data / "phondata").write_bytes(b"changed phonemes")
    after_files, after_hash = generator.hash_engine_data(data)
    assert before_hash != after_hash
    assert before_files[0]["sha256"] != after_files[0]["sha256"]
    assert before_files[1] == after_files[1]


@pytest.mark.parametrize("filename", ["de-01.wav", "de-01.source.json", "index.json"])
@pytest.mark.parametrize("destination", ["outside", "inside", "dangling"])
def test_generator_preflights_all_targets_and_refuses_file_symlinks(tmp_path, filename, destination):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "de-02.wav"
    sentinel.write_bytes(b"must not overwrite any earlier target")
    target = output / "local-target" if destination == "inside" else tmp_path / "outside-target"
    if destination != "dangling":
        target.write_bytes(b"must not overwrite symlink destination")
    try:
        (output / filename).symlink_to(target)
    except OSError as error:
        pytest.skip(f"File symlinks unavailable on this Windows account: {error}")
    before = {path.name for path in output.iterdir()}
    # Safety preflight must complete even without loading the optional engine.
    with pytest.raises(ValueError, match="symlink|path"):
        generator_module().generate(tmp_path / "absent-engine", output)
    assert sentinel.read_bytes() == b"must not overwrite any earlier target"
    assert {path.name for path in output.iterdir()} == before
    if destination != "dangling":
        assert target.read_bytes() == b"must not overwrite symlink destination"
    else:
        assert not target.exists()


def test_generator_preflights_unsafe_index_before_loading_engine_or_overwriting_audio(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "index.json").mkdir()
    (output / "de-01.wav").write_bytes(b"existing audio must remain untouched")
    with pytest.raises(ValueError, match="path"):
        generator_module().generate(tmp_path / "absent-engine", output)
    assert (output / "de-01.wav").read_bytes() == b"existing audio must remain untouched"
    assert sorted(path.name for path in output.iterdir()) == ["de-01.wav", "index.json"]
