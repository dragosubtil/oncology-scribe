"""Generate bounded fictional dialogue WAVs with an explicitly supplied local engine."""

import argparse
import ctypes
import hashlib
import io
import json
import os
import platform
import re
import sys
import tempfile
import wave
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "clinical_asr" / "assets" / "dialogues"
RATE_WPM = 160
PAUSE_SECONDS = 0.35
RANDOM_SEED = 42
CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p
)


def hash_engine_data(data_dir: Path) -> tuple[list[dict[str, str]], str]:
    """Hash the entire data tree and its canonical relative-path/hash inventory."""
    paths = sorted(data_dir.rglob("*"), key=lambda path: path.relative_to(data_dir).as_posix())
    inventory = [
        {"path": path.relative_to(data_dir).as_posix(),
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths if path.is_file()
    ]
    canonical = json.dumps(inventory, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return inventory, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _output_path(root: Path, name: str) -> Path:
    target = root / name
    if target.is_symlink():
        raise ValueError(f"Refusing symlink output path: {name}")
    resolved = target.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"Output path escapes catalogue: {name}")
    if target.exists() and not target.is_file():
        raise ValueError(f"Output path is not a regular file: {name}")
    return target


def _write_output(root: Path, name: str, content: bytes):
    _output_path(root, name)
    temporary = None
    try:
        # Exclusive temporary creation plus replacement never opens a target symlink.
        with tempfile.NamedTemporaryFile(dir=root, prefix=".dialogue-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, _output_path(root, name))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class ESpeak:
    """One synchronous engine per process; eSpeak's global state is not thread safe."""

    def __init__(self, engine_dir: Path):
        engine_dir = engine_dir.resolve()
        library_name = {
            "Windows": "espeak-ng.dll",
            "Linux": "libespeak-ng.so",
            "Darwin": "libespeak-ng.dylib",
        }[platform.system()]
        self.library_path = engine_dir / library_name
        if not self.library_path.is_file() or not (engine_dir / "espeak-ng-data").is_dir():
            raise FileNotFoundError("--engine-dir must contain the eSpeak NG library and espeak-ng-data")
        self.data_files, self.data_sha256 = hash_engine_data(engine_dir / "espeak-ng-data")
        self.lib = ctypes.CDLL(str(self.library_path))
        self.lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.espeak_Initialize.restype = ctypes.c_int
        self.lib.espeak_SetSynthCallback.argtypes = [CALLBACK]
        self.lib.espeak_SetSynthCallback.restype = None
        self.lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        self.lib.espeak_SetVoiceByName.restype = ctypes.c_int
        self.lib.espeak_SetParameter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.espeak_SetParameter.restype = ctypes.c_int
        self.lib.espeak_Synth.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int,
            ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p,
        ]
        self.lib.espeak_Synth.restype = ctypes.c_int
        self.lib.espeak_Synchronize.argtypes = []
        self.lib.espeak_Synchronize.restype = ctypes.c_int
        self.lib.espeak_Terminate.argtypes = []
        self.lib.espeak_Terminate.restype = ctypes.c_int
        self.lib.espeak_Info.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
        self.lib.espeak_Info.restype = ctypes.c_char_p
        self.lib.espeak_ng_SetRandSeed.argtypes = [ctypes.c_long]
        self.lib.espeak_ng_SetRandSeed.restype = ctypes.c_int
        # The engine requires the parent of espeak-ng-data, not that data folder.
        self.sample_rate = self.lib.espeak_Initialize(2, 0, str(engine_dir).encode("utf-8"), 0x8000)
        if self.sample_rate <= 0:
            raise RuntimeError("eSpeak NG initialization failed")
        # Initialization seeds from the clock; fix the seed for byte-repeatable noise.
        self._check(self.lib.espeak_ng_SetRandSeed(RANDOM_SEED), "set random seed")
        data_path = ctypes.c_char_p()
        self.version = self.lib.espeak_Info(ctypes.byref(data_path)).decode("utf-8")
        self.chunks = []
        self.callback_error = None
        self.callback = CALLBACK(self._copy_pcm)
        self.lib.espeak_SetSynthCallback(self.callback)
        self._check(self.lib.espeak_SetParameter(1, RATE_WPM, 0), "set speaking rate")
        metadata = sorted(engine_dir.parent.glob("espeakng_loader-*.dist-info/METADATA"))
        self.loader_version = None
        self.loader_metadata_sha256 = None
        if metadata:
            content = metadata[0].read_bytes()
            self.loader_version = BytesParser().parsebytes(content)["Version"]
            self.loader_metadata_sha256 = hashlib.sha256(content).hexdigest()

    @staticmethod
    def _check(result: int, operation: str):
        if result != 0:
            raise RuntimeError(f"eSpeak NG could not {operation}: error {result}")

    def _copy_pcm(self, samples, count, _events):
        try:
            if samples and count > 0:
                self.chunks.append(ctypes.string_at(samples, count * ctypes.sizeof(ctypes.c_short)))
            return 0
        except Exception as error:
            self.callback_error = error
            return 1

    def synthesize(self, text: str, voice: str) -> bytes:
        self._check(self.lib.espeak_SetVoiceByName(voice.encode("ascii")), f"select voice {voice}")
        self.chunks.clear()
        self.callback_error = None
        encoded = text.encode("utf-8") + b"\0"
        buffer = ctypes.create_string_buffer(encoded)
        identifier = ctypes.c_uint()
        self._check(
            self.lib.espeak_Synth(buffer, len(encoded), 0, 1, 0, 1 | 0x1000,
                                  ctypes.byref(identifier), None),
            "synthesize speech",
        )
        self._check(self.lib.espeak_Synchronize(), "finish speech")
        if self.callback_error:
            raise RuntimeError("PCM callback failed") from self.callback_error
        pcm = b"".join(self.chunks)
        if not pcm or not any(pcm):
            raise RuntimeError("eSpeak NG returned empty or silent speech")
        if sys.byteorder != "little":
            from array import array

            values = array("h")
            values.frombytes(pcm)
            values.byteswap()
            pcm = values.tobytes()
        return pcm

    def metadata(self) -> dict:
        return {
            "name": "eSpeak NG",
            "version": self.version,
            "version_source": "espeak_Info",
            "license": "GPL-3.0-or-later",
            "attribution": "Jonathan Duddington, Reece H. Dunn and eSpeak NG contributors",
            "license_url": "https://github.com/espeak-ng/espeak-ng/blob/master/COPYING",
            "library_sha256": hashlib.sha256(self.library_path.read_bytes()).hexdigest(),
            "data_files": self.data_files,
            "data_sha256": self.data_sha256,
            "data_hash_serialization": "UTF-8 JSON; ASCII escapes; sorted keys; compact separators",
            "loader_package": "espeakng-loader",
            "loader_version": self.loader_version,
            "loader_metadata_sha256": self.loader_metadata_sha256,
            "loader_metadata_license": "not declared in installed METADATA",
            "rate_wpm": RATE_WPM,
            "random_seed": RANDOM_SEED,
            "audio_output_mode": "AUDIO_OUTPUT_SYNCHRONOUS",
            "initialize_options": 0x8000,
            "synth_flags": 1 | 0x1000,
            "neural_weights": False,
        }

    def close(self):
        self._check(self.lib.espeak_Terminate(), "terminate")


def generate(engine_dir: Path, output_dir: Path) -> list[dict]:
    output_dir = output_dir.resolve()
    source_files = sorted(ASSETS.glob("*.source.json"))
    if len(source_files) != 8:
        raise ValueError("Expected exactly eight fictional source scripts")
    sources = []
    targets = ["index.json"]
    for source_path in source_files:
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes.decode("utf-8"))
        identifier = source["id"]
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"(en|de|fr|it)-0[12]", identifier)
            or identifier != source_path.name[:-12]
            or source["language"] != identifier[:2]
        ):
            raise ValueError("Invalid fictional source language or id")
        sources.append((source_path, source_bytes, source))
        targets.extend([identifier + ".wav", source_path.name])
    # Check every output, including the final index, before creating or writing files.
    for name in targets:
        _output_path(output_dir, name)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = ESpeak(engine_dir)
    entries = []
    try:
        for source_path, source_bytes, source in sources:
            language = source["language"]
            identifier = source["id"]
            voices = {"doctor": language + "+m3", "patient": language + "+f3"}
            pause_frames = round(engine.sample_rate * PAUSE_SECONDS)
            pause = b"\0\0" * pause_frames
            pcm_parts = []
            cursor = 0
            turns = []
            for turn_index, turn in enumerate(source["turns"]):
                if turn_index:
                    pcm_parts.append(pause)
                    cursor += pause_frames
                voice = voices[turn["role"]]
                pcm = engine.synthesize(turn["text"], voice)
                frames = len(pcm) // 2
                turns.append({
                    "role": turn["role"], "voice": voice,
                    "start_frame": cursor, "end_frame": cursor + frames,
                    "start_seconds": cursor / engine.sample_rate,
                    "end_seconds": (cursor + frames) / engine.sample_rate,
                })
                pcm_parts.append(pcm)
                cursor += frames
            duration = cursor / engine.sample_rate
            if not 35 <= duration <= 80:
                raise ValueError(f"{identifier}: {duration:.2f}s exceeds the 35-80 second bound")
            audio_name = identifier + ".wav"
            audio_buffer = io.BytesIO()
            with wave.open(audio_buffer, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(engine.sample_rate)
                audio.writeframes(b"".join(pcm_parts))
            audio_bytes = audio_buffer.getvalue()
            _write_output(output_dir, audio_name, audio_bytes)
            if source_path.resolve() != (output_dir / source_path.name).resolve():
                _write_output(output_dir, source_path.name, source_bytes)
            entry = {key: source[key] for key in (
                "id", "label", "language", "synthetic", "fictional", "clinically_reviewed",
                "native_language_reviewed", "reference_kind", "notice", "role_labels",
            )}
            entry.update({
                "audio_file": audio_name, "source_file": source_path.name,
                "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "voices": voices, "engine": engine.metadata(),
                "sample_rate": engine.sample_rate, "channels": 1, "sample_width_bytes": 2,
                "frames": cursor, "duration_seconds": duration,
                "pause_frames": pause_frames, "turns": turns,
                "timestamp_kind": "synthesis_chunk_boundaries_including_engine_silence_not_word_alignment",
            })
            entries.append(entry)
            print(f"{identifier}: {duration:.2f}s, {engine.sample_rate} Hz, {voices}")
        entries.sort(key=lambda entry: ({"en": 0, "de": 1, "fr": 2, "it": 3}[entry["language"]], entry["id"]))
        manifest = {"schema_version": 1, "examples": entries}
        _write_output(
            output_dir, "index.json",
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        print(f"Engine: eSpeak NG {engine.version}; loader: {engine.loader_version}")
    finally:
        engine.close()
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True,
                        help="Existing directory containing the library and espeak-ng-data")
    parser.add_argument("--output-dir", type=Path, default=ASSETS,
                        help="Catalogue output directory (defaults to packaged dialogue assets)")
    args = parser.parse_args()
    generate(args.engine_dir, args.output_dir)


if __name__ == "__main__":
    main()
