"""Validate audio and resample without amplifying silence or discarding speech."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class PreparedAudio:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    rms: float
    is_silent: bool
    warnings: tuple[str, ...]
    sha256: str


def decoded_pcm_sha256(path: Path | str) -> str:
    """Container-independent identity, streamed without inference duration limits."""
    digest = hashlib.sha256()
    with sf.SoundFile(path) as stream:
        digest.update(f"decoded-f32le-v1:{stream.samplerate}:{stream.channels}:{stream.frames}\n".encode("ascii"))
        while True:
            block = stream.read(65536, dtype="float32", always_2d=True)
            if not len(block):
                break
            digest.update(np.ascontiguousarray(block, dtype="<f4").tobytes())
    return digest.hexdigest()


def prepare_audio(samples, sample_rate: int, max_duration_seconds: float = 1800) -> PreparedAudio:
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float, np.integer, np.floating))
        or not math.isfinite(sample_rate)
        or sample_rate <= 0
        or int(sample_rate) != sample_rate
    ):
        raise ValueError("Invalid sample rate: expected a positive integer")
    array = np.asarray(samples)
    if array.ndim not in (1, 2) or (array.ndim == 2 and not 1 <= array.shape[1] <= 32):
        raise ValueError("Invalid audio channel layout")
    if array.dtype.kind not in "fiu":
        raise ValueError("Audio must contain finite numeric samples")
    if not np.isfinite(array).all():
        raise ValueError("Audio must contain finite samples")
    if array.shape[0] / sample_rate > max_duration_seconds:
        raise ValueError(f"Audio exceeds the {max_duration_seconds:g} second limit")
    if array.dtype.kind in "iu":
        limits = np.iinfo(array.dtype)
        if array.dtype.kind == "u":
            midpoint = (limits.max + 1) / 2
            array = (array.astype(np.float32) - midpoint) / midpoint
        else:
            array = array.astype(np.float32) / max(abs(limits.min), limits.max)
    else:
        array = array.astype(np.float32)
    warnings = []
    if array.size and np.max(np.abs(array)) > 1.0:
        raise ValueError("Floating audio must be in the [-1, 1] PCM range")
    if array.size and np.mean(np.abs(array) >= 0.999) > 0.001:
        warnings.append("possible_clipping")
    if array.ndim == 2:
        mono = array.mean(axis=1)
        if len(array):
            channel_rms = np.sqrt(np.mean(array.astype(np.float64) ** 2, axis=0)).max()
            mono_rms = np.sqrt(np.mean(mono.astype(np.float64) ** 2))
            if channel_rms > 1e-8 and mono_rms < 0.05 * channel_rms:
                raise ValueError("Severe channel cancellation: select or repair channels before ASR")
        array = mono
        warnings.append("downmixed_to_mono")
    if len(array) and sample_rate != 16000:
        divisor = math.gcd(int(sample_rate), 16000)
        array = resample_poly(array, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
        warnings.append("resampled_to_16000")
    array = np.ascontiguousarray(array, dtype=np.float32)
    rms = float(np.sqrt(np.mean(array.astype(np.float64) ** 2))) if len(array) else 0.0
    if 0 < rms < 0.001:
        warnings.append("low_volume")
    return PreparedAudio(
        samples=array, sample_rate=16000, duration_seconds=len(array) / 16000,
        rms=rms, is_silent=not len(array) or rms < 1e-8, warnings=tuple(warnings),
        sha256=hashlib.sha256(array.tobytes()).hexdigest(),
    )


def load_audio(path: Path | str, max_duration_seconds: float = 1800) -> PreparedAudio:
    path = Path(path)
    if not path.is_file():
        raise ValueError("Audio file does not exist")
    info = sf.info(path)
    if info.duration > max_duration_seconds:
        raise ValueError(f"Audio exceeds the {max_duration_seconds:g} second limit")
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return prepare_audio(samples, sample_rate, max_duration_seconds)
